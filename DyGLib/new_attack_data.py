import random
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from distutils.dir_util import copy_tree
from collections import Counter, defaultdict
from scipy import stats
from scipy.spatial import cKDTree
from typing import Dict, Tuple, Set

from tqdm.auto import tqdm

from utils.adv_utils import AdvNegativeEdgeSampler as AdversarialEdgeSampler
from preprocess_data.sparsify_data import EL_sparsify
from preprocess_data.preprocess_data import check_data

save_location = rf'/raid/t2/TGN_adv/' # '/home/t2/TGN_adv'
scratch_location = rf'/raid/t2/TGN_adv/DyGlib'


def sample_timestamp(sparsified_graph, current_time, next_time, rng, time_window=1200):
    df_sorted = sparsified_graph.sort_values('ts').reset_index(drop=True)
    inter_event_times = np.diff(df_sorted['ts'])
    kde = stats.gaussian_kde(inter_event_times)
    
    if next_time is None:
        next_time = current_time + time_window

    # Sample from KDE and clip to ensure it's within the allowed range
    sampled_delta = kde.resample(1, rng)[0][0]
    sampled_delta = np.clip(sampled_delta, 0, min(time_window, next_time - current_time))

    # Add some randomness to avoid exact KDE sampling
    jitter = rng.uniform(0, min(time_window, next_time - current_time) * 0.1)

    return current_time + sampled_delta + jitter


class TemporalSampler:
    def __init__(self, original_graph, seed=None, time_window=1200):
        self.original_graph = original_graph
        self.rng = np.random.default_rng(seed)
        self.time_window = time_window
        self.min_time = original_graph['ts'].min()
        self.max_time = original_graph['ts'].max()
        self.initialize_kde()
        self.initialize_kd_tree()

    def initialize_kde(self):
        df_sorted = self.original_graph.sort_values('ts')
        inter_event_times = np.diff(df_sorted['ts'])
        self.kde = stats.gaussian_kde(inter_event_times)

    def initialize_kd_tree(self):
        self.temporal_kd_tree = cKDTree(self.original_graph['ts'].values.reshape(-1, 1))

    def assign_timestamps(self, num_samples):
        # Generate a large batch of samples from KDE
        kde_samples = self.kde.resample(num_samples, self.rng)[0]

        # Create an array of cumulative times
        cumulative_times = np.cumsum(kde_samples) + self.min_time

        # Clip the times to ensure they're within the graph's time range
        cumulative_times = np.clip(cumulative_times, self.min_time, self.max_time)

        # Add jitter to avoid exact KDE sampling
        jitter = self.rng.uniform(0, self.time_window * 0.1, num_samples)
        sampled_timestamps = cumulative_times + jitter

        # Ensure the timestamps are sorted and within the original time range
        sampled_timestamps.sort()
        sampled_timestamps = np.clip(sampled_timestamps, self.min_time, self.max_time)

        return sampled_timestamps

    def find_next_events(self, timestamps):
        _, indices = self.temporal_kd_tree.query(timestamps.reshape(-1, 1), k=1)
        return self.original_graph['ts'].values[indices]


class DegreePreservingGraphProcessor:
    def __init__(self, graph_df: pd.DataFrame, full_graph: pd.DataFrame, edge_features: np.ndarray, node_features: np.ndarray,
                 sparsification_strategy: str = 'random', sparsification_ratio: float = 0.7,
                 negative_sample_strategy: str = 'random', new_test_node_set: set = set(), 
                 num_negative_samples: int = 5,
                 dataset_name:str='wikipedia', seed: int = None):
        self.original_graph = graph_df
        self.full_graph = full_graph
        self.edge_features = edge_features
        self.node_features = node_features
        self.sparsification_strategy = sparsification_strategy
        self.sparsification_ratio = sparsification_ratio
        self.negative_sample_strategy = negative_sample_strategy
        self.new_test_node_set = new_test_node_set
        self.num_negative_samples = num_negative_samples
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.dataset_name = dataset_name
        self.unique_dst_node_ids = np.unique(full_graph['i'])
        self.earliest_time = min(0, full_graph['ts'].min())
        self.latest_time = full_graph['ts'].max()

        self.node_degrees: Dict[int, int] = defaultdict(int)
        self.edges: Set[Tuple[int, int]] = set()
        self.temporal_kd_tree = None
        
        print(f'Processing {self.dataset_name} {self.sparsification_strategy} {self.negative_sample_strategy} {self.sparsification_ratio}')

    def process_graph(self):
        self.compute_original_degrees()
        self.sparsify_graph()
        self.add_negative_samples()
        self.preserve_degrees()
        return self.create_final_graph()

    def compute_original_degrees(self):
        for _, row in self.original_graph.iterrows():
            self.node_degrees[row['u']] += 1
            self.node_degrees[row['i']] += 1

    def sparsify_graph(self):
        self.sparsified_graph, self.sparsified_edge_features = EL_sparsify(
            self.original_graph, self.edge_features,
            strategy=self.sparsification_strategy, upto=self.sparsification_ratio,
            dataset_name=self.dataset_name
        )
        print(f'original graph is {len(self.original_graph)}', end=' ')
        print(f'sparsified graph is {len(self.sparsified_graph)}')
        
        # DONE: remove test node test <--- Do we really need  it?  This reduces dataset. Should we keep it? --> Yes!!
        
        self.sparsified_graph = self.sparsified_graph[~self.sparsified_graph['u'].isin(self.new_test_node_set)]
        self.sparsified_graph = self.sparsified_graph[~self.sparsified_graph['i'].isin(self.new_test_node_set)]
        
        print(f'Checkpoint: sparsified graph is {len(self.sparsified_graph)}')
        
        for _, row in self.sparsified_graph.iterrows():
            self.edges.add((row['u'], row['i']))

    def add_negative_samples(self):

        # Early negative sampling
        early_samples = self.add_early_negative_samples()
        
        print('*'*11, ' early sampling done ', '*'*11)
        
        remaining_samples = self.add_remaining_negative_samples()
        
        print('*'*11, ' remaining sampling done ', '*'*11)
        

        self.negative_samples = pd.concat([early_samples, remaining_samples], ignore_index=True)

    def add_early_negative_samples(self):
        
        early_start_time = min(0, self.sparsified_graph['ts'].iloc[0])  # This is correct.
        early_end_time = self.sparsified_graph['ts'].iloc[1]  # This is correct.
        
        # print(f'{early_start_time = }, {early_end_time = }')
        
        early_graph = self.original_graph[self.original_graph['ts'] < early_end_time]
        
        # print(f'\t{len(early_graph)}')
        
        original_edges = set(zip(self.original_graph['u'], self.original_graph['i']))
        
        early_src_nodes = early_graph['u'].values
        early_dst_nodes = early_graph['i'].values
        early_timestamps = early_graph['ts'].values
        early_labels = early_graph[early_graph['ts']<early_end_time]['label'].values
        early_idx = early_graph[early_graph['ts']<early_end_time]['idx'].values
        
        sampler = AdversarialEdgeSampler(
            src_node_ids=early_src_nodes,
            dst_node_ids=early_dst_nodes,
            interact_times=early_timestamps,
            original_edges=original_edges,
            last_observed_time=early_end_time,
            negative_sample_strategy='random',  # always random
            seed=self.seed
        )

        # Assuming early_src_nodes is your NumPy ndarray
        unique_values, counts = np.unique(early_src_nodes, return_counts=True)

        # Print the results
        value_counts = dict(zip(unique_values, counts))
        
        early_samples = []
        for u, count in tqdm(value_counts.items(), desc='Processing early samples', leave=False):
            # print(f'{u = } {count = }')
            neg_src, neg_dst = sampler.sample(
                size=count,
                batch_src_node_ids=np.array(u),
                batch_dst_node_ids=early_dst_nodes,
                current_batch_start_time=early_start_time,
                current_batch_end_time=early_end_time
            )
            # print(set(neg_src), len(neg_src), len(neg_dst))
            # cnt  = 0
            assert len(neg_dst) == count, f'Discrepancy in negative sampling generation for node {u}.'
            for src, dst in zip(neg_src, neg_dst):
                early_samples.append([src, dst])
                # cnt+=1

        early_df = pd.DataFrame(early_samples, columns=['u', 'i'])
        # Sort the DataFrame based on the 'u' column
        # early_df = early_df.sort_values(by=['u'])   # sorting should not be done otherwise all the nodes would colasce to consecutive timestamps
        early_df['ts'] = early_timestamps  # np.sort(early_timestamps)
        early_df['label'] = early_labels
        early_df['idx'] = early_idx
        early_df['is_negative'] = 1
        return early_df

    def add_remaining_negative_samples(self):
        start_time, end_time = self.original_graph['ts'].min(), self.original_graph['ts'].max()  # this is correct
        self.temporal_sampler = TemporalSampler(self.sparsified_graph, seed=0)
        
        
        # Remaining negative sampling
        self.sampler = AdversarialEdgeSampler(
            src_node_ids=self.full_graph['u'].values,
            dst_node_ids=self.full_graph['i'].values,
            interact_times=self.full_graph['ts'].values,
            last_observed_time=max(self.full_graph['ts'].values),
            negative_sample_strategy=self.negative_sample_strategy,
            seed=self.seed)
        
        # Extract arrays from DataFrame for batch processing
        batch_src_node_ids = self.sparsified_graph['u'].to_numpy()
        batch_dst_node_ids = self.sparsified_graph['i'].to_numpy()
        
        remaining_samples = self.sampler.sample(
            size=self.num_negative_samples * len(self.sparsified_graph),
            batch_src_node_ids=batch_src_node_ids,
            batch_dst_node_ids=batch_dst_node_ids,
            current_batch_start_time=start_time,
            current_batch_end_time=end_time
        )
        
        remaining_df = pd.DataFrame(zip(*remaining_samples), columns=['u', 'i'])
        remaining_df['ts'] = self.temporal_sampler.assign_timestamps(len(remaining_df))  # self.assign_timestamps(len(remaining_df))
        remaining_df['is_negative'] = 1
        return remaining_df

    def assign_timestamps(self, num_samples):
        # TODO: Add timesample to follow C3.
        if self.temporal_kd_tree is None:
            self.temporal_kd_tree = cKDTree(self.original_graph['ts'].values.reshape(-1, 1))
        
        random_times = self.rng.uniform(
            self.original_graph['ts'].min(),
            self.original_graph['ts'].max(),
            num_samples
        )
        _, indices = self.temporal_kd_tree.query(random_times.reshape(-1, 1))
        return self.original_graph['ts'].values[indices]

    def old_preserve_degrees(self):
        current_degrees = defaultdict(int)
        for _, row in pd.concat([self.sparsified_graph, self.negative_samples]).iterrows():
            current_degrees[row['u']] += 1
            current_degrees[row['i']] += 1

        degree_deficits = {node: self.node_degrees[node] - current_degrees[node] 
                           for node in self.node_degrees}

        # Add edges for nodes with deficits
        deficit_nodes = [node for node, deficit in degree_deficits.items() if deficit > 0]
        new_negative_samples = []
        all_nodes = list(self.node_degrees.keys())
        
        while deficit_nodes:
            u = deficit_nodes.pop(0)
            # v = self.rng.choice([n for n in deficit_nodes if n != u])
            possible_v = [n for n in deficit_nodes if n != u]  # doesnt take care of bipartiteness of a graph
        
            if possible_v:
                v = self.rng.choice(possible_v)
            else:
                # If no suitable v in deficit_nodes, choose from all nodes
                v = self.rng.choice([n for n in all_nodes if n != u])
                
            new_negative_samples.append({
                'u': u,
                'i': v, 
                'ts': self.assign_timestamps(1)[0],
                'is_negative': 1
            })
            
            degree_deficits[u] -= 1
            degree_deficits[v] -= 1
            if degree_deficits[v] == 0:
                deficit_nodes.remove(v)
            if degree_deficits[u] > 0:
                deficit_nodes.append(u)
                
        # Add new negative samples to the existing ones
        self.negative_samples = pd.concat([self.negative_samples, pd.DataFrame(new_negative_samples)], ignore_index=True)
        
        # Remove excess edges for nodes with surpluses
        for node, surplus in degree_deficits.items():
            if surplus < 0:
                excess_mask = (self.negative_samples['u'] == node) | (self.negative_samples['i'] == node)
                excess_edges = self.negative_samples[excess_mask]
                
                # Ensure we're not trying to remove more edges than available
                num_to_remove = min(-surplus, len(excess_edges))
                
                if num_to_remove > 0:
                    edges_to_remove = excess_edges.sample(n=num_to_remove)
                    self.negative_samples = self.negative_samples.drop(edges_to_remove.index)
                
        self.negative_samples = self.negative_samples.reset_index(drop=True)

    
    def preserve_degrees(self):
        # Counter(E['u'])
        print('*'*10, f'    Inside preserve_degrees    ', '*'*10)
        
        attacked_df = pd.concat([self.sparsified_graph, self.negative_samples])
        original_degrees = Counter(self.full_graph['u'])
        current_degrees = Counter(attacked_df['u'])
        
        deficit_nodes = {node: degree - current_degrees[node] for node, degree in original_degrees.items()}
        
        print(f'{len(deficit_nodes) = }')
        
        new_negative_samples = []
        
        # counter = 0
        
        for node, degree in deficit_nodes.items():
            print(f'processing {node = } {degree = }')
            
            # if counter > 5:
            #     print('*'*10, f'breaking as {counter = }', '*'*10)
            #     exit()
            # counter+=1
            
            if degree > 0:
                # generate more sample for this node 
                neg_src, neg_dst = self.sampler.sample(
                    size=degree,
                    batch_src_node_ids=np.atleast_1d(np.array(node)),
                    batch_dst_node_ids=self.unique_dst_node_ids,
                    current_batch_start_time=self.earliest_time,
                    current_batch_end_time=self.latest_time
                )
                
                print(f'\t{len(neg_src) = }, {degree = }')
                
                for u, v in zip(neg_src, neg_dst):
                    new_negative_samples.append({
                    'u': u,
                    'i': v, 
                    'ts': self.temporal_sampler.assign_timestamps(degree),  # self.assign_timestamps(1)[0],
                    'is_negative': 1
                })
            elif degree < 0:
                excess_mask = (self.negative_samples['u'] == node) | (self.negative_samples['i'] == node)
                excess_edges = self.negative_samples[excess_mask]
                # remove randomly upto degree
                 # Ensure we're not trying to remove more edges than available
                num_to_remove = min(-degree, len(excess_edges))
                print('\t', num_to_remove, ' degrees will be removed')
                
                if num_to_remove > 0:
                    edges_to_remove = excess_edges.sample(n=num_to_remove)
                    self.negative_samples = self.negative_samples.drop(edges_to_remove.index)
            else:
                continue
        
        # add new negative samples into negative samples
        self.negative_samples = pd.concat([self.negative_samples, pd.DataFrame(new_negative_samples)], ignore_index=True)
        self.negative_samples = self.negative_samples.reset_index(drop=True)
        
    
    def create_final_graph_(self):
        self.final_graph = pd.concat([self.sparsified_graph, self.negative_samples], ignore_index=True)
        self.final_graph = self.final_graph.sort_values('ts').reset_index(drop=True)
        
        # Ensure the number of edges matches the original graph
        if len(self.final_graph) > len(self.original_graph):
            self.final_graph = self.final_graph.sample(n=len(self.original_graph))
        elif len(self.final_graph) < len(self.original_graph):
            additional_needed = len(self.original_graph) - len(self.final_graph)
            additional_samples = self.add_remaining_negative_samples().sample(n=additional_needed)
            self.final_graph = pd.concat([self.final_graph, additional_samples], ignore_index=True)
        
        self.final_graph = self.final_graph.sort_values('ts').reset_index(drop=True)
        return self.final_graph
    
    def create_final_graph(self):
        # Concatenate sparsified graph and negative samples
        self.final_graph = pd.concat([self.sparsified_graph, self.negative_samples], ignore_index=True)
        
        # Handle potential NaN values in the 'ts' column
        self.final_graph['ts'] = pd.to_numeric(self.final_graph['ts'], errors='coerce')
        
        # Sort the DataFrame, handling NaN values
        self.final_graph = self.final_graph.sort_values('ts', na_position='last').reset_index(drop=True)
        
        # Remove any rows with NaN timestamps
        self.final_graph = self.final_graph.dropna(subset=['ts'])
        
        # Ensure the number of edges matches the original graph
        original_length = len(self.original_graph)
        current_length = len(self.final_graph)
        
        if current_length > original_length:
            self.final_graph = self.final_graph.head(original_length)
        elif current_length < original_length:
            additional_needed = original_length - current_length
            additional_samples = self.add_remaining_negative_samples()
            additional_samples = additional_samples.sample(n=min(additional_needed, len(additional_samples)), replace=True)
            self.final_graph = pd.concat([self.final_graph, additional_samples], ignore_index=True)
        
        # Final sort and reset index
        self.final_graph = self.final_graph.sort_values('ts', na_position='last').reset_index(drop=True)
        
        return self.final_graph

def load_data(dataset_name):
    graph = pd.read_csv(f'{scratch_location}/processed_data/{dataset_name}/ml_{dataset_name}.csv')
    edge_features = np.load(f'{scratch_location}/processed_data/{dataset_name}/ml_{dataset_name}.npy')
    node_features = np.load(f'{scratch_location}/processed_data/{dataset_name}/ml_{dataset_name}_node.npy')
    return graph, edge_features, node_features

def process_data(dataset_name, strategy, upto, negative_sample_strategy, val_ratio=0.15, test_ratio=0.15):
    graph, edge_features, node_features = load_data(dataset_name)
    
    # DONE: Add train/val/test split and attack only train part
    
    val_time, test_time = list(np.quantile(graph.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))
    # =====================================  Change it to train dataset only  =====================================
    print('Attacking train only')
    src_node_ids = graph.u.values.astype(np.longlong)
    dst_node_ids = graph.i.values.astype(np.longlong)
    node_interact_times = graph.ts.values.astype(np.float64)
    
    node_set = set(src_node_ids) | set(dst_node_ids)
    num_total_unique_node_ids = len(node_set)
    
    # compute nodes which appear at test time
    test_node_set = set(src_node_ids[node_interact_times > val_time]).union(set(dst_node_ids[node_interact_times > val_time]))
    # sample nodes which we keep as new nodes (to test inductiveness), so then we have to remove all their edges from training
    # new_test_node_set = set(random.sample(test_node_set, int(0.1 * num_total_unique_node_ids)))  # changed it to list
    new_test_node_set = set(random.sample(sorted(test_node_set), int(0.1 * num_total_unique_node_ids)))
    
    train_graph_df = graph[graph['ts'] < val_time]
    print(f'length of full training graph is {len(train_graph_df)} and full graph is {len(graph)}.')
    
    print(f'{len(train_graph_df) / len(graph):.2%}')
    assert (len(train_graph_df) / len(graph)) > 0.7, 'Not training graph'
    
    val_graph_df = graph[np.logical_and(graph['ts'] <= test_time, graph['ts'] > val_time)]
    test_graph_df = graph[graph['ts'] > test_time]
    
    processor = DegreePreservingGraphProcessor(
        graph_df=train_graph_df,
        full_graph=graph,
        edge_features=edge_features,
        node_features=node_features,
        sparsification_strategy=strategy,
        sparsification_ratio=upto,
        negative_sample_strategy=negative_sample_strategy,
        new_test_node_set = new_test_node_set,
        num_negative_samples=5,  # You might want to make this configurable
        dataset_name=dataset_name,
        seed=0  # You might want to make this configurable
    )
    
    processed_graph = processor.process_graph()
    
    
    # print('breaking intentionally.')
    
    # exit()
    
    # DONE: Concat all three parts here.
    
    processed_graph = pd.concat([processed_graph, val_graph_df, test_graph_df], ignore_index=True)
    processed_graph = processed_graph.sort_values('ts').reset_index(drop=True)
    
    # Save the processed data
    out_dir = f'{save_location}/poisoned_data/nss_{negative_sample_strategy}/{dataset_name}/{strategy}'
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    processed_graph.to_csv(f'{out_dir}/{dataset_name}_{strategy}_sparsified_{upto}.csv', index=False)
    np.save(f'{out_dir}/ml_{dataset_name}.npy', processor.sparsified_edge_features)
    np.save(f'{out_dir}/ml_{dataset_name}_node.npy', processor.node_features)
    print(f'saved at {out_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Interface for poisoning datasets')
    parser.add_argument('--dataset_name', type=str,
                        choices=['wikipedia', 'reddit', 'mooc', 'lastfm', 'myket', 'enron', 'SocialEvo', 'uci',
                                'Flights', 'CanParl', 'USLegis', 'UNtrade', 'UNvote', 'Contacts'],
                        help='Dataset name', default='wikipedia')
    parser.add_argument('--upto', type=float, help='sparsify upto', default=0.7)
    parser.add_argument('--strategy', type=str, default='random', choices=['random',
                        'tpr_remove', 'ts_tpr_remove_ss', 'ts_tpr_remove_inc', 'ts_tpr_remove_MSS',
                        'ts_tpr_remove_mss_2', 'ts_tpr_remove_cosine', 'ts_tpr_remove_euclidean',
                        'ts_tpr_remove_jaccard', 'ts_tpr_remove_wasserstein', 'ts_tpr_remove_kl_divergence', 'ts_tpr_remove_chebyshev',
                        'ts_tpr_remove_jensen_shannon_divergence', 'ts_tpr_remove_TER', 'ts_tpr_remove_combined_ter', 'ts_tpr_remove_rec_mss'],
                        help='strategy for the sparsification')
    parser.add_argument('--negative_sample_strategy', type=str, default='random', choices=['random', 'historical', 'inductive'],
                        help='strategy for the negative edge sampling')

    args = parser.parse_args()

    print(f'Processing dataset {args.dataset_name}...')
    if args.dataset_name in ['enron', 'SocialEvo']:
        Path(f"{save_location}/poisoned_data/{args.dataset_name}/").mkdir(parents=True, exist_ok=True)
        copy_tree(f"{scratch_location}/DG_data/{args.dataset_name}/", f"{save_location}/poisoned_data/{args.dataset_name}/")
        print(f'Not implemented for enron, SocialEvo graph yet.')
    else:
        Path(f"{save_location}/poisoned_data/{args.dataset_name}/").mkdir(parents=True, exist_ok=True)
        copy_tree(f"{scratch_location}/DG_data/{args.dataset_name}/", f"{save_location}/poisoned_data/{args.dataset_name}/")
        
        process_data(args.dataset_name, args.strategy, args.upto, args.negative_sample_strategy)
        
        print(f'{args.dataset_name} is processed successfully.')

        if args.dataset_name not in ['myket']:
            check_data(args.dataset_name)
        print(f'{args.dataset_name} passes the checks successfully.')


