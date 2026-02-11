import argparse
import os
import random
import networkx as nx
from itertools import product
import pandas as pd
import numpy as np
from pathlib import Path
from distutils.dir_util import copy_tree
from preprocess_data.preprocess_data import check_data
from collections import defaultdict
from scipy.spatial import KDTree
from copy import deepcopy
from tqdm.auto import tqdm
from functools import partial
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed


# good_seed = 1729928109

random.seed(1729928109)
np.random.seed(1729928109)

from hh_attack_utils import calculate_node_degrees, create_node_mapping, find_missing_edges,\
    get_nodes_in_time_window, EdgeTimestampSelector, TemporalSampler, TPRTimestampSelector
    

# Add the project root to the Python path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(project_root)

from preprocess_data.sparsify_data import EL_sparsify

save_location = rf'/raid/t2/TGN_adv/'
scratch_location = rf'/raid/t2/TGN_adv/DyGlib'


def _process_single_batch(missing_edges_df, full_graph, idx, hh_edge_flag=True, seed=0):
    print(f'Processing batch {idx+1}: length of missing edges {len(missing_edges_df)}')

    removed_edges = missing_edges_df[['u', 'i']].values.tolist()
    

    deficit_src_degree, deficit_dst_degree = calculate_node_degrees(missing_edges_df)

    last_hh_graph = nx.bipartite.havel_hakimi_graph(
        aseq=list(deficit_src_degree.values()),
        bseq=list(deficit_dst_degree.values()),
        create_using=nx.Graph()
        )
    hh_edges = list(last_hh_graph.edges())

    print('\tfull edges: ', len(hh_edges), end=' ')

    # +++ RUNNING TEMPORAL SAMPLER +++

    time_window = 1200
    seed = 0

    # Initialize TemporalSampler
    sampler = TemporalSampler(missing_edges_df, time_window=time_window, seed=seed)

    # Generate timestamps for all potential negative samples
    num_samples = sum(deficit_src_degree.values())
    sampled_timestamps = sampler.assign_timestamps(num_samples)
    
    # orig_src_degrees, orig_dst_degrees
    n = len(deficit_src_degree)
    degrees = dict(last_hh_graph.degree())
    degrees_top = {k: degrees[k] for k in list(degrees.keys())[:n]}
    degrees_bottom = {k: degrees[k] for k in list(degrees.keys())[n:]}

    src_mapping = create_node_mapping(deficit_src_degree, degrees_top)
    dst_mapping = create_node_mapping(deficit_dst_degree, degrees_bottom)

    df_sorted =full_graph.sort_values('ts')  # missing_edges_df
    original_edges = set(tuple(x) for x in df_sorted[['u', 'i']].values)
    original_edge_set = set(original_edges)

    min_src, max_src = full_graph['u'].min(), full_graph['u'].max()
    min_dst, max_dst = full_graph['i'].min(), full_graph['i'].max()
    
    map_edges_to_ts = defaultdict(list)

    # ADD HH EDGES
    # Track node usage counts for balance
    src_node_counts = defaultdict(int)
    dst_node_counts = defaultdict(int)

    # Set a threshold for under-connected nodes
    
    hh_edges_set = {(src_mapping[edge[0]], dst_mapping[edge[1]]) for edge in set(hh_edges)}  # this operation is slow, this is its right place
    print('Unique edges: ', len(hh_edges_set))
    
    for timestamp in tqdm(sampled_timestamps, desc='\tProcessing Timestamps', leave=False, ncols=80):
        nodes_in_window = get_nodes_in_time_window(df_sorted, timestamp, time_window)
        
        # Filter nodes by their connection counts to prioritize under-connected nodes
        src_nodes_orig = [
            n for n in nodes_in_window if min_src <= n <= max_src # and src_node_counts[n] < connectivity_threshold
        ]
        dst_nodes_orig = [
            n for n in nodes_in_window if min_dst <= n < max_dst # and dst_node_counts[n] < connectivity_threshold
        ]
        
        # Targeted edge generation for under-connected nodes only ignoring HH edges
        if not hh_edge_flag:
            targeted_edges = [
                    (src, dst) for src, dst in product(src_nodes_orig, dst_nodes_orig)
                    if (src, dst) not in original_edge_set and (src, dst) not in hh_edges_set
            ]
        else:
            targeted_edges = [
            (src, dst) for src, dst in product(src_nodes_orig, dst_nodes_orig)
            if (src, dst) not in original_edge_set
        ]
        
        for src, dst in targeted_edges:
            map_edges_to_ts[(src, dst)].append(timestamp)
            src_node_counts[src] += 1
            dst_node_counts[dst] += 1

            
    degree_dict = deepcopy(deficit_src_degree)
    degree_dict.update(deficit_dst_degree)

    selector = EdgeTimestampSelector(
        feasible_edges_dict=map_edges_to_ts,
        original_df=missing_edges_df,
        degree_dict=degree_dict,
        removed_edges=removed_edges,
        src_nodes=set(deficit_src_degree.keys()),
        dst_nodes=set(deficit_dst_degree.keys()),
        time_window=time_window
    )

    gen_edges = selector.select_edges_timestamps()
    print(f'For batch {idx+1} generated edges: {len(gen_edges)}/{len(missing_edges_df)}.')
    return gen_edges


def _process_batches_concurrently(all_early_missing_edges_df, full_graph, hh_edge_flag=True):
    results = []
    
    # Use ProcessPoolExecutor to parallelize the batch processing
    with ProcessPoolExecutor() as executor:
        # Submit each batch to the executor
        futures = [
            executor.submit(process_single_batch, missing_edges_df, full_graph, idx, hh_edge_flag)
            for idx, missing_edges_df in enumerate(all_early_missing_edges_df)
        ]
        
        # Retrieve the results as they complete
        for future in as_completed(futures):
            try:
                result = future.result()
                results.extend(result)  # Collect generated edges from each batch
            except Exception as e:
                print(f"\tAn error occurred: {e}")
    
    return results


def add_negative_samples(sprs_df, orig_df, full_graph, hh_edge_flag=True):
    
    # Add negative samples to the graph for first few missing timestamps
    early_start_time, early_end_time = sprs_df['ts'].iloc[0], sprs_df['ts'].iloc[1]

    all_early_missing_edges_df = find_missing_edges(orig_df, sprs_df, start_time=early_start_time, end_time=early_end_time, batch_size=1)

    print(len(pd.concat(all_early_missing_edges_df)), end=' ')
    
    flag = False
    
    new_edges = process_batches_concurrently(all_early_missing_edges_df, full_graph, hh_edge_flag=True)

    if len(new_edges) == 0:
        print('No new edges generated for first batch.')
        flag = True
    print(f'flag is {flag}.')    
    # Add negative samples to the graph for first few missing timestamps
    
    if not flag:
        start_time, end_time = sprs_df['ts'].iloc[1], sprs_df['ts'].max()
        all_late_missing_edges_df = find_missing_edges(orig_df, sprs_df, start_time=start_time, end_time=end_time)
    else:
        start_time, end_time = min(0, sprs_df['ts'].iloc[0]), sprs_df['ts'].max()
        all_late_missing_edges_df = find_missing_edges(orig_df, sprs_df, batch_size=2, start_time=start_time, end_time=end_time)
        

    new_edges_ = process_batches_concurrently(all_late_missing_edges_df, orig_df, hh_edge_flag=hh_edge_flag)
    print(f"Total new edges for last df generated: {len(new_edges_)}")
    new_edges.extend(new_edges_)
    
    all_missing_edges_df = all_early_missing_edges_df + all_late_missing_edges_df
    print(f"Total missing edges: {len(pd.concat(all_missing_edges_df))}", end=' ')
    print(f"Total new edges generated: {len(new_edges)}")
    return new_edges


def get_poisoned_data(dataset_name='wikipedia', strategy='random', upto=0.7,
                    negative_sample_strategy='random', val_ratio=0.15, test_ratio=0.15, 
                    random_seed = 0):
    random.seed(random_seed)
    # Load data and train val test split
    graph = pd.read_csv('{}/processed_data/{}/ml_{}.csv'.format(scratch_location, dataset_name, dataset_name))
    edge_raw_features = np.load('{}/processed_data/{}/ml_{}.npy'.format(scratch_location, dataset_name, dataset_name))
    node_raw_features = np.load('{}/processed_data/{}/ml_{}_node.npy'.format(scratch_location, dataset_name, dataset_name))
    print('length of full graph is ', len(graph))
    
    os.makedirs(f'{save_location}/poisoned_data/{dataset_name}/{strategy}', exist_ok=True)
    
    OUT_DF = f'{save_location}/poisoned_data/{dataset_name}/{strategy}/{dataset_name}_{strategy}_sparsified_{upto}.csv'
    OUT_FEAT = '{}/poisoned_data/{}/{}/ml_{}.npy'.format(save_location, dataset_name, strategy, dataset_name)
    OUT_NODE_FEAT = '{}/poisoned_data/{}/{}/ml_{}_node.npy'.format(save_location, dataset_name, strategy, dataset_name)
    
    val_time, test_time = list(np.quantile(graph.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))
    new_test_node_set=set()
    
    # Only training data is used for sparsification and validation and test data are kept as it is
    
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
    print(f'percentage of train data: {len(train_graph_df) / len(graph):.2%}')
    assert (len(train_graph_df) / len(graph)) > 0.7, 'Not training graph'

    EL_graph, EL_edge_raw_features, _ = EL_sparsify(train_graph_df, edge_raw_features,
                                                strategy=strategy, upto=upto,
                                                dataset_name=dataset_name)
    
    print(f'length of sparsified graph: {len(EL_graph)}, percentage of sparsified train data: {len(EL_graph) / len(train_graph_df):.2%}')
    print('Sparsification done')
    
    # get val and test splits too
    val_graph_df = graph[np.logical_and(graph['ts'] <= test_time, graph['ts'] > val_time)]
    test_graph_df = graph[graph['ts'] > test_time]
    
        
    print('starting negative sampling', end='\r')
    
    
    negative_samples = add_negative_samples(EL_graph, train_graph_df, graph, hh_edge_flag=True)
    
    print('done negative sampling     ')
    
    print('Full training data length: ', len(train_graph_df), len(EL_graph)+len(negative_samples))
    # TODO: Remove after testing
    
    negative_samples = pd.DataFrame(negative_samples, columns=['u', 'i', 'ts'])
    negative_samples.sort_values('ts', inplace=True)
    negative_samples.reset_index(drop=True, inplace=True)
        
    att_graph = pd.concat([EL_graph, negative_samples], ignore_index=True)
    att_graph = att_graph.sort_values('ts').reset_index(drop=True)
    
    #TODO: add val and test data to attack graph and save it

    # print(f"Final length of full_graph: {len(full_graph)}")
    # if len(graph) != len(full_graph):
    #     print(f"Warning: Length mismatch. Original: {len(graph)}, New: {len(full_graph)}")

    # assert len(graph) == len(full_graph), f'Make dataset same again, Original graph is {len(graph)=}, while is reconstructed {len(full_graph)=}'
    
    
    # TODO: save train/val/test splits separately or same file
    print('Before: ', len(att_graph))
    att_graph = pd.concat([att_graph, val_graph_df, test_graph_df], ignore_index=True)
    print('After: ', len(att_graph))
    att_graph.to_csv(OUT_DF)  # edge-list
    np.save(OUT_FEAT, EL_edge_raw_features)  # edge features
    np.save(OUT_NODE_FEAT, node_raw_features)  # node features
    padding_length = abs((80 - (15 + len(dataset_name))) // 2)
    print('* ' * padding_length, f'Saved Attacked {dataset_name}.', ' *' * padding_length)
    

def process_single_batch(missing_edges_df, full_graph, idx, hh_edge_flag=True):
    """Modified batch processing with better error handling"""
    # try:
    print(f'Processing batch {idx+1}')

    # Calculate degrees and generate HH graph
    removed_edges = missing_edges_df[['u', 'i']].values.tolist()
    deficit_src_degree, deficit_dst_degree = calculate_node_degrees(missing_edges_df)

    last_hh_graph = nx.bipartite.havel_hakimi_graph(
        aseq=list(deficit_src_degree.values()),
        bseq=list(deficit_dst_degree.values()),
        create_using=nx.Graph()
    )
    hh_edges = list(last_hh_graph.edges())
    print('\tfull edges: ', len(hh_edges))

    # Initialize temporal sampler
    time_window = 1200
    sampler = TemporalSampler(missing_edges_df, time_window=time_window, seed=0)
    num_samples = sum(deficit_src_degree.values())
    sampled_timestamps = sampler.assign_timestamps(num_samples)

    # Create mappings
    n = len(deficit_src_degree)
    degrees = dict(last_hh_graph.degree())
    degrees_top = {k: degrees[k] for k in list(degrees.keys())[:n]}
    degrees_bottom = {k: degrees[k] for k in list(degrees.keys())[n:]}

    src_mapping = create_node_mapping(deficit_src_degree, degrees_top)
    dst_mapping = create_node_mapping(deficit_dst_degree, degrees_bottom)

    # Process edges and timestamps
    df_sorted = full_graph.sort_values('ts')
    original_edges = set(tuple(x) for x in df_sorted[['u', 'i']].values)
    min_src, max_src = missing_edges_df['u'].min(), missing_edges_df['u'].max()
    min_dst, max_dst = missing_edges_df['i'].min(), missing_edges_df['i'].max()
    
    map_edges_to_ts = defaultdict(list)
    src_node_counts = defaultdict(int)
    dst_node_counts = defaultdict(int)

    # Generate and process edges
    hh_edges_set = {(src_mapping[edge[0]], dst_mapping[edge[1]]) 
                    for edge in set(hh_edges)}
    print('Unique edges: ', len(hh_edges_set))

    for timestamp in tqdm(sampled_timestamps, desc='\tProcessing Timestamps', ncols=80):
        nodes_in_window = get_nodes_in_time_window(df_sorted, timestamp, time_window)
        
        src_nodes_orig = [n for n in nodes_in_window 
                        if min_src <= n <= max_src]
        dst_nodes_orig = [n for n in nodes_in_window 
                        if min_dst <= n < max_dst]
        
        if not hh_edge_flag:
            targeted_edges = [
                (src, dst) for src, dst in product(src_nodes_orig, dst_nodes_orig)
                if (src, dst) not in original_edges 
                and (src, dst) not in hh_edges_set
            ]
        else:
            targeted_edges = [
                (src, dst) for src, dst in product(src_nodes_orig, dst_nodes_orig)
                if (src, dst) not in original_edges
            ]
        
        for src, dst in targeted_edges:
            map_edges_to_ts[(src, dst)].append(timestamp)
            src_node_counts[src] += 1
            dst_node_counts[dst] += 1

    # Create selector and generate edges
    degree_dict = deepcopy(deficit_src_degree)
    degree_dict.update(deficit_dst_degree)

    selector = TPRTimestampSelector(
        feasible_edges_dict=map_edges_to_ts,
        original_df=missing_edges_df,
        degree_dict=degree_dict,
        removed_edges=removed_edges,
        src_nodes=set(deficit_src_degree.keys()),
        dst_nodes=set(deficit_dst_degree.keys()),
        time_window=1200,
        beta=0.5,
        alpha=0.15,
        dataset_name=f"wikipedia_batch_{idx}"
    )

    gen_edges = selector.select_edges_timestamps()
    return gen_edges

    # except Exception as e:
    #     print(f"Error processing batch {idx}: {str(e)}")
    #     return []


def process_batches_concurrently(all_early_missing_edges_df, full_graph, hh_edge_flag=True):
    """Process batches with better error handling and multiprocessing support"""
    
    def process_batch_wrapper(args):
        """Wrapper function for batch processing with error handling"""
        try:
            idx, missing_edges_df = args
            print(f'Processing batch {idx+1}')
            return process_single_batch(missing_edges_df, full_graph, idx, hh_edge_flag)
        except Exception as e:
            print(f"Error in batch {idx}: {str(e)}")
            return []

    results = []
    
    # Process batches sequentially if small number of batches
    if len(all_early_missing_edges_df) <= 2:
        for idx, df in enumerate(all_early_missing_edges_df):
            # try:
            batch_results = process_single_batch(df, full_graph, idx, hh_edge_flag)
            results.extend(batch_results)
            # except Exception as e:
            #     print(f"Error in batch {idx}: {str(e)}")
            #     continue
    else:
        # Use Pool for parallel processing
        try:
            batch_args = list(enumerate(all_early_missing_edges_df))
            with Pool() as pool:
                batch_results = pool.map(process_batch_wrapper, batch_args)
                
            # Combine results
            for result in batch_results:
                if result:  # Only extend if result is not empty
                    results.extend(result)
                    
        except Exception as e:
            print(f"Error in parallel processing: {str(e)}")
            # Fallback to sequential processing
            print("Falling back to sequential processing...")
            for idx, df in enumerate(all_early_missing_edges_df):
                # try:
                batch_results = process_single_batch(df, full_graph, idx, hh_edge_flag)
                results.extend(batch_results)
                # except Exception as e:
                #     print(f"Error in batch {idx}: {str(e)}")
                #     continue

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Interface for poisoning datasets')
    parser.add_argument('-d', '--dataset_name', type=str,
                        choices=['wikipedia', 'reddit', 'mooc', 'lastfm', 'myket', 'enron', 'SocialEvo', 'uci',
                                'Flights', 'CanParl', 'USLegis', 'UNtrade', 'UNvote', 'Contacts'],
                        help='Dataset name', default='wikipedia')
    # parser.add_argument('--node_feat_dim', type=int, default=172, help='Number of node raw features')
    parser.add_argument('-u', '--upto', type=float, help='sparsify upto', default=0.7)
    parser.add_argument('-s', '--strategy', type=str, default='random', choices=['random',
                        'tpr_remove', 'ts_tpr_remove_ss', 'ts_tpr_remove_inc', 'ts_tpr_remove_mss',
                        'ts_tpr_remove_mss_2', 'ts_tpr_remove_cosine', 'ts_tpr_remove_euclidean',
                        'ts_tpr_remove_jaccard', 'ts_tpr_remove_wasserstein', 'ts_tpr_remove_kl_divergence', 'ts_tpr_remove_chebyshev',
                        'ts_tpr_remove_jensen_shannon_divergence', 'ts_tpr_remove_TER', 'ts_tpr_remove_combined_ter', 'ts_tpr_remove_rec_mss'],
                        help='strategy for the sparsification')
        

    args = parser.parse_args()
    print(args)

    if args.dataset_name in ['enron', 'SocialEvo']:  # DONE: remove UCI from here
        Path("{}/poisoned_data/{}/".format(save_location, args.dataset_name)).mkdir(parents=True, exist_ok=True)
        copy_tree("{}/DG_data/{}/".format(scratch_location, args.dataset_name), "{}/poisoned_data/{}/".format(save_location, args.dataset_name))
        print(f'Not implemented for enron, SocialEvo graph yet.')
    else:
        Path("{}/poisoned_data/{}/".format(scratch_location, args.dataset_name)).mkdir(parents=True, exist_ok=True)
        copy_tree("{}/DG_data/{}/".format(scratch_location, args.dataset_name), "{}/poisoned_data/{}/".format(save_location, args.dataset_name))
        # bipartite dataset
        if args.dataset_name in ['wikipedia', 'reddit', 'mooc', 'lastfm', 'myket']:
            get_poisoned_data(dataset_name=args.dataset_name, strategy=args.strategy, upto=args.upto)
        else:
            get_poisoned_data(dataset_name=args.dataset_name, strategy=args.strategy, upto=args.upto)
        print(f'{args.dataset_name} is processed successfully.')

        if args.dataset_name not in ['myket']:
            check_data(args.dataset_name)
        print(f'{args.dataset_name} passes the checks successfully.')