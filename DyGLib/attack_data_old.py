import argparse
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from distutils.dir_util import copy_tree
from preprocess_data.preprocess_data import check_data
from scipy.spatial import KDTree

# Add the project root to the Python path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(project_root)

from utils.utils import NegativeEdgeSampler
from preprocess_data.sparsify_data import EL_sparsify

save_location = rf'/raid/t2/TGN_adv/' # '/home/t2/TGN_adv'
scratch_location = rf'/raid/t2/TGN_adv/DyGlib'

# TODO: Make it C3 Compliant
# TODO: Check for time taken for single dataset run (TimeAnalysis)
# TODO: 
random_seed = 0

def add_negative_samples(df: pd.DataFrame, start_time: float, end_time: float, 
                         negative_sample_strategy: str = 'random', 
                         num_negative_samples: int = 1, 
                         seed: int = None) -> pd.DataFrame:
    """
    Add negative samples to the input DataFrame using the NegativeEdgeSampler.
    
    :param df: Input DataFrame with columns 'source_nodes', 'target_nodes', 'interact_times'
    :param start_time: Start time for the time window
    :param end_time: End time for the time window
    :param negative_sample_strategy: Strategy for negative sampling ('random', 'historical', or 'inductive')
    :param num_negative_samples: Number of negative samples to generate for each positive sample
    :param seed: Random seed for reproducibility
    :return: DataFrame with added negative samples
    """
    # Assumption: The input DataFrame is sorted by 'interact_times'
    assert all(df['ts'].diff()[1:] >= 0), "DataFrame must be sorted by 'interact_times'"
    
    # Filter the DataFrame based on the time window
    df_filtered = df[(df['ts'] >= start_time) & (df['ts'] <= end_time)]
    
    # Initialize the NegativeEdgeSampler
    sampler = NegativeEdgeSampler(
        src_node_ids=df['u'].values,
        dst_node_ids=df['i'].values,
        interact_times=df['ts'].values,
        last_observed_time=end_time,
        negative_sample_strategy=negative_sample_strategy,
        seed=seed
    )
    
    # Generate negative samples
    negative_samples = []
    
    # for _, row in df_filtered.iterrows():
    #     neg_src, neg_dst = sampler.sample(
    #         size=num_negative_samples,
    #         batch_src_node_ids=np.array([row['u']]),
    #         batch_dst_node_ids=np.array([row['i']]),
    #         current_batch_start_time=row['ts'],
    #         current_batch_end_time=row['ts']
    #     )
    #     for src, dst in zip(neg_src, neg_dst):
    #         negative_samples.append({
    #             'u': src,
    #             'i': dst,
    #             'ts': row['ts'],
    #             'is_negative': 1
    #         })
    
    # Extract arrays from DataFrame for batch processing
    batch_src_node_ids = df_filtered['u'].to_numpy()
    batch_dst_node_ids = df_filtered['i'].to_numpy()
    current_batch_times = df_filtered['ts'].to_numpy()

    # Call the sampler once with the entire batch
    neg_src, neg_dst = sampler.sample(
        size=num_negative_samples * len(df_filtered),  # Multiply by the number of rows to cover all samples
        batch_src_node_ids=batch_src_node_ids,
        batch_dst_node_ids=batch_dst_node_ids,
        current_batch_start_time=start_time,  #current_batch_times.min(),  # Assuming you want to span the whole time range
        current_batch_end_time=end_time  # C3 compliant
    )
    
    # TODO: C4

    # Efficiently create negative samples
    for i in range(len(neg_src)):
        negative_samples.append({
            'u': neg_src[i],
            'i': neg_dst[i],
            'ts': current_batch_times[i // num_negative_samples],  # Adjust for sampling size
            'is_negative': 1
        })
    
    # Create a DataFrame from negative samples
    df_negative = pd.DataFrame(negative_samples)
    
    # Add 'is_negative' column to the original filtered DataFrame
    df_filtered['is_negative'] = 0
    
    # Combine positive and negative samples
    df_combined = pd.concat([df_filtered, df_negative], ignore_index=True)
    
    # Sort the combined DataFrame by 'interact_times'
    df_combined = df_combined.sort_values('ts').reset_index(drop=True)
    
    return df_combined

def get_poisoned_data(dataset_name='wikipedia', strategy='random', upto=0.7, negative_sample_strategy='random'):
    # Load data and train val test split
    graph = pd.read_csv('{}/processed_data/{}/ml_{}.csv'.format(scratch_location, dataset_name, dataset_name))
    edge_raw_features = np.load('{}/processed_data/{}/ml_{}.npy'.format(scratch_location, dataset_name, dataset_name))
    node_raw_features = np.load('{}/processed_data/{}/ml_{}_node.npy'.format(scratch_location, dataset_name, dataset_name))
    
    # OUT_DF = '{}/sparsified_data/{}/ml_{}.csv'.format(scratch_location, dataset_name, dataset_name)
    # OUT_DF = f'{scratch_location}/sparsified_data/{dataset_name}/{dataset_name}_{strategy}_sparsified_{upto}.csv'
    os.makedirs(f'{save_location}/poisoned_data/nss_{negative_sample_strategy}/{dataset_name}/{strategy}', exist_ok=True)
    
    OUT_DF = f'{save_location}/poisoned_data/nss_{negative_sample_strategy}/{dataset_name}/{strategy}/{dataset_name}_{strategy}_sparsified_{upto}.csv'
    
    OUT_FEAT = '{}/poisoned_data/nss_{}/{}/{}/ml_{}.npy'.format(save_location, negative_sample_strategy, dataset_name, strategy, dataset_name)
    OUT_NODE_FEAT = '{}/poisoned_data/nss_{}/{}/{}/ml_{}_node.npy'.format(save_location, negative_sample_strategy, dataset_name, strategy, dataset_name)
    
    print(f'In get_poisoned_data {dataset_name=}')

    
    EL_graph, EL_edge_raw_features = EL_sparsify(graph, edge_raw_features,
                                                strategy=strategy, upto=upto, dataset_name=dataset_name)
    
    print('Sparsification done')
    
    start_time, end_time = min(0, graph['ts'].min()), graph['ts'].max()
    
    print('starting negative sampling', end='\r')
    
    full_graph = add_negative_samples(EL_graph, start_time, end_time, 
                                      negative_sample_strategy=negative_sample_strategy, 
                                      num_negative_samples=5,  # how to choose this ??
                                      seed=random_seed)
    
    print('done negative sampling     ')
    
    full_ts_list = graph['ts'].unique()
    remainder_ts_list = EL_graph['ts'].unique()
    to_generate_ts_list = set(full_ts_list).difference(remainder_ts_list)
    
    negative_full_graph = full_graph[full_graph['is_negative'] == 1]
    
    print(f"Length of original graph: {len(graph)}")
    print(f"Length of EL_graph: {len(EL_graph)}")
    print(f"Length of negative_full_graph: {len(negative_full_graph)}")
    print(f"Number of timestamps to generate: {len(to_generate_ts_list)}")
    
    num_samples_needed = len(graph) - len(EL_graph)
    print(f"Number of samples needed: {num_samples_needed}")
    
    # ====================================================   Old Code   ========================================
    
    # # Create a dictionary mapping each timestamp in to_generate_ts_list to the closest timestamp in negative_full_graph
    # negative_ts_array = negative_full_graph['ts'].unique()
    # ts_mapping = {ts: negative_ts_array[np.argmin(np.abs(negative_ts_array - ts))] for ts in to_generate_ts_list}
    
    # # Select negative samples based on the mapped timestamps
    # selected_negative_samples = []
    # for original_ts, mapped_ts in ts_mapping.items():
    #     samples = negative_full_graph[negative_full_graph['ts'] == mapped_ts]
    #     if len(samples) > 0:
    #         selected_sample = samples.sample(n=1)
    #         selected_sample['ts'] = original_ts  # Replace the timestamp with the original one
    #         selected_negative_samples.append(selected_sample)
            
    # ===================================== We implement a KDTree solution ====================================
    print('running KD Tree implementation.........')
    full_ts_list = graph['ts'].unique()
    remainder_ts_list = EL_graph['ts'].unique()
    
    to_generate_ts_list = set(full_ts_list) - set(remainder_ts_list)
    to_generate_ts_array = np.array(list(to_generate_ts_list))

    negative_full_graph = full_graph[full_graph['is_negative'] == 1]
    negative_ts_array = negative_full_graph['ts'].values

    # Create a KDTree for the negative timestamps
    negative_ts_tree = KDTree(negative_ts_array.reshape(-1, 1))

    # Find the nearest neighbors for each timestamp in to_generate_ts_list
    _, indices = negative_ts_tree.query(to_generate_ts_array.reshape(-1, 1))
    mapped_ts_array = negative_ts_array[indices.flatten()]

    # # Select negative samples based on the mapped timestamps
    # selected_negative_samples = negative_full_graph.loc[negative_full_graph['ts'].isin(mapped_ts_array)]
    # selected_negative_samples['ts'] = to_generate_ts_array
    

    # Create a dictionary to map original to mapped timestamps
    ts_mapping = {orig_ts: mapped_ts for orig_ts, mapped_ts in zip(to_generate_ts_array, mapped_ts_array)}

    # Select negative samples based on the mapped timestamps
    selected_negative_samples = negative_full_graph.loc[negative_full_graph['ts'].isin(mapped_ts_array)]

    # Replace the mapped timestamps with the original ones using the mapping
    selected_negative_samples['ts'] = selected_negative_samples['ts'].map({v: k for k, v in ts_mapping.items()})
    
    
    # =========================================================================================================
    
    selected_negative_full_graph = selected_negative_samples  # pd.concat(selected_negative_samples, ignore_index=True)
    print(f"Length of selected_negative_full_graph: {len(selected_negative_full_graph)}")
    
    if len(selected_negative_full_graph) < num_samples_needed:
        print("Warning: Not enough negative samples in selected_negative_full_graph")
        print("Sampling additional negatives to reach the required number")
        additional_samples_needed = num_samples_needed - len(selected_negative_full_graph)
        additional_samples = negative_full_graph.sample(n=additional_samples_needed, replace=False)
        selected_negative_full_graph = pd.concat([selected_negative_full_graph, additional_samples], ignore_index=True)
    elif len(selected_negative_full_graph) > num_samples_needed:
        print("Warning: Too many negative samples, sampling down to the required number")
        selected_negative_full_graph = selected_negative_full_graph.sample(n=num_samples_needed, replace=False)
    
    full_graph = pd.concat([EL_graph, selected_negative_full_graph], ignore_index=True)
    full_graph = full_graph.sort_values('ts').reset_index(drop=True)

    print(f"Final length of full_graph: {len(full_graph)}")
    if len(graph) != len(full_graph):
        print(f"Warning: Length mismatch. Original: {len(graph)}, New: {len(full_graph)}")

    assert len(graph) == len(full_graph), f'Make dataset same again, Original graph is {len(graph)=}, while is reconstructed {len(full_graph)=}'
    
    
    full_graph.to_csv(OUT_DF)  # edge-list
    np.save(OUT_FEAT, EL_edge_raw_features)  # edge features
    np.save(OUT_NODE_FEAT, node_raw_features)  # node features
    
    print(f'Sparsified {dataset_name}.')

if __name__ == '__main__':

    # # Example usage:
    # df_input = pd.DataFrame({
    #     'source_nodes': [1, 2, 3, 4],
    #     'target_nodes': [5, 6, 7, 8],
    #     'interact_times': [1.0, 2.0, 5.0, 7.0]
    # })
    # start_time = 0
    # end_time = 9
    # df_with_negatives = add_negative_samples(df_input, start_time, end_time, 
    #                                          negative_sample_strategy='random', 
    #                                          num_negative_samples=2, 
    #                                          seed=42)
    # print(df_with_negatives)
    
    parser = argparse.ArgumentParser('Interface for poisoning datasets')
    parser.add_argument('--dataset_name', type=str,
                        choices=['wikipedia', 'reddit', 'mooc', 'lastfm', 'myket', 'enron', 'SocialEvo', 'uci',
                                'Flights', 'CanParl', 'USLegis', 'UNtrade', 'UNvote', 'Contacts'],
                        help='Dataset name', default='wikipedia')
    # parser.add_argument('--node_feat_dim', type=int, default=172, help='Number of node raw features')
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

    print(f'Sparsifying dataset {args.dataset_name}...')
    if args.dataset_name in ['enron', 'SocialEvo']:  # DONE: remove UCI from here
        Path("{}/poisoned_data/{}/".format(save_location, args.dataset_name)).mkdir(parents=True, exist_ok=True)
        copy_tree("{}/DG_data/{}/".format(scratch_location, args.dataset_name), "{}/poisoned_data/{}/".format(save_location, args.dataset_name))
        print(f'Not implemented for enron, SocialEvo graph yet.')
    else:
        Path("{}/poisoned_data/{}/".format(scratch_location, args.dataset_name)).mkdir(parents=True, exist_ok=True)
        copy_tree("{}/DG_data/{}/".format(scratch_location, args.dataset_name), "{}/poisoned_data/{}/".format(save_location, args.dataset_name))
        # bipartite dataset
        if args.dataset_name in ['wikipedia', 'reddit', 'mooc', 'lastfm', 'myket']:
            get_poisoned_data(dataset_name=args.dataset_name, strategy=args.strategy, upto=args.upto, negative_sample_strategy=args.negative_sample_strategy)
        else:
            get_poisoned_data(dataset_name=args.dataset_name, strategy=args.strategy, upto=args.upto, negative_sample_strategy=args.negative_sample_strategy)
        print(f'{args.dataset_name} is processed successfully.')

        if args.dataset_name not in ['myket']:
            check_data(args.dataset_name)
        print(f'{args.dataset_name} passes the checks successfully.')