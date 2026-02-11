import argparse
from collections import deque
import os
import random
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from distutils.dir_util import copy_tree
from preprocess_data.preprocess_data import check_data
from scipy.spatial import KDTree
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from scipy import stats

# Add the project root to the Python path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(project_root)

# from utils.utils import AdversarialEdgeSampler
from utils.adv_utils import AdvNegativeEdgeSampler as AdversarialEdgeSampler
from preprocess_data.sparsify_data import EL_sparsify

save_location = rf'/raid/t2/TGN_adv/' # '/home/t2/TGN_adv'
scratch_location = rf'/raid/t2/TGN_adv/DyGlib'

# TODO: Make it C3 Compliant
# TODO: Check for time taken for single dataset run (TimeAnalysis)
# TODO: 
random_seed = 0



def validate_negative_samples(df: pd.DataFrame, time_window: float = 1200) -> bool:
    """
    Validate if negative samples follow the time window constraint.
    
    :param df: DataFrame with columns 'u', 'i', 'ts', and 'is_negative'
    :param time_window: Time window in milliseconds (default 1200)
    :return: Boolean indicating if all negative samples are valid
    """
    print('Inside validate_negative_samples')
    # Sort the DataFrame by timestamp
    df_sorted = df.sort_values('ts').reset_index(drop=True)
    
    # Separate positive and negative samples
    positive_samples = df_sorted[df_sorted['is_negative'] == 0]
    negative_samples = df_sorted[df_sorted['is_negative'] == 1]
    
    # Function to get valid nodes within the time window
    def get_valid_nodes(current_time, current_index):
        valid_nodes = set()
        w_edges_count = 0
        for i in range(current_index - 1, -1, -1):
            if df_sorted.loc[i, 'is_negative'] == 0:
                if current_time - df_sorted.loc[i, 'ts'] <= time_window:
                    valid_nodes.add(df_sorted.loc[i, 'u'])
                    valid_nodes.add(df_sorted.loc[i, 'i'])
                    w_edges_count += 1
                else:
                    break
        return valid_nodes

    # Check each negative sample
    for idx, row in negative_samples.iterrows():
        print(f'\tFor {idx=} {row=}')
        time = row['ts']
        current_index = df_sorted.index.get_loc(idx)
        
        # Get valid nodes within the time window
        print(f'\t{time=}')
        valid_nodes = get_valid_nodes(time, current_index)
        print('\t\tnumber of valid nodes', len(valid_nodes))
        
        # Check if both source and destination nodes are in the valid set
        if row['u'] not in valid_nodes or row['i'] not in valid_nodes:
            print(f"Invalid negative sample found at time {time}: src: {row['u']} -> tgt: {row['i']}")
            return False
        
        # Check if the negative sample is between two positive samples
        next_positive = positive_samples[positive_samples['ts'] > time].iloc[0] if not positive_samples[positive_samples['ts'] > time].empty else None
        if next_positive is not None and next_positive['ts'] - time > time_window:
            print(f"Invalid negative sample found at time {time}: too far from next positive sample")
            return False
    
    return True

def add_negative_samples_old(df: pd.DataFrame, start_time: float, end_time: float, 
                         negative_sample_strategy: str = 'random', 
                         num_negative_samples: int = 1, 
                         seed: int = None, time_window: float = 1200) -> pd.DataFrame:
    """
    Add negative samples to the input DataFrame using the AdversarialEdgeSampler.
    
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
    df_filtered = df[(df['ts'] >= start_time) & (df['ts'] <= end_time)].sort_values('ts')

    
    # Initialize the AdversarialEdgeSampler
    sampler = AdversarialEdgeSampler(
        src_node_ids=df['u'].values,
        dst_node_ids=df['i'].values,
        interact_times=df['ts'].values,
        last_observed_time=end_time,
        negative_sample_strategy=negative_sample_strategy,
        seed=seed
    )
    
    # Generate negative samples
    negative_samples = []
    batch_size = 10000
    
    # Extract arrays from DataFrame for batch processing
    sources = df_filtered['u'].to_numpy()
    destinations = df_filtered['i'].to_numpy()
    times = df_filtered['ts'].to_numpy()
    
    print(len(sources), len(destinations), len(times))
    
    print(f'{start_time=} {end_time=}')
    
    # Implementing sliding window
    window_start = np.searchsorted(times, times - time_window, side='left')
    
    # Create arrays of unique nodes within each sliding window
    unique_nodes = [np.unique(np.concatenate([sources[start:i+1], destinations[start:i+1]]))
                    for i, start in enumerate(window_start)]
        
    
    for i in tqdm(range(0, len(df_filtered), batch_size), desc='Processing', ncols=100):
        batch_end = min(i+batch_size, len(df_filtered))
        
        batch_times = times[i:batch_end]
        batch_sources = sources[i:batch_end]
        batch_destinations = destinations[i:batch_end]
        batch_unique_nodes = unique_nodes[i:batch_end]
    

        # Call the sampler once with the entire batch
        neg_src, neg_dst = sampler.sample(
            size=num_negative_samples * (batch_end - i),  # Multiply by the number of rows to cover all samples
            batch_src_node_ids=batch_sources,
            batch_dst_node_ids=batch_destinations,
            current_batch_start_time=batch_times[0],  # Changing it from start time --> endtime - 1200ms to make it C3 compliant
            current_batch_end_time=batch_times[-1]  # C3 compliant
        )
        
        for j in range(batch_end - i):
            active_nodes = batch_unique_nodes[j]
            for k in range(num_negative_samples):
                idx = j * num_negative_samples + k
                neg_src_node = active_nodes[neg_src[idx] % len(active_nodes)]
                neg_dst_node = active_nodes[neg_dst[idx] % len(active_nodes)]
                negative_samples.append({
                        'u': neg_src_node,
                        'i': neg_dst_node,
                        'ts': batch_times[j],
                        'is_negative': 1
                    })
    
    # Create a DataFrame from negative samples
    df_negative = pd.DataFrame(negative_samples)
    
    print(len(df_negative))
        
    # Add 'is_negative' column to the original filtered DataFrame
    df_filtered['is_negative'] = 0
    
    # Combine positive and negative samples
    df_combined = pd.concat([df_filtered, df_negative], ignore_index=True)
    
    # Sort the combined DataFrame by 'interact_times'
    df_combined = df_combined.sort_values('ts').reset_index(drop=True)
    
    # Validate the negative samples
    is_valid = validate_negative_samples(df_combined, time_window)
    assert is_valid, "Warning: Some negative samples do not follow the 1200ms constraint!"
    exit()
    
    return df_combined



    return df_combined


def add_negative_samples_without_timesampling(df: pd.DataFrame, orig_df: pd.DataFrame, start_time: float, end_time: float, 
                         negative_sample_strategy: str = 'random', 
                         num_negative_samples: int = 1, 
                         seed: int = None,
                         time_window: float = 1200) -> pd.DataFrame:
    
    df_sorted = df.sort_values('ts').reset_index(drop=True)
    
    sampler = AdversarialEdgeSampler(
        src_node_ids=orig_df['u'].values,
        dst_node_ids=orig_df['i'].values,
        interact_times=orig_df['ts'].values,
        last_observed_time=max(orig_df['ts'].values),
        negative_sample_strategy=negative_sample_strategy,
        seed=seed
    )
    
    rng = np.random.default_rng(seed)
    negative_samples = []
    
    # Calculate the number of batches
    batch_size = 10000  # You can adjust this based on your memory constraints
    num_batches = len(df_sorted) // batch_size + (1 if len(df_sorted) % batch_size != 0 else 0)
    
    for batch in tqdm(range(num_batches), desc='Generating neg samples', leave=False, ncols=100):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, len(df_sorted))
        
        batch_df = df_sorted.iloc[start_idx:end_idx]
        
        neg_src, neg_dst = sampler.sample(
            size=num_negative_samples,
            batch_src_node_ids=batch_df['u'].values,
            batch_dst_node_ids=batch_df['i'].values,
            current_batch_start_time=min(start_time, batch_df['ts'].min()),
            current_batch_end_time=min(end_time, batch_df['ts'].max())
        )
        
        for i in range(len(neg_src)):
            row_idx = i // num_negative_samples
            row = batch_df.iloc[row_idx]
            neg_ts = max(start_time, min(end_time, int(rng.uniform(row['ts'] - time_window, row['ts']))))
            
            negative_samples.append({
                'u': neg_src[i],
                'i': neg_dst[i],
                'ts': neg_ts,
                'is_negative': 1,
            })

    df_negative = pd.DataFrame(negative_samples)
    df_combined = pd.concat([df_sorted, df_negative], ignore_index=True)
    df_combined = df_combined.sort_values('ts').reset_index(drop=True)
    
    print(len(df_combined))
    
    # Validate the negative samples (you need to implement this function)
    is_valid = validate_negative_samples(df_combined, time_window)
    assert is_valid, "Warning: Some negative samples do not follow the time window constraint!"
    
    return df_combined


def add_negative_samples_unopt(df: pd.DataFrame, orig_df: pd.DataFrame, start_time: float, end_time: float, 
                         negative_sample_strategy: str = 'random', 
                         num_negative_samples: int = 1, 
                         seed: int = None,
                         time_window: float = 1200) -> pd.DataFrame:
    
    df_sorted = df.sort_values('ts').reset_index(drop=True)
    
    sampler = AdversarialEdgeSampler(
        src_node_ids=orig_df['u'].values,
        dst_node_ids=orig_df['i'].values,
        interact_times=orig_df['ts'].values,
        last_observed_time=max(orig_df['ts'].values),
        negative_sample_strategy=negative_sample_strategy,
        seed=seed
    )
    
    rng = np.random.default_rng(seed)
    negative_samples = []
    
    # Calculate inter-event time distribution
    inter_event_times = np.diff(df_sorted['ts'])  # what if we gave it original df
    kde = stats.gaussian_kde(inter_event_times)
    
    def sample_timestamp(current_time, next_time, rng):
        if next_time is None:
            next_time = current_time + time_window
        
        # Sample from KDE and clip to ensure it's within the allowed range
        sampled_delta = kde.resample(1, rng)[0][0]
        sampled_delta = np.clip(sampled_delta, 0, min(time_window, next_time - current_time))
        
        # Add some randomness to avoid exact KDE sampling
        jitter = rng.uniform(0, min(time_window, next_time - current_time) * 0.1)
        
        return current_time + sampled_delta + jitter
    
    # Calculate the number of batches
    batch_size = 1000  # You can adjust this based on your memory constraints
    num_batches = len(df_sorted) // batch_size + (1 if len(df_sorted) % batch_size != 0 else 0)
    
    for batch in tqdm(range(num_batches), desc='Generating neg samples', leave=False, ncols=100):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, len(df_sorted))
        print(f'\tfor {batch = } {start_idx = } {end_idx = }')
        
        batch_df = df_sorted.iloc[start_idx:end_idx]
        actual_batch_size = len(batch_df)
        
        print(f'{(batch_size) = }, {(actual_batch_size) = }')
        
        neg_src, neg_dst = sampler.sample(
            size=num_negative_samples * actual_batch_size,
            batch_src_node_ids=batch_df['u'].values,
            batch_dst_node_ids=batch_df['i'].values,
            current_batch_start_time=min(start_time, batch_df['ts'].min()),
            current_batch_end_time=min(end_time, batch_df['ts'].max())
        )
        
        print(f'\tgenerated {len(neg_src)} for {len(batch_df["u"].values)}')
        
        for i in range(len(neg_src)):
            row_idx = i // (num_negative_samples)
            try:
                row = batch_df.iloc[row_idx]
            except Exception as e:
                print(f'for {i}th neg sample {row_idx = }')
                assert 0 == 1
            next_row = batch_df.iloc[row_idx + 1] if row_idx + 1 < len(batch_df) else None
            next_time = next_row['ts'] if next_row is not None else None
            
            neg_ts = sample_timestamp(row['ts'], next_time, rng)
            neg_ts = max(start_time, min(end_time, int(neg_ts)))
            
            negative_samples.append({
                'u': neg_src[i],
                'i': neg_dst[i],
                'ts': neg_ts,
                'is_negative': 1,
            })

    df_negative = pd.DataFrame(negative_samples)
    df_combined = pd.concat([df_sorted, df_negative], ignore_index=True)
    df_combined = df_combined.sort_values('ts').reset_index(drop=True)
    
    print('Combined df length: ', len(df_combined), 'no of neg samples are: ', len(negative_samples))
    
    # Validate the negative samples
    # is_valid = validate_negative_samples(df_combined, time_window)
    # assert is_valid, "Warning: Some negative samples do not follow the time window constraint!"
    
    # df_combined.to_csv('./testing.csv')
    
    # exit()
    
    return df_combined


def add_negative_samples(df: pd.DataFrame, orig_df: pd.DataFrame, start_time: float, end_time: float, 
                         negative_sample_strategy: str = 'random', 
                         num_negative_samples: int = 1, 
                         seed: int = None,
                         time_window: float = 1200) -> pd.DataFrame:
    
    df_sorted = df.sort_values('ts').reset_index(drop=True)
    
    sampler = AdversarialEdgeSampler(
        src_node_ids=orig_df['u'].values,
        dst_node_ids=orig_df['i'].values,
        interact_times=orig_df['ts'].values,
        last_observed_time=max(orig_df['ts'].values),
        negative_sample_strategy=negative_sample_strategy,
        seed=seed
    )
    
    rng = np.random.default_rng(seed)
    
    # Calculate inter-event time distribution
    inter_event_times = np.diff(df_sorted['ts'])
    kde = stats.gaussian_kde(inter_event_times)
    
    def sample_timestamp(current_times, next_times, rng):
        # Replace None values with a large number (e.g., end_time + time_window)
        next_times = np.where(next_times == None, end_time + time_window, next_times)
        next_times = next_times.astype(float)  # Ensure all values are float
        current_times = current_times.astype(float)
        
        time_diffs = np.minimum(time_window, next_times - current_times)
        sampled_deltas = kde.resample(len(current_times), rng)[0]
        sampled_deltas = np.clip(sampled_deltas, 0, time_diffs)
        jitters = rng.uniform(0, time_diffs * 0.1)
        return current_times + sampled_deltas + jitters
    
    # Extract arrays from DataFrame for batch processing
    batch_src_node_ids = df_sorted['u'].to_numpy()
    batch_dst_node_ids = df_sorted['i'].to_numpy()
    current_batch_times = df_sorted['ts'].to_numpy()
    
    print(f'{start_time=} {end_time=}')

    # Call the sampler once with the entire batch
    neg_src, neg_dst = sampler.sample(
        size=num_negative_samples * len(df_sorted),
        batch_src_node_ids=batch_src_node_ids,
        batch_dst_node_ids=batch_dst_node_ids,
        current_batch_start_time=start_time,
        current_batch_end_time=end_time
    )
    
    print(f'Generated {len(neg_src)} negative samples for {len(df_sorted)} positive samples')
    
    # Prepare timestamps for negative samples
    pos_timestamps = np.repeat(current_batch_times, num_negative_samples)
    next_timestamps = np.append(current_batch_times[1:], [end_time + time_window])
    next_timestamps = np.repeat(next_timestamps, num_negative_samples)
    
    neg_timestamps = sample_timestamp(pos_timestamps, next_timestamps, rng)
    neg_timestamps = np.clip(neg_timestamps, start_time, end_time).astype(int)

    # Create negative samples DataFrame
    negative_samples = pd.DataFrame({
        'u': neg_src,
        'i': neg_dst,
        'ts': neg_timestamps,
        'is_negative': 1
    })
    
    # Add 'is_negative' column to the original DataFrame
    df_sorted['is_negative'] = 0
    
    # Combine positive and negative samples
    df_combined = pd.concat([df_sorted, negative_samples], ignore_index=True)
    
    # Sort the combined DataFrame by 'ts'
    df_combined = df_combined.sort_values('ts').reset_index(drop=True)
    
    print('Combined df length:', len(df_combined), 'Number of negative samples:', len(negative_samples))
    
    return df_combined


def get_poisoned_data(dataset_name='wikipedia', strategy='random', upto=0.7, negative_sample_strategy='random',
                      val_ratio=0.15, test_ratio=0.15, train_only=True):
    # Load data and train val test split
    graph = pd.read_csv('{}/processed_data/{}/ml_{}.csv'.format(scratch_location, dataset_name, dataset_name))
    edge_raw_features = np.load('{}/processed_data/{}/ml_{}.npy'.format(scratch_location, dataset_name, dataset_name))
    node_raw_features = np.load('{}/processed_data/{}/ml_{}_node.npy'.format(scratch_location, dataset_name, dataset_name))
    
    
    print(f'length of full graph is {len(graph)}')
    
    random.seed(2020)
    
    val_time, test_time = list(np.quantile(graph.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))
    new_test_node_set=set()
    # =====================================  Change it to train dataset only  =====================================
    if train_only:  # remove  `not`
        print('In train only')
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
        _graph = train_graph_df.copy(deep=True)
        print(f'length of full training graph is {len(_graph)}')
        
        print(f'{len(_graph) / len(graph):.2%}')
        assert (len(_graph) / len(graph)) > 0.7, 'Not training graph'
    else:
        _graph = graph.copy(deep=True)
    
    # ===================================================================================================================
    os.makedirs(f'{save_location}/poisoned_data/nss_{negative_sample_strategy}/{dataset_name}/{strategy}', exist_ok=True)
    
    OUT_DF = f'{save_location}/poisoned_data/nss_{negative_sample_strategy}/{dataset_name}/{strategy}/{dataset_name}_{strategy}_sparsified_{upto}.csv'
    
    OUT_FEAT = '{}/poisoned_data/nss_{}/{}/{}/ml_{}.npy'.format(save_location, negative_sample_strategy, dataset_name, strategy, dataset_name)
    OUT_NODE_FEAT = '{}/poisoned_data/nss_{}/{}/{}/ml_{}_node.npy'.format(save_location, negative_sample_strategy, dataset_name, strategy, dataset_name)
    # ===================================================================================================================
    
    
    print(f'In get_poisoned_data {dataset_name=}')

    
    EL_graph, EL_edge_raw_features = EL_sparsify(_graph, edge_raw_features,
                                                strategy=strategy, upto=upto, dataset_name=dataset_name,
                                                save=True)
    
    # DONE: remove test node test <--- Do we really need  it?
    # EL_graph = EL_graph[~EL_graph['u'].isin(new_test_node_set)]
    # EL_graph = EL_graph[~EL_graph['i'].isin(new_test_node_set)]
    
    print(f'length of sparsified graph is {len(EL_graph)}')
    
    # exit()
    
    print('Sparsification done')
    
    # get val and test splits too
    # val_mask = np.logical_and(graph['ts'] <= test_time, graph['ts'] > val_time)
    val_graph_df = graph[np.logical_and(graph['ts'] <= test_time, graph['ts'] > val_time)]
    test_graph_df = graph[graph['ts'] > test_time]
    
        
    print('starting negative sampling', end='\r')
    
    start_time, end_time = min(0, _graph['ts'].min()), _graph['ts'].max()  # it should correspond to train data only
    
    
    # we add early negative samples first
    print(f'\twe are adding negative samples first')
    early_start_time = min(0, EL_graph['ts'].iloc[0])
    early_end_time = EL_graph['ts'].iloc[1]
    
    early_src_nodes = graph[graph['ts']<early_end_time]['u'].values
    early_dst_nodes = graph[graph['ts']<early_end_time]['i'].values
    early_interact_times = graph[graph['ts']<early_end_time]['ts'].values
    early_labels = graph[graph['ts']<early_end_time]['label'].values
    early_idx = graph[graph['ts']<early_end_time]['idx'].values
    
    # Assuming early_src_nodes is your NumPy ndarray
    unique_values, counts = np.unique(early_src_nodes, return_counts=True)

    # Print the results
    value_counts = dict(zip(unique_values, counts))
    
    early_sampler = AdversarialEdgeSampler(
        src_node_ids=early_src_nodes,
        dst_node_ids=early_dst_nodes,
        interact_times=early_interact_times, ## What if we give orig_df['ts'].values
        original_edges=set(list(graph[['u', 'i']].to_dict()['u'].items())),
        last_observed_time=early_end_time,
        negative_sample_strategy='random',  # early sampling should always be `random`
        seed=0
    )
    
    early_negative_samples_list = []


    for u, count in tqdm(value_counts.items(), desc='Processing', leave=False):
        neg_src, neg_dst = early_sampler.sample(
                size=count,
                batch_src_node_ids=np.array(u),
                batch_dst_node_ids=early_dst_nodes,
                current_batch_start_time=early_start_time,
                current_batch_end_time=early_end_time
        )
        
        for src, dst in zip(neg_src, neg_dst):
                early_negative_samples_list.append([src, dst])
        
    early_negative_samples_df = pd.DataFrame(early_negative_samples_list, columns=['u', 'i'])
    early_negative_samples_df['ts'] = sorted(early_interact_times)
    early_negative_samples_df['label'] = early_labels
    early_negative_samples_df['idx'] = early_idx
    early_negative_samples_df['is_negative'] = 1
    
    
    EL_graph['is_negative'] = 0 
    
    # Concat EL_graph and early_negative_samples_df
    EL_graph = pd.concat([early_negative_samples_df, EL_graph])
    EL_graph = EL_graph.sort_values(by='ts')
    # Reset the index and drop the old index
    EL_graph = EL_graph.reset_index(drop=True)
    
    print(f'\tlength of sparsified graph after filling early sample is {len(EL_graph)}')
    print(f'\tnegative only samples are {len(early_negative_samples_df[early_negative_samples_df["is_negative"] == 1])}', end='\n\t')
    # ===============================================================================
    
    # then remaining with 1200ms window
    full_graph = add_negative_samples(EL_graph, graph, start_time, end_time, 
                                      negative_sample_strategy=negative_sample_strategy, 
                                      num_negative_samples=5,  # how to choose this ??
                                      seed=random_seed)  # we provide original dataframe too.
    
    
    print('done negative sampling     ')
    
    full_ts_list = _graph['ts'].unique()  # full unique timestamp list of training data
    remainder_ts_list = EL_graph['ts'].unique() # full unique timestamp list of sparsified training data
    to_generate_ts_list = set(full_ts_list).difference(remainder_ts_list)
    
    negative_full_graph = full_graph[full_graph['is_negative'] == 1]
    
    print(f"Length of ns generated graph: {len(full_graph)}")
    print(f"Length of original graph: {len(_graph)}")
    print(f"Length of EL_graph: {len(EL_graph)}")
    print(f"Length of negative_full_graph: {len(negative_full_graph)}")  # 21212
    print(f"Number of timestamps to generate: {len(to_generate_ts_list)}")
    
    num_samples_needed = len(_graph) - len(EL_graph)
    print(f"Number of samples needed: {num_samples_needed}")
    
    # =======================================   Degree nodes to be kept same   =================================
    # 
    
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
    print('********** running KD Tree implementation **********')
    # full_ts_list = _graph['ts'].unique()  # original training graph
    # remainder_ts_list = EL_graph['ts'].unique()  # sparsified training graph
    
    # to_generate_ts_list = set(full_ts_list) - set(remainder_ts_list)
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
    
    
    full_graph = pd.concat([EL_graph, selected_negative_full_graph], ignore_index=True)  # Training sparsified graph + negative samples
    
    print('training: ', len(full_graph), 'Original: ', len(_graph))
    print('val_graph_df: ', len(val_graph_df), 'test_graph_df: ', len(test_graph_df))
        
    # TODO: Next --> check for duplicate    
    
    assert len(full_graph) == len(_graph), "Length of generated training data does not match original training data."
    
    # TODO: Maybe Merge the all split and save.
    full_graph = pd.concat([full_graph, val_graph_df, test_graph_df])
    full_graph = full_graph.sort_values(by='ts')
    full_graph.reset_index(drop=True, inplace=True)
    
    full_graph = full_graph.sort_values('ts').reset_index(drop=True)

    print(f"Final length of full_graph: {len(full_graph)}")
    if len(graph) != len(full_graph):
        print(f"Warning: Length mismatch. Original: {len(graph)}, New: {len(full_graph)}")

    assert len(graph) == len(full_graph), f'Make dataset same again, Original graph is {len(graph)=}, while reconstructed is {len(full_graph)=}'    
    
    full_graph.to_csv(OUT_DF)  # edge-list
    np.save(OUT_FEAT, EL_edge_raw_features)  # edge features
    np.save(OUT_NODE_FEAT, node_raw_features)  # node features
    
    print(f'Sparsified {dataset_name}.')

if __name__ == '__main__':
    
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
                        'ts_tpr_remove_jensen_shannon_divergence', 'ts_tpr_remove_TER', 'ts_tpr_remove_combined_ter', 'ts_tpr_remove_rec_mss',
                        'preference', 'jaccard', 'pagerank', 'degree'],
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