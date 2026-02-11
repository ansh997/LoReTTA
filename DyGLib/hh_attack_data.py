import os
import random
import argparse
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from tqdm.auto import tqdm
from itertools import product
from multiprocessing import Pool
from collections import defaultdict
from distutils.dir_util import copy_tree
from preprocess_data.preprocess_data import check_data
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# good_seed = 1729928109

random.seed(1729928109)
np.random.seed(1729928109)

from hh_attack_utils import calculate_node_degrees, create_node_mapping, find_missing_edges,\
    get_nodes_in_time_window, EdgeTimestampSelector, TemporalSampler, TPRTimestampSelector, NonBipartiteEdgeSelector
    

from preprocess_data.sparsify_data import EL_sparsify

save_location = rf'/raid/t2/TGN_adv/'
scratch_location = rf'/raid/t2/TGN_adv/DyGlib'

def _add_negative_samples(sprs_df, orig_df, full_graph):
    
    # Add negative samples to the graph for first few missing timestamps
    early_start_time, early_end_time = sprs_df['ts'].iloc[0], sprs_df['ts'].iloc[1]
    
    print(early_start_time, early_end_time)

    all_early_missing_edges_df = find_missing_edges(orig_df, sprs_df, start_time=early_start_time, end_time=early_end_time, batch_size=24)

    print('Chck 1: ', len(pd.concat(all_early_missing_edges_df)))
    
    assert 0 == 1, 'Stop here'
    
    flag = False
    
    new_edges = process_batches_concurrently(all_early_missing_edges_df, full_graph, max_workers=24)
    new_edges = list(itertools.chain.from_iterable(new_edges))

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
        all_late_missing_edges_df = find_missing_edges(orig_df, sprs_df, batch_size=24, start_time=start_time, end_time=end_time)
        

    new_edges_ = process_batches_concurrently(all_late_missing_edges_df, orig_df, max_workers=24)
    new_edges_ = list(itertools.chain.from_iterable(new_edges_))
    print(f"Total new edges for last df generated: {len(new_edges_)}")
    new_edges.extend(new_edges_)
    
    all_missing_edges_df = all_early_missing_edges_df + all_late_missing_edges_df
    print(f"Total missing edges: {len(pd.concat(all_missing_edges_df))}", end=' ')
    print(f"Total new edges generated: {len(new_edges)}")
    return new_edges

def add_negative_samples(sprs_df, orig_df, full_graph, is_bipartite=True):
    """
    Add negative samples to the graph across the full time range in one process.
    
    Parameters:
    -----------
    sprs_df : pd.DataFrame 
        Sparse dataframe with subset of edges
    orig_df : pd.DataFrame
        Original dataframe with all edges
    full_graph : NetworkX graph
        Complete graph structure
        
    Returns:
    --------
    list
        List of new negative sample edges
    """
    print(f'{is_bipartite = }')
    # Use the full time range from start to end
    start_time = orig_df['ts'].min()  # min(0, sprs_df['ts'].min())  # Include 0 if it's earlier
    end_time = sprs_df['ts'].max()
    
    # Process all batches to generate negative samples
    if is_bipartite:
        print('\tProcessing here since Bipartie.')
        # Find all missing edges across the full time range
        all_missing_edges_df = find_missing_edges(
            orig_df, 
            sprs_df,
            start_time=start_time,
            end_time=end_time,
            batch_size=24  # Can adjust batch size as needed
        )
        
        new_edges = process_batches_concurrently(
            all_missing_edges_df,
            full_graph,
            max_workers=24,
            is_bipartite=is_bipartite
        )
    else:
        print('\tProcessing here since Non-Bipartie.')
        all_missing_edges_df = find_missing_edges(
        orig_df, 
        sprs_df,
        start_time=start_time,
        end_time=end_time,
        batch_size=4  # Can adjust batch size as needed
        )
        
        # process_batches_sequentially(all_later_missing_edges_df, orig_df, is_bipartite=False, sequential=True)
        new_edges = process_batches_sequentially(
            all_missing_edges_df,
            full_graph,
            is_bipartite=is_bipartite,
            sequential=True
        )
    
    # Flatten the list of edges
    new_edges = list(itertools.chain.from_iterable(new_edges))
    
    # Print summary statistics
    print(f"Total missing edges identified: {len(pd.concat(all_missing_edges_df))}")
    print(f"Total new negative edges generated: {len(new_edges)}")
    
    return new_edges

def get_poisoned_data(dataset_name='wikipedia', strategy='random', upto=0.7, val_ratio=0.15, test_ratio=0.15, 
                    random_seed = 0, is_bipartite = True):
    random.seed(random_seed)
    # Load data and train val test split
    graph = pd.read_csv('{}/processed_data/{}/ml_{}.csv'.format(scratch_location, dataset_name, dataset_name))
    graph['ts'] = graph['ts'].astype(int)
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
    print(f'Length of train data: {len(train_graph_df)} {len(graph)}')
    assert np.round(len(train_graph_df) / len(graph), 2) >= 0.7, 'Not training graph'

    EL_graph, EL_edge_raw_features, _ = EL_sparsify(train_graph_df, edge_raw_features,
                                                strategy=strategy, upto=upto,
                                                dataset_name=dataset_name)
    
    print(f'length of sparsified graph: {len(EL_graph)}, percentage of sparsified train data: {len(EL_graph) / len(train_graph_df):.2%}')
    print('Sparsification done')
    print('*'*80)
    
    # get val and test splits too
    val_graph_df = graph[np.logical_and(graph['ts'] <= test_time, graph['ts'] > val_time)]
    test_graph_df = graph[graph['ts'] > test_time]
    
        
    print('starting negative sampling', end='\r')
    
    
    negative_samples = add_negative_samples(EL_graph, train_graph_df, graph, is_bipartite)
    
    print('done negative sampling     ')
    
    print('Full training data length: ', len(train_graph_df), len(EL_graph)+len(negative_samples))
    # TODO: Remove after testing
    
    # TODO: add a column adversary to the negative samples
    
    negative_samples = pd.DataFrame(negative_samples, columns=['u', 'i', 'ts'])
    negative_samples['adv'] = 1
    negative_samples.sort_values('ts', inplace=True)
    negative_samples.reset_index(drop=True, inplace=True)
        
    EL_graph['adv'] = 0
    att_graph = pd.concat([EL_graph, negative_samples], ignore_index=True)
    att_graph = att_graph.sort_values('ts').reset_index(drop=True)
    
    #TODO: add val and test data to attack graph and save it

    # print(f"Final length of full_graph: {len(full_graph)}")
    # if len(graph) != len(full_graph):
    #     print(f"Warning: Length mismatch. Original: {len(graph)}, New: {len(full_graph)}")

    
    
    # TODO: save train/val/test splits separately or same file
    print('Before: ', len(att_graph))
    val_graph_df['adv'] = 0
    test_graph_df['adv'] = 0
    att_graph = pd.concat([att_graph, val_graph_df, test_graph_df], ignore_index=True)
    print('After: ', len(att_graph))
    # assert len(att_graph) == len(graph), f'Make dataset same again, Original graph is {len(graph)=}, while is reconstructed {len(att_graph)=}'
    
    att_graph.to_csv(OUT_DF)  # edge-list
    np.save(OUT_FEAT, EL_edge_raw_features)  # edge features
    np.save(OUT_NODE_FEAT, node_raw_features)  # node features
    padding_length = abs((80 - (15 + len(dataset_name))) // 2)
    print('* ' * padding_length, f'Saved Attacked {dataset_name} at {OUT_DF}.', ' *' * padding_length)
    
def fast_but_not_working_process_single_batch(missing_edges_df, full_graph, idx, tpr_selector=False):
    """Optimized batch processing"""
    print(f'Processing batch {idx + 1}')

    # Calculate degrees and generate HH graph
    removed_edges = missing_edges_df[['u', 'i']].values.tolist()
    deficit_src_degree, deficit_dst_degree = calculate_node_degrees(missing_edges_df)

    # Initialize temporal sampler
    time_window = 1200
    sampler = TemporalSampler(missing_edges_df, time_window=time_window, seed=0)
    num_samples = sum(deficit_src_degree.values())
    sampled_timestamps = sampler.assign_timestamps(num_samples)

    # Prepare full graph data
    df_sorted_array = full_graph[['u', 'i', 'ts']].to_numpy()  # Ensure this aligns with numpy indexing
    original_edges = set(map(tuple, full_graph[['u', 'i']].values))
    min_src, max_src = missing_edges_df['u'].min(), missing_edges_df['u'].max()
    min_dst, max_dst = missing_edges_df['i'].min(), missing_edges_df['i'].max()

    map_edges_to_ts = defaultdict(list)
    src_node_counts = defaultdict(int)
    dst_node_counts = defaultdict(int)

    for timestamp in tqdm(sampled_timestamps, desc='\tProcessing Timestamps', ncols=80):
        nodes_in_window = get_nodes_in_time_window(df_sorted_array, timestamp, time_window)
        
        # Filter nodes within ranges
        src_nodes_orig = nodes_in_window[(nodes_in_window >= min_src) & (nodes_in_window <= max_src)]
        dst_nodes_orig = nodes_in_window[(nodes_in_window >= min_dst) & (nodes_in_window < max_dst)]

        # Vectorized feasible edges generation
        feasible_edges = np.array(list(product(src_nodes_orig, dst_nodes_orig)))
        feasible_edges = feasible_edges[~np.isin(feasible_edges, original_edges)]

        for src, dst in feasible_edges:
            map_edges_to_ts[(src, dst)].append(timestamp)
            src_node_counts[src] += 1
            dst_node_counts[dst] += 1

    # Selector logic remains unchanged
    degree_dict = {**deficit_src_degree, **deficit_dst_degree}
    selector_cls = TPRTimestampSelector if tpr_selector else EdgeTimestampSelector
    selector = selector_cls(
        feasible_edges_dict=map_edges_to_ts,
        original_df=missing_edges_df,
        degree_dict=degree_dict,
        removed_edges=removed_edges,
        src_nodes=set(deficit_src_degree.keys()),
        dst_nodes=set(deficit_dst_degree.keys()),
        time_window=time_window,
    )
    if tpr_selector:
        selector.beta = 0.5
        selector.alpha = 0.15
        selector.dataset_name = f"wikipedia_batch_{idx}"

    gen_edges = selector.select_edges_timestamps()
    return gen_edges

def process_single_batch(missing_edges_df, full_graph, idx, already_added=set(), is_bipartite=True):
    """Modified batch processing with better error handling
    
    Works faster for wiki like small dataset but very slow for large datasets like reddit
    
    """
    # try:
    print(f'Processing batch {idx+1},')

    # Calculate degrees and generate HH graph
    removed_edges = missing_edges_df[['u', 'i']].values.tolist()
    deficit_src_degree, deficit_dst_degree = calculate_node_degrees(missing_edges_df)
    


    # Initialize temporal sampler
    time_window = 1200
    sampler = TemporalSampler(missing_edges_df, time_window=time_window, seed=0)
    num_samples = sum(deficit_src_degree.values())
    sampled_timestamps = sampler.assign_timestamps(num_samples)

    # Process edges and timestamps
    df_sorted = full_graph.sort_values('ts')
    original_edges = set(tuple(x) for x in df_sorted[['u', 'i']].values)
    min_src, max_src = full_graph['u'].min(), full_graph['u'].max()
    min_dst, max_dst = full_graph['i'].min(), full_graph['i'].max()
    
    print(f'{min_src = }, {max_src = }')
    print(f'{min_dst = }, {max_dst = }')
    
    assert all(min_src <= key <= max_src for key in deficit_src_degree.keys()), \
            "Not all deficit_src_degree keys are within the range [min_src, max_src]"
    
    assert all(min_dst <= key <= max_dst for key in deficit_dst_degree.keys()), \
            "Not all deficit_dst_degree keys are within the range [min_dst, max_dst]"
    print("Assertion passed: All deficit_degree keys are within the specified source range.")
    
    map_edges_to_ts = defaultdict(list)
    src_node_counts = defaultdict(int)
    dst_node_counts = defaultdict(int)

    for timestamp in tqdm(sampled_timestamps, desc='\tProcessing Timestamps', ncols=80, leave=False):
        nodes_in_window = get_nodes_in_time_window(df_sorted, timestamp, time_window)

        
        src_nodes_orig = [n for n in nodes_in_window 
                        if min_src <= n <= max_src]
        dst_nodes_orig = [n for n in nodes_in_window 
                        if min_dst <= n < max_dst]
        
        assert all(min_src <= key <= max_src for key in src_nodes_orig), \
            "Not all deficit_src_degree keys are within the range [min_src, max_src]"
    
        assert all(min_dst <= key <= max_dst for key in dst_nodes_orig), \
                "Not all deficit_dst_degree keys are within the range [min_dst, max_dst]"
        # print("Assertion passed: All deficit_degree keys are within the specified source range.")

        targeted_edges = []
        
        # Do we want self loops?
        if is_bipartite:
            targeted_edges = [
            (src, dst) for src, dst in product(src_nodes_orig, dst_nodes_orig)
            if (src, dst) not in original_edges
            ]
        
        else:
            
            for src, dst in product(src_nodes_orig, dst_nodes_orig):
            # Skip self-loops
            # print(src, dst, end=' ')
                if src != dst:
                    # Check if neither the edge nor its reverse exists in the original graph
                    if (src, dst) not in original_edges and (dst, src) not in original_edges:
                        # Check if neither the edge nor its reverse has already been added
                        if (src, dst) not in already_added and (dst, src) not in already_added:
                            # Assertions to verify the integrity
                            assert (dst, src) not in already_added, f'Intentional, {already_added}'
                            assert (src, dst) not in already_added, f'Intentional, {already_added}'
                            assert (dst, src) not in original_edges, 'Intentional'
                            assert (src, dst) not in original_edges, 'Intentional'
                            assert src != dst, 'Intentional'
                            # print('Added')
                            targeted_edges.append((src, dst))
                            # Mark both the edge and its reverse as added
                            already_added.add((src, dst))
                            already_added.add((dst, src))
        # print(len(targeted_edges))

        for src, dst in targeted_edges:
            map_edges_to_ts[(src, dst)].append(timestamp)
            src_node_counts[src] += 1
            dst_node_counts[dst] += 1

    # Create selector and generate edges
    degree_dict = deepcopy(deficit_src_degree)
    degree_dict.update(deficit_dst_degree)

    if not is_bipartite:  # this has to be used for non-bipartite
        print(f'{is_bipartite = } Using NonBipartiteEdgeSelector')
        selector = NonBipartiteEdgeSelector(
        feasible_edges_dict=map_edges_to_ts,
        original_df=missing_edges_df,
        degree_dict=degree_dict,
        removed_edges=removed_edges,
        time_window=time_window,)
    else:
        print(f'{is_bipartite = } Using EdgeTimestampSelector')
        selector = EdgeTimestampSelector(
        feasible_edges_dict=map_edges_to_ts,
        original_df=missing_edges_df,
        degree_dict=degree_dict,
        removed_edges=removed_edges,
        src_nodes=set(deficit_src_degree.keys()),  
        dst_nodes=set(deficit_dst_degree.keys()),  
        time_window=time_window,
    )
    
    gen_edges = selector.select_edges_timestamps()
    return gen_edges

def process_batches_concurrently(missing_edges_df, full_graph, max_workers=None, is_bipartite=True):
    """Process all batches in parallel"""
    full_neg_samples = []
    already_added = set()   # For non-bipartite graphs
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create futures for each batch
        futures = [
            executor.submit(process_single_batch, batch_df, full_graph, idx, already_added=already_added, is_bipartite=is_bipartite)
            for idx, batch_df in enumerate(missing_edges_df)
        ]
        
        # Collect results as they complete
        for future in as_completed(futures):
            new_edges = future.result()
            full_neg_samples.append(new_edges)
                
    return full_neg_samples


def process_batches_sequentially(missing_edges_df, full_graph, max_workers=None, is_bipartite=True, sequential=False):
    """Process all batches in parallel"""
    
    def _process_single_batch(missing_edges_df, full_graph, idx, already_added, is_bipartite=True):
        """Modified batch processing with better error handling
        
        Works faster for wiki like small dataset but very slow for large datasets like reddit
        
        """
        # try:
        print(f'Processing batch {idx+1}')

        # Calculate degrees and generate HH graph
        # removed_edges = missing_edges_df[['u', 'i']].values.tolist()
        deficit_src_degree, deficit_dst_degree = calculate_node_degrees(missing_edges_df)
        # Create selector and generate edges
        degree_dict = deepcopy(deficit_src_degree)
        degree_dict.update(deficit_dst_degree)


        # Initialize temporal sampler
        time_window = 1200
        sampler = TemporalSampler(missing_edges_df, time_window=time_window, seed=0)
        num_samples = sum(degree_dict.values())
        sampled_timestamps = sampler.assign_timestamps(num_samples)

        # Process edges and timestamps
        df_sorted = full_graph.sort_values('ts')
        original_edges = set(tuple(x) for x in df_sorted[['u', 'i']].values)
        
        map_edges_to_ts = defaultdict(list)
        src_node_counts = defaultdict(int)
        dst_node_counts = defaultdict(int)
         

        for timestamp in tqdm(sampled_timestamps, desc='\tProcessing Timestamps', ncols=80, leave=False):
            nodes_in_window = get_nodes_in_time_window(df_sorted, timestamp, time_window)

            targeted_edges = []

            for src, dst in product(nodes_in_window, nodes_in_window):
                # Skip self-loops
                # print(src, dst, end=' ')
                if src != dst:
                    # Check if neither the edge nor its reverse exists in the original graph
                    if (src, dst) not in original_edges and (dst, src) not in original_edges:
                        # Check if neither the edge nor its reverse has already been added
                        if (src, dst) not in already_added and (dst, src) not in already_added:
                            # Assertions to verify the integrity
                            assert (dst, src) not in already_added, f'Intentional, {already_added}'
                            assert (src, dst) not in already_added, f'Intentional, {already_added}'
                            assert (dst, src) not in original_edges, 'Intentional'
                            assert (src, dst) not in original_edges, 'Intentional'
                            assert src != dst, 'Intentional'
                            # print('Added')
                            targeted_edges.append((src, dst))
                            # Mark both the edge and its reverse as added
                            already_added.add((src, dst))
                            already_added.add((dst, src))
            
            for edge in targeted_edges:
                assert edge not in original_edges, f"Invalid edge: {edge} in original edges."
                assert edge[::-1] not in original_edges, f"Invalid reverse edge: {edge[::-1]} in original edges."

            
            for src, dst in targeted_edges:
                assert (src, dst) not in original_edges, f"Invalid edge: {src, dst} in original edges."
                assert (dst, src) not in original_edges, f"Invalid edge: {dst, src} in original edges."
                map_edges_to_ts[(src, dst)].append(timestamp)
                src_node_counts[src] += 1
                dst_node_counts[dst] += 1
        
        selector = NonBipartiteEdgeSelector(
            feasible_edges_dict=map_edges_to_ts,
            original_df=missing_edges_df,
            degree_dict=degree_dict,
            removed_edges=original_edges,
            time_window=time_window,
        )
        gen_edges = selector.select_edges_timestamps()
        
        # breaking_point = 0
        
        # # TODO: If len(gen_edges) < len(missing_edges_df) --> resample again
        # while len(missing_edges_df) - len(gen_edges) > 1000:
        #     if breaking_point >= 250:
        #         break
            
        #     breaking_point+=1
            
        #     # TODO: Update the degree_dict here.
        #     gen_edges_df = pd.DataFrame(gen_edges, columns=['u', 'i', 'ts'])
        #     new_deficit_src_degree, new_deficit_dst_degree = calculate_node_degrees(gen_edges_df)
            
        #     # Update the degree for generate edges
        #     new_degree_dict = deepcopy(new_deficit_src_degree)
        #     new_degree_dict.update(new_deficit_dst_degree)
            
        #     all_nodes_in_consideration = set(degree_dict.keys()) | set(new_degree_dict.keys())
            
        #     remaining_degree_dict = {}
            
        #     for node in all_nodes_in_consideration:
        #         initial_deficit_for_node = degree_dict.get(node, 0)  # How many connections this node originally needed in this batch
        #         generated_degree_for_node = new_degree_dict.get(node, 0)  # How many connections this node received in the first pass

        #         # Calculate how many connections are still needed
        #         still_needed = initial_deficit_for_node - generated_degree_for_node

        #         # Only include in remaining_degree_dict if more connections are actually needed
        #         if still_needed > 0:
        #             remaining_degree_dict[node] = still_needed
        #         else:
        #             remaining_degree_dict[node] = 0
        #     print(f'We are resampling {breaking_point}th time.')
        #     sampler = TemporalSampler(missing_edges_df, time_window=time_window, seed=breaking_point+1)
        #     new_sampled_timestamps = sampler.assign_timestamps(len(missing_edges_df) - len(gen_edges))
        #     map_edges_to_ts = defaultdict(list)
        #     src_node_counts = defaultdict(int)
        #     dst_node_counts = defaultdict(int)
            

        #     for timestamp in tqdm(new_sampled_timestamps, desc='\tProcessing Timestamps', ncols=80, leave=False):
        #         nodes_in_window = get_nodes_in_time_window(df_sorted, timestamp, time_window)

        #         targeted_edges = []

        #         for src, dst in product(nodes_in_window, nodes_in_window):
        #             if src != dst:
        #                 # Check if neither the edge nor its reverse exists in the original graph
        #                 if (src, dst) not in original_edges and (dst, src) not in original_edges:
        #                     # Check if neither the edge nor its reverse has already been added
        #                     if (src, dst) not in already_added and (dst, src) not in already_added:
        #                         # Assertions to verify the integrity
        #                         assert (dst, src) not in already_added, f'Intentional, {already_added}'
        #                         assert (src, dst) not in already_added, f'Intentional, {already_added}'
        #                         assert (dst, src) not in original_edges, 'Intentional'
        #                         assert (src, dst) not in original_edges, 'Intentional'
        #                         assert src != dst, 'Intentional'
        #                         # print('Added')
        #                         targeted_edges.append((src, dst))
        #                         # Mark both the edge and its reverse as added
        #                         already_added.add((src, dst))
        #                         already_added.add((dst, src))
                
        #         for edge in targeted_edges:
        #             assert edge not in original_edges, f"Invalid edge: {edge} in original edges."
        #             assert edge[::-1] not in original_edges, f"Invalid reverse edge: {edge[::-1]} in original edges."

                
        #         for src, dst in targeted_edges:
        #             assert (src, dst) not in original_edges, f"Invalid edge: {src, dst} in original edges."
        #             assert (dst, src) not in original_edges, f"Invalid edge: {dst, src} in original edges."
        #             map_edges_to_ts[(src, dst)].append(timestamp)
        #             src_node_counts[src] += 1
        #             dst_node_counts[dst] += 1
            
        #     selector = NonBipartiteEdgeSelector(
        #         feasible_edges_dict=map_edges_to_ts,
        #         original_df=missing_edges_df,
        #         degree_dict=remaining_degree_dict,
        #         removed_edges=original_edges,
        #         time_window=time_window,
        #     )
        #     regen_edges = selector.select_edges_timestamps()
        #     gen_edges.extend(regen_edges)
        #     print(f'\t{len(regen_edges) = } {len(gen_edges) = }')
            
        return gen_edges

    full_neg_samples = []
    already_added = set()   # For non-bipartite graphs
    
    if sequential:           
        for idx, batch_df in enumerate(missing_edges_df):
            new_edges = _process_single_batch(batch_df, full_graph, idx, already_added=already_added, is_bipartite=is_bipartite)
            print()
            full_neg_samples.append(new_edges) 
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Create futures for each batch
            futures = [
                executor.submit(_process_single_batch, batch_df, full_graph, idx, already_added=already_added, is_bipartite=is_bipartite)
                for idx, batch_df in enumerate(missing_edges_df)
            ]
            
            # Collect results as they complete
            for future in as_completed(futures):
                new_edges = future.result()
                full_neg_samples.append(new_edges)
        
    return full_neg_samples

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser('Interface for poisoning datasets')
    parser.add_argument('-d', '--dataset_name', type=str,
                        choices=['wikipedia', 'reddit', 'mooc', 'lastfm', 'myket', 'enron', 'SocialEvo', 'uci',
                                'Flights', 'CanParl', 'USLegis', 'UNtrade', 'UNvote', 'Contacts', 'bitcoin'],
                        help='Dataset name', default='wikipedia')
    # parser.add_argument('--node_feat_dim', type=int, default=172, help='Number of node raw features')
    parser.add_argument('-u', '--upto', type=float, help='sparsify upto', default=0.7)
    parser.add_argument('-s', '--strategy', type=str, default='random', choices=['random',
                        'tpr_remove', 'ts_tpr_remove_ss', 'ts_tpr_remove_inc', 'ts_tpr_remove_mss',
                        'ts_tpr_remove_mss_2', 'ts_tpr_remove_cosine', 'ts_tpr_remove_euclidean',
                        'ts_tpr_remove_jaccard', 'ts_tpr_remove_wasserstein', 'ts_tpr_remove_kl_divergence', 'ts_tpr_remove_chebyshev',
                        'ts_tpr_remove_jensen_shannon_divergence', 'ts_tpr_remove_TER', 'ts_tpr_remove_combined_ter', 'ts_tpr_remove_rec_mss',
                        'preference', 'jaccard', 'pagerank', 'degree', 'ts_tpr_remove_mean_TER'],
                        help='strategy for the sparsification')
        

    args = parser.parse_args()
    print(args)

    if args.dataset_name in []:  # DONE: remove UCI from here
        Path("{}/poisoned_data/{}/".format(save_location, args.dataset_name)).mkdir(parents=True, exist_ok=True)
        copy_tree("{}/DG_data/{}/".format(scratch_location, args.dataset_name), "{}/poisoned_data/{}/".format(save_location, args.dataset_name))
        print(f'Not implemented for enron, SocialEvo graph yet.')
    else:
        Path("{}/poisoned_data/{}/".format(scratch_location, args.dataset_name)).mkdir(parents=True, exist_ok=True)
        copy_tree("{}/DG_data/{}/".format(scratch_location, args.dataset_name), "{}/poisoned_data/{}/".format(save_location, args.dataset_name))
        # bipartite dataset
        if args.dataset_name in ['wikipedia', 'reddit', 'mooc', 'lastfm', 'myket']:
            print(f'{args.dataset_name} is_bipartite = True')
            get_poisoned_data(dataset_name=args.dataset_name, strategy=args.strategy, upto=args.upto, is_bipartite = True)
        else:
            print(f'{args.dataset_name} is_bipartite = False')
            get_poisoned_data(dataset_name=args.dataset_name, strategy=args.strategy, upto=args.upto, is_bipartite = False)
        print(f'{args.dataset_name} is processed successfully.')

        if args.dataset_name not in ['myket']:
            check_data(args.dataset_name)
        print(f'{args.dataset_name} passes the checks successfully.')
    end_time = time.time()
    print("Total Time taken: ", end_time-start_time)
