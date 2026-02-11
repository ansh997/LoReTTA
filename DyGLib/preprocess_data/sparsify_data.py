from collections import Counter
import itertools
import os
import random
import sys
import getpass
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from distutils.dir_util import copy_tree
from .preprocess_data import check_data
from typing import Any, Tuple
import time

random.seed(1729928109)
np.random.seed(1729928109)


# remove  relative import from here
from .temporal_pr import temporal_pagerank_with_timestamps, calc_timestamp_pagerank,\
    calc_inc_timestamp_pagerank, optimized_calc_inc_timestamp_pagerank,\
    get_temporal_pagerank, mean_shift_removal, mean_shift_removal2, LW_compute_mean_shifts_with_metrics, calculate_temporal_edge_rank_working,\
    calculate_combined_temporal_edgerank, calculate_temporal_edge_rank_mean
import networkx as nx

# Set the working directory to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # this might cause issue
sys.path.append(project_root)

# scratch_location = r'/scratch/hmnshpl'
# scratch_location = rf'/scratch/{getpass.getuser()}'
save_location = rf'/raid/t2/TGN_adv' # '/home/t2/TGN_adv'
scratch_location = rf'/raid/t2/TGN_adv/DyGlib'

# '/raid/t2/TGN_adv'

def Old_Working_EL_sparsify(graph, edge_raw_features, strategy='random', upto=0.7, dataset_name='', save=False, seed=0):
    """_summary_

    Args:
        graph (_type_): Original graph df
        edge_raw_features (_type_): Original edge raw features

    Returns:
        _type_: sparsified graph with edge features
    """
    if dataset_name == '':
        raise ValueError('Please pass a dataset name.')
    else:
        print(f'\tIn EL_sparsify {dataset_name=}')
    
    # strategy = strategy.lower() # making it case insensitive
    tmp_graph = graph.copy(deep=True)
    tmp_graph = tmp_graph.sort_values(by=['u', 'i', 'ts'])
    
    save_dir = f'{save_location}/sparsified_data/{dataset_name}/{strategy}'
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Exclude the first and last rows based on 'u' and 'i'
    # grouped = tmp_graph.groupby(['u', 'i'])
    modified_df = tmp_graph.copy(deep=True)  # grouped.apply(lambda x: x.iloc[1:-1]).reset_index(drop=True)
    
    sample_size = int(len(modified_df) * upto)
    
    already_sparsified = False
    
    # # Group by 'u' and 'i' and capture the first and last interactions
    # first_interactions = grouped.first().reset_index()
    # last_interactions = grouped.last().reset_index()
    
    # TODO: add random interactions --> 10% - 30%
    # we can do different selection strategy - right now random only - always keep strategy in small caps here
    
    if strategy == 'random':
        # Randomly sample rows without replacement
        sampled_df = modified_df.sample(n=sample_size, random_state=seed)
    elif strategy == 'tpr_remove':
        # calculate page rank of a dataset
        # sample upto given percentage to be removed - since we are rejecting hence it should be different than selecting
        # use top % nodes and remove (1-upto) nodes
        # naive method
        # graph = build_graph(tmp_graph)
        # page_rank_scores = temporal_page_rank(graph)
        # Official method
        page_rank_scores = get_temporal_pagerank(tmp_graph)
        
        # Sort nodes by PageRank scores
        sorted_nodes = sorted(page_rank_scores.items(), key=lambda item: item[1], reverse=True)
        # Calculate the top upto% of nodes
        top_x_percent_count = int(len(sorted_nodes) * (1-upto))
        top_x_percent_nodes = sorted_nodes[:top_x_percent_count]
        # Extract the node IDs from the top 30 percent nodes
        top_x_percent_node_ids = {node for node, _ in top_x_percent_nodes}

        # Remove rows from graph_df where either source or target node is in the top 30% nodes
        sampled_df = modified_df[~modified_df['u'].isin(top_x_percent_node_ids) & ~modified_df['i'].isin(top_x_percent_node_ids)]
    elif strategy == 'ts_tpr_remove_ss':
        # calculate timestamp level tpr
        # ts_level_tpr = temporal_pagerank_with_timestamps(tmp_graph)
        
        ts_level_tpr = calc_timestamp_pagerank(tmp_graph)  # snapshot implementation
        
        ts_aggregated_scores= {}
        for ts, scores in ts_level_tpr.items():
            agg_scores=sum(scores.values())
            ts_aggregated_scores[ts]=agg_scores
        
        # Sort timestamps by aggregated PageRank scores
        sorted_timestamps = sorted(ts_aggregated_scores.items(),
                                key=lambda item: item[1], reverse=True)
        
        top_x_percent_count = int(len(sorted_timestamps) *(1-upto))
        
        top_x_percent_timestamps = [timestamp for timestamp, _ in sorted_timestamps[:top_x_percent_count]]
        
        sampled_df = modified_df[~modified_df['ts'].isin(top_x_percent_timestamps)]  # should we keep full training data - as we are already dropping duplicates
    elif strategy == 'ts_tpr_remove_inc':
        # incremental 
        # ts_level_tpr = calc_inc_timestamp_pagerank(tmp_graph)  # incremental implementation
        ts_level_tpr = optimized_calc_inc_timestamp_pagerank(tmp_graph)  # incremental implementation
        
        ts_aggregated_scores= {}
        for ts, scores in ts_level_tpr.items():
            agg_scores=sum(scores.values()) # what different aggregation can I try here?
            ts_aggregated_scores[ts]=agg_scores
        
        # Sort timestamps by aggregated PageRank scores
        sorted_timestamps = sorted(ts_aggregated_scores.items(),
                                key=lambda item: item[1], reverse=True)
        
        top_x_percent_count = int(len(sorted_timestamps) *(1-upto))
        
        top_x_percent_timestamps = [timestamp for timestamp, _ in sorted_timestamps[:top_x_percent_count]]
        
        sampled_df = modified_df[~modified_df['ts'].isin(top_x_percent_timestamps)]
    elif strategy == 'ts_tpr_remove_mss':  # problem started from here
        # based in maximum mean shift strategy
        metric = strategy
        filename = f'{save_location}/sparsified_data/{dataset_name}/{strategy}/{dataset_name}_{metric}_sparsified_{upto}.csv'
        print('Save:', save)
        if os.path.exists(filename):
            print(f'\treading {os.path.basename(filename)}...', end=' ')
            sampled_df=pd.read_csv(filename)
            sampled_df = sampled_df.loc[:, ~sampled_df.columns.str.contains('^Unnamed')]
            already_sparsified = True
            print(' done')
        else:
            mean_shifts = mean_shift_removal(tmp_graph, dataset_name=dataset_name)            
            threshold_index = int(len(mean_shifts) * (1-upto))
            top_mean_shifts = mean_shifts[:threshold_index]
            
            top_x_percent_timestamps = [ts for ts, _ in top_mean_shifts]
            
            sampled_df = modified_df[~modified_df['ts'].isin(top_x_percent_timestamps)]
            
            if save:
                sampled_df.drop(['Unnamed: 0'], axis=1).to_csv(filename)
        # print('data sampling successful.')
    elif strategy == 'ts_tpr_remove_mss_2':
        # based in maximum mean shift strategy
        metric = strategy
        filename = f'{save_location}/sparsified_data/{dataset_name}/{strategy}/{dataset_name}_{metric}_sparsified_{upto}.csv'
        if os.path.exists(filename):
            print(f'\treading {os.path.basename(filename)}...', end=' ')
            sampled_df=pd.read_csv(filename)
            sampled_df = sampled_df.loc[:, ~sampled_df.columns.str.contains('^Unnamed')]
            already_sparsified = True
            print(' done')
        else:
            save=True
            mean_shifts = mean_shift_removal2(tmp_graph, dataset_name=dataset_name)
        
            
            threshold_index = int(len(mean_shifts) * (1-upto))
            top_mean_shifts = mean_shifts[:threshold_index]
            
            top_x_percent_timestamps = [ts for ts, _ in top_mean_shifts]
            
            sampled_df = modified_df[~modified_df['ts'].isin(top_x_percent_timestamps)]
            if save:
                sampled_df.drop(['Unnamed: 0'], axis=1).to_csv(filename)
                print(filename, ' saved.')
        # print('data sampling successful.')
    elif strategy == 'ts_tpr_remove_cosine':
        # based on maximum mean shift strategy
        metric = strategy
        filename = f'{save_location}/sparsified_data/{dataset_name}/{strategy}/{dataset_name}_{metric}_sparsified_{upto}.csv'
        if os.path.exists(filename):
            print(f'reading {filename}...', end=' ')
            sampled_df=pd.read_csv(filename)
            sampled_df = sampled_df.loc[:, ~sampled_df.columns.str.contains('^Unnamed')]
            already_sparsified = True
            print(' done')
        else:
            save=True
            mean_shifts = compute_mean_shifts_with_metrics(tmp_graph, metric='cosine', dataset_name=dataset_name)
            
            print('back to sparsify_data file....')
            
            threshold_index = int(len(mean_shifts) * (1-upto))
            top_mean_shifts = mean_shifts[:threshold_index]
            
            top_x_percent_timestamps = [ts for ts, _ in top_mean_shifts]
            
            sampled_df = modified_df[~modified_df['ts'].isin(top_x_percent_timestamps)]
            if save:
                sampled_df.drop(['Unnamed: 0'], axis=1).to_csv(filename)
                print(filename, ' saved.')
        # print('data sampling successful.')
    elif strategy == 'ts_tpr_remove_euclidean':
        # based on maximum mean shift strategy
        metric = strategy
        filename = f'{save_location}/sparsified_data/{dataset_name}/{strategy}/{dataset_name}_{metric}_sparsified_{upto}.csv'
        if os.path.exists(filename):
            print(f'reading {filename}...', end=' ')
            sampled_df=pd.read_csv(filename)
            sampled_df = sampled_df.loc[:, ~sampled_df.columns.str.contains('^Unnamed')]
            already_sparsified = True
            print(' done')
        else:
            save=True
            mean_shifts = compute_mean_shifts_with_metrics(tmp_graph, metric='euclidean', dataset_name=dataset_name)
            
            print('back to sparsify_data file....')
            
            threshold_index = int(len(mean_shifts) * (1-upto))
            top_mean_shifts = mean_shifts[:threshold_index]
            
            top_x_percent_timestamps = [ts for ts, _ in top_mean_shifts]
            
            sampled_df = modified_df[~modified_df['ts'].isin(top_x_percent_timestamps)]
            if save:
                sampled_df.drop(['Unnamed: 0'], axis=1).to_csv(filename)
                print(filename, ' saved.')
        # print('data sampling successful.')
    elif strategy == 'ts_tpr_remove_jaccard':
        # based on maximum mean shift strategy
        metric = strategy
        # filename = f'{save_location}/sparsified_data/{dataset_name}/{strategy}/{dataset_name}_{strategy}_sparsified_{upto}.csv'
        filename = f'{save_location}/sparsified_data/{dataset_name}/{strategy}/{dataset_name}_{strategy}_sparsified_{upto}.csv'
        print(filename)
        # exit()
        if os.path.exists(filename):
            print(f'reading {filename}...', end=' ')
            sampled_df=pd.read_csv(filename)
            sampled_df = sampled_df.loc[:, ~sampled_df.columns.str.contains('^Unnamed')]
            already_sparsified = True
            print(' done')
        else:
            save=True
            mean_shifts = compute_mean_shifts_with_metrics(tmp_graph, metric='jaccard', dataset_name=dataset_name)
        
            print('back to sparsify_data file....')
            
            threshold_index = int(len(mean_shifts) * (1-upto))
            top_mean_shifts = mean_shifts[:threshold_index]
            
            top_x_percent_timestamps = [ts for ts, _ in top_mean_shifts]
            
            sampled_df = modified_df[~modified_df['ts'].isin(top_x_percent_timestamps)]
            
            if save:
                sampled_df.drop(['Unnamed: 0'], axis=1).to_csv(filename)
                print(filename, 'saved.')
            
        # print('data sampling successful.')
    elif strategy == 'ts_tpr_remove_wasserstein':
        # based on maximum mean shift strategy
        metric = strategy
        filename = f'{save_location}/sparsified_data/{dataset_name}/{strategy}/{dataset_name}_{strategy}_sparsified_{upto}.csv'
        if os.path.exists(filename):
            print(f'reading {filename}...', end=' ')
            sampled_df=pd.read_csv(filename)
            sampled_df = sampled_df.loc[:, ~sampled_df.columns.str.contains('^Unnamed')]
            already_sparsified = True
            print(' done')
        else:
            save=True
            mean_shifts = compute_mean_shifts_with_metrics(tmp_graph, metric='wasserstein', dataset_name=dataset_name)
        
            print('back to sparsify_data file....')
            
            threshold_index = int(len(mean_shifts) * (1-upto))
            top_mean_shifts = mean_shifts[:threshold_index]
            
            top_x_percent_timestamps = [ts for ts, _ in top_mean_shifts]
            
            sampled_df = modified_df[~modified_df['ts'].isin(top_x_percent_timestamps)]
            if save:
                sampled_df.drop(['Unnamed: 0'], axis=1).to_csv(filename)
                print(filename, ' saved.')
        # print('data sampling successful.')
    elif strategy == 'ts_tpr_remove_kl_divergence':
        # based on maximum mean shift strategy
        metric = 'kl_divergence'
        filename = f'{save_location}/sparsified_data/{dataset_name}/{strategy}/{dataset_name}_{strategy}_sparsified_{upto}.csv'
        if os.path.exists(filename):
            print(f'reading {filename}...', end=' ')
            sampled_df=pd.read_csv(filename)
            sampled_df = sampled_df.loc[:, ~sampled_df.columns.str.contains('^Unnamed')]
            already_sparsified = True
            print(' done')
        else:
            save=True
            mean_shifts = compute_mean_shifts_with_metrics(tmp_graph, metric=metric, dataset_name=dataset_name)
        
            print('back to sparsify_data file....')
            
            threshold_index = int(len(mean_shifts) * (1-upto))
            top_mean_shifts = mean_shifts[:threshold_index]
            
            top_x_percent_timestamps = [ts for ts, _ in top_mean_shifts]
            
            sampled_df = modified_df[~modified_df['ts'].isin(top_x_percent_timestamps)]
            if save:
                sampled_df.drop(['Unnamed: 0'], axis=1).to_csv(filename)
                print(filename, ' saved.')
        # print('data sampling successful.')
    elif strategy == 'ts_tpr_remove_jensen_shannon_divergence':
        # based on maximum mean shift strategy
        metric = 'jensen_shannon_divergence'
        # filename = f'{scratch_location}/sparsified_data/{dataset_name}_{metric}_sparsified_{upto}.csv'
        filename = f'{save_location}/sparsified_data/{dataset_name}/{strategy}/{dataset_name}_{strategy}_sparsified_{upto}.csv'
        if os.path.exists(filename):
            print(f'reading {filename}...', end=' ')
            sampled_df=pd.read_csv(filename)
            sampled_df = sampled_df.loc[:, ~sampled_df.columns.str.contains('^Unnamed')]
            already_sparsified = True
            print(' done')
        else:
            save=True
            mean_shifts = compute_mean_shifts_with_metrics(tmp_graph, metric=metric, dataset_name=dataset_name)
        
            print('back to sparsify_data file....')
            
            threshold_index = int(len(mean_shifts) * (1-upto))
            top_mean_shifts = mean_shifts[:threshold_index]
            
            top_x_percent_timestamps = [ts for ts, _ in top_mean_shifts]
            
            sampled_df = modified_df[~modified_df['ts'].isin(top_x_percent_timestamps)]
            if save:
                sampled_df.drop(['Unnamed: 0'], axis=1).to_csv(filename)
                print(filename, ' saved.')
        # print('data sampling successful.')
    elif strategy == 'ts_tpr_remove_chebyshev':
        # based on maximum mean shift strategy
        metric = 'chebyshev'
        # filename = f'{scratch_location}/sparsified_data/{dataset_name}_{metric}_sparsified_{upto}.csv'
        filename = f'{save_location}/sparsified_data/{dataset_name}/{strategy}/{dataset_name}_{strategy}_sparsified_{upto}.csv'
        if os.path.exists(filename):
            print(f'reading {filename}...', end=' ')
            sampled_df=pd.read_csv(filename)
            sampled_df = sampled_df.loc[:, ~sampled_df.columns.str.contains('^Unnamed')]
            already_sparsified = True
            print(' done')
        else:
            save=True
            mean_shifts = compute_mean_shifts_with_metrics(tmp_graph, metric=metric, dataset_name=dataset_name)
        
            print('back to sparsify_data file....')
            
            threshold_index = int(len(mean_shifts) * (1-upto))
            top_mean_shifts = mean_shifts[:threshold_index]
            
            top_x_percent_timestamps = [ts for ts, _ in top_mean_shifts]
            
            sampled_df = modified_df[~modified_df['ts'].isin(top_x_percent_timestamps)]
            if save:
                sampled_df.drop(['Unnamed: 0'], axis=1).to_csv(filename)
                print(filename, ' saved.')
        # print('data sampling successful.')
    elif strategy == 'ts_tpr_remove_TER':
        # based on maximum mean shift strategy
        metric = 'TER'
        # filename = f'{scratch_location}/sparsified_data/{dataset_name}_{metric}_sparsified_{upto}.csv'
        filename = f'{save_location}/sparsified_data/{dataset_name}/{strategy}/{dataset_name}_{strategy}_sparsified_{upto}.csv'
        if os.path.exists(filename):
            print(f'reading {filename}...', end=' ')
            sampled_df=pd.read_csv(filename)
            sampled_df = sampled_df.loc[:, ~sampled_df.columns.str.contains('^Unnamed')]
            already_sparsified = True
            print(' done')
        else:
            save=True
            print(f'In ts_tpr_remove_TER {dataset_name=}')
            
            ter_dict = calculate_temporal_edge_rank(tmp_graph, dataset_name=dataset_name)
            
            sorted_ter_dict = dict(sorted(ter_dict.items(), key=lambda x: x[1], reverse=True))
            sorted_ter_dict = list(sorted_ter_dict.items())
        
            print('back to sparsify_data file....')

            threshold_index = int(len(sorted_ter_dict) * (1-upto))

            top_mean_shifts = sorted_ter_dict[:threshold_index]

            top_x_percent_timestamps = [ts for ts, _ in top_mean_shifts]

            sampled_df = tmp_graph[~tmp_graph['ts'].isin(top_x_percent_timestamps)]
            if save:
                sampled_df.drop(['Unnamed: 0'], axis=1).to_csv(filename)
                print(filename, ' saved.')
        # print('data sampling successful.')
    elif strategy == 'ts_tpr_remove_combined_ter':
        metric = 'Combined_TER'
        # filename = f'{scratch_location}/sparsified_data/{dataset_name}_{metric}_sparsified_{upto}.csv'
        filename = f'{save_location}/sparsified_data/{dataset_name}/{strategy}/{dataset_name}_{strategy}_sparsified_{upto}.csv'
        if os.path.exists(filename):
            print(f'reading {filename}...', end=' ')
            sampled_df=pd.read_csv(filename)
            sampled_df = sampled_df.loc[:, ~sampled_df.columns.str.contains('^Unnamed')]
            already_sparsified = True
            print(' done')
        else:
            save=True
            ter_dict = calculate_combined_temporal_edgerank(tmp_graph, dataset_name=dataset_name)
            
            sorted_ter_dict = dict(sorted(ter_dict.items(), key=lambda x: x[1], reverse=True))
            sorted_ter_dict = list(sorted_ter_dict.items())
        
            print('back to sparsify_data file....')

            threshold_index = int(len(sorted_ter_dict) * (1-upto))

            top_mean_shifts = sorted_ter_dict[:threshold_index]

            top_x_percent_timestamps = [ts for ts, _ in top_mean_shifts]

            sampled_df = tmp_graph[~tmp_graph['ts'].isin(top_x_percent_timestamps)]
            if save:
                sampled_df.drop(['Unnamed: 0'], axis=1).to_csv(filename)
                print(filename, ' saved.')
        # print('data sampling successful.')
    elif strategy == 'ts_tpr_remove_rec_mss':
        metric = 'rec_mss'
        # filename = f'{scratch_location}/sparsified_data/{dataset_name}_{metric}_sparsified_{upto}.csv'
        filename = f'{save_location}/sparsified_data/{dataset_name}/{strategy}/{dataset_name}_{strategy}_sparsified_{upto}.csv'
        if os.path.exists(filename):
            print(f'reading {filename}...', end=' ')
            sampled_df=pd.read_csv(filename)
            sampled_df = sampled_df.loc[:, ~sampled_df.columns.str.contains('^Unnamed')]
            already_sparsified = True
            print(' done')
        else:
            raise ValueError('Not implemented yet.')
        # print('data sampling successful.')
    else:
        raise ValueError(f'Unknown strategy {strategy}')
    
    print('data sampling successful.')
    EL_graph = (pd.concat([
        # first_interactions,
        sampled_df,
        # last_interactions
        ])
        .drop_duplicates()
        .reset_index(drop=True)
        # .drop(['Unnamed: 0'], axis=1)
        .sort_values(['idx'], ascending=True) # this fixed it.
                )
    
    EL_edge_raw_features = edge_raw_features # edge_raw_features[sorted(EL_graph['idx'].values)]
    # EL_graph['idx'] = [i for i in range(1, len(EL_graph['idx'])+1)]
    # assert EL_graph.shape[0] == EL_edge_raw_features.shape[0]
    
    return EL_graph, EL_edge_raw_features, already_sparsified

def EL_sparsify(
    graph: pd.DataFrame,
    edge_raw_features: Any,
    strategy: str = 'random',
    upto: float = 0.7,
    dataset_name: str = '',
    save: bool = False,
    seed: int = 0,
    selection: str = 'top'
) -> Tuple[pd.DataFrame, Any, bool]:
    """
    Deterministic graph sparsification using various sampling strategies.
    Added tie-breaking mechanisms for consistent results.
    """
    print(f'running EL_sparsify with {selection=}')
    if not dataset_name:
        raise ValueError('dataset_name must be provided')
    
    if selection not in ['top', 'middle', 'bottom']:
        raise ValueError("selection must be one of 'top', 'middle', 'bottom'")
    
    # `Bottom` should be called next k% or something like that
    
    print(f'\tIn EL_sparsify {dataset_name=}')
    
    graph['ts'] = graph['ts'].astype(int)
    
    # Initialize working copy of graph with deterministic sorting
    tmp_graph = graph.copy().sort_values(
        ['u', 'i', 'ts', 'idx'],  # Added 'idx' as final tie-breaker
        ascending=[True, True, True, True]
    )
    modified_df = tmp_graph.copy()
    
    save_dir = f'{save_location}/sparsified_data/{dataset_name}/{strategy}'
    os.makedirs(save_dir, exist_ok=True)
    
    sample_size = int(len(modified_df) * upto)
    already_sparsified = False
    
    if strategy == 'random':
        np.random.seed(seed)  # Ensure random operations are seeded
        print(f'{len(modified_df) = }')
        sampled_df = modified_df.sample(n=sample_size, random_state=seed)
        print(f'{len(sampled_df) = } --> {len(sampled_df)/len(modified_df):.2%}')
        print('*'*25)
        # exit()
    
    elif strategy == 'tpr_remove':
        page_rank_scores = get_temporal_pagerank(tmp_graph)
        # Add deterministic tie-breaking using node IDs
        sorted_nodes = sorted(
            page_rank_scores.items(),
            key=lambda x: (x[1], x[0]),  # Use node ID as tie-breaker
            reverse=True
        )
        top_nodes_count = int(len(sorted_nodes) * (1 - upto))
        top_nodes = {node for node, _ in sorted_nodes[:top_nodes_count]}
        sampled_df = modified_df[~modified_df['u'].isin(top_nodes) & ~modified_df['i'].isin(top_nodes)]
    
    elif strategy in ['ts_tpr_remove_ss', 'ts_tpr_remove_inc']:
        ts_level_tpr = (calc_timestamp_pagerank(tmp_graph) if strategy == 'ts_tpr_remove_ss' 
                       else optimized_calc_inc_timestamp_pagerank(tmp_graph))
        
        # Deterministic score aggregation
        ts_scores = {}
        for ts in sorted(ts_level_tpr.keys()):  # Sort timestamps first
            scores_dict = ts_level_tpr[ts]
            # Sort by node ID before summing to ensure consistent order
            ts_scores[ts] = sum(dict(sorted(scores_dict.items())).values())
        
        # Add timestamp as tie-breaker
        sorted_timestamps = sorted(
            ts_scores.items(),
            key=lambda x: (x[1], x[0]),  # Use timestamp as tie-breaker
            reverse=True
        )
        timestamps_to_remove = select_timestamps(sorted_timestamps, upto, selection)
        sampled_df = modified_df[~modified_df['ts'].isin(timestamps_to_remove)]
    
    else:
        filename = f'{save_dir}/{dataset_name}_{strategy}_sparsified_{upto}.csv'
        
        if os.path.exists(filename):
            print(f'\treading {os.path.basename(filename)}...', end=' ')
            sampled_df = pd.read_csv(filename)
            sampled_df = sampled_df.loc[:, ~sampled_df.columns.str.contains('^Unnamed')]
            already_sparsified = True
            print(' done')
        # ===========================    Original above this   ===========================
        elif strategy in ['preference', 'jaccard', 'pagerank', 'degree']:
            print(f'{strategy = }')
            # G is already created from tmp_graph
            G = nx.from_pandas_edgelist(tmp_graph, source='u', target='i')
            
            # print(tmp_graph.head())

            # Get the existing edges from the graph G
            existing_edges = np.array(tmp_graph[['u', 'i']])   # list(G.edges())
            
            # TODO: another issue
            print(f'\t{len(existing_edges) = }, {len(tmp_graph) = }')  # len(existing_edges) = 13545, len(tmp_graph) = 110232
            
            # exit()

            # Prepare lists to store results
            u_list = []
            i_list = []
            score_list = []

            if strategy == 'preference':
                # Calculate preferential attachment for existing edges
                scores_iterator = nx.preferential_attachment(G, existing_edges)
                for u, i, score in scores_iterator:
                    u_list.append(u)
                    i_list.append(i)
                    score_list.append(score)

            elif strategy == 'jaccard':
                # Calculate jaccard coefficient for existing edges
                if dataset_name.lower() in ['wikipedia', 'mooc']:
                    raise ValueError('Not applicable for bipartite graph')
                scores_iterator = nx.jaccard_coefficient(G, existing_edges)
                for u, i, score in scores_iterator:
                    u_list.append(u)
                    i_list.append(i)
                    score_list.append(score)

            elif strategy == 'degree':
                # Calculate degree sum for existing edges
                # Getting degree dictionary once is efficient
                deg_dict = dict(G.degree())
                get_deg = deg_dict.get
                for u, i in existing_edges:
                    u_list.append(u)
                    i_list.append(i)
                    # Get degree, defaulting to 0 if node somehow not in dict (shouldn't happen for G.edges())
                    score = get_deg(u, 0) + get_deg(i, 0)
                    score_list.append(score)

            elif strategy == 'pagerank':
                # Calculate pagerank for all nodes once
                # Using the same max_iter and tol from your original code
                pr_dict = nx.pagerank(G, max_iter=10, tol=1e-3)
                get_pr = pr_dict.get
                # Calculate pagerank sum for existing edges
                for u, i in existing_edges:
                    u_list.append(u)
                    i_list.append(i)
                    # Get pagerank, defaulting to 0.0 if node somehow not in dict
                    score = get_pr(u, 0.0) + get_pr(i, 0.0)
                    score_list.append(score)

            # Create a DataFrame for scores and edges (only existing ones)
            edge_score_df = pd.DataFrame({
                'u': u_list,
                'i': i_list,
                'score': score_list
            })
            # ================================================================
            modified_df = modified_df.reset_index(drop=True)
            edge_score_df = edge_score_df.reset_index(drop=True)

            modified_df['row_id'] = modified_df.index
            edge_score_df['row_id'] = edge_score_df.index
            # ================================================================
            merged = pd.merge(modified_df, edge_score_df[['row_id', 'score']], on='row_id', how='left')
            merged.drop(columns=['row_id'], inplace=True)
            print(f'\t{len(merged) = }')
            merged = merged.sort_values(by=['score', 'ts', 'idx'], ascending=[selection != 'top', True, True]) # Assuming 'ts', 'idx' are in modified_df
            sampled_df = merged.iloc[:sample_size]
            print(f'\t{len(sampled_df) = } --> {len(sampled_df) / len(merged):.2%}')
        # ===========================     original below this   ===========================
        else:
            # Calculate shifts with deterministic operations
            if strategy == 'ts_tpr_remove_mss':
                shifts = mean_shift_removal(tmp_graph, dataset_name=dataset_name)
            elif strategy == 'ts_tpr_remove_mss_2':
                shifts = mean_shift_removal2(tmp_graph, dataset_name=dataset_name)
            elif strategy == 'ts_tpr_remove_TER':
                ter_dict = calculate_temporal_edge_rank_working(tmp_graph, dataset_name=dataset_name)
                shifts = sorted(ter_dict.items(), key=lambda x: (x[1], x[0]), reverse=True)
            elif strategy == 'ts_tpr_remove_mean_TER':
                ter_dict = calculate_temporal_edge_rank_mean(tmp_graph, dataset_name=dataset_name)
                shifts = sorted(ter_dict.items(), key=lambda x: (x[1], x[0]), reverse=True)
            elif strategy == 'ts_tpr_remove_combined_ter':
                ter_dict = calculate_combined_temporal_edgerank(tmp_graph, dataset_name=dataset_name)
                shifts = sorted(ter_dict.items(), key=lambda x: (x[1], x[0]), reverse=True)
            else:
                metric = strategy.replace('ts_tpr_remove_', '')
                shifts = LW_compute_mean_shifts_with_metrics(tmp_graph, metric=metric, dataset_name=dataset_name)
            
            # Ensure shifts are sorted deterministically
            if isinstance(shifts, list):
                shifts = sorted(shifts, key=lambda x: (x[1], x[0]), reverse=True)
            
            timestamps_to_remove = select_timestamps(shifts, upto, selection)  # remove edges not timestamps
            candidate_edges = modified_df[modified_df['ts'].isin(timestamps_to_remove)]
            n_to_remove = int(len(modified_df) * 0.3)
            # edges_to_remove = candidate_edges.sample(n=min(n_to_remove, len(candidate_edges)), random_state=seed)
            edges_to_remove = candidate_edges.iloc[:min(n_to_remove, len(candidate_edges))]  # Keeping it deterministic: removing only first 30%
            sampled_df = modified_df.drop(edges_to_remove.index)
            
            if save:
                save_df = sampled_df.copy()
                if 'Unnamed: 0' in save_df.columns:
                    save_df = save_df.drop(['Unnamed: 0'], axis=1)
                save_df.to_csv(filename)
                print(f'{filename} saved.')
    
    print('data sampling successful.')
    
    # Ensure deterministic final sorting
    sparsified_graph = (sampled_df
        .drop_duplicates()
        .reset_index(drop=True)
        .sort_values(['idx'], ascending=True))
    print(f'{len(sparsified_graph) = } --> {len(sparsified_graph)/len(modified_df):.2%}')
    # exit()
    
    return sparsified_graph, edge_raw_features, already_sparsified

def select_timestamps(sorted_items: list, upto: float, selection: str) -> list:
    """Deterministic timestamp selection."""
    total_len = len(sorted_items)
    remove_count = int(total_len * (1 - upto))
    
    if selection == 'top':
        selected_items = sorted_items[:remove_count]
    elif selection == 'bottom':
        selected_items = sorted_items[-remove_count:]
    else:  # middle
        start_idx = remove_count # (total_len - remove_count) // 2
        selected_items = sorted_items[start_idx:start_idx + remove_count]
    
    # Ensure deterministic order of returned timestamps
    return sorted([ts for ts, _ in selected_items])


def EL_sparsify_data(dataset_name='wikipedia', strategy='random',
                    upto=0.7, selection='top', val_ratio=0.15, test_ratio=0.15, train_only=True):
    print('This is a simple')
    # Load data and train val test split
    graph = pd.read_csv('{}/processed_data/{}/ml_{}.csv'.format(scratch_location, dataset_name, dataset_name))
    edge_raw_features = np.load('{}/processed_data/{}/ml_{}.npy'.format(scratch_location, dataset_name, dataset_name))
    node_raw_features = np.load('{}/processed_data/{}/ml_{}_node.npy'.format(scratch_location, dataset_name, dataset_name))
    
    print(f'full graph length is {len(graph)}.')
    
    print(f'{val_ratio = } {test_ratio = }')
    
    val_time, test_time = list(np.quantile(graph.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))
    new_test_node_set=set()
    # =====================================  Change it to train dataset only  =====================================
    if train_only:
        
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
        print(f'Length of training only graph length is {len(_graph)}.')
        
        orig_src_degree, orig_dst_degree  = Counter(_graph['u']), Counter(_graph['i'])
    else:
        _graph = graph.copy(deep=True)
    # ================================================================================================================
    
    # OUT_DF = '{}/sparsified_data/{}/ml_{}.csv'.format(scratch_location, dataset_name, dataset_name)
    # OUT_DF = f'{scratch_location}/sparsified_data/{dataset_name}/{dataset_name}_{strategy}_sparsified_{upto}.csv'
    os.makedirs(f'{save_location}/sparsified_data/{dataset_name}/{strategy}', exist_ok=True)
    
    OUT_DF = f'{save_location}/sparsified_data/{dataset_name}/{strategy}/{dataset_name}_{strategy}_sparsified_{upto}.csv'
    
    OUT_FEAT = '{}/sparsified_data/{}/{}/ml_{}.npy'.format(save_location, dataset_name, strategy, dataset_name)
    OUT_NODE_FEAT = '{}/sparsified_data/{}/{}/ml_{}_node.npy'.format(save_location, dataset_name, strategy, dataset_name)
    
    print(f'In EL_sparsify_data {dataset_name=}')

    
    # EL_graph, EL_edge_raw_features = EL_sparsify(_graph, edge_raw_features,
    #                                             strategy=strategy, upto=upto, dataset_name=dataset_name)
    
    EL_graph, EL_edge_raw_features, already_sparsified = EL_sparsify(_graph, edge_raw_features, strategy=strategy,
                                                                    upto=upto, dataset_name=dataset_name, selection=selection)
    
    print(f'{already_sparsified = } {len(EL_graph)}')
    # exit()
    
    # DONE: Don't remove test node test <--- Do we really need  it?
    # if not already_sparsified:
    #     print(f'We are removing test node also.')
    #     EL_graph = EL_graph[~EL_graph['u'].isin(new_test_node_set)]
    #     EL_graph = EL_graph[~EL_graph['i'].isin(new_test_node_set)]
    
    print(f'After Sparsifcation training full graph length is {len(EL_graph)}.')
    # Real Issue lies above.
    # exit()
    
    # get val and test splits too
    # val_mask = np.logical_and(graph['ts'] <= test_time, graph['ts'] > val_time)
    val_graph_df = graph[np.logical_and(graph['ts'] <= test_time, graph['ts'] > val_time)]
    test_graph_df = graph[graph['ts'] > test_time]
    
    # DONE: Maybe Merge the all split and save.
    print('Before concat: ', f'{len(EL_graph) = }, {len(val_graph_df) = }, {len(test_graph_df) = }')
    # EL_graph = pd.concat([EL_graph, val_graph_df, test_graph_df])  # TODO: Later turn it on
    EL_graph = EL_graph.sort_values(by='ts')
    EL_graph.reset_index(drop=True, inplace=True)
    
    # print(f'{len(EL_graph) = }')
    
    assert len(EL_graph) < len(graph), "sparsified graph should be smaller than full graph."
    
    # TODO: Check for invalid edges
    # Check if sparsified edges are subset of original
    orig_edges = set(zip(graph['u'], graph['i']))
    sprs_edges = set(zip(EL_graph['u'], EL_graph['i']))
    
    invalid_edges = sprs_edges - orig_edges
    
    assert len(invalid_edges) == 0, "Error: Found edges in sparse network that don't exist in original"
    
    
    # TODO: Check for change in node degree in sparsified graph and origingal graph
    orig_src_degree, orig_dst_degree  = Counter(graph['u']), Counter(graph['i'])
    sprc_src_degree, sprc_dst_degree  = Counter(EL_graph['u']), Counter(EL_graph['i'])
    
    print(sum(orig_src_degree.values()), sum(orig_dst_degree.values()))
    print(sum(sprc_src_degree.values()), sum(sprc_dst_degree.values()))
    
    assert sum(orig_src_degree.values()) == sum(orig_dst_degree.values()), 'Check 1'
    assert sum(sprc_src_degree.values()) == sum(sprc_dst_degree.values()), 'Check 2'
    
    assert sum(sprc_src_degree.values()) < sum(orig_src_degree.values()), 'Check 3'
    assert sum(sprc_dst_degree.values()) < sum(orig_dst_degree.values()), 'Check 4'
    
    assert len(orig_src_degree) > len(sprc_src_degree), "length of orig src degrees is less than sprs ones."
    assert len(orig_dst_degree) > len(sprc_dst_degree), "length of orig dst degrees is less than sprs ones."
    
    check_src_def = {node: degree - sprc_src_degree.get(node, 0) for node, degree in orig_src_degree.items()}
    check_dst_def = {node: degree - sprc_dst_degree.get(node, 0) for node, degree in orig_dst_degree.items()}
    
    assert not any(value < 0 for value in check_src_def.values()), "Error: Negative deficit degree found in source"
    assert not any(value < 0 for value in check_dst_def.values()), "Error: Negative deficit degree found in source"
    
    print('ending before saving, all checks are passed.')

    print(f'Sparsified full graph length is {len(EL_graph)}.')
    
    # save it into the location
    EL_graph.to_csv(OUT_DF, index=False)  # edge-list
    print('saved the sparsified graph at {}'.format(OUT_DF))
    np.save(OUT_FEAT, EL_edge_raw_features)  # edge features
    print('saved the sparsified graph at {}'.format(OUT_FEAT))
    np.save(OUT_NODE_FEAT, node_raw_features)  # node features
    print('saved the sparsified graph at {}'.format(OUT_NODE_FEAT))
    
    print(f'Sparsified {dataset_name}.')
    

def temporal_page_rank(G, alpha=0.85, max_iter=100, tol=1e-6):
    nodes = G.nodes()
    num_nodes = G.number_of_nodes()
    
    # Initialize PageRank scores
    pr = {node: 1.0 / num_nodes for node in nodes}
    temp_pr = pr.copy()

    for _ in range(max_iter):
        change = 0
        for node in nodes:
            rank_sum = sum(pr[neighbor] / len(G[neighbor]) for neighbor in G.neighbors(node) if 'timestamp' in G[node][neighbor])
            temp_pr[node] = (1 - alpha) / num_nodes + alpha * rank_sum
        
        # Calculate change for convergence check
        change = sum(abs(temp_pr[node] - pr[node]) for node in nodes)
        
        if change < tol:
            break
        
        pr = temp_pr.copy()

    return pr


def build_graph(graph):
    # Extract nodes, edges, and timestamps
    edges = graph[['u', 'i', 'ts']].values
    
    G = nx.Graph()
    for edge in edges:
        source, target, timestamp = edge
        G.add_edge(source, target, timestamp=timestamp)
    return G


if __name__ == "__main__":    
    start_time = time.time()
    parser = argparse.ArgumentParser('Interface for preprocessing datasets')
    parser.add_argument('--dataset_name', type=str,
                        choices=['wikipedia', 'reddit', 'mooc', 'lastfm', 'myket', 'enron', 'SocialEvo', 'uci',
                                'Flights', 'CanParl', 'USLegis', 'UNtrade', 'UNvote', 'Contacts', 'bitcoin'],
                        help='Dataset name', default='wikipedia')
    # parser.add_argument('--node_feat_dim', type=int, default=172, help='Number of node raw features')
    parser.add_argument('--upto', type=float, help='sparsify upto', default=0.7)
    parser.add_argument('--val_ratio', type=float, help='val upto', default=0.15)
    parser.add_argument('--test_ratio', type=float, help='test upto', default=0.15)
    parser.add_argument('--strategy', type=str, help='sparsification strategy', default='random')
    parser.add_argument('--selection', type=str, help='selection strategy', default='top')
    

    args = parser.parse_args()

    print(f'Sparsifying dataset {args.dataset_name}...')
    if args.dataset_name in []:  # DONE: remove UCI from here
        Path("{}/sparsified_data/{}/".format(scratch_location, args.dataset_name)).mkdir(parents=True, exist_ok=True)
        copy_tree("{}/DG_data/{}/".format(scratch_location, args.dataset_name), "{}/sparsified_data/{}/".format(save_location, args.dataset_name))
        print(f'Not implemented for enron, SocialEvo graph yet.')
    else:
        Path("{}/sparsified_data/{}/".format(scratch_location, args.dataset_name)).mkdir(parents=True, exist_ok=True)
        copy_tree("{}/DG_data/{}/".format(scratch_location, args.dataset_name), "{}/sparsified_data/{}/".format(save_location, args.dataset_name))
        # bipartite dataset
        EL_sparsify_data(dataset_name=args.dataset_name, strategy=args.strategy, upto=args.upto, selection=args.selection,
                        val_ratio=args.val_ratio, test_ratio=args.test_ratio,
                        # train_only=False
                        )
        
        # if args.dataset_name in ['wikipedia', 'reddit', 'mooc', 'lastfm', 'myket']:
        #     EL_sparsify_data(dataset_name=args.dataset_name, strategy=args.strategy, upto=args.upto, selection=args.selection)
        # else:
        #     EL_sparsify_data(dataset_name=args.dataset_name, strategy=args.strategy, upto=args.upto, selection=args.selection)
        print(f'{args.dataset_name} is processed successfully.')

        if args.dataset_name not in ['myket']:
            check_data(args.dataset_name)
        print(f'{args.dataset_name} passes the checks successfully.')
    end_time = time.time()
    print("Total Time taken: ", end_time-start_time)
