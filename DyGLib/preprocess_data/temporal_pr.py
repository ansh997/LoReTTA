import getpass
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import heapq
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon, chebyshev
from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm
import heapq
from typing import Tuple, Optional, Any

np.random.seed(1729928109)

# Set the working directory to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # this might cause issue
sys.path.append(project_root)

from TPR import temporal_pagerank
# scratch_location = '/home/t2/TGN_adv'  # rf'/scratch/{getpass.getuser()}'
save_location = '/raid/t2/TGN_adv'  # rf'/scratch/{getpass.getuser()}'

# Function to convert results to dictionary
def get_pagerank_scores(tpr_computer):
    scores = {}
    for i, node in enumerate(tpr_computer.active_mass[:, 0]):
        scores[node] = tpr_computer.temp_pr[i, 1]
    return scores


def get_temporal_pagerank(_graph_df, alpha = 0.85, beta = 0.1):
    
    graph_df = _graph_df.copy(deep=True)
    
    # Extract nodes, edges, and timestamps
    edges = graph_df[['u', 'i', 'ts']].values
    nodes = np.unique(edges[:, :2])  # Get unique nodes from edges

    # Temporal PageRank Parameters
    params = temporal_pagerank.TemporalPageRankParams(alpha, beta)

    # Initialize TemporalPageRankComputer
    tpr_computer = temporal_pagerank.TemporalPageRankComputer(nodes, [params])
    
    # Update PageRank Scores
    for edge in edges:
        src, trg, timestamp = edge
        tpr_computer.update((src, trg))
    
    # Get PageRank scores
    page_rank_scores = get_pagerank_scores(tpr_computer)
    return page_rank_scores


def build_graph(graph):
    # Extract nodes, edges, and timestamps
    edges = graph[['u', 'i', 'ts']].values
    
    G = nx.Graph()
    for edge in edges:
        source, target, timestamp = edge
        G.add_edge(source, target, timestamp=timestamp)
    return G


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


# Function to convert results to dictionary
def get_pagerank_scores(tpr_computer):
    scores = {}
    for i, node in enumerate(tpr_computer.active_mass[:, 0]):
        scores[node] = tpr_computer.temp_pr[i, 1]
    return scores


def temporal_pagerank_with_timestamps(graph, alpha=0.85, beta=0.1):
    
    # Extract nodes, edges, and timestamps
    edges = graph[['u', 'i', 'ts']].values
    nodes = np.unique(edges[:, :2])  # Get unique nodes from edges
    
    params = temporal_pagerank.TemporalPageRankParams(alpha, beta)
    
    # Initialize TemporalPageRankComputer
    tpr_computer = temporal_pagerank.TemporalPageRankComputer(nodes, [params])
    
    # Initialize a dictionary to store PageRank scores for each timestamp
    timestamp_pagerank = {}

    # Update PageRank Scores for each edge at each timestamp
    for edge in edges:
        src, trg, timestamp = edge
        tpr_computer.update((src, trg))
        
        # Store the PageRank scores for the current timestamp
        if timestamp not in timestamp_pagerank:
            timestamp_pagerank[timestamp] = get_pagerank_scores(tpr_computer)
        else:
            timestamp_pagerank[timestamp].update(get_pagerank_scores(tpr_computer))
    
    return timestamp_pagerank


def calc_timestamp_pagerank(graph):
    G = build_graph(graph)
    
    edges = graph[['u', 'i', 'ts']].values
    
    # Calculate Temporal PageRank for each timestamp
    timestamp_pagerank = {}
    for edge in edges:
        _, _, timestamp = edge
        if timestamp not in timestamp_pagerank:
            # Extract subgraph for the current timestamp
            subgraph_edges = [e for e in G.edges(data=True) if e[2]['timestamp'] == timestamp]
            subgraph = nx.Graph()
            subgraph.add_edges_from((e[0], e[1]) for e in subgraph_edges)
            
            # Calculate PageRank for the subgraph
            pr_scores = temporal_page_rank(subgraph)
            timestamp_pagerank[timestamp] = pr_scores
    return timestamp_pagerank


def calc_inc_timestamp_pagerank_prev(graph):
    # Full vectorization is challenging due to the nature of PageRank computation, which requires iterative convergence checks.
    G = build_graph(graph)
    
    edges = graph[['u', 'i', 'ts']].values
    
    timestamp_pagerank = {}
    current_edges = []
    
    for edge in sorted(edges, key=lambda x: x[2]): # sorted edges by timestamps
        src, trg, ts = edge
        current_edges.append((src, trg))
        # G.add_edge(current_edges)
        
        if ts not in timestamp_pagerank:
            G.add_edges_from(current_edges)
            timestamp_pagerank[ts] = temporal_page_rank(G)
    return timestamp_pagerank


# def calc_inc_timestamp_pagerank(graph):
#     G = build_graph(graph)
    
#     edges = graph[['u', 'i', 'ts']].values
#     edges = sorted(edges, key=lambda x: x[2])  # Sort edges by timestamp
    
#     timestamp_pagerank = {}
#     current_edges = []

#     timestamps = np.unique(edges[:, 2])
#     for ts in timestamps:
#         # Filter edges up to the current timestamp
#         edges_up_to_ts = edges[edges[:, 2] <= ts]
#         current_edges.extend(edges_up_to_ts)

#         # Add new edges to the graph
#         G.add_edges_from((src, trg) for src, trg, _ in edges_up_to_ts)

#         # Compute PageRank for the current state of the graph
#         timestamp_pagerank[ts] = temporal_page_rank(G)
    
#     return timestamp_pagerank


def calc_inc_timestamp_pagerank_2(graph):
    G = build_graph(graph)
    
    # Initialize edges as a NumPy array
    edges = np.array(graph[['u', 'i', 'ts']].values)
    edges = sorted(edges, key=lambda x: x[2])  # Sort edges by timestamp
    
    timestamp_pagerank = {}
    current_edges = []

    # No need to check if edges is a list; it should remain a NumPy array
    timestamps = np.unique(graph['ts'].values)  # np.unique(edges[:, 2])
    for ts in timestamps:
        # Find indices where the timestamp column is less than or equal to ts
        indices = np.where(edges[:, 2] <= ts)[0]

        # Select rows from edges using these indices
        edges_up_to_ts = edges[indices]
        
        # Filter edges up to the current timestamp
        # edges_up_to_ts = edges[edges[:, 2] <= ts]
        
        current_edges.extend((src, trg) for src, trg, _ in edges_up_to_ts)

        # Add new edges to the graph
        G.add_edges_from((src, trg) for src, trg, _ in edges_up_to_ts)

        # Compute PageRank for the current state of the graph
        timestamp_pagerank[ts] = temporal_page_rank(G)
    
    return timestamp_pagerank

def calc_inc_timestamp_pagerank_3(df):
    G = build_graph(df)
    
    # Convert DataFrame columns to appropriate types if necessary
    df['u'] = df['u'].astype(int)  # Assuming 'u' needs to be integer
    df['i'] = df['i'].astype(int)  # Assuming 'i' needs to be integer
    # df['ts'] = pd.to_datetime(df['ts'])  # Assuming 'ts' is a datetime

    # Sort DataFrame by timestamp
    df.sort_values(by='ts', inplace=True)
    
    timestamp_pagerank = {}
    current_edges = []

    timestamps = df['ts'].unique()  # Get unique timestamps
    for ts in timestamps:
        # Filter DataFrame rows up to the current timestamp
        df_up_to_ts = df[df['ts'] <= ts]
        current_edges.extend(list(zip(df_up_to_ts['u'], df_up_to_ts['i'])))

        # Add new edges to the graph
        G.add_edges_from(current_edges)

        # Compute PageRank for the current state of the graph
        timestamp_pagerank[ts] = temporal_page_rank(G)
    
    return timestamp_pagerank



def calc_inc_timestamp_pagerank_slower(df):
    G = build_graph(df)
    
    # # Convert DataFrame columns to appropriate types if necessary
    # df['u'] = df['u'].astype(int)  # Assuming 'u' needs to be integer
    # df['i'] = df['i'].astype(int)  # Assuming 'i' needs to be integer
    # df['ts'] = pd.to_datetime(df['ts'])  # Assuming 'ts' is a datetime

    # Sort DataFrame by timestamp
    df.sort_values(by='ts', inplace=True)
    
    timestamp_pagerank = {}
    current_edges = []

    timestamps = df['ts'].unique()  # Get unique timestamps

    # Wrap the loop with tqdm for progress visualization
    for ts in tqdm(timestamps, desc="Calculating PageRank"):
        # Filter DataFrame rows up to the current timestamp
        df_up_to_ts = df[df['ts'] <= ts]
        current_edges.extend(list(zip(df_up_to_ts['u'], df_up_to_ts['i'])))

        # Add new edges to the graph
        G.add_edges_from(current_edges)

        # Compute PageRank for the current state of the graph
        timestamp_pagerank[ts] = temporal_page_rank(G)
    
    return timestamp_pagerank


def calc_inc_timestamp_pagerank(df):
    G = build_graph(df)
    
    # Convert DataFrame columns to appropriate types if necessary
    df['u'] = df['u'].astype(int)
    df['i'] = df['i'].astype(int)
    # df['ts'] = pd.to_datetime(df['ts'])

    # Sort DataFrame by timestamp
    df.sort_values(by='ts', inplace=True)
    
    timestamp_pagerank = {}
    current_edges = []

    timestamps = df['ts'].unique()

    # Preallocate memory for edge lists to avoid resizing in each iteration
    max_edges = len(df)
    current_edges = [[] for _ in range(max_edges)]

    # Wrap the loop with tqdm for progress visualization
    for i, ts in enumerate(tqdm(timestamps, desc="Calculating PageRank")):
        # Filter DataFrame rows up to the current timestamp
        df_up_to_ts = df[df['ts'] <= ts]
        # Update current_edges in place to avoid creating new lists
        current_edges[i] = list(zip(df_up_to_ts['u'], df_up_to_ts['i']))

        # Add new edges to the graph in batches
        G.add_edges_from(sum(current_edges[:i+1], []))

        # Compute PageRank for the current state of the graph
        timestamp_pagerank[ts] = temporal_page_rank(G)
    
    return timestamp_pagerank


def optimized_temporal_page_rank(G, alpha=0.85, max_iter=100, tol=1e-6):
    num_nodes = G.number_of_nodes()
    pr = np.full(num_nodes, 1.0 / num_nodes)
    adj_matrix = nx.adjacency_matrix(G).T.tocsr()  # Use nx.adjacency_matrix to get the adjacency matrix

    for _ in range(max_iter):
        pr_old = pr.copy()
        # Perform the dot product and scale by alpha
        scaled_dot_product = alpha * adj_matrix.dot(pr)
        # Add the teleportation factor
        pr = scaled_dot_product + ((1 - alpha) / num_nodes)
        change = np.sum(np.abs(pr - pr_old))

        if change < tol:
            break

    return pr

def optimized_calc_inc_timestamp_pagerank(graph):
    G = build_graph(graph)
    edges = np.array(graph[['u', 'i', 'ts']].values)
    # Ensure edges_sorted_by_ts remains a NumPy array after sorting
    edges_sorted_by_ts = np.array(sorted(edges, key=lambda x: x[2]))

    timestamp_pagerank = {}
    current_edges = []
    unique_timestamps = np.unique(edges_sorted_by_ts[:, 2])

    for ts in tqdm(unique_timestamps, desc="Calculating PageRank"):
        edges_up_to_ts = edges_sorted_by_ts[edges_sorted_by_ts[:, 2] <= ts]
        current_edges.extend((src, trg) for src, trg, _ in edges_up_to_ts)
        G.add_edges_from(current_edges)

        timestamp_pagerank[ts] = optimized_temporal_page_rank(G)
        current_edges.clear()  # Clear current_edges for the next timestamp

    return timestamp_pagerank

# def optimized_calc_inc_timestamp_pagerank(graph):
#     G = build_graph(graph)
#     edges = np.array(graph[['u', 'i', 'ts']].values)
#     # Ensure edges_sorted_by_ts remains a NumPy array after sorting
#     edges_sorted_by_ts = np.array(sorted(edges, key=lambda x: x[2]))

#     timestamp_pagerank = {}
#     unique_timestamps = np.unique(edges_sorted_by_ts[:, 2])

#     for ts in tqdm(unique_timestamps, desc="Calculating PageRank"):
#         # edges_up_to_ts = edges_sorted_by_ts[edges_sorted_by_ts[:, 2] == ts] # its upto that ts
#         edges_up_to_ts = edges_sorted_by_ts[edges_sorted_by_ts[:, 2] <= ts]
#         current_edges = [(src, trg) for src, trg, _ in edges_up_to_ts]
#         G.add_edges_from(current_edges)

#         timestamp_pagerank[ts] = optimized_temporal_page_rank(G)

#     return timestamp_pagerank


def working_temporal_pagerank_heap_np(E, beta, alpha, check_evolution=False, mmap_file=f'{save_location}/ts_tpr_data.dat', dataset_name=''):
    if dataset_name == '':
        raise ValueError('Please provide dataset name')
    else:
        mmap_file=f'{save_location}/{dataset_name}_ts_tpr_data.dat'
        print(f'{mmap_file=}')
        
    
    # Convert edges to a NumPy array
    E = np.array(E, dtype=[('u', int), ('v', int), ('t', float)])
    
    # Get the maximum node index to size the r and s arrays appropriately
    max_node = max(E['u'].max(), E['v'].max())
    
    ts_tpr_mmap = None
    if check_evolution:
        if os.path.exists(mmap_file):
            print('\tLoading precomputed memory-mapped array...', end='\r')
            ts_tpr_mmap = np.memmap(mmap_file,
                                # dtype=[('t', float), ('r', float, max_node + 1)],
                                dtype = np.dtype([('t', float), ('r', float, (max_node + 1,))]),
                                mode='r+')
            print('\tPrecomputed memory-mapped array loaded.      ')
            check_evolution=False
            # return r, ts_tpr_mmap
    
    # Initialize r and s arrays
    r = np.zeros(max_node + 1)
    s = np.zeros(max_node + 1)
    
    ts_tpr = [] if check_evolution else None
    
    # Use a heap to efficiently process edges in time order
    heap = [(t, u, v) for u, v, t in E]
    heapq.heapify(heap)
    # print('\t heapify successful')
    while heap:
        t, u, v = heapq.heappop(heap)
        
        # Update r and s values
        delta = 1 - alpha
        r[u] += delta
        s[u] += delta
        r[v] += s[u] * alpha
        
        if beta < 1:
            s_v_increment = s[u] * (1 - beta) * alpha
            s[v] += s_v_increment
            s[u] *= beta
        else:
            s[v] += s[u] * alpha
            s[u] = 0
        
        # Store evolution if required
        if check_evolution:
            # ts_tpr.append((t, r.copy()))  # Store r values at current timestamp
            # Normalize r before appending
            total_r = r.sum()
            if total_r > 0:
                ts_tpr.append((t, r.copy() / total_r))
    # print('\t out of loop.')
    
    # Normalize r
    total_r = r.sum()
    if total_r > 0:
        r /= total_r
    
    if check_evolution:
        # ts_tpr = np.array(ts_tpr, dtype=[('t', float), ('r', float, max_node + 1)])
        # print('Creating a memory mapped array...', end='\r')
        
        # Create a memory-mapped array with an estimated size
        num_entries = len(ts_tpr)  # This is the number of timestamps collected
        ts_tpr_mmap = np.memmap(mmap_file,
                            # dtype=[('t', float), ('r', float, max_node + 1)],
                            dtype = np.dtype([('t', float), ('r', float, (max_node + 1,))]),
                            mode='w+', shape=(num_entries,))
        
        # Fill the memory-mapped array with collected data
        for idx, (t, r_array) in enumerate(ts_tpr):
            ts_tpr_mmap[idx]['t'] = t
            ts_tpr_mmap[idx]['r'][:len(r_array)] = r_array
        
        # Ensure all data is written to disk
        ts_tpr_mmap.flush()
        
        # Optionally, reopen the memory-mapped array to resize it
        ts_tpr_mmap = np.memmap(mmap_file,
                            # dtype=[('t', float), ('r', float, max_node + 1)],
                            dtype = np.dtype([('t', float), ('r', float, (max_node + 1,))]),
                            mode='r+', shape=(num_entries,))  # Resize to the actual number of entries

        # print('Memory mapped array created.         ')
    
    return r, ts_tpr_mmap

# NOTE: DO NOT REMOVE THIS FUNCTION
def old_temporal_pagerank_heap_np(
    E: np.ndarray,
    beta: float,
    alpha: float,
    check_evolution: bool = False,
    mmap_file: str = f'{save_location}/ts_tpr_data.dat',
    dataset_name: str = ''
) -> Tuple[np.ndarray, Optional[np.memmap]]:
    """
    Calculate Old implementation temporal PageRank using a heap-based approach with NumPy arrays.
    
    Args:
        E: Edge array with columns [source, target, timestamp]
        beta: Time decay factor
        alpha: Damping factor
        check_evolution: Whether to track PageRank evolution over time
        mmap_file: Path to memory-mapped file
        dataset_name: Name of the dataset (required)
    
    Returns:
        Tuple containing:
        - Final PageRank scores array
        - Memory-mapped array of temporal evolution (if check_evolution=True)
    """
    if not dataset_name:
        raise ValueError('dataset_name must be provided')
    
    mmap_file = f'{save_location}/{dataset_name}_ts_tpr_data.dat'
    print(f'{mmap_file=}')
    
    # Ensure E is properly formatted numpy array
    if not isinstance(E, np.ndarray):
        E = np.array(E, dtype=[('u', np.int64), ('v', np.int64), ('t', np.float64)])
    
    # Get maximum node index
    max_node = max(np.max(E['u']), np.max(E['v']))
    
    # Check for existing memory-mapped file
    ts_tpr_mmap = None
    if check_evolution and (os.path.exists(mmap_file)):
        print('\tLoading precomputed memory-mapped array...', end='\r')
        ts_tpr_mmap = np.memmap(
            mmap_file,
            dtype=np.dtype([('t', np.float64), ('r', np.float64, (max_node + 1,))]),
            mode='r+'
        )
        print('\tPrecomputed memory-mapped array loaded.      ')
        return None, ts_tpr_mmap
    
    # Initialize score arrays with zeros
    r = np.zeros(max_node + 1, dtype=np.float64)
    s = np.zeros(max_node + 1, dtype=np.float64)
    
    # Create and sort the heap entries for deterministic processing
    # Add a counter to break timestamp ties
    heap_entries = []
    for idx, (u, v, t) in enumerate(E):
        # Use tuple comparison for stable sorting
        heap_entries.append((t, idx, u, v))
    
    # Sort entries deterministically
    heap_entries.sort()
    heapq.heapify(heap_entries)
    
    # Store temporal evolution if requested
    temporal_states = [] if check_evolution else None
    
    # Process edges in temporal order
    while heap_entries:
        t, _, u, v = heapq.heappop(heap_entries)
        
        # Update PageRank scores
        delta = 1 - alpha
        r[u] += delta
        s[u] += delta
        r[v] += s[u] * alpha
        
        # Handle time decay
        if beta < 1:
            s_v_increment = s[u] * (1 - beta) * alpha
            s[v] += s_v_increment
            s[u] *= beta
        else:
            s[v] += s[u] * alpha
            s[u] = 0
        
        # Store normalized scores at this timestamp if tracking evolution
        if check_evolution:
            total_r = np.sum(r)
            if total_r > 0:
                temporal_states.append((t, r.copy() / total_r))
    
    # Normalize final scores
    total_r = np.sum(r)
    if total_r > 0:
        r /= total_r
    
    # Create memory-mapped file if tracking evolution
    if check_evolution:
        num_states = len(temporal_states)
        ts_tpr_mmap = np.memmap(
            mmap_file,
            dtype=np.dtype([('t', np.float64), ('r', np.float64, (max_node + 1,))]),
            mode='w+',
            shape=(num_states,)
        )
        
        # Fill memory-mapped array deterministically
        for idx, (t, r_array) in enumerate(temporal_states):
            ts_tpr_mmap[idx]['t'] = t
            ts_tpr_mmap[idx]['r'] = r_array
        
        
        # Ensure data is written to disk
        ts_tpr_mmap.flush()
    return r, ts_tpr_mmap


def temporal_pagerank_heap_np(
    E: np.ndarray,
    beta: float,
    alpha: float,
    check_evolution: bool = False,
    mmap_file: str = "",
    dataset_name: str = ''
) -> Tuple[np.ndarray, Optional[np.memmap]]:
    """
    (Improved Version)
    Calculate temporal PageRank using a heap-based approach with NumPy arrays.
    """
    if not dataset_name:
        raise ValueError('dataset_name must be provided')
    
    mmap_file = f'{save_location}/{dataset_name}_ts_tpr_data.dat' if  mmap_file == "" else f'{save_location}/{mmap_file}'
    print(f'{mmap_file=}')
    
    # Ensure E is properly formatted numpy array
    if not isinstance(E, np.ndarray):
        E = np.array(E, dtype=[('u', np.int64), ('v', np.int64), ('t', np.float64)])
    
    # Get maximum node index
    max_node = max(np.max(E['u']), np.max(E['v']))
    
    # Initialize score arrays with zeros
    r = np.zeros(max_node + 1, dtype=np.float64)
    s = np.zeros(max_node + 1, dtype=np.float64)
    
    # Create and sort the heap entries for deterministic processing
    heap_entries = []
    for idx, (u, v, t) in enumerate(E):
        heap_entries.append((t, idx, u, v))
    
    # Sort entries deterministically
    heap_entries.sort()
    heapq.heapify(heap_entries)
    
    # Initialize temporal evolution tracking
    ts_tpr_mmap = None
    temporal_states = []
    
    # Check for existing memory-mapped file for temporal evolution
    if check_evolution and os.path.exists(mmap_file):
        print('\tLoading precomputed memory-mapped array...', end='\r')
        ts_tpr_mmap = np.memmap(
            mmap_file,
            dtype=np.dtype([('t', np.float64), ('r', np.float64, (max_node + 1,))]),
            mode='r+'
        )
        print('\tReturning precomputed memory-mapped array. type(ts_tpr_mmap) =', type(ts_tpr_mmap))
    else:
        # Process edges in temporal order
        while heap_entries:
            t, _, u, v = heapq.heappop(heap_entries)
            
            # Update PageRank scores
            delta = 1 - alpha
            r[u] += delta
            s[u] += delta
            r[v] += s[u] * alpha
            
            # Handle time decay
            if beta < 1:
                s_v_increment = s[u] * (1 - beta) * alpha
                s[v] += s_v_increment
                s[u] *= beta
            else:
                s[v] += s[u] * alpha
                s[u] = 0
            
            # Store normalized scores at this timestamp if tracking evolution
            if check_evolution:
                total_r = np.sum(r)
                if total_r > 0:
                    temporal_states.append((t, r.copy() / total_r))
        
        # Normalize final scores
        total_r = np.sum(r)
        if total_r > 0:
            r /= total_r
        
        # Create memory-mapped file if tracking evolution
        if check_evolution and not os.path.exists(mmap_file):
            num_states = len(temporal_states)
            ts_tpr_mmap = np.memmap(
                mmap_file,
                dtype=np.dtype([('t', np.float64), ('r', np.float64, (max_node + 1,))]),
                mode='w+',
                shape=(num_states,)
            )
            
            # Fill memory-mapped array deterministically
            for idx, (t, r_array) in enumerate(temporal_states):
                ts_tpr_mmap[idx]['t'] = t
                ts_tpr_mmap[idx]['r'] = r_array
            
            # Ensure data is written to disk
            ts_tpr_mmap.flush()
    return r, ts_tpr_mmap



def mean_shift_removal(_graph_df, beta = 0.85, alpha = 0.15, dataset_name=''):
    # print(f'\t\tin mss removal method {dataset_name=}')
    graph_df = _graph_df.copy(deep=True)
    
    # Extract nodes, edges, and timestamps
    edges = graph_df[['u', 'i', 'ts']].values
    # nodes = np.unique(edges[:, :2])  # Get unique nodes from edges

    # Convert E to a more readable format if needed
    edges_new = [(int(u), int(v), float(t)) for u, v, t in edges]
    print('running temporal page rank method', end=' ')
    _, ts_tpr= temporal_pagerank_heap_np(edges_new, beta, alpha, True, dataset_name=dataset_name)
    print('Done.')
    
    print('sorting started', end=' ')
    sorted(ts_tpr, key=lambda x: x[0])
    print('sorting Completed.')
    

    # Extract timestamps and PageRank values
    # timestamps, pagerank_arrays = zip(*ts_tpr)
    # timestamps = np.array(timestamps)
    # pagerank_arrays = np.array(pagerank_arrays)
    
    # Since ts_tpr_mmap is already a memory-mapped array, we can use it directly
    # Extract timestamps and PageRank values from the memory-mapped array
    timestamps = ts_tpr['t']
    pagerank_arrays = ts_tpr['r']

    # Calculate mean shifts between consecutive timestamps
    mean_shifts = []
    for i in tqdm(range(1, len(timestamps)), desc='running mean shift'):
        # if i % 100 ==0:
        # print('calculating mean shift....')
        prev_pagerank = pagerank_arrays[i - 1]
        curr_pagerank = pagerank_arrays[i]
        mean_shift = np.mean(np.abs(curr_pagerank - prev_pagerank))
        mean_shifts.append((timestamps[i], mean_shift))

    # Identify timesteps with highest mean shift
    mean_shifts.sort(key=lambda x: x[1], reverse=True)
    
    # print('mean shift calc done....')
    
    return mean_shifts


def mean_shift_removal2(_graph_df, beta = 0.85, alpha = 0.15, batch_size=100, dataset_name=''):
    graph_df = _graph_df.copy(deep=True)
    
    # Extract nodes, edges, and timestamps
    edges = graph_df[['u', 'i', 'ts']].values
    # nodes = np.unique(edges[:, :2])  # Get unique nodes from edges

    # Convert E to a more readable format if needed
    edges_new = [(int(u), int(v), float(t)) for u, v, t in edges]
    print('running temporal page rank method', end=' ')
    _, ts_tpr= temporal_pagerank_heap_np(edges_new, beta, alpha, True, dataset_name=dataset_name)
    print('Done.')
    
    print('sorting started', end=' ')
    sorted(ts_tpr, key=lambda x: x[0])
    print('sorting Completed.')
    

    # # Extract timestamps and PageRank values
    # timestamps, pagerank_arrays = zip(*ts_tpr)
    # timestamps = np.array(timestamps)
    # pagerank_arrays = np.array(pagerank_arrays)
    
    # Since ts_tpr_mmap is already a memory-mapped array, we can use it directly
    # Extract timestamps and PageRank values from the memory-mapped array
    timestamps = ts_tpr['t']
    pagerank_arrays = ts_tpr['r']

    # Calculate mean shifts between consecutive timestamps
    mean_shifts = []
    total_batches = int(np.ceil(len(timestamps) / batch_size))    
    for batch_start in range(0, len(timestamps), batch_size):
        batch_end = min(batch_start + batch_size, len(timestamps))
        
        for i in tqdm(range(batch_start + 1, batch_end), desc=f'Processing batch {batch_start // batch_size + 1} / {total_batches}', ncols=80, leave=False):
            prev_pagerank = np.array(pagerank_arrays[i - 1])
            curr_pagerank = np.array(pagerank_arrays[i])
            mean_shift = np.mean(np.abs(curr_pagerank - prev_pagerank))
            mean_shifts.append((timestamps[i], mean_shift))
    
    # for i in tqdm(range(1, len(timestamps)), desc='running mean shift'):
    #     # if i % 100 ==0:
    #     # print('calculating mean shift....')
    #     prev_pagerank = np.mean(pagerank_arrays[i - 1])
    #     curr_pagerank = np.mean(pagerank_arrays[i])
        # mean_shift = np.mean(np.abs(curr_pagerank - prev_pagerank))
        # mean_shifts.append((timestamps[i], mean_shift))

    # Identify timesteps with highest mean shift
    print('Done.')
    mean_shifts.sort(key=lambda x: x[1], reverse=True)
    
    print('mean shift calc done....')
    
    return mean_shifts


def _compute_mean_shifts_with_metrics(graph_df, beta=0.85, alpha=0.15, metric='mean_shift'):
    """
    Compute mean shifts and distance metrics between consecutive PageRank vectors 
    in a continuous time dynamic graph.
    
    Parameters:
        graph_df (pd.DataFrame): DataFrame containing the graph edges with columns ['u', 'i', 'ts'].
        beta (float): Damping factor for PageRank.
        alpha (float): Alpha factor for temporal PageRank.
        metric (str): Metric to use for mean shift ('mean_shift', 'euclidean', 'jaccard', 'cosine').
    
    Returns:
        list: List of tuples containing timestamp and the selected metric.
    """
    print('Starting mean shift and metrics computation...')
    
    # Extract edges from DataFrame
    edges = graph_df[['u', 'i', 'ts']].values
    edges_new = [(int(u), int(v), float(t)) for u, v, t in edges]
    
    print('Running temporal PageRank computation...')
    _, ts_tpr = temporal_pagerank_heap_np(edges_new, beta, alpha, check_evolution=True)
    print('Temporal PageRank computation completed.')
    
    print('Sorting timestamps...')
    sorted(ts_tpr, key=lambda x: x[0])    
    print('Sorting completed.')
    
    # Extract timestamps and PageRank values
    # TODO: Batched implementations of this
    timestamps, pagerank_arrays = zip(*ts_tpr)
    timestamps = np.array(timestamps)
    pagerank_arrays = np.array(pagerank_arrays)
    
    # Calculate mean shifts and distance metrics between consecutive timestamps
    results = []
    print('Calculating mean shifts and distance metrics...')
    for i in tqdm(range(1, len(timestamps)), desc=f'Processing for {metric}...'):
        prev_pagerank = np.array(pagerank_arrays[i - 1])
        curr_pagerank = np.array(pagerank_arrays[i])
        
        if metric == 'mean_shift':
            # Mean Shift
            value = np.mean(np.abs(curr_pagerank - prev_pagerank))
        elif metric == 'euclidean':
            # Euclidean Distance
            value = np.linalg.norm(curr_pagerank - prev_pagerank)
        elif metric == 'jaccard':
            # Jaccard Distance
            # value = pairwise_distances([curr_pagerank], [prev_pagerank], metric='jaccard')[0][0]
            value = pairwise_distances(curr_pagerank.reshape(1, -1), prev_pagerank.reshape(1, -1), metric='jaccard')[0][0]
        elif metric == 'cosine':
            # Cosine Similarity
            value = np.dot(curr_pagerank, prev_pagerank) / (np.linalg.norm(curr_pagerank) * np.linalg.norm(prev_pagerank))
        elif metric == 'wasserstein':
            value = wasserstein_distance(prev_pagerank, curr_pagerank)
        elif metric == 'kl_divergence':
            value = np.sum(rel_entr(prev_pagerank, curr_pagerank))
        elif metric == 'jensen_shannon_divergence':
            value = jensenshannon(prev_pagerank, curr_pagerank) 
        elif metric == 'chebyshev':
            value = chebyshev(prev_pagerank, curr_pagerank)
        else:
            raise ValueError(f"Unsupported metric: {metric} should be in mean_shift, euclidean, jaccard, cosine.")
        
        results.append((timestamps[i], value))
    
    print('Mean shifts and metrics calculation completed.')
    
    # Sort results by the selected metric in descending order if applicable
    if metric != 'cosine':  # Cosine similarity might not need descending order
        results.sort(key=lambda x: x[1], reverse=True)
    print('Sorting by the selected metric completed.')
    
    return results


def LW_compute_mean_shifts_with_metrics(graph_df, beta=0.85, alpha=0.15, metric='mean_shift', batch_size=100, dataset_name=''):
    """
    LW means Last working, before adding knowledge ablation.
    Compute mean shifts and distance metrics between consecutive PageRank vectors 
    in a continuous time dynamic graph, processing in batches.
    
    Parameters:
        graph_df (pd.DataFrame): DataFrame containing the graph edges with columns ['u', 'i', 'ts'].
        beta (float): Damping factor for PageRank.
        alpha (float): Alpha factor for temporal PageRank.
        metric (str): Metric to use for mean shift ('mean_shift', 'euclidean', 'jaccard', 'cosine').
        batch_size (int): Number of timestamps to process in each batch.
    
    Returns:
        list: List of tuples containing timestamp and the selected metric.
    """
    print(f'Starting mean shift and metrics computation with {batch_size=}...')
    
    # Extract edges from DataFrame
    edges = graph_df[['u', 'i', 'ts']].values
    edges_new = [(int(u), int(v), float(t)) for u, v, t in edges]
    
    print('Running temporal PageRank computation...')
    _, ts_tpr = temporal_pagerank_heap_np(edges_new, beta, alpha, check_evolution=True, dataset_name=dataset_name)
    print('Temporal PageRank computation completed.')
    
    print('Sorting timestamps...')
    # ts_tpr.sort(key=lambda x: x[0])
    sorted(ts_tpr, key=lambda x: x[0])    
    print('Sorting completed.')
    
    # Extract timestamps and PageRank values
    # timestamps, pagerank_arrays = zip(*ts_tpr)
    # timestamps = np.array(timestamps)
    # pagerank_arrays = np.array(pagerank_arrays)
    
    # Since ts_tpr_mmap is already a memory-mapped array, we can use it directly
    # Extract timestamps and PageRank values from the memory-mapped array
    timestamps = ts_tpr['t']
    pagerank_arrays = ts_tpr['r']
    
    results = []
    total_batches = int(np.ceil(len(timestamps) / batch_size))
    
    for batch_start in range(0, len(timestamps), batch_size):
        batch_end = min(batch_start + batch_size, len(timestamps))
        
        for i in tqdm(range(batch_start + 1, batch_end),
                desc=f'Processing for {metric} batch {batch_start // batch_size + 1} / {total_batches}', ncols=80, leave=False):
            prev_pagerank = np.array(pagerank_arrays[i - 1])
            curr_pagerank = np.array(pagerank_arrays[i])
            
            if metric == 'mean_shift':
                value = np.mean(np.abs(curr_pagerank - prev_pagerank))
            elif metric == 'euclidean':
                value = np.linalg.norm(curr_pagerank - prev_pagerank)
            elif metric == 'jaccard':
                value = pairwise_distances(curr_pagerank.reshape(1, -1), prev_pagerank.reshape(1, -1), metric='jaccard')[0][0]
            elif metric == 'cosine':
                value = np.dot(curr_pagerank, prev_pagerank) / (np.linalg.norm(curr_pagerank) * np.linalg.norm(prev_pagerank))
            elif metric == 'wasserstein':
                value = wasserstein_distance(prev_pagerank, curr_pagerank)
            elif metric == 'kl_divergence':
                value = np.sum(rel_entr(prev_pagerank, curr_pagerank))
            elif metric == 'jensen_shannon_divergence':
                value = jensenshannon(prev_pagerank, curr_pagerank) 
            elif metric == 'chebyshev':
                value = chebyshev(prev_pagerank, curr_pagerank)
            else:
                raise ValueError(f"Unsupported metric: {metric} should be in mean_shift, euclidean, jaccard, cosine.")
            
            results.append((timestamps[i], value))
    
    print('Mean shifts and metrics calculation completed.')
    
    if metric != 'cosine':
        results.sort(key=lambda x: x[1], reverse=True)
    print('Sorting by the selected metric completed.')
    
    return results


def compute_mean_shifts_with_metrics(graph_df, beta=0.85, alpha=0.15, metric='mean_shift', batch_size=100, dataset_name='', k_split='split_0.7'):
    """
    Compute mean shifts and distance metrics between consecutive PageRank vectors 
    in a continuous time dynamic graph, processing in batches.
    
    Parameters:
        graph_df (pd.DataFrame): DataFrame containing the graph edges with columns ['u', 'i', 'ts'].
        beta (float): Damping factor for PageRank.
        alpha (float): Alpha factor for temporal PageRank.
        metric (str): Metric to use for mean shift ('mean_shift', 'euclidean', 'jaccard', 'cosine').
        batch_size (int): Number of timestamps to process in each batch.
    
    Returns:
        list: List of tuples containing timestamp and the selected metric.
    """
    print(f'Starting mean shift and metrics computation with {batch_size=}...')
    
    # Extract edges from DataFrame
    edges = graph_df[['u', 'i', 'ts']].values
    edges_new = [(int(u), int(v), float(t)) for u, v, t in edges]

    mmap_file = f'{dataset_name}_ts_tpr_data_{k_split}.dat'
    
    print('Running temporal PageRank computation...')
    _, ts_tpr = temporal_pagerank_heap_np(edges_new, beta, alpha, check_evolution=True, dataset_name=dataset_name, mmap_file=mmap_file)
    print('Temporal PageRank computation completed.')
    
    print('Sorting timestamps...')
    # ts_tpr.sort(key=lambda x: x[0])
    sorted(ts_tpr, key=lambda x: x[0])    
    print('Sorting completed.')
    
    # Extract timestamps and PageRank values
    # timestamps, pagerank_arrays = zip(*ts_tpr)
    # timestamps = np.array(timestamps)
    # pagerank_arrays = np.array(pagerank_arrays)
    
    # Since ts_tpr_mmap is already a memory-mapped array, we can use it directly
    # Extract timestamps and PageRank values from the memory-mapped array
    timestamps = ts_tpr['t']
    pagerank_arrays = ts_tpr['r']
    
    results = []
    total_batches = int(np.ceil(len(timestamps) / batch_size))
    
    for batch_start in range(0, len(timestamps), batch_size):
        batch_end = min(batch_start + batch_size, len(timestamps))
        
        for i in tqdm(range(batch_start + 1, batch_end),
                desc=f'Processing for {metric} batch {batch_start // batch_size + 1} / {total_batches}', ncols=80, leave=False):
            prev_pagerank = np.array(pagerank_arrays[i - 1])
            curr_pagerank = np.array(pagerank_arrays[i])
            
            if metric == 'mean_shift':
                value = np.mean(np.abs(curr_pagerank - prev_pagerank))
            elif metric == 'euclidean':
                value = np.linalg.norm(curr_pagerank - prev_pagerank)
            elif metric == 'jaccard':
                value = pairwise_distances(curr_pagerank.reshape(1, -1), prev_pagerank.reshape(1, -1), metric='jaccard')[0][0]
            elif metric == 'cosine':
                value = np.dot(curr_pagerank, prev_pagerank) / (np.linalg.norm(curr_pagerank) * np.linalg.norm(prev_pagerank))
            elif metric == 'wasserstein':
                value = wasserstein_distance(prev_pagerank, curr_pagerank)
            elif metric == 'kl_divergence':
                value = np.sum(rel_entr(prev_pagerank, curr_pagerank))
            elif metric == 'jensen_shannon_divergence':
                value = jensenshannon(prev_pagerank, curr_pagerank) 
            elif metric == 'chebyshev':
                value = chebyshev(prev_pagerank, curr_pagerank)
            else:
                raise ValueError(f"Unsupported metric: {metric} should be in mean_shift, euclidean, jaccard, cosine.")
            
            results.append((timestamps[i], value))
    
    print('Mean shifts and metrics calculation completed.')
    
    if metric != 'cosine':
        results.sort(key=lambda x: x[1], reverse=True)
    print('Sorting by the selected metric completed.')
    
    return results


def compute_overall_outgoing_degree(E):
    return Counter(E['u'])

def calculate_temporal_edge_rank(_graph_df, beta=0.85, alpha=0.15, batch_size=1000, dataset_name='', k_split='split_0.7'):
    # had added K-split for knowledge ablation
    graph_df = _graph_df.copy(deep=True)
    
    # Extract nodes, edges, and timestamps as a list of tuples
    edges = graph_df[['u', 'i', 'ts']].values.tolist()
    edges = [(int(u), int(v), float(t)) for u, v, t in edges]
    
    print(f'In calculate_temporal_edge_rank {dataset_name=}')

    mmap_file = f'{dataset_name}_ts_tpr_data_{k_split}.dat'
    
    _, ts_tpr = temporal_pagerank_heap_np(edges, beta, alpha, True, dataset_name=dataset_name, mmap_file=mmap_file)
    
    # Create numpy array for efficient operations
    edges_array = np.array(edges, dtype=[('u', int), ('v', int), ('t', float)])
    temporal_outgoing_degree = compute_overall_outgoing_degree(edges_array)
    
    # Extract timestamps and PageRank values
    # timestamps, pagerank_arrays = zip(*ts_tpr)
    # timestamps = np.array(timestamps)
    # pagerank_arrays = np.array(pagerank_arrays)
    
    # Since ts_tpr_mmap is already a memory-mapped array, we can use it directly
    # Extract timestamps and PageRank values from the memory-mapped array
    timestamps = ts_tpr['t']
    pagerank_arrays = ts_tpr['r']
    
    ts_to_node_dict = graph_df.groupby('ts')['u'].apply(np.array).to_dict()
    num_timestamps = len(timestamps)
    
    ter_dict = {}
    
    # Process in batches
    for start_idx in (range(0, num_timestamps, batch_size)):
        end_idx = min(start_idx + batch_size, num_timestamps)
        
        # Get the batch of timestamps and corresponding PageRank arrays
        batch_timestamps = timestamps[start_idx:end_idx]
        batch_pagerank_arrays = pagerank_arrays[start_idx:end_idx]
        
        # Prepare to store results for this batch
        batch_ter_dict = {}
        
        for i in tqdm(range(len(batch_timestamps)), desc='Calculating TER', ncols=80, leave=False):
            ts = batch_timestamps[i]
            r = batch_pagerank_arrays[i]
            
            # Get node list for the current timestamp
            node_list = ts_to_node_dict[ts]
            
            # Compute Temporal EdgeRank (TER) for the current batch
            node_out_deg = np.array([temporal_outgoing_degree[node] for node in node_list])
            ter = r[node_list] / node_out_deg

            # get TER of participating nodes (u, v) and sum it
            
            # Store result for the current timestamp
            batch_ter_dict[ts] = np.sum(ter)  # Better Function that can get more info
        
        # Update the global ter_dict with batch results
        ter_dict.update(batch_ter_dict)
    
    # for i in tqdm(range(len(timestamps)), desc='Calculating TER'):
    #     r = pagerank_arrays[i]
    #     ts = timestamps[i]
    #     node_list = ts_to_node_dict[ts]
    #     ter = r[node_list] / np.vectorize(temporal_outgoing_degree.__getitem__)(node_list)
    #     ter_dict[ts] = np.sum(ter)
    
    return ter_dict


def calculate_temporal_edge_rank_working(_graph_df, beta=0.85, alpha=0.15, batch_size=1000, dataset_name=''):
    # This was before knowledge ablation
    graph_df = _graph_df.copy(deep=True)
    
    # Extract nodes, edges, and timestamps as a list of tuples
    edges = graph_df[['u', 'i', 'ts']].values.tolist()
    edges = [(int(u), int(v), float(t)) for u, v, t in edges]
    
    print(f'In calculate_temporal_edge_rank {dataset_name=}')
    
    _, ts_tpr = temporal_pagerank_heap_np(edges, beta, alpha, True, dataset_name=dataset_name)
    
    # Create numpy array for efficient operations
    edges_array = np.array(edges, dtype=[('u', int), ('v', int), ('t', float)])
    temporal_outgoing_degree = compute_overall_outgoing_degree(edges_array)
    
    # Extract timestamps and PageRank values
    # timestamps, pagerank_arrays = zip(*ts_tpr)
    # timestamps = np.array(timestamps)
    # pagerank_arrays = np.array(pagerank_arrays)
    
    # Since ts_tpr_mmap is already a memory-mapped array, we can use it directly
    # Extract timestamps and PageRank values from the memory-mapped array
    timestamps = ts_tpr['t']
    pagerank_arrays = ts_tpr['r']
    
    ts_to_node_dict = graph_df.groupby('ts')['u'].apply(np.array).to_dict()
    num_timestamps = len(timestamps)
    
    ter_dict = {}
    
    # Process in batches
    for start_idx in (range(0, num_timestamps, batch_size)):
        end_idx = min(start_idx + batch_size, num_timestamps)
        
        # Get the batch of timestamps and corresponding PageRank arrays
        batch_timestamps = timestamps[start_idx:end_idx]
        batch_pagerank_arrays = pagerank_arrays[start_idx:end_idx]
        
        # Prepare to store results for this batch
        batch_ter_dict = {}
        
        for i in tqdm(range(len(batch_timestamps)), desc='Calculating TER', ncols=80, leave=False):
            ts = batch_timestamps[i]
            r = batch_pagerank_arrays[i]
            
            # Get node list for the current timestamp
            node_list = ts_to_node_dict[ts]
            
            # Compute Temporal EdgeRank (TER) for the current batch
            node_out_deg = np.array([temporal_outgoing_degree[node] for node in node_list])
            ter = r[node_list] / node_out_deg

            # get TER of participating nodes (u, v) and sum it
            
            # Store result for the current timestamp
            batch_ter_dict[ts] = np.sum(ter)  # Better Function that can get more info
        
        # Update the global ter_dict with batch results
        ter_dict.update(batch_ter_dict)
    
    # for i in tqdm(range(len(timestamps)), desc='Calculating TER'):
    #     r = pagerank_arrays[i]
    #     ts = timestamps[i]
    #     node_list = ts_to_node_dict[ts]
    #     ter = r[node_list] / np.vectorize(temporal_outgoing_degree.__getitem__)(node_list)
    #     ter_dict[ts] = np.sum(ter)
    
    return ter_dict

def calculate_temporal_edge_rank_mean(_graph_df, beta=0.85, alpha=0.15, batch_size=1000, dataset_name=''):

    """ 
    We want to test what if we don't take isolate the nodes and define edgerank for the timestamp itself.
    """
    graph_df = _graph_df.copy(deep=True)
    
    # Extract nodes, edges, and timestamps as a list of tuples
    edges = graph_df[['u', 'i', 'ts']].values.tolist()
    edges = [(int(u), int(v), float(t)) for u, v, t in edges]
    
    print(f'In calculate_temporal_edge_rank {dataset_name=}')
    
    _, ts_tpr = temporal_pagerank_heap_np(edges, beta, alpha, True, dataset_name=dataset_name)
    
    # Create numpy array for efficient operations
    edges_array = np.array(edges, dtype=[('u', int), ('v', int), ('t', float)])
    temporal_outgoing_degree = compute_overall_outgoing_degree(edges_array)
    
    
    # Since ts_tpr_mmap is already a memory-mapped array, we can use it directly
    # Extract timestamps and PageRank values from the memory-mapped array
    timestamps = ts_tpr['t']
    pagerank_arrays = ts_tpr['r']
    
    ts_to_node_dict = graph_df.groupby('ts')['u'].apply(np.array).to_dict()
    num_timestamps = len(timestamps)
    
    ter_dict = {}
    
    # Process in batches
    for start_idx in (range(0, num_timestamps, batch_size)):
        end_idx = min(start_idx + batch_size, num_timestamps)
        
        # Get the batch of timestamps and corresponding PageRank arrays
        batch_timestamps = timestamps[start_idx:end_idx]
        batch_pagerank_arrays = pagerank_arrays[start_idx:end_idx]
        
        # Prepare to store results for this batch
        batch_ter_dict = {}
        
        for i in tqdm(range(len(batch_timestamps)), desc='Calculating TER', ncols=80, leave=False):
            ts = batch_timestamps[i]
            r = batch_pagerank_arrays[i]
            
            # Get node list for the current timestamp
            node_list = ts_to_node_dict[ts]
            
            # Compute Temporal EdgeRank (TER) for the current batch
            node_out_deg = np.array([temporal_outgoing_degree[node] for node in node_list])
            ter = r[node_list] / node_out_deg

            # get TER of all nodes in that timestamp and sum it            
            # Store result for the current timestamp
            batch_ter_dict[ts] = np.mean(ter)  # Better Function that can get more info
        
        # Update the global ter_dict with batch results
        ter_dict.update(batch_ter_dict)
    
    return ter_dict


# Function to compute the global Temporal PageRank
def compute_global_temporal_pagerank(edges, beta=0.85, alpha=0.15):
    # Use your existing Temporal PageRank calculation method (e.g., temporal_pagerank_heap_np)
    # This should return a dictionary with nodes as keys and global TPR as values
    global_tpr, _ = temporal_pagerank_heap_np(edges, beta, alpha, False)
    return global_tpr

# Function to compute timestamp-specific Temporal PageRank
def compute_timestamp_specific_temporal_pagerank(edges, beta=0.85, alpha=0.15):
    # Use your existing Temporal PageRank calculation method with timestamp-specific mode
    # This should return a list of (timestamp, dict) where dict contains node-specific TPR
    _, ts_tpr = temporal_pagerank_heap_np(edges, beta, alpha, True)
    return ts_tpr

# Function to compute the outgoing degree of nodes globally
def compute_global_outgoing_degree(edges):
    edges = np.array(edges, dtype=[('u', int), ('v', int), ('t', float)])
    u_values = edges['u'].tolist()
    return dict(Counter(u_values))

# Function to compute the outgoing degree of nodes at specific timestamps
def compute_timestamp_specific_outgoing_degree(edges):
    edges_by_ts = defaultdict(list)
    for u, v, t in edges:
        edges_by_ts[t].append((u, v))
    
    ts_outgoing_degree = {}
    for t, edge_list in tqdm(edges_by_ts.items()):
        ts_outgoing_degree[t] = dict(Counter([u for u, _ in edge_list]))
    
    return ts_outgoing_degree


def compute_cumulative_timestamp_specific_outgoing_degree(graph_df, batch_size=1000):
    # Sort by timestamp to ensure cumulative counting is correct
    graph_df = graph_df.sort_values('ts')
    
    # Initialize cumulative outgoing degree dictionary
    cumulative_outgoing_degree = defaultdict(int)
    
    # Initialize result dictionary
    ts_outgoing_degree = {}
    
    # # Iterate through the DataFrame row by row
    # for ts, group in tqdm(graph_df.groupby('ts'), desc="Calculating cumulative outgoing degree"):
    #     # Update cumulative counts
    #     for u in group['u']:
    #         cumulative_outgoing_degree[u] += 1
            
    #     # Store the current cumulative counts in the result dictionary
    #     ts_outgoing_degree[ts] = dict(cumulative_outgoing_degree)
    
    # Get unique timestamps
    unique_timestamps = graph_df['ts'].unique()
    
    # Process the timestamps in batches
    for start_idx in tqdm(range(0, len(unique_timestamps), batch_size), desc="Calculating cumulative outgoing degree"):
        end_idx = min(start_idx + batch_size, len(unique_timestamps))
        
        # Get the batch of timestamps
        batch_timestamps = unique_timestamps[start_idx:end_idx]
        
        # Filter the DataFrame for the current batch of timestamps
        batch_df = graph_df[graph_df['ts'].isin(batch_timestamps)]
        
        # Group by timestamp and update cumulative counts
        for ts, group in batch_df.groupby('ts'):
            for u in group['u']:
                cumulative_outgoing_degree[u] += 1
                
            # Store the current cumulative counts in the result dictionary
            ts_outgoing_degree[ts] = dict(cumulative_outgoing_degree)
    
    return ts_outgoing_degree



# Function to compute Temporal EdgeRank for each edge
def _compute_temporal_edgerank(graph_df, beta=0.85, alpha=0.15, gamma=0.5, dataset_name=''):
    edges_list = graph_df[['u', 'i', 'ts']].values
    edges = np.array(
        list(zip(edges_list[:, 0].astype(int), edges_list[:, 1].astype(int), edges_list[:, 2].astype(float))),
        dtype=[('u', int), ('v', int), ('t', float)]
    )
    
    # edges
    print(f'Calculating TPR...', end='\r')
    global_tpr, ts_tpr = temporal_pagerank_heap_np(edges, beta, alpha, True, dataset_name)
    # Convert ts_tpr to a dictionary with timestamps as keys
    ts_tpr = {row['t']: row['r'] for row in ts_tpr}
    print(f'Calculated TPR      ')
    
    # ts_tpr = compute_timestamp_specific_temporal_pagerank(edges, beta, alpha)  # redundant
    print(f'Calculating global_out_deg...', end='\r')
    global_out_deg = compute_global_outgoing_degree(edges)
    print(f'Calculated global_out_deg      ')
    print(f'Calculating cumulative_timestamp_specific_outgoing_degree...', end='\r')
    ts_out_deg = compute_cumulative_timestamp_specific_outgoing_degree(graph_df[['u', 'i', 'ts']])
    print(f'Calculated cumulative_timestamp_specific_outgoing_degree       ')
    
    ter_dict = {}

    for u, _, t in tqdm((edges), desc='Processing'):
        TER_global = global_tpr[u] / global_out_deg[u]
        TER_ts = ts_tpr[t][u] / ts_out_deg[t][u]
        ter_ts = gamma * TER_global + (1 - gamma) * TER_ts
        ter_dict[t] = ter_ts
        
    return ter_dict


# Function to compute Temporal EdgeRank for each edge
def compute_combined_temporal_edgerank(graph_df, beta=0.85, alpha=0.15, gamma=0.5, batch_size=1000, dataset_name=''):
    # Extract nodes, edges, and timestamps as a list of tuples
    edges = graph_df[['u', 'i', 'ts']].values.tolist()
    edges = [(int(u), int(v), float(t)) for u, v, t in edges]
    
    print(f'Calculating TPR...', end='\r')
    global_tpr, ts_tpr_mmap = temporal_pagerank_heap_np(edges, beta, alpha, True, dataset_name=dataset_name)
    
    print(f'Calculated TPR      ')
    
    # Convert global_tpr and ts_out_deg to appropriate formats
    global_out_deg = compute_global_outgoing_degree(edges)
    ts_out_deg = compute_cumulative_timestamp_specific_outgoing_degree(graph_df[['u', 'i', 'ts']])
    
    print(len(global_out_deg))
    ter_dict = {}

    # Access the memory-mapped array
    ts_tpr_shape = len(ts_tpr_mmap)
    
    # Process edges in batches
    for start_idx in tqdm(range(0, len(edges), batch_size), desc='Processing'):
        end_idx = min(start_idx + batch_size, len(edges))
        
        # Get the batch of edges
        batch_edges = edges[start_idx:end_idx]
        
        # Extract timestamps from the batch
        batch_timestamps = np.array([t for _, _, t in batch_edges])
        
        # Find the indices of these timestamps in the memory-mapped array
        indices = np.searchsorted(ts_tpr_mmap['t'], batch_timestamps)
        
        # Process each edge in the batch
        for i, (u, _, t) in enumerate(batch_edges):
            idx = indices[i]
            
            if idx < ts_tpr_shape and ts_tpr_mmap['t'][idx] == t:
                # Access the PageRank values for the given timestamp
                TER_global = global_tpr[u] / global_out_deg[u]
                TER_ts = ts_tpr_mmap[idx]['r'][u] / ts_out_deg[t][u]
                ter_ts = gamma * TER_global + (1 - gamma) * TER_ts
                ter_dict[t] = ter_ts
            else:
                # Handle the case where the timestamp is not found in the memory-mapped array
                ter_dict[t] = 0  # Or another suitable default value
    
    return ter_dict



# Main function to call and calculate Combined Temporal EdgeRank
def calculate_combined_temporal_edgerank(graph_df, beta=0.85, alpha=0.15, gamma=0.5, dataset_name=''):
    
    return compute_combined_temporal_edgerank(graph_df, beta, alpha, gamma, dataset_name=dataset_name)


