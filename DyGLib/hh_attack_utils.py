from copy import deepcopy
from scipy import stats
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional
from itertools import chain, product
import bisect
from collections import defaultdict

np.random.seed(1729928109)

def compute_overall_outgoing_degree(E):
    return Counter(E['u'])


def get_nodes_in_time_window(df: pd.DataFrame, current_time: float, window: int) -> Set[int]:
    """
    Get the set of nodes that were active within the time window before current_time.
    
    Args:
    df (pd.DataFrame): DataFrame with interactions ('u' for source, 'i' for destination, 'ts' for timestamp)
    current_time (float): Current timestamp
    window (int): Time window size
    
    Returns:
    Set[int]: Set of nodes active within the time window
    """
    window_start = max(current_time - window, df['ts'].min())
    window_df = df[(df['ts'] >= window_start) & (df['ts'] <= current_time)]
    return set(window_df['u']) | set(window_df['i'])

def create_node_mapping(original_degrees: Dict[int, int], generated_degrees: Dict[int, int]) -> Dict[int, int]:
    """
    Create a one-to-one mapping between generated and original nodes based on degree.
    
    Args:
    original_degrees (Dict[int, int]): Degree distribution of original nodes
    generated_degrees (Dict[int, int]): Degree distribution of generated nodes
    
    Returns:
    Dict[int, int]: Mapping from generated nodes to original nodes
    """
    degree_to_nodes = defaultdict(list)
    for node, degree in original_degrees.items():
        degree_to_nodes[degree].append(node)
    
    generated_degree_to_nodes = defaultdict(list)
    for node, degree in generated_degrees.items():
        generated_degree_to_nodes[degree].append(node)
    
    mapping = {}
    for degree, nodes in degree_to_nodes.items():
        generated_nodes = generated_degree_to_nodes.get(degree, [])
        for original_node, generated_node in zip(nodes, generated_nodes):
            mapping[generated_node] = original_node
    
    return mapping


class TemporalSampler:
    def __init__(self, original_graph, seed=None, time_window=1200, 
                 num_bins=1000, min_activity_threshold=3):
        """
        Enhanced Temporal Sampler with activity awareness and adaptive sampling
        
        Args:
            original_graph: DataFrame with 'ts', 'u', 'i' columns
            seed: Random seed
            time_window: Time window size for node availability
            num_bins: Number of temporal bins for activity analysis
            min_activity_threshold: Minimum number of nodes required in a time period
        """
        self.original_graph = original_graph
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.time_window = time_window
        self.num_bins = num_bins
        self.min_activity_threshold = min_activity_threshold
        
        # Time bounds
        self.min_time = min(0, original_graph['ts'].min())
        self.max_time = original_graph['ts'].max()
        self.time_range = self.max_time - self.min_time
        
        # Initialize components
        self._compute_activity_density()
        self._initialize_kde()
        self._initialize_node_windows()
        
    def _compute_activity_density(self):
        """Compute temporal activity density and identify active periods"""
        # Create temporal bins
        self.bin_edges = np.linspace(self.min_time, self.max_time, self.num_bins + 1)
        self.bin_width = (self.max_time - self.min_time) / self.num_bins
        
        # Compute activity density
        df_sorted = self.original_graph.sort_values('ts')
        hist, _ = np.histogram(df_sorted['ts'], bins=self.bin_edges)
        self.activity_density = hist / hist.sum()
        
        # Track node activities
        self.src_activity = defaultdict(list)
        self.dst_activity = defaultdict(list)
        
        for _, row in df_sorted.iterrows():
            bin_idx = np.searchsorted(self.bin_edges, row['ts']) - 1
            if bin_idx < self.num_bins:
                self.src_activity[row['u']].append(bin_idx)
                self.dst_activity[row['i']].append(bin_idx)
        
        # Identify active periods
        self.active_bins = np.where(hist >= self.min_activity_threshold)[0]
        self.active_periods = self._merge_active_periods()
        
    def _merge_active_periods(self):
        """Merge consecutive active periods"""
        if len(self.active_bins) == 0:
            return []
            
        periods = []
        start = self.active_bins[0]
        prev = start
        
        for bin_idx in self.active_bins[1:]:
            if bin_idx - prev > 1:
                periods.append((
                    self.bin_edges[start],
                    self.bin_edges[prev + 1]
                ))
                start = bin_idx
            prev = bin_idx
            
        periods.append((
            self.bin_edges[start],
            self.bin_edges[prev + 1]
        ))
        
        return periods
        
    def _initialize_kde(self):
        """Initialize adaptive KDE with local bandwidth"""
        df_sorted = self.original_graph.sort_values('ts')
        inter_event_times = np.diff(df_sorted['ts'])
        
        # Compute local density for adaptive bandwidth
        local_density = []
        window_size = len(inter_event_times) // 20  # 5% of data
        
        for i in range(len(inter_event_times)):
            start_idx = max(0, i - window_size//2)
            end_idx = min(len(inter_event_times), i + window_size//2)
            local_density.append(len(inter_event_times[start_idx:end_idx]))
            
        local_density = np.array(local_density)
        self.bandwidths = 1 / np.sqrt(local_density + 1)  # Adaptive bandwidth
        
        # Initialize KDE with adaptive bandwidth
        self.kde = stats.gaussian_kde(
            inter_event_times,
            bw_method=np.mean(self.bandwidths)
        )
        
    def _initialize_node_windows(self):
        """Track node availability windows"""
        df_sorted = self.original_graph.sort_values('ts')
        self.node_windows = defaultdict(list)
        
        window_starts = df_sorted['ts'].values
        window_ends = window_starts + self.time_window
        
        for ts, u, i in df_sorted[['ts', 'u', 'i']].values:
            self.node_windows[u].append((ts, ts + self.time_window))
            self.node_windows[i].append((ts, ts + self.time_window))
            
        # Merge overlapping windows
        for node in self.node_windows:
            merged = []
            current = None
            
            for window in sorted(self.node_windows[node]):
                if current is None:
                    current = list(window)
                elif window[0] <= current[1]:
                    current[1] = max(current[1], window[1])
                else:
                    merged.append(tuple(current))
                    current = list(window)
                    
            if current is not None:
                merged.append(tuple(current))
                
            self.node_windows[node] = merged

    def assign_timestamps(self, num_samples):
        """
        Assign timestamps using activity-aware sampling
        """
        sampled_timestamps = []
        required_samples = num_samples
        
        # Sample from active periods with higher probability
        while len(sampled_timestamps) < required_samples:
            # Choose active period weighted by activity density
            period_weights = [
                np.sum(self.activity_density[
                    np.searchsorted(self.bin_edges, start):
                    np.searchsorted(self.bin_edges, end)
                ])
                for start, end in self.active_periods
            ]
            period_weights = np.array(period_weights) / np.sum(period_weights)
            
            chosen_period = self.active_periods[
                self.rng.choice(len(self.active_periods), p=period_weights)
            ]
            
            # Sample timestamps within chosen period
            local_samples = min(
                required_samples - len(sampled_timestamps),
                max(1, int(required_samples * period_weights[0]))
            )
            
            # Generate samples using KDE
            kde_samples = self.kde.resample(local_samples, self.rng)[0]
            timestamps = np.cumsum(kde_samples) + chosen_period[0]
            timestamps = timestamps[timestamps <= chosen_period[1]]
            
            # Add jitter while maintaining window constraints
            jitter = self.rng.uniform(
                -self.time_window * 0.05,
                self.time_window * 0.05,
                len(timestamps)
            )
            timestamps += jitter
            
            # Ensure timestamps are within bounds
            timestamps = np.clip(timestamps, self.min_time, self.max_time)
            sampled_timestamps.extend(timestamps)
            
        sampled_timestamps = np.array(sorted(sampled_timestamps[:num_samples]))
        return sampled_timestamps
        
    def get_activity_metrics(self):
        """Return activity metrics for analysis"""
        return {
            'activity_density': self.activity_density,
            'active_periods': self.active_periods,
            'bin_edges': self.bin_edges,
            'total_active_bins': len(self.active_bins)
        }


class EdgeTimestampSelector:
   def __init__(self,
                feasible_edges_dict: Dict,
                original_df: pd.DataFrame, 
                degree_dict: Dict[int, int],
                removed_edges: Set[Tuple[int, int]],
                src_nodes: Optional[Set[int]] = None,
                dst_nodes: Optional[Set[int]] = None,
                time_window:int=1200):
       """Initialize Edge-Timestamp Selector"""
       self.feasible_edges_dict = feasible_edges_dict
       self.original_df = original_df
       self.degree_dict = degree_dict
       self.removed_edges = removed_edges
       self.is_bipartite = src_nodes is not None and dst_nodes is not None
       self.src_nodes = src_nodes
       self.dst_nodes = dst_nodes
       self.time_window = time_window
       
       # Initialize tracking structures
       self.node_usage = defaultdict(int)
       self.selected_pairs = []
       self.remaining_capacity = deepcopy(degree_dict)
       
       # Calculate edge budget and create lookup structures
       self.max_edges = self._calculate_edge_budget()
       self.node_to_edges = self._create_node_edge_mapping()
       self.feasible_adj_lists = self._create_adj_lists()
           
   def _calculate_edge_budget(self) -> int:
       """Calculate maximum possible edges"""
       if self.is_bipartite:
           src_sum = sum(self.degree_dict[n] for n in self.src_nodes)
           dst_sum = sum(self.degree_dict[n] for n in self.dst_nodes)
           return src_sum
       return sum(self.degree_dict.values()) // 2
       
   def _create_node_edge_mapping(self) -> Dict[int, List[Tuple[int, int]]]:
       """Create mapping from nodes to their edges"""
       node_to_edges = defaultdict(list)
       for edge in self.feasible_edges_dict:
           u, v = edge
           node_to_edges[u].append(edge)
           node_to_edges[v].append(edge)
       return node_to_edges

   def _create_adj_lists(self) -> Dict[int, Set[int]]:
       """Create adjacency lists for faster lookup"""
       adj_lists = defaultdict(set)
       for (u, v) in self.feasible_edges_dict:
           adj_lists[u].add(v)
           if not self.is_bipartite:
               adj_lists[v].add(u)
       return adj_lists
       
   def select_edges_timestamps(self) -> List[Tuple[int, int, int]]:
       """Main edge selection algorithm"""
       # Sort nodes by degree for priority
       priority_nodes = sorted(
           self.degree_dict.keys(),
           key=lambda x: (self.degree_dict[x], -len(self.node_to_edges[x])),
           reverse=True
       )
       
       selected_count = 0
       pbar = tqdm(total=self.max_edges, desc="\tSelecting edges", leave=False, ncols=80)
       
       # Initial edge selection phase
       for node in priority_nodes:
           if selected_count >= self.max_edges:
               break
               
           if self.remaining_capacity[node] <= 0:
               continue
               
           while self.remaining_capacity[node] > 0 and selected_count < self.max_edges:
               candidates = self._get_candidates(node)
               if not candidates:
                   break
                   
               scored_candidates = [
                   (self._calculate_score(edge, timestamps), edge, timestamps) 
                   for edge, timestamps in candidates
               ]
               scored_candidates.sort(reverse=True)
               
               selected = False
               for _, edge, timestamps in scored_candidates:
                   if self._can_select(edge, timestamps):
                       best_timestamp = self._pick_best_timestamp(timestamps, edge)
                       self.selected_pairs.append((edge[0], edge[1], best_timestamp))
                       self._update_tracking(edge)
                       selected_count += 1
                       pbar.update(1)
                       selected = True
                       break
                       
               if not selected:
                   break
       
       # Recovery phase if needed
       if selected_count < self.max_edges:
           print('\tStarting recovery phase...')
           remaining = self.max_edges - selected_count
           print(f"\tRemaining edges: {remaining}")
           additional_edges = self._fast_recovery_phase(remaining, self.original_df)
           print(f'\tRecovery phase complete. Added {len(additional_edges)} additional edges.')
           selected_count += len(additional_edges)
           pbar.update(len(additional_edges))
       
       pbar.close()
       self._validate_selections()
       return self.selected_pairs
   
   def _fast_recovery_phase(self, needed_edges: int, original_df: pd.DataFrame) -> List[Tuple[int, int, int]]:
       """Recovery phase with C3 compliance and optimized window queries"""
       additional_edges = []
       
       # Pre-process data
       df_sorted = original_df.sort_values('ts').set_index('ts')
       node_appearances = defaultdict(list)
       for idx, row in df_sorted.iterrows():
           node_appearances[row['u']].append(idx)
           node_appearances[row['i']].append(idx)
       
       available_src_nodes = [n for n in self.src_nodes if self.remaining_capacity[n] > 0]
       available_dst_nodes = [n for n in self.dst_nodes if self.remaining_capacity[n] > 0]
       used_pairs = {(e[0], e[1]) for e in self.selected_pairs}
       
       all_timestamps = sorted(list(set(
           ts for timestamps in self.feasible_edges_dict.values() 
           for ts in timestamps
       )))
       
       window_cache = {}
       window_size = self.time_window*1e3
    #    window_size = 1200*1e3  # original
       
       def get_nodes_in_window(timestamp: float) -> Set[int]:
           if timestamp in window_cache:
               return window_cache[timestamp]
               
           window_start = max(timestamp - window_size, df_sorted.index.min())
           nodes_in_window = set()
           
           for node in chain(available_src_nodes, available_dst_nodes):
               appearances = node_appearances[node]
               idx = bisect.bisect_left(appearances, window_start)
               if idx < len(appearances) and appearances[idx] <= timestamp:
                   nodes_in_window.add(node)
           
           window_cache[timestamp] = nodes_in_window
           return nodes_in_window
       
       # Process timestamps in batches
       batch_size = 1000
       for i in range(0, len(all_timestamps), batch_size):
           batch_timestamps = all_timestamps[i:i + batch_size]
           
           for ts in batch_timestamps:
               get_nodes_in_window(ts)
           
           for timestamp in batch_timestamps:
               if len(additional_edges) >= needed_edges:
                   break
                   
               nodes_in_window = window_cache[timestamp]
               valid_src = [n for n in available_src_nodes 
                          if n in nodes_in_window and self.remaining_capacity[n] > 0]
               valid_dst = [n for n in available_dst_nodes 
                          if n in nodes_in_window and self.remaining_capacity[n] > 0]
               
               for src_node in valid_src:
                   if len(additional_edges) >= needed_edges:
                       break
                       
                   for dst_node in valid_dst:
                       edge = (src_node, dst_node)
                       if edge not in used_pairs and self._can_select(edge, [timestamp]):
                           additional_edges.append((src_node, dst_node, timestamp))
                           used_pairs.add(edge)
                           self._update_tracking(edge)
                           
                           if len(additional_edges) >= needed_edges:
                               break
                               
           window_cache.clear()
       
       self.selected_pairs.extend(additional_edges)
       return additional_edges

   def _get_candidates(self, node: int) -> List[Tuple[Tuple[int, int], List[int]]]:
       """Get valid candidate edges for a node"""
       return [(edge, self.feasible_edges_dict[edge]) 
               for edge in self.node_to_edges[node]
               if self.remaining_capacity[self._get_other_node(edge, node)] > 0 
               and edge in self.feasible_edges_dict]

   def _get_other_node(self, edge: Tuple[int, int], node: int) -> int:
       """Get the other node in an edge"""
       u, v = edge
       return v if node == u else u
       
   def _calculate_score(self, edge: Tuple[int, int], timestamps: List[int]) -> float:
       """Calculate score for candidate edge"""
       u, v = edge
       score = (self.remaining_capacity[u] + self.remaining_capacity[v] + 
               len(timestamps) - (self.node_usage[u] + self.node_usage[v]) * 10)
       if self.remaining_capacity[u] > 0 and self.remaining_capacity[v] > 0:
           score += 50
       return score
       
   def _can_select(self, edge: Tuple[int, int], timestamps: List[int]) -> bool:
       """Check if edge can be selected"""
       u, v = edge
       if not timestamps or self.remaining_capacity[u] <= 0 or self.remaining_capacity[v] <= 0:
           return False
       if self.is_bipartite and ((u in self.src_nodes and v in self.src_nodes) or 
                                (u in self.dst_nodes and v in self.dst_nodes)):
           return False
       return True

   def _pick_best_timestamp(self, timestamps: List[int], edge: Tuple[int, int]) -> int:
       """Select best timestamp from available options"""
       return max(timestamps)  # min(timestamps)

   def _update_tracking(self, edge: Tuple[int, int]):
       """Update tracking after edge selection"""
       u, v = edge
       self.remaining_capacity[u] -= 1
       self.remaining_capacity[v] -= 1
       self.node_usage[u] += 1
       self.node_usage[v] += 1

   def _validate_selections(self):
       """Validate final selections"""
       node_counts = defaultdict(int)
       for u, v, _ in self.selected_pairs:
           node_counts[u] += 1
           node_counts[v] += 1
           
       src_used = len([n for n in node_counts if n in self.src_nodes])
       dst_used = len([n for n in node_counts if n in self.dst_nodes])
       
       if self.is_bipartite:
           print(f"\tSource nodes used: {src_used}")
           print(f"\tDestination nodes used: {dst_used}")


def calculate_node_degrees(df):
    """Calculates the degree for source and destination nodes separately."""
    src_degrees = Counter(df['u'])
    dst_degrees = Counter(df['i'])
    return src_degrees, dst_degrees


def check_c3_compliance(df, negative_samples, time_window=1200*1e3):
    df_sorted = df.sort_values('ts')
    compliant = []
    for u, i, ts in negative_samples:
        window_start = max(ts - time_window, df_sorted['ts'].min())
        window_df = df_sorted[(df_sorted['ts'] >= window_start) & (df_sorted['ts'] <= ts)]
        nodes_in_window = set(window_df['u']) | set(window_df['i'])
        compliant.append((u in nodes_in_window) and (i in nodes_in_window))
    return compliant


def get_filtered_dataframes(
    original_df: pd.DataFrame, 
    sparse_df: pd.DataFrame,
    start_time: float = None,
    end_time: float = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter both dataframes based on time window.
    
    Args:
        original_df: Original dataframe with columns ['u', 'i', 'ts']
        sparse_df: Sparse dataframe with columns ['u', 'i', 'ts']
        start_time: Starting timestamp for filtering. If None, uses sparse_df's second timestamp
        end_time: Ending timestamp for filtering. If None, uses max timestamp from both dataframes
    
    Returns:
        Tuple containing filtered original and sparse dataframes
    """
    # Use default values if not provided
    start_time = sparse_df['ts'].iloc[1] if start_time is None else start_time
    end_time = max(sparse_df['ts'].max(), original_df['ts'].max()) if end_time is None else end_time
    
    # Filter both dataframes
    filtered_original = original_df[
        (original_df['ts'] > start_time) & 
        (original_df['ts'] < end_time)
    ]
    filtered_sparse = sparse_df[
        (sparse_df['ts'] > start_time) & 
        (sparse_df['ts'] < end_time)
    ]
    
    return filtered_original, filtered_sparse


def find_missing_edges(
    original_df: pd.DataFrame, 
    sparse_df: pd.DataFrame, 
    batch_size: int = 4,
    start_time: float = None,
    end_time: float = None
) -> List[pd.DataFrame]:
    """
    Find edges present in original_df but missing in sparse_df and split into batches.
    
    Args:
        original_df: Original dataframe with columns ['u', 'i', 'ts']
        sparse_df: Sparse dataframe with columns ['u', 'i', 'ts']
        batch_size: Number of batches to split the missing edges into
        start_time: Starting timestamp for filtering
        end_time: Ending timestamp for filtering
    
    Returns:
        List of DataFrames, each containing a batch of missing edges
    """
    # Get filtered dataframes
    filtered_original, filtered_sparse = get_filtered_dataframes(
        original_df, 
        sparse_df,
        start_time,
        end_time
    )
    
    # Convert to sets for comparison
    original_edges = set(filtered_original[['u', 'i', 'ts']].apply(tuple, axis=1))
    sparse_edges = set(filtered_sparse[['u', 'i', 'ts']].apply(tuple, axis=1))
    
    # Find missing edges
    missing_edges = original_edges - sparse_edges
    
    # Convert to DataFrame
    missing_df = pd.DataFrame(list(missing_edges), columns=['u', 'i', 'ts'])
    missing_df['ts'] = missing_df['ts'].astype(int)

    
    # Merge with original data to get all columns
    complete_missing_df = missing_df.merge(
        filtered_original, 
        on=['u', 'i', 'ts'], 
        how='left'
    )
    complete_missing_df['ts'] = complete_missing_df['ts'].astype(int)
    
    
    # Calculate batch sizes
    total_edges = len(complete_missing_df)
    batch_cut = total_edges // batch_size
    
    # Create batches
    batches = []
    for i in range(batch_size):
        start_idx = i * batch_cut
        end_idx = start_idx + batch_cut if i < batch_size - 1 else total_edges
        batch = complete_missing_df.iloc[start_idx:end_idx].copy()
        batches.append(batch)
    
    return batches


def get_nodes_in_time_window(df: pd.DataFrame, 
                           timestamp: float, 
                           window_size: int = 1200) -> Set[int]:
    """
    Get all nodes that appear within the time window around given timestamp.
    """
    mask = (df['ts'] >= timestamp - window_size) & (df['ts'] <= timestamp)
    window_df = df[mask]
    return set(window_df['u'].unique()) | set(window_df['i'].unique())

def get_nodes_in_time_window_slow(df: pd.DataFrame, 
                            timestamp: float, 
                            num_interactions: int = 1200) -> Set[int]:
    """
    Get all nodes that appear within the last 'num_interactions' interactions up to the given timestamp.
    """
    # Filter for interactions up to the given timestamp
    filtered_df = df[df['ts'] <= timestamp]
    
    # Sort by timestamp in descending order
    sorted_df = filtered_df.sort_values(by='ts', ascending=False)
    
    # Select the last 'num_interactions' rows
    last_interactions_df = sorted_df.head(num_interactions)
    
    # Extract unique nodes
    return set(last_interactions_df['u'].unique()) | set(last_interactions_df['i'].unique())

def not_used_get_nodes_in_time_window(df: pd.DataFrame, 
                                        timestamp: float, 
                                        num_interactions: int = 1200) -> Set[int]:
    """
    Efficiently get all nodes that appear within the last 'num_interactions' interactions up to the given timestamp.
    """
    # Filter and select the last 'num_interactions' rows with timestamp <= given timestamp
    last_interactions_df = df[df['ts'] <= timestamp].nlargest(num_interactions, 'ts')
    
    # Extract unique nodes directly using vectorized operations
    return set(pd.concat([last_interactions_df['u'], last_interactions_df['i']]).unique())




def _get_nodes_in_time_window(df_array, timestamp, window_size):
    # Assuming 'ts' is the 3rd column (index 2)
    mask = (df_array[:, 2] >= timestamp - window_size) & (df_array[:, 2] <= timestamp)
    return df_array[mask][:, 0]  # Assuming 'u' or 'i' is the first column (index 0)



def old_build_temporal_mappings(
    missing_edges_df: pd.DataFrame, 
    sampled_timestamps: List[float],
    hh_edges: List[Tuple[int, int]], 
    time_window: int = 1200,
    node_expansion_factor: float = 1.5  # New parameter
) -> Dict[Tuple[int, int], List[float]]:
    """
    Build mappings between Havel-Hakimi edges and their possible timestamps.
    
    Args:
        missing_edges_df: DataFrame with missing edges
        sampled_timestamps: List of timestamps to consider
        hh_edges: List of Havel-Hakimi edges
        time_window: Time window size
        node_expansion_factor: Factor to expand the time window to include more nodes
    """
    df_sorted = missing_edges_df.sort_values('ts')
    hh_edges_set = set(hh_edges)
    
    min_src, max_src = df_sorted['u'].min(), df_sorted['u'].max()
    min_dst, max_dst = df_sorted['i'].min(), df_sorted['i'].max()
    
    edges_to_timestamps = defaultdict(list)
    
    # Calculate expanded time window
    expanded_window = int(time_window * node_expansion_factor)
    
    # Track node availability
    src_node_counts = defaultdict(int)
    dst_node_counts = defaultdict(int)
    
    for timestamp in tqdm(sampled_timestamps, desc='Processing timestamps', leave=False):
        # Get nodes with expanded window
        nodes = get_nodes_in_time_window(df_sorted, int(timestamp), expanded_window)
        
        # Split nodes into source and destination
        src_nodes_orig = [n for n in nodes if min_src <= n <= max_src]
        dst_nodes_orig = [n for n in nodes if min_dst <= n <= max_dst]
        
        # Sort nodes by availability (prefer less used nodes)
        src_nodes = sorted(src_nodes_orig, key=lambda x: src_node_counts[x])
        dst_nodes = sorted(dst_nodes_orig, key=lambda x: dst_node_counts[x])
        
        # Process HH edges first
        # for edge in hh_edges:
        #     src, dst = edge
        #     if src in nodes and dst in nodes:
        #         edges_to_timestamps[edge].append(timestamp)
        #         src_node_counts[src] += 1
        #         dst_node_counts[dst] += 1
        
        # Generate additional edges if needed
        possible_edges = product(src_nodes, dst_nodes)
        # for src, dst in possible_edges:
        #     if (src, dst) not in hh_edges_set:
        #         # Check node usage to ensure balanced distribution
        #         if (src_node_counts[src] < len(sampled_timestamps) / 2 and 
        #             dst_node_counts[dst] < len(sampled_timestamps) / 2):
        #             edges_to_timestamps[(src, dst)].append(timestamp)
        #             src_node_counts[src] += 1
        #             dst_node_counts[dst] += 1
        
        for src, dst in possible_edges: 
            edges_to_timestamps[(src, dst)].append(timestamp)
            src_node_counts[src] += 1
            dst_node_counts[dst] += 1
    
    # Print statisti
    
    return edges_to_timestamps





class Slow_TPRTimestampSelector(EdgeTimestampSelector):
    def __init__(self, *args, beta=0.5, alpha=0.15, dataset_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.alpha = alpha
        self.dataset_name = dataset_name
        
        # Initialize TPR calculator and TER scores
        self._initialize_tpr_and_ter()
        
    def _compute_temporal_degrees(self, edges):
        """Compute temporal outgoing degrees for nodes"""
        temporal_outgoing_degree = defaultdict(int)
        ts_to_node_dict = defaultdict(list)
        
        # Sort edges by timestamp
        sorted_edges = sorted(edges, key=lambda x: x[2])
        
        # Calculate temporal degrees
        for u, v, t in sorted_edges:
            temporal_outgoing_degree[u] += 1
            ts_to_node_dict[t].append(u)
            
        return temporal_outgoing_degree, ts_to_node_dict
        
    def _compute_ter_scores(self, edges, pagerank_evolution, temporal_outgoing_degree, ts_to_node_dict):
        """Compute Temporal Edge Rank scores"""
        ter_dict = {}
        
        # Get unique timestamps
        timestamps = sorted(list(set(t for _, _, t in edges)))
        
        print("Computing TER scores...")
        for i, ts in enumerate(tqdm(timestamps, desc='Calculating TER')):
            # Get PageRank scores for current timestamp
            r = pagerank_evolution[i]['scores']
            
            # Get active nodes for current timestamp
            node_list = ts_to_node_dict[ts]
            if not node_list:
                continue
                
            # Compute TER for current timestamp
            node_out_deg = np.array([temporal_outgoing_degree[node] for node in node_list])
            # Add small epsilon to avoid division by zero
            node_out_deg = np.maximum(node_out_deg, 1e-10)
            ter = r[node_list] / node_out_deg
            
            # Store sum of TER scores for timestamp
            ter_dict[ts] = np.sum(ter)
            
        return ter_dict
        
    def _initialize_tpr_and_ter(self):
        """Initialize both TPR and TER calculations"""
        try:
            # Convert original dataframe to edge format
            edges = []
            for _, row in self.original_df.iterrows():
                edges.append((int(row['u']), int(row['i']), float(row['ts'])))
            
            # Sort edges by timestamp
            edges.sort(key=lambda x: x[2])
            
            # Compute TPR scores
            print("Computing TPR scores...")
            self.tpr_scores, self.tpr_evolution = self._compute_tpr_scores(edges)
            self.tpr_timestamps = self.tpr_evolution['timestamp']
            
            # Compute temporal degrees and node mappings
            self.temporal_outgoing_degree, self.ts_to_node_dict = self._compute_temporal_degrees(edges)
            
            # Compute TER scores
            self.ter_scores = self._compute_ter_scores(
                edges, 
                self.tpr_evolution,
                self.temporal_outgoing_degree,
                self.ts_to_node_dict
            )
            
        except Exception as e:
            print(f"Error in TPR/TER initialization: {str(e)}")
            self.tpr_scores = None
            self.tpr_evolution = None
            self.tpr_timestamps = None
            self.ter_scores = None

    def _calculate_ter_impact_score(self, edge: Tuple[int, int], timestamp: float) -> float:
        """Calculate impact score based on TER"""
        if self.ter_scores is None:
            return 1.0  # Fallback score
            
        # Find closest timestamp in TER scores
        closest_ts = min(self.ter_scores.keys(), 
                        key=lambda x: abs(x - timestamp))
        
        # Get TER score for timestamp
        ter_score = self.ter_scores.get(closest_ts, 1.0)
        
        # Calculate impact score (inverse of TER)
        # Higher TER means more important/influential edges
        # For attack, we want edges with low TER
        eps = 1e-10
        impact_score = 1.0 / (ter_score + eps)
        
        return impact_score

    def _pick_best_timestamp(self, timestamps: List[int], edge: Tuple[int, int]) -> int:
        """Select timestamp that maximizes attack potential using TER"""
        try:
            valid_timestamps = self._filter_valid_timestamps(timestamps)
            if not valid_timestamps:
                return max(timestamps)
            
            timestamp_scores = []
            for ts in valid_timestamps:
                if self.ter_scores is not None:
                    # Calculate inverse TER-based score
                    ter_score = self._calculate_ter_impact_score(edge, float(ts))
                    # Higher score = better for attack
                    timestamp_scores.append((ter_score, ts))
                else:
                    # Fallback to using latest timestamp if TER failed
                    timestamp_scores.append((1.0, ts))
            
            # Choose timestamp with highest score (lowest TER)
            return max(timestamp_scores, key=lambda x: x[0])[1]
            
        except Exception as e:
            print(f"Error in timestamp selection: {str(e)}")
            return max(timestamps)  # Fallback to latest timestamp

    def _filter_valid_timestamps(self, timestamps: List[int]) -> List[int]:
        """Filter timestamps within valid time window"""
        if not timestamps:
            return []
            
        window_end = float(max(timestamps))
        window_start = window_end - float(self.time_window * 1000)
        
        valid_ts = [ts for ts in timestamps 
                   if window_start <= float(ts) <= window_end]
        return valid_ts
    
    def _compute_tpr_scores(self, edges):
        """Compute TPR scores directly without mmap"""
        # Get max node index
        max_node = max(max(e[0] for e in edges), max(e[1] for e in edges))
        
        # Initialize arrays
        r = np.zeros(max_node + 1)
        s = np.zeros(max_node + 1)
        evolution = []
        
        # Process edges in temporal order
        for u, v, t in edges:
            # Update TPR scores
            delta = 1 - self.alpha
            r[u] += delta
            s[u] += delta
            r[v] += s[u] * self.alpha
            
            if self.beta < 1:
                s_v_increment = s[u] * (1 - self.beta) * self.alpha
                s[v] += s_v_increment
                s[u] *= self.beta
            else:
                s[v] += s[u] * self.alpha
                s[u] = 0
            
            # Store evolution as numpy array
            total_r = r.sum()
            if total_r > 0:
                normalized_r = r.copy() / total_r
                evolution.append((float(t), normalized_r))
        
        # Convert evolution to structured array
        dtype = np.dtype([
            ('timestamp', np.float64),
            ('scores', np.float64, (max_node + 1,))
        ])
        
        structured_evolution = np.zeros(len(evolution), dtype=dtype)
        for i, (t, scores) in enumerate(evolution):
            structured_evolution[i]['timestamp'] = t
            structured_evolution[i]['scores'] = scores
            
        return r, structured_evolution


from typing import List, Tuple, Dict, Set, Optional
import numpy as np
import time
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import numba
from concurrent.futures import ThreadPoolExecutor
from bisect import bisect_left
from copy import deepcopy

class _TPRTimestampSelector(EdgeTimestampSelector):
    """
    Optimized implementation of TPR Timestamp Selector
    """
    def __init__(self,
                 feasible_edges_dict: Dict,
                 original_df: pd.DataFrame,
                 degree_dict: Dict[int, int],
                 removed_edges: Set[Tuple[int, int]],
                 src_nodes: Optional[Set[int]] = None,
                 dst_nodes: Optional[Set[int]] = None,
                 time_window: int = 1200,
                 beta: float = 0.5,
                 alpha: float = 0.15,
                 dataset_name: Optional[str] = None):
        """
        Initialize the Fast TPR Timestamp Selector
        """
        print("\n=== Initializing Fast TPR Timestamp Selector ===")
        start_time = time.time()
        # Call parent class initialization
        super().__init__(
            feasible_edges_dict=feasible_edges_dict,
            original_df=original_df,
            degree_dict=degree_dict,
            removed_edges=removed_edges,
            src_nodes=src_nodes,
            dst_nodes=dst_nodes,
            time_window=time_window
        )
        
        # Initialize TPR-specific parameters
        self.beta = beta
        self.alpha = alpha
        self.dataset_name = dataset_name
        self._cache = {}
        
        print(f"Base initialization completed in {time.time() - start_time:.2f} seconds")
        
        # Initialize TPR and TER calculations
        self._initialize_tpr_and_ter()
    
    @staticmethod
    @numba.jit(nopython=True)
    def _fast_temporal_degrees(edges_array):
        """Compute temporal degrees using numba for speed"""
        max_node = int(max(np.max(edges_array[:, 0]), np.max(edges_array[:, 1])))
        degrees = np.zeros(max_node + 1, dtype=np.int32)
        
        # Vectorized degree calculation
        for i in range(len(edges_array)):
            degrees[int(edges_array[i, 0])] += 1
        
        return degrees

    @staticmethod
    @numba.jit(nopython=True)
    def _fast_ter_calculation(pagerank_scores, node_degrees, node_list):
        """Optimized TER calculation using numba"""
        node_out_deg = np.zeros(len(node_list))
        for i in range(len(node_list)):
            node_out_deg[i] = node_degrees[node_list[i]]
        
        node_out_deg = np.maximum(node_out_deg, 1e-10)
        pr_scores = np.zeros(len(node_list))
        for i in range(len(node_list)):
            pr_scores[i] = pagerank_scores[node_list[i]]
            
        return np.sum(pr_scores / node_out_deg)

    def _compute_temporal_degrees(self, edges):
        """Vectorized temporal degree computation"""
        edges_array = np.array(edges)
        temporal_degrees = self._fast_temporal_degrees(edges_array)
        
        # Create timestamp to node mapping using numpy operations
        timestamps = edges_array[:, 2]
        unique_ts = np.unique(timestamps)
        
        # Create dictionary mapping timestamps to nodes
        ts_to_node_dict = defaultdict(list)
        for i in range(len(edges_array)):
            ts_to_node_dict[edges_array[i, 2]].append(int(edges_array[i, 0]))
            
        return temporal_degrees, ts_to_node_dict

    def ___compute_ter_scores(self, edges, pagerank_evolution, temporal_degrees, ts_to_node_dict):
        """Parallel TER score computation"""
        print("Computing TER scores...", end='\r')
        ter_dict = {}
        timestamps = np.array(sorted(set(t for _, _, t in edges)))
        
        def process_timestamp_chunk(chunk):
            chunk_results = {}
            for ts in chunk:
                pr_idx = bisect_left([pe['timestamp'] for pe in pagerank_evolution], ts)
                if pr_idx >= len(pagerank_evolution):
                    pr_idx = len(pagerank_evolution) - 1
                    
                r = pagerank_evolution[pr_idx]['scores']
                node_list = np.array(ts_to_node_dict[ts], dtype=np.int32)
                
                if len(node_list) > 0:
                    ter_score = self._fast_ter_calculation(r, temporal_degrees, node_list)
                    chunk_results[ts] = ter_score
            return chunk_results
        
        # Parallel processing of timestamps
        chunk_size = min(1000, len(timestamps))
        timestamp_chunks = np.array_split(timestamps, max(1, len(timestamps) // chunk_size))
        
        with ThreadPoolExecutor() as executor:
            chunk_results = list(executor.map(process_timestamp_chunk, timestamp_chunks))
            
        # Merge results
        for chunk_result in chunk_results:
            ter_dict.update(chunk_result)
            
        print("TER scores computed successfully!     ")
        return ter_dict

    def _compute_ter_scores(self, edges, pagerank_evolution, temporal_degrees, ts_to_node_dict):
        """
        Simple TER score computation with basic parallelization
        """
        print("Computing TER scores...")
        ter_dict = {}
        
        # Get unique timestamps from edges
        timestamps = sorted(set(t for _, _, t in edges))
        total_timestamps = len(timestamps)
        
        # Cache pagerank timestamps for faster lookup
        pr_timestamps = [pe['timestamp'] for pe in pagerank_evolution]
        
        def process_timestamp(ts):
            """Process a single timestamp"""
            try:
                # Find closest pagerank index
                pr_idx = bisect_left(pr_timestamps, ts)
                if pr_idx >= len(pagerank_evolution):
                    pr_idx = len(pagerank_evolution) - 1
                    
                # Get pagerank scores
                r = pagerank_evolution[pr_idx]['scores']
                
                # Get nodes active at this timestamp
                nodes = ts_to_node_dict[ts]
                
                if not nodes:
                    return ts, 0.0
                    
                # Get degrees and scores for active nodes
                degrees = np.array([temporal_degrees[node] for node in nodes])
                degrees = np.maximum(degrees, 1e-10)  # Avoid division by zero
                scores = r[nodes]
                
                # Calculate TER score
                ter_score = np.sum(scores / degrees)
                
                return ts, ter_score
                
            except Exception as e:
                print(f"Error processing timestamp {ts}: {str(e)}")
                return ts, 0.0
        
        # Process all timestamps in parallel
        with ThreadPoolExecutor() as executor:
            # Use tqdm to show progress
            results = list(tqdm(
                executor.map(process_timestamp, timestamps),
                total=total_timestamps,
                desc="Processing timestamps"
            ))
        
        # Update ter_dict with results
        for ts, score in results:
            if score > 0:
                ter_dict[ts] = score
        
        print(f"TER scores computed for {len(ter_dict)} timestamps")
        return ter_dict
    
    @staticmethod
    @numba.jit(nopython=True)
    def _fast_tpr_computation(edges_array, alpha, beta, max_node):
        """Optimized TPR computation using numba"""
        r = np.zeros(max_node + 1)
        s = np.zeros(max_node + 1)
        evolution = np.zeros((len(edges_array), max_node + 2))  # +1 for timestamp, +1 for max_node
        evolution_idx = 0
        
        for i in range(len(edges_array)):
            u, v, t = int(edges_array[i, 0]), int(edges_array[i, 1]), edges_array[i, 2]
            
            delta = 1.0 - alpha
            r[u] += delta
            s[u] += delta
            r[v] += s[u] * alpha
            
            if beta < 1.0:
                s_v_increment = s[u] * (1.0 - beta) * alpha
                s[v] += s_v_increment
                s[u] *= beta
            else:
                s[v] += s[u] * alpha
                s[u] = 0.0
                
            total_r = np.sum(r)
            if total_r > 0:
                normalized_r = r / total_r
                evolution[evolution_idx, 0] = t
                evolution[evolution_idx, 1:] = normalized_r
                evolution_idx += 1
                
        return r, evolution[:evolution_idx]

    def _initialize_tpr_and_ter(self):
        """Vectorized initialization of TPR and TER"""
        try:
            # Convert dataframe to numpy array for faster processing
            edges_array = np.array([
                (int(row['u']), int(row['i']), float(row['ts']))
                for _, row in self.original_df.iterrows()
            ])
            
            # Sort edges by timestamp using numpy
            sort_idx = np.argsort(edges_array[:, 2])
            edges_array = edges_array[sort_idx]
            
            # Cache max node for future use
            self._cache['max_node'] = int(max(
                np.max(edges_array[:, 0]),
                np.max(edges_array[:, 1])
            ))
            
            # Compute TPR scores
            print("Computing TPR scores...", end='\r')
            self.tpr_scores, evolution_array = self._fast_tpr_computation(
                edges_array, self.alpha, self.beta, self._cache['max_node']
            )
            
            # Convert evolution to structured array efficiently
            dtype = np.dtype([
                ('timestamp', np.float64),
                ('scores', np.float64, (self._cache['max_node'] + 1,))
            ])
            self.tpr_evolution = np.zeros(len(evolution_array), dtype=dtype)
            
            for i in range(len(evolution_array)):
                self.tpr_evolution[i]['timestamp'] = evolution_array[i, 0]
                self.tpr_evolution[i]['scores'] = evolution_array[i, 1:]
            
            self.tpr_timestamps = self.tpr_evolution['timestamp']
            print("TPR scores computed successfully!     ")
            
            # Compute temporal degrees and TER scores
            print("Computing temporal degrees and TER scores...")
            self.temporal_degrees, self.ts_to_node_dict = self._compute_temporal_degrees(edges_array)
            self.ter_scores = self._compute_ter_scores(
                edges_array,
                self.tpr_evolution,
                self.temporal_degrees,
                self.ts_to_node_dict
            )
            
            print("Temporal degrees and TER scores computed successfully!     ")
            
        except Exception as e:
            print(f"Error in TPR/TER initialization: {str(e)}")
            self.tpr_scores = None
            self.tpr_evolution = None
            self.tpr_timestamps = None
            self.ter_scores = None

    def _calculate_ter_impact_score(self, edge: Tuple[int, int], timestamp: float) -> float:
        """Optimized TER impact score calculation"""
        if self.ter_scores is None:
            return 1.0
        
        # Use numpy for faster closest timestamp search
        timestamps = np.array(list(self.ter_scores.keys()))
        idx = np.searchsorted(timestamps, timestamp)
        if idx == len(timestamps):
            idx -= 1
        elif idx > 0:
            if abs(timestamps[idx] - timestamp) > abs(timestamps[idx-1] - timestamp):
                idx -= 1
                
        ter_score = self.ter_scores.get(timestamps[idx], 1.0)
        return 1.0 / (ter_score + 1e-10)

    def _pick_best_timestamp(self, timestamps: List[int], edge: Tuple[int, int]) -> int:
        """Optimized timestamp selection - overrides parent class method"""
        try:
            valid_timestamps = self._filter_valid_timestamps(timestamps)
            if not valid_timestamps:
                return max(timestamps)
            
            # Vectorized timestamp scoring
            valid_ts_array = np.array(valid_timestamps, dtype=np.float64)
            scores = np.array([
                self._calculate_ter_impact_score(edge, float(ts))
                for ts in valid_ts_array
            ])
            
            return valid_timestamps[np.argmax(scores)]
            
        except Exception as e:
            print(f"Error in timestamp selection: {str(e)}")
            return max(timestamps)

    def _filter_valid_timestamps(self, timestamps: List[int]) -> List[int]:
        """Vectorized timestamp filtering"""
        if not timestamps:
            return []
        
        timestamps_array = np.array(timestamps, dtype=np.float64)
        window_end = np.max(timestamps_array)
        window_start = window_end - float(self.time_window * 1000)
        
        mask = (timestamps_array >= window_start) & (timestamps_array <= window_end)
        return timestamps_array[mask].tolist()



class TPRTimestampSelector(EdgeTimestampSelector):
    """
    Optimized implementation of TPR Timestamp Selector
    """
    def __init__(self,
                 feasible_edges_dict: Dict,
                 original_df: pd.DataFrame,
                 degree_dict: Dict[int, int],
                 removed_edges: Set[Tuple[int, int]],
                 src_nodes: Optional[Set[int]] = None,
                 dst_nodes: Optional[Set[int]] = None,
                 time_window: int = 1200,
                 beta: float = 0.5,
                 alpha: float = 0.15,
                 dataset_name: Optional[str] = None):
        """
        Initialize the Fast TPR Timestamp Selector
        """
        print("\n=== Initializing Fast TPR Timestamp Selector ===")
        start_time = time.time()
        # Call parent class initialization
        super().__init__(
            feasible_edges_dict=feasible_edges_dict,
            original_df=original_df,
            degree_dict=degree_dict,
            removed_edges=removed_edges,
            src_nodes=src_nodes,
            dst_nodes=dst_nodes,
            time_window=time_window
        )
        
        # Initialize TPR-specific parameters
        self.beta = beta
        self.alpha = alpha
        self.dataset_name = dataset_name
        self._cache = {}
        
        print(f"Base initialization completed in {time.time() - start_time:.2f} seconds")
        
        # Initialize TPR and TER calculations
        self._initialize_tpr_and_ter()
        self._cached_ter_timestamps = None
        self._cached_ter_scores_array = None
        self._setup_ter_cache()
    
    @staticmethod
    @numba.jit(nopython=True)
    def _fast_temporal_degrees(edges_array):
        """Compute temporal degrees using numba for speed"""
        max_node = int(max(np.max(edges_array[:, 0]), np.max(edges_array[:, 1])))
        degrees = np.zeros(max_node + 1, dtype=np.int32)
        
        # Vectorized degree calculation
        for i in range(len(edges_array)):
            degrees[int(edges_array[i, 0])] += 1
        
        return degrees

    @staticmethod
    @numba.jit(nopython=True)
    def _fast_ter_calculation(pagerank_scores, node_degrees, node_list):
        """Optimized TER calculation using numba"""
        node_out_deg = np.zeros(len(node_list))
        for i in range(len(node_list)):
            node_out_deg[i] = node_degrees[node_list[i]]
        
        node_out_deg = np.maximum(node_out_deg, 1e-10)
        pr_scores = np.zeros(len(node_list))
        for i in range(len(node_list)):
            pr_scores[i] = pagerank_scores[node_list[i]]
            
        return np.sum(pr_scores / node_out_deg)

    def _compute_temporal_degrees(self, edges):
        """Vectorized temporal degree computation"""
        edges_array = np.array(edges)
        temporal_degrees = self._fast_temporal_degrees(edges_array)
        
        # Create timestamp to node mapping using numpy operations
        timestamps = edges_array[:, 2]
        unique_ts = np.unique(timestamps)
        
        # Create dictionary mapping timestamps to nodes
        ts_to_node_dict = defaultdict(list)
        for i in range(len(edges_array)):
            ts_to_node_dict[edges_array[i, 2]].append(int(edges_array[i, 0]))
            
        return temporal_degrees, ts_to_node_dict

    def ___compute_ter_scores(self, edges, pagerank_evolution, temporal_degrees, ts_to_node_dict):
        """Parallel TER score computation"""
        print("Computing TER scores...", end='\r')
        ter_dict = {}
        timestamps = np.array(sorted(set(t for _, _, t in edges)))
        
        def process_timestamp_chunk(chunk):
            chunk_results = {}
            for ts in chunk:
                pr_idx = bisect_left([pe['timestamp'] for pe in pagerank_evolution], ts)
                if pr_idx >= len(pagerank_evolution):
                    pr_idx = len(pagerank_evolution) - 1
                    
                r = pagerank_evolution[pr_idx]['scores']
                node_list = np.array(ts_to_node_dict[ts], dtype=np.int32)
                
                if len(node_list) > 0:
                    ter_score = self._fast_ter_calculation(r, temporal_degrees, node_list)
                    chunk_results[ts] = ter_score
            return chunk_results
        
        # Parallel processing of timestamps
        chunk_size = min(1000, len(timestamps))
        timestamp_chunks = np.array_split(timestamps, max(1, len(timestamps) // chunk_size))
        
        with ThreadPoolExecutor() as executor:
            chunk_results = list(executor.map(process_timestamp_chunk, timestamp_chunks))
            
        # Merge results
        for chunk_result in chunk_results:
            ter_dict.update(chunk_result)
            
        print("TER scores computed successfully!     ")
        return ter_dict

    def _compute_ter_scores(self, edges, pagerank_evolution, temporal_degrees, ts_to_node_dict):
        """
        Simple TER score computation with basic parallelization
        """
        print("Computing TER scores...")
        ter_dict = {}
        
        # Get unique timestamps from edges
        timestamps = sorted(set(t for _, _, t in edges))
        total_timestamps = len(timestamps)
        
        # Cache pagerank timestamps for faster lookup
        pr_timestamps = [pe['timestamp'] for pe in pagerank_evolution]
        
        def process_timestamp(ts):
            """Process a single timestamp"""
            try:
                # Find closest pagerank index
                pr_idx = bisect_left(pr_timestamps, ts)
                if pr_idx >= len(pagerank_evolution):
                    pr_idx = len(pagerank_evolution) - 1
                    
                # Get pagerank scores
                r = pagerank_evolution[pr_idx]['scores']
                
                # Get nodes active at this timestamp
                nodes = ts_to_node_dict[ts]
                
                if not nodes:
                    return ts, 0.0
                    
                # Get degrees and scores for active nodes
                degrees = np.array([temporal_degrees[node] for node in nodes])
                degrees = np.maximum(degrees, 1e-10)  # Avoid division by zero
                scores = r[nodes]
                
                # Calculate TER score
                ter_score = np.sum(scores / degrees)
                
                return ts, ter_score
                
            except Exception as e:
                print(f"Error processing timestamp {ts}: {str(e)}")
                return ts, 0.0
        
        # Process all timestamps in parallel
        with ThreadPoolExecutor() as executor:
            # Use tqdm to show progress
            results = list(tqdm(
                executor.map(process_timestamp, timestamps),
                total=total_timestamps,
                desc="Processing timestamps"
            ))
        
        # Update ter_dict with results
        for ts, score in results:
            if score > 0:
                ter_dict[ts] = score
        
        print(f"TER scores computed for {len(ter_dict)} timestamps")
        return ter_dict
    
    @staticmethod
    @numba.jit(nopython=True)
    def _fast_tpr_computation(edges_array, alpha, beta, max_node):
        """Optimized TPR computation using numba"""
        r = np.zeros(max_node + 1)
        s = np.zeros(max_node + 1)
        evolution = np.zeros((len(edges_array), max_node + 2))  # +1 for timestamp, +1 for max_node
        evolution_idx = 0
        
        for i in range(len(edges_array)):
            u, v, t = int(edges_array[i, 0]), int(edges_array[i, 1]), edges_array[i, 2]
            
            delta = 1.0 - alpha
            r[u] += delta
            s[u] += delta
            r[v] += s[u] * alpha
            
            if beta < 1.0:
                s_v_increment = s[u] * (1.0 - beta) * alpha
                s[v] += s_v_increment
                s[u] *= beta
            else:
                s[v] += s[u] * alpha
                s[u] = 0.0
                
            total_r = np.sum(r)
            if total_r > 0:
                normalized_r = r / total_r
                evolution[evolution_idx, 0] = t
                evolution[evolution_idx, 1:] = normalized_r
                evolution_idx += 1
                
        return r, evolution[:evolution_idx]

    def _initialize_tpr_and_ter(self):
        """Vectorized initialization of TPR and TER"""
        try:
            # Convert dataframe to numpy array for faster processing
            edges_array = np.array([
                (int(row['u']), int(row['i']), float(row['ts']))
                for _, row in self.original_df.iterrows()
            ])
            
            # Sort edges by timestamp using numpy
            sort_idx = np.argsort(edges_array[:, 2])
            edges_array = edges_array[sort_idx]
            
            # Cache max node for future use
            self._cache['max_node'] = int(max(
                np.max(edges_array[:, 0]),
                np.max(edges_array[:, 1])
            ))
            
            # Compute TPR scores
            print("Computing TPR scores...", end='\r')
            self.tpr_scores, evolution_array = self._fast_tpr_computation(
                edges_array, self.alpha, self.beta, self._cache['max_node']
            )
            
            # Convert evolution to structured array efficiently
            dtype = np.dtype([
                ('timestamp', np.float64),
                ('scores', np.float64, (self._cache['max_node'] + 1,))
            ])
            self.tpr_evolution = np.zeros(len(evolution_array), dtype=dtype)
            
            for i in range(len(evolution_array)):
                self.tpr_evolution[i]['timestamp'] = evolution_array[i, 0]
                self.tpr_evolution[i]['scores'] = evolution_array[i, 1:]
            
            self.tpr_timestamps = self.tpr_evolution['timestamp']
            print("TPR scores computed successfully!     ")
            
            # Compute temporal degrees and TER scores
            print("Computing temporal degrees and TER scores...")
            self.temporal_degrees, self.ts_to_node_dict = self._compute_temporal_degrees(edges_array)
            self.ter_scores = self._compute_ter_scores(
                edges_array,
                self.tpr_evolution,
                self.temporal_degrees,
                self.ts_to_node_dict
            )
            
            print("Temporal degrees and TER scores computed successfully!     ")
            
        except Exception as e:
            print(f"Error in TPR/TER initialization: {str(e)}")
            self.tpr_scores = None
            self.tpr_evolution = None
            self.tpr_timestamps = None
            self.ter_scores = None

    def __calculate_ter_impact_score(self, edge: Tuple[int, int], timestamp: float) -> float:
        """Optimized TER impact score calculation"""
        if self.ter_scores is None:
            return 1.0
        
        # Use numpy for faster closest timestamp search
        timestamps = np.array(list(self.ter_scores.keys()))
        idx = np.searchsorted(timestamps, timestamp)
        if idx == len(timestamps):
            idx -= 1
        elif idx > 0:
            if abs(timestamps[idx] - timestamp) > abs(timestamps[idx-1] - timestamp):
                idx -= 1
                
        ter_score = self.ter_scores.get(timestamps[idx], 1.0)
        return 1.0 / (ter_score + 1e-10)

    def __pick_best_timestamp(self, timestamps: List[int], edge: Tuple[int, int]) -> int:
        """Optimized timestamp selection - overrides parent class method"""
        try:
            valid_timestamps = self._filter_valid_timestamps(timestamps)
            if not valid_timestamps:
                return max(timestamps)
            
            # Vectorized timestamp scoring
            valid_ts_array = np.array(valid_timestamps, dtype=np.float64)
            scores = np.array([
                self._calculate_ter_impact_score(edge, float(ts))
                for ts in valid_ts_array
            ])
            
            return valid_timestamps[np.argmax(scores)]
            
        except Exception as e:
            print(f"Error in timestamp selection: {str(e)}")
            return max(timestamps)

    def _filter_valid_timestamps(self, timestamps: List[int]) -> List[int]:
        """Vectorized timestamp filtering"""
        if not timestamps:
            return []
        
        timestamps_array = np.array(timestamps, dtype=np.float64)
        window_end = np.max(timestamps_array)
        window_start = window_end - float(self.time_window * 1000)
        
        mask = (timestamps_array >= window_start) & (timestamps_array <= window_end)
        return timestamps_array[mask].tolist()
    
    def _setup_ter_cache(self):
        """Setup cached arrays for faster TER calculations"""
        if self.ter_scores is not None:
            # Sort timestamps once
            self._cached_ter_timestamps = np.array(sorted(self.ter_scores.keys()))
            # Create corresponding scores array
            self._cached_ter_scores_array = np.array([
                self.ter_scores[ts] for ts in self._cached_ter_timestamps
            ])

    def _calculate_ter_impact_score(self, edge: Tuple[int, int], timestamp: float) -> float:
        """Vectorized TER impact score calculation"""
        if self.ter_scores is None:
            return 1.0
        
        if self._cached_ter_timestamps is None:
            self._setup_ter_cache()
        
        idx = np.searchsorted(self._cached_ter_timestamps, timestamp)
        if idx == len(self._cached_ter_timestamps):
            idx -= 1
        elif idx > 0:
            if abs(self._cached_ter_timestamps[idx] - timestamp) > abs(self._cached_ter_timestamps[idx-1] - timestamp):
                idx -= 1
                
        ter_score = self._cached_ter_scores_array[idx]
        return 1.0 / (ter_score + 1e-10)

    def _pick_best_timestamp(self, timestamps: List[int], edge: Tuple[int, int]) -> int:
        """Optimized timestamp selection with vectorized operations"""
        try:
            valid_timestamps = self._filter_valid_timestamps(timestamps)
            if not valid_timestamps:
                return max(timestamps)
            
            # Convert to numpy array once
            valid_ts_array = np.array(valid_timestamps, dtype=np.float64)
            
            # Vectorized search for all timestamps at once
            indices = np.searchsorted(self._cached_ter_timestamps, valid_ts_array)
            
            # Handle edge cases
            indices = np.clip(indices, 0, len(self._cached_ter_timestamps) - 1)
            
            # Calculate distances to next and previous timestamps
            curr_dists = np.abs(self._cached_ter_timestamps[indices] - valid_ts_array)
            prev_indices = np.maximum(indices - 1, 0)
            prev_dists = np.abs(self._cached_ter_timestamps[prev_indices] - valid_ts_array)
            
            # Use previous index where it's closer
            use_prev = prev_dists < curr_dists
            indices[use_prev] = prev_indices[use_prev]
            
            # Get TER scores directly from cached array
            ter_scores = self._cached_ter_scores_array[indices]
            scores = 1.0 / (ter_scores + 1e-10)
            
            return valid_timestamps[np.argmax(scores)]
            
        except Exception as e:
            print(f"Error in timestamp selection: {str(e)}")
            return max(timestamps)
    def select_edges_timestamps(self) -> List[Tuple[int, int, int]]:
        """
        Optimized edge selection algorithm with proper type handling
        """
        print("\n\nStarting optimized edge selection...")
        start_time = time.time()

        # Convert to numpy arrays for faster operations
        feasible_edges = np.array(list(self.feasible_edges_dict.keys()))
        if len(feasible_edges) == 0:
            return []

        # Pre-compute degrees and sort nodes by degree
        degrees = np.array([(node, self.degree_dict[node]) 
                        for node in self.degree_dict.keys()])
        priority_nodes = degrees[np.argsort(-degrees[:, 1])][:, 0].astype(np.int32)

        # Initialize tracking arrays for faster lookups
        max_node_id = int(max(self.degree_dict.keys())) + 1
        remaining_capacity = np.zeros(max_node_id, dtype=np.int32)
        node_usage = np.zeros(max_node_id, dtype=np.int32)
        for node, degree in self.degree_dict.items():
            remaining_capacity[int(node)] = int(degree)
        

        selected_pairs = []
        selected_edges_set = set()
        
        # Create efficient lookup structures
        node_to_edges = defaultdict(list)
        for edge in feasible_edges:
            u, v = edge
            node_to_edges[int(u)].append(tuple(edge))
            node_to_edges[int(v)].append(tuple(edge))

        # Vectorized candidate scoring
        def score_candidates(candidates, timestamps_list):
            scores = []
            for edge, timestamps in zip(candidates, timestamps_list):
                u, v = map(int, edge)
                score = (remaining_capacity[u] + remaining_capacity[v] + 
                        len(timestamps) - (node_usage[u] + node_usage[v]) * 10)
                if remaining_capacity[u] > 0 and remaining_capacity[v] > 0:
                    score += 50
                scores.append(score)
            return np.array(scores)

        selected_count = 0
        progress_bar = tqdm(total=self.max_edges, desc=f"\tSelecting edges {selected_count}/{self.max_edges}", position=0, ncols=80, leave=False)

        # Main selection loop with optimizations
        for node in priority_nodes:
            if selected_count >= self.max_edges:
                break

            node = int(node)
            if remaining_capacity[node] <= 0:
                continue

            # Get all valid candidates for this node at once
            candidates = []
            timestamps_list = []
            for edge in node_to_edges[node]:
                if tuple(edge) in selected_edges_set:
                    continue
                    
                # u, v = map(int, edge)
                other_node = v if u == node else u
                
                if remaining_capacity[other_node] <= 0:
                    continue

                if self.is_bipartite:
                    if ((node in self.src_nodes and other_node in self.src_nodes) or 
                        (node in self.dst_nodes and other_node in self.dst_nodes)):
                        continue

                timestamps = [int(ts) for ts in self.feasible_edges_dict[edge]]
                if timestamps:
                    candidates.append(edge)
                    timestamps_list.append(timestamps)

            if not candidates:
                continue

            # Score all candidates at once
            scores = score_candidates(candidates, timestamps_list)
            
            # Sort candidates by score
            sorted_indices = np.argsort(-scores)
            
            # Try candidates in order of score
            for idx in sorted_indices:
                if selected_count >= self.max_edges:
                    break
                    
                if remaining_capacity[node] <= 0:
                    break
                    
                edge = candidates[idx]
                timestamps = timestamps_list[idx]
                
                if edge not in selected_edges_set:
                    u, v = map(int, edge)
                    if remaining_capacity[u] > 0 and remaining_capacity[v] > 0:
                        try:
                            # start = time.process_time()
                            best_timestamp = int(self._pick_best_timestamp(timestamps, edge))
                            # end = time.process_time()
                            # cpu_time = end - start
                            # tqdm.write(f"CPU time for _pick_best_timestamp: {cpu_time:.6f} seconds")
                            selected_pairs.append((u, v, best_timestamp))
                            selected_edges_set.add(tuple(edge))
                            
                            # Update tracking
                            remaining_capacity[u] -= 1
                            remaining_capacity[v] -= 1
                            node_usage[u] += 1
                            node_usage[v] += 1
                            
                            selected_count += 1
                            progress_bar.update(1)
                        except Exception as e:
                            print(f"Error processing edge {edge}: {str(e)}")
                            continue

        progress_bar.close()
        
        print(f'selected {selected_count}.')

        # Recovery phase if needed
        if selected_count < self.max_edges:
            print("\nStarting recovery phase...")
            remaining = self.max_edges - selected_count
            recovery_progress = tqdm(total=remaining, desc=f"Recovery phase", position=0, ncols=80, leave=False)
            
            try:
                additional_edges = self._fast_recovery_phase(remaining, self.original_df)
                selected_pairs.extend(additional_edges)
                recovery_progress.update(len(additional_edges))
            except Exception as e:
                print(f"Error in recovery phase: {str(e)}")
            finally:
                recovery_progress.close()

        end_time = time.time()
        print(f"\nEdge selection completed in {(end_time - start_time)/3600:.2f} hours")
        print(f"Selected {len(selected_pairs)} edges")
        
        return selected_pairs



def build_temporal_mappings(
    missing_edges_df: pd.DataFrame, 
    full_df: pd.DataFrame, 
    sampled_timestamps: List[float],
    time_window: int = 1200,
    node_expansion_factor: float = 1.5,  # New parameter,
    print_stats: bool = True
) -> Dict[Tuple[int, int], List[float]]:
    """
    Build mappings between Havel-Hakimi edges and their possible timestamps.
    
    Args:
        missing_edges_df: DataFrame with missing edges
        full_df: Dataframe with full graph edges to sample only negative edges
        sampled_timestamps: List of timestamps to consider
        hh_edges: List of Havel-Hakimi edges
        time_window: Time window size
        node_expansion_factor: Factor to expand the time window to include more nodes
    """
    df_sorted = missing_edges_df.sort_values('ts')
    full_edge_set = set((int(row['u']), int(row['i'])) for _, row in full_df.iterrows())    
    
    min_src, max_src = df_sorted['u'].min(), df_sorted['u'].max()
    min_dst, max_dst = df_sorted['i'].min(), df_sorted['i'].max()
    
    edges_to_timestamps = defaultdict(list)
    
    # Calculate expanded time window
    # expanded_window = int(time_window * node_expansion_factor)
    expanded_window = time_window
    
    # Track node availability
    src_node_counts = defaultdict(int)
    dst_node_counts = defaultdict(int)
    
    for timestamp in tqdm(sampled_timestamps, desc='Processing timestamps', leave=False):
        # Get nodes with expanded window
        nodes = get_nodes_in_time_window(df_sorted, int(timestamp), expanded_window)
        
        # Split nodes into source and destination
        src_nodes_orig = [n for n in nodes if min_src <= n <= max_src]
        dst_nodes_orig = [n for n in nodes if min_dst <= n <= max_dst]
        
        # Sort nodes by availability (prefer less used nodes)
        src_nodes = sorted(src_nodes_orig, key=lambda x: src_node_counts[x])
        dst_nodes = sorted(dst_nodes_orig, key=lambda x: dst_node_counts[x])
        
        # Generate negative edges
        possible_edges = [edge
            for edge in product(src_nodes, dst_nodes)
            if edge not in full_edge_set
            ]
        for src, dst in possible_edges: 
            edges_to_timestamps[(src, dst)].append(timestamp)
            src_node_counts[src] += 1
            dst_node_counts[dst] += 1
    
    # Print statistics
    if print_stats:
        print(f"Source nodes used: {len([n for n, c in src_node_counts.items() if c > 0])}")
        print(f"Destination nodes used: {len([n for n, c in dst_node_counts.items() if c > 0])}")    
    return edges_to_timestamps


class Old_working_NonBipartiteEdgeSelector:
    def __init__(self,
                 feasible_edges_dict: Dict,
                 original_df: pd.DataFrame,
                 degree_dict: Dict[int, int],
                 removed_edges: Set[Tuple[int, int]],
                 time_window: int = 1200):
        """Initialize Edge-Timestamp Selector for non-bipartite graphs"""
        self.feasible_edges_dict = feasible_edges_dict
        self.original_df = original_df
        self.degree_dict = degree_dict
        self.removed_edges = removed_edges
        self.time_window = time_window
        
        # Initialize tracking structures
        self.node_usage = defaultdict(int)
        self.selected_pairs = []
        self.remaining_capacity = deepcopy(degree_dict)
        
        # Calculate edge budget and create lookup structures
        self.max_edges = self._calculate_edge_budget()
        self.node_to_edges = self._create_node_edge_mapping()
        self.feasible_adj_lists = self._create_adj_lists()

    def _calculate_edge_budget(self) -> int:
        """Calculate maximum possible edges for non-bipartite graphs"""
        return sum(self.degree_dict.values()) // 2

    def _create_node_edge_mapping(self) -> Dict[int, List[Tuple[int, int]]]:
        """Create mapping from nodes to their edges"""
        node_to_edges = defaultdict(list)
        for edge in self.feasible_edges_dict:
            u, v = edge
            node_to_edges[u].append(edge)
            node_to_edges[v].append(edge)
        return node_to_edges

    def _create_adj_lists(self) -> Dict[int, Set[int]]:
        """Create adjacency lists for faster lookup"""
        adj_lists = defaultdict(set)
        for (u, v) in self.feasible_edges_dict:
            adj_lists[u].add(v)
            adj_lists[v].add(u)
        return adj_lists

    def select_edges_timestamps(self) -> List[Tuple[int, int, int]]:
        """Main edge selection algorithm for non-bipartite graphs"""
        # Sort nodes by degree for priority
        priority_nodes = sorted(
            self.degree_dict.keys(),
            key=lambda x: (self.degree_dict[x], -len(self.node_to_edges[x])),
            reverse=True
        )

        selected_count = 0
        pbar = tqdm(total=self.max_edges, desc="\tSelecting edges", leave=False, ncols=80)

        # Initial edge selection phase
        for node in priority_nodes:
            if selected_count >= self.max_edges:
                break

            if self.remaining_capacity[node] <= 0:
                continue

            while self.remaining_capacity[node] > 0 and selected_count < self.max_edges:
                candidates = self._get_candidates(node)
                if not candidates:
                    break

                scored_candidates = [
                    (self._calculate_score(edge, timestamps), edge, timestamps)
                    for edge, timestamps in candidates
                ]
                scored_candidates.sort(reverse=True)

                selected = False
                for _, edge, timestamps in scored_candidates:
                    if self._can_select(edge, timestamps):
                        best_timestamp = self._pick_best_timestamp(timestamps, edge)
                        self.selected_pairs.append((edge[0], edge[1], best_timestamp))
                        self._update_tracking(edge)
                        selected_count += 1
                        pbar.update(1)
                        selected = True
                        break

                if not selected:
                    break

        # Recovery phase if needed
        if selected_count < self.max_edges:
            print('\tStarting recovery phase...')
            remaining = self.max_edges - selected_count
            print(f"\tRemaining edges: {remaining}")
            additional_edges = self._fast_recovery_phase(remaining, self.original_df)
            print(f'\tRecovery phase complete. Added {len(additional_edges)} additional edges.')
            selected_count += len(additional_edges)
            pbar.update(len(additional_edges))

        pbar.close()
        self._validate_selections()
        return self.selected_pairs

    def _fast_recovery_phase(self, needed_edges: int, original_df: pd.DataFrame) -> List[Tuple[int, int, int]]:
        """Recovery phase with optimized window queries for non-bipartite graphs"""
        additional_edges = []

        # Pre-process data
        df_sorted = original_df.sort_values('ts').set_index('ts')
        node_appearances = defaultdict(list)
        for idx, row in df_sorted.iterrows():
            node_appearances[row['u']].append(idx)
            node_appearances[row['i']].append(idx)

        available_nodes = [n for n in self.degree_dict.keys() if self.remaining_capacity[n] > 0]
        used_pairs = {(e[0], e[1]) for e in self.selected_pairs}

        all_timestamps = sorted(list(set(
            ts for timestamps in self.feasible_edges_dict.values()
            for ts in timestamps
        )))

        window_cache = {}
        window_size = self.time_window * 1e3

        def get_nodes_in_window(timestamp: float) -> Set[int]:
            if timestamp in window_cache:
                return window_cache[timestamp]

            window_start = max(timestamp - window_size, df_sorted.index.min())
            nodes_in_window = set()

            for node in available_nodes:
                appearances = node_appearances[node]
                idx = bisect.bisect_left(appearances, window_start)
                if idx < len(appearances) and appearances[idx] <= timestamp:
                    nodes_in_window.add(node)

            window_cache[timestamp] = nodes_in_window
            return nodes_in_window

        # Process timestamps in batches
        batch_size = 1000
        for i in range(0, len(all_timestamps), batch_size):
            batch_timestamps = all_timestamps[i:i + batch_size]

            for ts in batch_timestamps:
                get_nodes_in_window(ts)

            for timestamp in batch_timestamps:
                if len(additional_edges) >= needed_edges:
                    break

                nodes_in_window = window_cache[timestamp]
                valid_nodes = [n for n in available_nodes if n in nodes_in_window]

                for node_u in valid_nodes:
                    if len(additional_edges) >= needed_edges:
                        break

                    for node_v in valid_nodes:
                        if node_u == node_v:
                            continue
                        edge = (node_u, node_v)
                        if edge not in used_pairs and self._can_select(edge, [timestamp]):
                            additional_edges.append((node_u, node_v, timestamp))
                            used_pairs.add(edge)
                            self._update_tracking(edge)

                            if len(additional_edges) >= needed_edges:
                                break

            window_cache.clear()

        self.selected_pairs.extend(additional_edges)
        return additional_edges

    def _get_candidates(self, node: int) -> List[Tuple[Tuple[int, int], List[int]]]:
        """Get valid candidate edges for a node"""
        return [(edge, self.feasible_edges_dict[edge])
                for edge in self.node_to_edges[node]
                if self.remaining_capacity[self._get_other_node(edge, node)] > 0
                and edge in self.feasible_edges_dict]

    def _get_other_node(self, edge: Tuple[int, int], node: int) -> int:
        """Get the other node in an edge"""
        u, v = edge
        return v if node == u else u

    def _calculate_score(self, edge: Tuple[int, int], timestamps: List[int]) -> float:
        """Calculate score for candidate edge"""
        u, v = edge
        score = (self.remaining_capacity[u] + self.remaining_capacity[v] +
                 len(timestamps) - (self.node_usage[u] + self.node_usage[v]) * 10)
        if self.remaining_capacity[u] > 0 and self.remaining_capacity[v] > 0:
            score += 50
        return score

    def _can_select(self, edge: Tuple[int, int], timestamps: List[int]) -> bool:
        """Check if edge can be selected"""
        u, v = edge
        if not timestamps or self.remaining_capacity[u] <= 0 or self.remaining_capacity[v] <= 0:
            return False
        return True

    def _pick_best_timestamp(self, timestamps: List[int], edge: Tuple[int, int]) -> int:
        """Select best timestamp from available options"""
        return max(timestamps)

    def _update_tracking(self, edge: Tuple[int, int]):
        """Update tracking after edge selection"""
        u, v = edge
        self.remaining_capacity[u] -= 1
        self.remaining_capacity[v] -= 1
        self.node_usage[u] += 1
        self.node_usage[v] += 1

    def _validate_selections(self):
        """Validate final selections"""
        node_counts = defaultdict(int)
        for u, v, _ in self.selected_pairs:
            node_counts[u] += 1
            node_counts[v] += 1

        print(f"\tNodes used: {len(node_counts)}")


# 99%C3, 100%C4

class NonBipartiteEdgeSelector:
    def __init__(self,
                 feasible_edges_dict: Dict,
                 original_df: pd.DataFrame,
                 degree_dict: Dict[int, int],
                 removed_edges: Set[Tuple[int, int]],
                 time_window: int = 1200):
        """Initialize Edge-Timestamp Selector for non-bipartite graphs"""
        self.feasible_edges_dict = feasible_edges_dict
        self.original_df = original_df
        self.degree_dict = degree_dict
        self.removed_edges = removed_edges
        self.time_window = time_window
        
        # Initialize tracking structures
        self.node_usage = defaultdict(int)
        self.selected_pairs = []
        self.remaining_capacity = deepcopy(degree_dict)
        
        # Calculate edge budget and create lookup structures
        self.max_edges = self._calculate_edge_budget()
        self.node_to_edges = self._create_node_edge_mapping()
        self.feasible_adj_lists = self._create_adj_lists()

    def _calculate_edge_budget(self) -> int:
        """Calculate maximum possible edges for non-bipartite graphs"""
        return sum(self.degree_dict.values()) // 2

    def _create_node_edge_mapping(self) -> Dict[int, List[Tuple[int, int]]]:
        """Create mapping from nodes to their edges"""
        node_to_edges = defaultdict(list)
        for edge in self.feasible_edges_dict:
            u, v = edge
            node_to_edges[u].append(edge)
            node_to_edges[v].append(edge)
        return node_to_edges

    def _create_adj_lists(self) -> Dict[int, Set[int]]:
        """Create adjacency lists for faster lookup"""
        adj_lists = defaultdict(set)
        for (u, v) in self.feasible_edges_dict:
            adj_lists[u].add(v)
            adj_lists[v].add(u)
        return adj_lists

    def select_edges_timestamps(self) -> List[Tuple[int, int, int]]:
        """Main edge selection algorithm for non-bipartite graphs"""
        # Sort nodes by degree for priority
        priority_nodes = sorted(
            self.degree_dict.keys(),
            key=lambda x: (self.degree_dict[x], -len(self.node_to_edges[x])),
            reverse=True
        )

        selected_count = 0
        pbar = tqdm(total=self.max_edges, desc="\tSelecting edges", leave=False, ncols=80)

        # Initial edge selection phase
        for node in priority_nodes:
            if selected_count >= self.max_edges:
                break

            if self.remaining_capacity[node] <= 0:
                continue

            while self.remaining_capacity[node] > 0 and selected_count < self.max_edges:
                candidates = self._get_candidates(node)
                if not candidates:
                    break

                scored_candidates = [
                    (self._calculate_score(edge, timestamps), edge, timestamps)
                    for edge, timestamps in candidates
                ]
                scored_candidates.sort(reverse=True)

                selected = False
                for _, edge, timestamps in scored_candidates:
                    if self._can_select(edge, timestamps):
                        best_timestamp = self._pick_best_timestamp(timestamps, edge)
                        self.selected_pairs.append((edge[0], edge[1], best_timestamp))
                        self._update_tracking(edge)
                        selected_count += 1
                        pbar.update(1)
                        selected = True
                        break

                if not selected:
                    break

        # Recovery phase if needed
        if selected_count < self.max_edges:
            print('\tStarting recovery phase...')
            remaining = self.max_edges - selected_count
            print(f"\tRemaining edges: {remaining}")
            additional_edges = self._fast_recovery_phase(remaining, self.original_df)
            print(f'\tRecovery phase complete. Added {len(additional_edges)} additional edges.')
            selected_count += len(additional_edges)
            pbar.update(len(additional_edges))

        pbar.close()
        self._validate_selections()
        return self.selected_pairs

    def _fast_recovery_phase(self, needed_edges: int, original_df: pd.DataFrame) -> List[Tuple[int, int, int]]:
        """Recovery phase with optimized window queries for non-bipartite graphs"""
        additional_edges = []

        # Pre-process data
        df_sorted = original_df.sort_values('ts').set_index('ts')
        node_appearances = defaultdict(list)
        for idx, row in df_sorted.iterrows():
            node_appearances[row['u']].append(idx)
            node_appearances[row['i']].append(idx)

        available_nodes = [n for n in self.degree_dict.keys() if self.remaining_capacity[n] > 0]
        used_pairs = {(e[0], e[1]) for e in self.selected_pairs}

        all_timestamps = sorted(list(set(
            ts for timestamps in self.feasible_edges_dict.values()
            for ts in timestamps
        )))

        window_cache = {}
        window_size = self.time_window * 1e3

        def get_nodes_in_window(timestamp: float) -> Set[int]:
            if timestamp in window_cache:
                return window_cache[timestamp]

            window_start = max(timestamp - window_size, df_sorted.index.min())
            nodes_in_window = set()

            for node in available_nodes:
                appearances = node_appearances[node]
                idx = bisect.bisect_left(appearances, window_start)
                if idx < len(appearances) and appearances[idx] <= timestamp:
                    nodes_in_window.add(node)

            window_cache[timestamp] = nodes_in_window
            return nodes_in_window

        # Process timestamps in batches
        batch_size = 1000
        for i in range(0, len(all_timestamps), batch_size):
            batch_timestamps = all_timestamps[i:i + batch_size]

            for ts in batch_timestamps:
                get_nodes_in_window(ts)

            for timestamp in batch_timestamps:
                if len(additional_edges) >= needed_edges:
                    break

                nodes_in_window = window_cache[timestamp]
                valid_nodes = [n for n in available_nodes if n in nodes_in_window]

                for node_u in valid_nodes:
                    if len(additional_edges) >= needed_edges:
                        break

                    for node_v in valid_nodes:
                        if node_u == node_v:
                            continue
                        edge = (node_u, node_v)
                        reversed_edge = (node_v, node_u)
                        if edge not in used_pairs and self._can_select(edge, [timestamp]) and edge not in self.removed_edges and reversed_edge not in self.removed_edges:
                            additional_edges.append((node_u, node_v, timestamp))
                            used_pairs.add(edge)
                            self._update_tracking(edge)

                            if len(additional_edges) >= needed_edges:
                                break

            window_cache.clear()

        self.selected_pairs.extend(additional_edges)
        return additional_edges

    def _get_candidates(self, node: int) -> List[Tuple[Tuple[int, int], List[int]]]:
        """Get valid candidate edges for a node"""
        return [(edge, self.feasible_edges_dict[edge])
                for edge in self.node_to_edges[node]
                if self.remaining_capacity[self._get_other_node(edge, node)] > 0
                and edge in self.feasible_edges_dict]

    def _get_other_node(self, edge: Tuple[int, int], node: int) -> int:
        """Get the other node in an edge"""
        u, v = edge
        return v if node == u else u

    def _calculate_score(self, edge: Tuple[int, int], timestamps: List[int]) -> float:
        """Calculate score for candidate edge"""
        u, v = edge
        score = (self.remaining_capacity[u] + self.remaining_capacity[v] +
                 len(timestamps) - (self.node_usage[u] + self.node_usage[v]) * 10)
        if self.remaining_capacity[u] > 0 and self.remaining_capacity[v] > 0:
            score += 50
        return score

    def _can_select(self, edge: Tuple[int, int], timestamps: List[int]) -> bool:
        """Check if edge can be selected"""
        u, v = edge
        if not timestamps or self.remaining_capacity[u] <= 0 or self.remaining_capacity[v] <= 0:
            return False
        return True

    def _pick_best_timestamp(self, timestamps: List[int], edge: Tuple[int, int]) -> int:
        """Select best timestamp from available options"""
        return min(timestamps)  # max(timestamps)

    def _update_tracking(self, edge: Tuple[int, int]):
        """Update tracking after edge selection"""
        u, v = edge
        self.remaining_capacity[u] -= 1
        self.remaining_capacity[v] -= 1
        self.node_usage[u] += 1
        self.node_usage[v] += 1

    def _validate_selections(self):
        """Validate final selections"""
        node_counts = defaultdict(int)
        for u, v, _ in self.selected_pairs:
            node_counts[u] += 1
            node_counts[v] += 1

        print(f"\tNodes used: {len(node_counts)}")


class _NonBipartiteEdgeSelector:
   def __init__(self,
                feasible_edges_dict: Dict,
                original_df: pd.DataFrame, 
                degree_dict: Dict[int, int],
                removed_edges: Set[Tuple[int, int]],
                time_window: int = 1200):
       """Initialize Edge-Timestamp Selector"""
       self.feasible_edges_dict = feasible_edges_dict
       self.original_df = original_df
       self.degree_dict = degree_dict
       self.removed_edges = removed_edges
       self.time_window = time_window
       
       # Initialize tracking structures
       self.node_usage = defaultdict(int)
       self.selected_pairs = []
       self.remaining_capacity = deepcopy(degree_dict)
       
       # Calculate edge budget and create lookup structures
       self.max_edges = self._calculate_edge_budget()
       self.node_to_edges = self._create_node_edge_mapping()
       self.feasible_adj_lists = self._create_adj_lists()
           
   def _calculate_edge_budget(self) -> int:
       """Calculate maximum possible edges"""
       return sum(self.degree_dict.values()) // 2  # non-bipartite graph
       
   def _create_node_edge_mapping(self) -> Dict[int, List[Tuple[int, int]]]:
       """Create mapping from nodes to their edges"""
       node_to_edges = defaultdict(list)
       for edge in self.feasible_edges_dict:
           u, v = edge
           node_to_edges[u].append(edge)
           node_to_edges[v].append(edge)
       return node_to_edges

   def _create_adj_lists(self) -> Dict[int, Set[int]]:
       """Create adjacency lists for faster lookup"""
       adj_lists = defaultdict(set)
       for (u, v) in self.feasible_edges_dict:
           adj_lists[u].add(v)
           adj_lists[v].add(u)
       return adj_lists
       
   def select_edges_timestamps(self) -> List[Tuple[int, int, int]]:
       """Main edge selection algorithm"""
       # Sort nodes by degree for priority
       priority_nodes = sorted(
           self.degree_dict.keys(),
           key=lambda x: (self.degree_dict[x], -len(self.node_to_edges[x])),
           reverse=True
       )
       
       selected_count = 0
       pbar = tqdm(total=self.max_edges, desc="\tSelecting edges", leave=False, ncols=80)
       
       for node in priority_nodes:
           if selected_count >= self.max_edges:
               break
               
           if self.remaining_capacity[node] <= 0:
               continue
               
           while self.remaining_capacity[node] > 0 and selected_count < self.max_edges:
               candidates = self._get_candidates(node)
               if not candidates:
                   break
                   
               scored_candidates = [
                   (self._calculate_score(edge, timestamps), edge, timestamps) 
                   for edge, timestamps in candidates
               ]
               scored_candidates.sort(reverse=True)  # we are selecting high score candidates first
               
               selected = False
               for _, edge, timestamps in scored_candidates:
                   if self._can_select(edge, timestamps):
                       best_timestamp = self._pick_best_timestamp(timestamps, edge)
                       self.selected_pairs.append((edge[0], edge[1], best_timestamp))
                       self._update_tracking(edge)
                       selected_count += 1
                       pbar.update(1)
                       selected = True
                       break
                       
               if not selected:
                   break
       
       pbar.close()
       self._validate_selections()
       return self.selected_pairs

   def _get_candidates(self, node: int) -> List[Tuple[Tuple[int, int], List[int]]]:
       """Get valid candidate edges for a node"""
       return [(edge, self.feasible_edges_dict[edge]) 
               for edge in self.node_to_edges[node]
               if self.remaining_capacity[self._get_other_node(edge, node)] > 0 
               and edge in self.feasible_edges_dict]

   def _get_other_node(self, edge: Tuple[int, int], node: int) -> int:
       """Get the other node in an edge"""
       u, v = edge
       return v if node == u else u
       
   def _calculate_score(self, edge: Tuple[int, int], timestamps: List[int]) -> float:
       """Calculate score for candidate edge"""
       u, v = edge
       score = (self.remaining_capacity[u] + self.remaining_capacity[v] + 
               len(timestamps) - (self.node_usage[u] + self.node_usage[v]) * 10)
       if self.remaining_capacity[u] > 0 and self.remaining_capacity[v] > 0:
           score += 50  # bonus for both nodes having capacity, should increase
       return score
       
   def _can_select(self, edge: Tuple[int, int], timestamps: List[int]) -> bool:
       """Check if edge can be selected"""
       u, v = edge
       if not timestamps or self.remaining_capacity[u] <= 0 or self.remaining_capacity[v] <= 0:
           return False
       return True

   def _pick_best_timestamp(self, timestamps: List[int], edge: Tuple[int, int]) -> int:
       """Select best timestamp from available options"""
       return max(timestamps)  # min(timestamps)

   def _update_tracking(self, edge: Tuple[int, int]):
       """Update tracking after edge selection"""
       u, v = edge
       self.remaining_capacity[u] -= 1
       self.remaining_capacity[v] -= 1
       self.node_usage[u] += 1
       self.node_usage[v] += 1

   def _validate_selections(self):
       """Validate final selections"""
       node_counts = defaultdict(int)
       for u, v, _ in self.selected_pairs:
           node_counts[u] += 1
           node_counts[v] += 1
       print(f"Total nodes used: {len(node_counts)}")