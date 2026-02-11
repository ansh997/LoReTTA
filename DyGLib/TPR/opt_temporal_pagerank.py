import os
import numpy as np
from .base_computer import *

class TemporalPageRankParams:
    def __init__(self, alpha, beta):
        if not (0 < alpha < 1):
            raise ValueError("'alpha' must be from interval (0,1)!")
        if not (0 <= beta < 1):
            raise ValueError("'beta' must be from interval [0,1)!")
        self.alpha = alpha
        self.beta = beta
        
    def __str__(self):
        return f"tpr_a{self.alpha:.2f}_b{self.beta:.2f}"

class TemporalPageRankComputer(BaseComputer):
    def __init__(self, nodes, param_list):
        """Input: list of TemporalPageRankParams objects"""
        self.param_list = param_list
        self.num_of_nodes = len(nodes)
        self.node_indexes = {node: index for index, node in enumerate(nodes)}
        self.active_mass = np.zeros((self.num_of_nodes, len(param_list) + 1))
        self.active_mass[:, 0] = nodes
        self.temp_pr = np.zeros((self.num_of_nodes, len(param_list) + 1))
        self.temp_pr[:, 0] = nodes
    
    def update(self, edge, time=None, graph=None, snapshot_graph=None):
        """edge=(src,trg)"""
        src, trg = edge
        src_idx, trg_idx = self.node_indexes[src], self.node_indexes[trg]
        for i, param in enumerate(self.param_list):
            self.temp_pr[src_idx, i + 1], self.temp_pr[trg_idx, i + 1], self.active_mass[src_idx, i + 1], self.active_mass[trg_idx, i + 1] = \
                self.update_with_param(i + 1, src_idx, trg_idx, param)
            
    def update_with_param(self, idx, src_idx, trg_idx, param):
        """Apply temporal pagerank update rule"""
        alpha, beta = param.alpha, param.beta
        tpr_src, tpr_trg = self.temp_pr[src_idx, idx], self.temp_pr[trg_idx, idx]
        mass_src, mass_trg = self.active_mass[src_idx, idx], self.active_mass[trg_idx, idx]

        # Update formula
        delta = 1.0 * (1.0 - alpha)
        tpr_src += delta
        tpr_trg += (mass_src + delta) * alpha
        mass_trg += (mass_src + delta) * alpha * (1 - beta)
        mass_src *= beta

        return tpr_src, tpr_trg, mass_src, mass_trg
        
    def save_snapshot(self, experiment_folder, snapshot_index, time=None, graph=None, snapshot_graph=None):
        os.makedirs(experiment_folder, exist_ok=True)
        for j, param in enumerate(self.param_list):
            output_folder = os.path.join(experiment_folder, str(param))
            os.makedirs(output_folder, exist_ok=True)
            pos_idx = self.temp_pr[:, j + 1] > 0
            active_arr = self.temp_pr[pos_idx][:, [0, j + 1]]
            scores2file(active_arr, f"{output_folder}/tpr_{snapshot_index}.csv")
