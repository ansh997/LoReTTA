import os
import numpy as np
from .base_computer import *

class TemporalPageRankParams():
    def __init__(self,alpha,beta):
        if alpha > 0 and alpha < 1:
            self.alpha = alpha
        else:
            raise RuntimeError("'alpha' must be from interval (0,1)!")
        if beta >= 0 and beta < 1:
            self.beta = beta
        else:
            raise RuntimeError("'beta' must be from interval [0,1)!")
        
    def __str__(self):
        return "tpr_a%0.2f_b%0.2f" % (self.alpha,self.beta)

    
class TemporalPageRankComputer(BaseComputer):
    def __init__(self,nodes,param_list):
        """Input: list of TemporalPageRankParams objects"""
        self.param_list = param_list
        self.num_of_nodes = len(nodes)
        self.node_indexes = dict(zip(nodes,range(self.num_of_nodes)))
        self.active_mass = np.zeros((self.num_of_nodes,len(self.param_list)+1))
        self.active_mass[:,0] = nodes
        self.temp_pr = np.zeros((self.num_of_nodes,len(self.param_list)+1))
        self.temp_pr[:,0] = nodes
        self.timestamps_pr = {}
    
    def update(self,edge,time=None,graph=None,snapshot_graph=None):
        """edge=(src,trg)"""
        # print('param list info: ', str(self.param_list), len(self.param_list))
        src, trg = edge
        for i in range(0,len(self.param_list)):
            # print('index: ', i, end='\n\t')
            param = self.param_list[i]
            # print('what param: ', param, end='\n\t')
            edge_source_index, edge_target_index = self.node_indexes[src], self.node_indexes[trg]
            # print(f'{edge_source_index=}, {edge_target_index=}', end='\n\t')
            self.temp_pr[edge_source_index,i+1], self.temp_pr[edge_target_index,i+1], self.active_mass[edge_source_index,i+1], self.active_mass[edge_target_index,i+1] = self.update_with_param(i+1,src,trg,param)
            # print(f"PageRank for src {edge_source_index} at param index {i+1}: {self.temp_pr[edge_source_index, i+1]}", end='\n\t')
            # print(f"PageRank for trg {edge_target_index} at param index {i+1}: {self.temp_pr[edge_target_index, i+1]}", end='\n\t')
            self.store_timestamp_pr(time)
            
    def update_with_param(self,idx,src,trg,param):
        "apply temporal pagerank update rule"
        alpha, beta = param.alpha, param.beta
        edge_source_index, edge_target_index = self.node_indexes[src], self.node_indexes[trg]
        tpr_src = self.temp_pr[edge_source_index,idx]
        tpr_trg = self.temp_pr[edge_target_index,idx]
        mass_src = self.active_mass[edge_source_index,idx]
        mass_trg = self.active_mass[edge_target_index,idx]
        # update formula
        tpr_src = tpr_src + 1.0 * (1.0 - alpha)
        tpr_trg = tpr_trg + (mass_src + 1.0 * (1.0 - alpha)) * alpha
        mass_trg = mass_trg + (mass_src + 1.0 * (1.0 - alpha)) * alpha * (1 - beta)
        mass_src = mass_src * beta
        return tpr_src, tpr_trg, mass_src, mass_trg
    
    def store_timestamp_pr(self, time):
        pr_at_time = self.temp_pr[:, 1:].copy()  # Deep copy to ensure independent state
        self.timestamps_pr[time] = pr_at_time
    
    def get_current_pagerank(self):
        return self.temp_pr[:, 1:]
        
    def save_snapshot(self,experiment_folder,snapshot_index,time=None,graph=None,snapshot_graph=None):
        if not os.path.exists(experiment_folder):
            os.makedirs(experiment_folder)
        for j, param in enumerate(self.param_list):
            output_folder = "%s/%s" % (experiment_folder,param)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            pos_idx = self.temp_pr[:,j+1] > 0 
            active_arr = self.temp_pr[pos_idx][:,[0,j+1]]
            scores2file(active_arr,"%s/tpr_%i.csv" % (output_folder,snapshot_index))