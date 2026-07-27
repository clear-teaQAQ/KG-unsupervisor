import sys
import time
from typing import List

import dgl
import torch
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm
from utils import load_all_graphs, load_labels, load_ged
import matplotlib.pyplot as plt
from gedgnn_kbest import KBestMSolver
from math import exp
from scipy.stats import spearmanr, kendalltau

from models import  GEDIOT, GEDGW
from loss_fn import mapping_loss

from torch_geometric.data import Data,Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse,to_undirected,sort_edge_index,coalesce,to_dense_adj,remove_self_loops,to_dense_batch,group_argsort,to_networkx
import torch_geometric as pyg
from torch_geometric.nn.pool import global_add_pool,global_mean_pool
import networkx as nx
import operator
import json

class Trainer(object):
    """
    A general model trainer.
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.args = args
        self.load_data_time = 0.0
        self.to_torch_time = 0.0
        self.results = []

        self.use_gpu = torch.cuda.is_available()
        print("use_gpu =", self.use_gpu)
        self.device = torch.device('cuda') if self.use_gpu else torch.device('cpu')
        
        self.load_data()
        self.transfer_data_to_torch()
      
        self.init_graph_pairs()

        self.training_data_loader = DataLoader(self.training_graphs_large,batch_size=self.args.batch_size,shuffle=True)  
        self.testing_data_large_loader = DataLoader(self.testing_graphs_large,batch_size=1,shuffle=False)

    def load_data(self):
        t1 = time.time()
        dataset_name = self.args.dataset
        self.train_num, self.val_num, self.test_num, self.graphs = load_all_graphs(self.args.abs_path, dataset_name)
        print("Load {} graphs. ({} for training)".format(len(self.graphs), self.train_num))

        self.number_of_labels = 0
        if dataset_name in ['AIDS']:
            self.global_labels, self.features = load_labels(self.args.abs_path, dataset_name)
            self.number_of_labels = len(self.global_labels)
        if self.number_of_labels == 0:
            self.number_of_labels = 1
            self.features = []
            for g in self.graphs:
                self.features.append([[2.0] for u in range(g['n'])])
        
        t2 = time.time()
        self.load_data_time = t2 - t1
    
    def transfer_data_to_torch(self):
        t1 = time.time()

        self.edge_index = []
        for g in self.graphs:
            edge = g['graph']
            edge = edge + [[y, x] for x, y in edge]
            edge = edge + [[x, x] for x in range(g['n'])]
            edge = torch.tensor(edge).t().long()
            self.edge_index.append(edge)
        
        self.features = [torch.tensor(x).float() for x in self.features]
        print("Feature shape of 1st graph:", self.features[0].shape)

        n = len(self.graphs)
        mapping = [[None for i in range(n)] for j in range(n)]
        ged = [[(0., 0., 0., 0.) for i in range(n)] for j in range(n)]
        gid = [g['gid'] for g in self.graphs]
        self.gid = gid
        # number of nodes
        self.gn = [g['n'] for g in self.graphs]
        # number of edges
        self.gm = [g['m'] for g in self.graphs]
        for i in range(n):
            mapping[i][i] = torch.eye(self.gn[i], dtype=torch.float)
            for j in range(i + 1, n):
                id_pair = (gid[i], gid[j])
                n1, n2 = self.gn[i], self.gn[j]
                if n1 > n2:
                    n1,n2 = n2,n1
                
                mapping_list = [[0 for y in range(n2)] for x in range(n1)]
                mapping_matrix = torch.tensor(mapping_list).float()
                mapping[i][j] = mapping[j][i] = mapping_matrix
        
        self.mapping = mapping
        
        t2 = time.time()
        self.to_torch_time = t2 - t1
    
    
    
    def pack_graph_pair(self,pair):
        new_data = Data()
        id_1, id_2 = pair
        
        new_data.i_j = torch.tensor([[id_1,id_2]])
        
    
       

        n1,m1 = self.gn[id_1],self.gm[id_1]
        n2,m2 = self.gn[id_2],self.gm[id_2]
        new_data.n = torch.tensor([[n1,n2]])
        new_data.m = torch.tensor([[m1,m2]])
        new_data.avg_n = torch.tensor([[(n1+n2)/2]])
        new_data.higher_bound = torch.tensor([[max(n1, n2) + max(m1, m2)]])
        # add dummy nodes
        new_data.x = torch.cat([self.features[id_1],self.features[id_2]],dim=0)
        
        new_data.edge_index = torch.cat([self.edge_index[id_1],self.edge_index[id_2]+n1],dim=1)
        # (G,G'): If G, then x_indicator=0. If G', x_indicator=1
        new_data.x_indicator = torch.cat([torch.zeros((n1,1)),torch.ones((n2,1))],dim=0)

        # transfer mapping to edge index between G and G'
        mapping = self.mapping[id_1][id_2]
        
        
        mapping = mapping + 0.1
        mapping_edge_index,mapping_edge_attr = dense_to_sparse(mapping)
        
        mapping_edge_index[1] += n1
        new_data.edge_index_mapping = mapping_edge_index
        new_data.edge_attr_mapping = (mapping_edge_attr-0.1).unsqueeze(-1)
      
        
        
        
        
       
        best_mapping_label = torch.rand_like(new_data.edge_attr_mapping)
        
        new_data.best_mapping_label = best_mapping_label
        
        return new_data
    
    
    def init_graph_pairs(self):
        start = time.time()
        random.seed(1)
        
        self.training_graphs_large = []
        self.testing_graphs_large = []
       
        train_num = self.train_num
        val_num = train_num + self.val_num
        test_num = len(self.graphs)
        
        
        # each large training graph is paired with all other training large graphs 
        for i in range(train_num):
            if self.gn[i] > 10:
                for j in range(i, train_num):
                    if self.gn[j] > 10:
                        if self.gn[i] > self.gn[j]:
                            pair = self.pack_graph_pair((j,i))
                        else:
                            pair = self.pack_graph_pair((i,j))
                        self.training_graphs_large.append(pair)
        
        # each large testing graph is paired with 100 large training large graphs 
        li = []
        
        for i in range(train_num):
            if self.gn[i] > 10:
                li.append(i)
        
        
    
                   
        for i in range(val_num, test_num):
            if self.gn[i] > 10:
                random.shuffle(li)
                for j in li[:self.args.num_testing_graphs]:
                    if self.gn[i] > self.gn[j]:
                        pair = self.pack_graph_pair((j,i))
                    else:
                        pair = self.pack_graph_pair((i,j))
                    self.testing_graphs_large.append(pair)
                    
            

        


        end = time.time()
        print("Generate {} training large graph pairs.".format(len(self.training_graphs_large)))
        
        print("Generate {} testing large graph pairs.".format(len(self.testing_graphs_large)))
       
        print("Generation time:",end-start)
    
    def sequential_topk_ged(self,batch,test_k):
        start = time.time()
        gt_mapping_idx = batch.edge_index_mapping
        n1 = batch.n[0,0].item()
        n2 = batch.n[0,1].item()
        x1 = batch.x[:n1]
        x2 = batch.x[n1:]

        x1_edge = batch.edge_index[:,batch.edge_index[0]<n1]
        x2_edge = batch.edge_index[:,batch.edge_index[0]>=n1] - n1

        g1 = dgl.graph((x1_edge[0],x1_edge[1]),num_nodes=n1)
        g2 = dgl.graph((x2_edge[0],x2_edge[1]),num_nodes=n2)
        g1.ndata['f'] = x1
        g2.ndata['f'] = x2
        if self.args.model_name =="GEDGW":
            gw = GEDGW(batch, self.args)
            mapping_t,pre_geds = gw.process()
            mapping_t = mapping_t*1e9+1
            pred_matching_matrix = mapping_t
        else:
            with torch.no_grad():
                prediction, pred_geds, mapping_t = self.model(batch)
            
            mapping_t = (mapping_t * 1e9 + 1).round()
            pred_matching_matrix = torch.zeros((n1,n2),device=self.device)
            pred_matching_matrix[batch.edge_index_mapping[0],batch.edge_index_mapping[1]-n1] = mapping_t.squeeze(-1)

        solver = KBestMSolver(pred_matching_matrix, g1, g2)
        solver.get_matching(test_k)
        min_ged = solver.min_ged
        end = time.time()
        return min_ged,end-start
    
    def score(self,testing_graph_set='test', test_k=100):
        print(len(self.testing_graphs_large))
        exit()
       
        loader = self.testing_data_large_loader
        
        print("\n\nEvalute GedGNN with topk {} on {} set.\n".format(test_k,testing_graph_set))
        if self.args.model_name != 'GEDGW':
            self.model.eval()
        num = 0  # total testing number
        time_usage = 0
        
        ged = []  # ged mae
        
        for batch in tqdm(loader,file=sys.stdout):
            batch.to(self.device)
            
            model_out = self.sequential_topk_ged(batch,test_k)

            pre_ged,running_time = model_out[0],model_out[1]
            num += 1
            time_usage += running_time
  
            
            
            ged.append(abs(pre_ged))
            

       

        time_usage = round(time_usage / num, 5)
        ged = round(np.mean(ged), 3)
        

        self.results.append(('model_name', 'dataset', 'graph_set', '#testing_pairs', 'time_usage(s/p)', 'ged'))
        self.results.append((self.args.model_name, self.args.dataset, testing_graph_set, num, time_usage, ged))

        print(*self.results[-2], sep='\t')
        print(*self.results[-1], sep='\t')

        with open(self.args.abs_path + self.args.result_path + f'result_GedGNN_{self.args.dataset}_{testing_graph_set}_{test_k}.json','w') as f:
            json.dump({'time':time_usage,'ged':ged},f)
    

