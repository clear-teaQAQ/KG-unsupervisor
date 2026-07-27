from lapjv import lapjv
from scipy.optimize import linear_sum_assignment
import sys
import time
from typing import List

import dgl
import torch

import random
import numpy as np
from tqdm import tqdm
from utils import load_all_graphs, load_labels, load_ged
import matplotlib.pyplot as plt

from math import exp
from scipy.stats import spearmanr, kendalltau

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

    def to_nx(self,data):
        x1 = torch.argmax(data.x[(data.x_indicator==0).squeeze(1)],dim=-1)
        x2 = torch.argmax(data.x[(data.x_indicator==1).squeeze(1)],dim=-1)
        edge1 = data.edge_index[:,(data.x_indicator[data.edge_index[0]]==0).squeeze(1)]
        edge1 = remove_self_loops(edge1)[0]
        edge2 = data.edge_index[:,(data.x_indicator[data.edge_index[0]]==1).squeeze(1)] - data.n[0,0].item()
        edge2 = remove_self_loops(edge2)[0]
        g1 = to_networkx(Data(x=x1,edge_index=edge1),to_undirected=True,node_attrs='x')
        g2 = to_networkx(Data(x=x2,edge_index=edge2),to_undirected=True,node_attrs='x')
        
        return g1,g2

    def cost_matrix_construction(self,G1, G2, dname:str):
        INF = G1.number_of_nodes() + G1.number_of_edges() + G2.number_of_nodes() + G2.number_of_edges() + 1
        ns1 = G1.number_of_nodes()
        ns2 = G2.number_of_nodes()
        cost_matrix = np.zeros((ns1 + ns2, ns1 + ns2), dtype=float)
        if dname == 'AIDS':
            node_label = {i: G1.nodes[i]['x'] for i in G1.nodes}
            node_label = sorted(node_label.items(), key=operator.itemgetter(0))
            g1_labels = np.array([k[1] for k in node_label])    
            node_label = {i: G2.nodes[i]['x'] for i in G2.nodes}
            node_label = sorted(node_label.items(), key=operator.itemgetter(0))
            g2_labels = np.array([k[1] for k in node_label]) 
            g1_labels = np.expand_dims(g1_labels, axis=1)
            g2_labels = np.expand_dims(g2_labels, axis=0)
            label_substitution_cost = np.abs(g1_labels - g2_labels)
            label_substitution_cost[np.nonzero(label_substitution_cost)] = 1
            cost_matrix[0:ns1, 0:ns2] = label_substitution_cost

        cost_matrix[0:ns1, ns2:ns1+ns2] = np.array([1 if i == j else INF for i in range(ns1) for j in range(ns1) ]).reshape(ns1, ns1)
        cost_matrix[ns1:ns1+ns2, 0:ns2] = np.array([1 if i == j else INF for i in range(ns2) for j in range(ns2) ]).reshape(ns2, ns2)


        # do not consider node and edge labels, i.e., the cost of edge Eui equals to the degree difference
        g1_degree = np.array([G1.degree(n) for n in range(ns1)], dtype=int)
        g2_degree = np.array([G2.degree(n) for n in range(ns2)], dtype=int)
        g1_degree = np.expand_dims(g1_degree, axis=1)
        g2_degree = np.expand_dims(g2_degree, axis=0)
        degree_substitution_cost = np.abs(g1_degree - g2_degree)
        cost_matrix[0:ns1, 0:ns2] += degree_substitution_cost
        return cost_matrix

    def bipartite_for_cost_matrix(self,G1, G2, cost_matrix, alg_type:str, dname:str):
        if G1.number_of_nodes() == G2.number_of_nodes():
            cost_matrix = cost_matrix[0:G1.number_of_nodes(), 0:G1.number_of_nodes()]
        mapping_str = ""
        can_used_for_AStar = True
        if alg_type == 'hungarian':
            row, col = linear_sum_assignment(cost_matrix)
        elif alg_type == 'vj':
            row, col, _ = lapjv(cost_matrix)
        node_match = {}
        cost = 0
        common = 0
        for i, n in enumerate(row):
            if n < G1.number_of_nodes():
                if col[i] < G2.number_of_nodes():
                    node_match[n] = col[i]
                    if G1.nodes[n]['x'] != G2.nodes[col[i]]['x'] and dname in ['AIDS']:
                        cost += 1
                    mapping_str += "{}|{} ".format(n, col[i])
                else:                      
                    node_match[n] = None
                    cost +=1
                    can_used_for_AStar = False

        for n in G2.nodes:
            if n not in node_match.values(): cost += 1

        for edge in G1.edges():
            (p, q) = (node_match[edge[0]], node_match[edge[1]])
            if (p, q) in G2.edges():
                common += 1
        cost = cost + G1.number_of_edges() + G2.number_of_edges() - 2 * common
        # generate mapping string
        return cost, can_used_for_AStar, mapping_str
    
    def score(self,testing_graph_set='test', algo='hungarian'):
        
       
        loader = self.testing_data_large_loader
        
        print("\n\nEvalute traditional {} on {} set.\n".format(algo,testing_graph_set))
        
        num = 0  # total testing number
        time_usage = 0
        
        ged = []  # ged mae
        
        for batch in tqdm(loader,file=sys.stdout):
            batch.to(self.device)
            g1,g2 = self.to_nx(batch)
            start_time = time.time()
            cost_matrix = self.cost_matrix_construction(g1,g2,self.args.dataset)
            pre_ged,valid,mapping = self.bipartite_for_cost_matrix(g1,g2,cost_matrix,algo,self.args.dataset)
            end_time = time.time()
            num += 1
            time_usage += (end_time-start_time)
  
            
            
            ged.append(abs(pre_ged))
            

       

        time_usage = round(time_usage / num, 5)
        ged = float(np.sum(ged))
        

        self.results.append(('model_name', 'dataset', 'graph_set', '#testing_pairs', 'time_usage(s/p)', 'ged'))
        self.results.append((self.args.model_name, self.args.dataset, testing_graph_set, num, time_usage, ged))

        print(*self.results[-2], sep='\t')
        print(*self.results[-1], sep='\t')

       