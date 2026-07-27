import sys
import time


import dgl
import torch
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm
from utils import load_all_graphs, load_labels, load_ged

from gedgnn_kbest import KBestMSolver

from scipy.stats import spearmanr, kendalltau

from models import DiffMatch, Discriminator
from loss_fn import mapping_loss, roll_out,roll_out_gumbel, bpr_loss, hinge_loss
from diffusion_schedulers import CategoricalDiffusion,InferenceSchedule
from torch_geometric.data import Data,Batch,Dataset
from torch_geometric.loader import DataLoader

from torch_geometric.utils import dense_to_sparse,to_dense_adj,remove_self_loops
import torch_geometric as pyg
import json

class Dataset_with_idx(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
       
        return self.data[index], index 

    def __len__(self):
        return len(self.data)

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
        self.founded_ged = []
        self.use_gpu = torch.cuda.is_available()
        print("use_gpu =", self.use_gpu)
        self.device = torch.device('cuda') if self.use_gpu else torch.device('cpu')
        
        self.load_data()
        self.transfer_data_to_torch()
        self.setup_model()
        self.init_graph_pairs()
        self.training_data_loader = DataLoader(Dataset_with_idx(self.training_graphs),batch_size=self.args.batch_size,shuffle=True)  
        self.init_roll_out()
        self.testing_data_loader = DataLoader(self.testing_graphs,batch_size=1,shuffle=False)
        self.validation_data_loader = DataLoader(self.val_graphs,batch_size=1,shuffle=False)
        

    
    def setup_model(self):
        self.model = DiffMatch(self.args, self.number_of_labels).to(self.device)
        self.D = Discriminator(self.args, self.number_of_labels).to(self.device)
        
        self.optimizer = torch.optim.RMSprop(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        self.optimizerD = torch.optim.RMSprop(self.D.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        self.diffusion = CategoricalDiffusion(T=self.args.diffusion_steps)
       
    
    def load_data(self):
        t1 = time.time()
        self.train_num, self.val_num, self.test_num, self.graphs = load_all_graphs(self.args.dataset_root, self.args.dataset)
        print("Load {} graphs. ({} for training)".format(len(self.graphs), self.train_num))

        self.global_labels, self.features = load_labels(self.args.dataset_root, self.args.dataset)
        self.number_of_labels = len(self.global_labels)
        
        ged_dict = dict()
        self.split_pairs = load_ged(ged_dict, self.args.dataset_root, self.args.dataset, 'TaGED.json')
        self.ged_dict = ged_dict
        print("Load ged dict.")
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
                reversed_pair = False
                if id_pair not in self.ged_dict:
                    id_pair = (gid[j], gid[i])
                    reversed_pair = True
                if id_pair not in self.ged_dict:
                    ged[i][j] = ged[j][i] = None
                    mapping[i][j] = mapping[j][i] = None
                else:
                    ta_ged, gt_mappings = self.ged_dict[id_pair]
                    ged[i][j] = ged[j][i] = ta_ged
                    if gt_mappings:
                        if reversed_pair:
                            base_n1, base_n2 = self.gn[j], self.gn[i]
                        else:
                            base_n1, base_n2 = self.gn[i], self.gn[j]
                        mapping_list = [[0 for _ in range(base_n2)] for _ in range(base_n1)]
                        gt_mapping = gt_mappings[0]
                        for x, y in enumerate(gt_mapping):
                            if x < base_n1 and y < base_n2:
                                mapping_list[x][y] = 1
                        mapping_matrix = torch.tensor(mapping_list).float()
                        if reversed_pair:
                            mapping[i][j] = mapping_matrix.t()
                            mapping[j][i] = mapping_matrix
                        else:
                            mapping[i][j] = mapping_matrix
                            mapping[j][i] = mapping_matrix.t()
                    else:
                        mapping[i][j] = torch.ones((self.gn[i], self.gn[j]), dtype=torch.float)
                        mapping[j][i] = torch.ones((self.gn[j], self.gn[i]), dtype=torch.float)
        
        self.ged = ged
        self.mapping = mapping
        
        t2 = time.time()
        self.to_torch_time = t2 - t1
    
    def pack_graph_pair(self,pair):
        new_data = Data()
        orig_id_1, orig_id_2, real_ged = pair
        new_data.i_j = torch.tensor([[orig_id_1, orig_id_2]])

        id_1, id_2 = orig_id_1, orig_id_2
        if self.gn[id_1] > self.gn[id_2]:
            id_1, id_2 = id_2, id_1

        n1,m1 = self.gn[id_1],self.gm[id_1]
        n2,m2 = self.gn[id_2],self.gm[id_2]
        new_data.n = torch.tensor([[n1,n2]])
        new_data.m = torch.tensor([[m1,m2]])
        new_data.avg_n = torch.tensor([[(n1+n2)/2]])
        new_data.higher_bound = torch.tensor([[max(n1, n2) + max(m1, m2)]])
        new_data.x = torch.cat([self.features[id_1],self.features[id_2]],dim=0)
        
        new_data.edge_index = torch.cat([self.edge_index[id_1],self.edge_index[id_2]+n1],dim=1)
        new_data.x_indicator = torch.cat([torch.zeros((n1,1)),torch.ones((n2,1))],dim=0)

        mapping = torch.ones((n1, n2), dtype=torch.float)
        mapping = mapping + 0.1
        mapping_edge_index,mapping_edge_attr = dense_to_sparse(mapping)
        
        mapping_edge_index[1] += n1
        new_data.edge_index_mapping = mapping_edge_index
        new_data.edge_attr_mapping = (mapping_edge_attr-0.1).unsqueeze(-1)
        
        new_data.ged = real_ged
       
        best_mapping_label = torch.rand_like(new_data.edge_attr_mapping)
        
        new_data.best_mapping_label = best_mapping_label
        
        return new_data
    
    def init_graph_pairs(self):
        start = time.time()
        self.training_graphs = [self.pack_graph_pair(pair) for pair in self.split_pairs["train"]]
        self.val_graphs = [self.pack_graph_pair(pair) for pair in self.split_pairs["val"]]
        self.testing_graphs = [self.pack_graph_pair(pair) for pair in self.split_pairs["test"]]

        end = time.time()
        print("Generate {} training graph pairs.".format(len(self.training_graphs)))
        print("Generate {} val graph pairs.".format(len(self.val_graphs)))
        print("Generate {} testing graph pairs.".format(len(self.testing_graphs)))
        print("Generation time:",end-start)
    
    def init_roll_out(self):
        
        training_data_loader = self.training_data_loader
    
        print('start initial roll out')
        
        
        for batch,indices in training_data_loader:
            batch.to(self.device)
        
            pred_ged,pred_solution = roll_out(batch.best_mapping_label,batch)
        
            for index in range(len(indices)):
            
                self.training_graphs[indices[index]].best_ged = pred_ged[index] 
                self.training_graphs[indices[index]].best_mapping_label = pred_solution[batch.batch[batch.edge_index_mapping[0]]==index].to(self.training_graphs[indices[index]].best_mapping_label.device)
                self.training_graphs[indices[index]].last_mapping_label = pred_solution[batch.batch[batch.edge_index_mapping[0]]==index].to(self.training_graphs[indices[index]].best_mapping_label.device)
                self.training_graphs[indices[index]].last_ged = pred_ged[index]
       
        print('roll out finished')
        
         

    def fit(self):
       
        training_data_loader = self.training_data_loader
        data_len = len(self.training_graphs)
        
    
        print("\nModel training.\n")
        t1 = time.time()

        self.model.train()
        
        with tqdm(total= data_len, unit="graph_pairs", leave=True, desc="Epoch",file=sys.stdout) as pbar:
            
            g_loss_sum = 0
            d_loss_sum = 0
            map_loss_sum = 0
            ged_loss_sum = 0
            main_index = 0
            index = 0
           
            total_new_solution = 0

            total_pred_ged = 0
            total_gt_ged = 0
            total_curr_best_ged = 0
            for batch,idx in training_data_loader:
                
                batch.to(self.device)
                batch_total_loss,batch_D_loss, rollout_ged,gt_ged,curr_best_ged,batch_map_loss,batch_ged_loss,new_solution = self.process_batch(batch,idx)
                total_curr_best_ged += curr_best_ged
                
                total_new_solution += new_solution
                total_pred_ged += rollout_ged
                total_gt_ged += gt_ged
                g_loss_sum += batch_total_loss
                d_loss_sum += batch_D_loss
                map_loss_sum += batch_map_loss
                ged_loss_sum += batch_ged_loss
                main_index += (torch.max(batch.batch)+1).item()
                loss = g_loss_sum / main_index
                d_loss = d_loss_sum / main_index
                map_loss = map_loss_sum / main_index
                ged_loss = ged_loss_sum / main_index
                pbar.update(len(batch))
                pbar.set_description(
                    "Epoch_{}: g_loss={} d_loss={} map_loss={} ged_loss={} - Batch_{}: generator loss={}, discrimator loss = {}, {} new solutions".format(self.cur_epoch + 1, round(1000 * loss, 3),round(1000 * d_loss, 3),round(1000 * map_loss, 3),round(1000 * ged_loss, 3),
                                                                    index,
                                                                    round(1000 * batch_total_loss / len(batch), 3),round(1000 * batch_D_loss / len(batch), 3), new_solution))
                index += 1
            tqdm.write("Epoch {}: generator loss={}, discriminator loss={}, mapping loss={}, ged loss={}".format(self.cur_epoch + 1, round(1000 * loss, 3),round(1000 * d_loss, 3),round(1000 * map_loss, 3),round(1000 * ged_loss, 3)))
            training_loss = round(1000 * loss, 3)
            training_d_loss = round(1000 * d_loss, 3)
            training_map_loss = round(1000 * map_loss, 3)
            training_ged_loss = round(1000 * ged_loss, 3)
            
        t2 = time.time()
        training_time = t2 - t1
        self.founded_ged.append(total_curr_best_ged)
        self.results.append(
            ('model_name', 'dataset', 'graph_set', "epoch", "training_time", "generator training_loss","discriminator training_loss","mapping training_loss","ged training loss","new solutions","pred ged","current best ged",'gt_ged'))
        self.results.append(
            (self.args.model_name, self.args.dataset, "train", self.cur_epoch + 1, training_time, training_loss,training_d_loss,training_map_loss,training_ged_loss,total_new_solution,total_pred_ged,total_curr_best_ged,total_gt_ged))

        print(*self.results[-2], sep='\t')
        print(*self.results[-1], sep='\t')
        with open(self.args.abs_path + self.args.result_path + f'pathlength_GEDRanker_{self.args.dataset}_{self.args.unsupervised_approach}_{self.args.run_timestamp}.json','w') as f:
            json.dump(self.founded_ged, f)
       
    def process_batch(self,batch,indices):
        batch_size = (torch.max(batch.batch) + 1).item()
        gt_mapping_idx,best_mapping_label,best_ged = batch.edge_index_mapping,batch.best_mapping_label,batch.best_ged
        
        # sample random time steps t
        t = np.random.randint(1, self.diffusion.T + 1, batch_size).astype(int)
        # one-hot encoding of ground-truth matching matrix
        best_mapping_onehot = torch.nn.functional.one_hot(best_mapping_label.long(), num_classes=2).float()
        mapping_batch = batch.batch[gt_mapping_idx[0]]
        # sample noisy matching matrix
        diffused_mapping = self.diffusion.sample(best_mapping_onehot, t,mapping_batch)
        t = torch.from_numpy(t).float()
        # predict matching matrix
        pred_mapping_label = self.model(batch,diffused_mapping.to(self.device),t.to(self.device))
        
        if self.args.unsupervised_approach == 'BPR' or self.args.unsupervised_approach == 'Hinge' or self.args.unsupervised_approach == 'GED':
            # pred_mapping_prob -> pred_solution_gumbel -> pred_solution -> pred_ged
            pred_ged,pred_solution,pred_solution_gumbel_sparse = roll_out_gumbel(pred_mapping_label,batch,self.args.tau,self.args.gumbel_iteration)
        else:
            pred_ged,pred_solution = roll_out(pred_mapping_label,batch)

        # normalize
        normalize_curr_ged = torch.exp(-pred_ged / batch.avg_n.squeeze(-1))
        normalize_best_ged = torch.exp(-best_ged / batch.avg_n.squeeze(-1))
        normalize_last_ged = torch.exp(-batch.last_ged / batch.avg_n.squeeze(-1))
        if self.args.unsupervised_approach == 'plain':
            alpha = 0
        else:
            alpha = max(1 - 1 * self.cur_epoch / (self.args.model_epoch_end/2),0)
           
        
        if alpha > 0 and (self.args.unsupervised_approach == 'BPR' or self.args.unsupervised_approach == 'Hinge' or self.args.unsupervised_approach == 'GED'):
           
            D_pred_ged_curr = self.D(batch,pred_solution_gumbel_sparse.detach())
            if self.args.unsupervised_approach == 'BPR' or self.args.unsupervised_approach == 'Hinge':
                D_pred_ged_best = self.D(batch,batch.best_mapping_label)
                D_pred_ged_last = self.D(batch,batch.last_mapping_label)
            
            if self.args.unsupervised_approach == 'BPR':
                D_loss = bpr_loss(D_pred_ged_curr,D_pred_ged_best,D_pred_ged_last,normalize_curr_ged,normalize_best_ged,normalize_last_ged)
            elif self.args.unsupervised_approach == 'Hinge':
                D_loss = hinge_loss(D_pred_ged_curr,D_pred_ged_best,D_pred_ged_last,normalize_curr_ged,normalize_best_ged,normalize_last_ged)
            elif self.args.unsupervised_approach == 'GED':
                D_loss = ((D_pred_ged_curr-normalize_curr_ged)**2).sum()
            self.optimizerD.zero_grad()
            D_loss.backward()
            self.optimizerD.step()
       
        else:
            D_loss = torch.tensor([0])

        map_loss = mapping_loss(pred_mapping_label,batch,best_mapping_label) 
        
        if self.args.unsupervised_approach == 'BPR' or self.args.unsupervised_approach == 'Hinge' or self.args.unsupervised_approach == 'GED':
            D_pred_ged = self.D(batch,pred_solution_gumbel_sparse)
            ged_loss = -(D_pred_ged).sum()
        else:
            ged_loss = torch.tensor([0],device=map_loss.device)


        losses = map_loss + ged_loss * alpha 
        
        self.optimizer.zero_grad()
        losses.backward()
        
       
        self.optimizer.step()
       
        new_solution = 0
        
        for index in range(len(indices)):
            self.training_graphs[indices[index]].last_mapping_label = pred_solution[batch.batch[batch.edge_index_mapping[0]]==index].to(self.training_graphs[indices[index]].last_mapping_label.device)
            self.training_graphs[indices[index]].last_ged = (pred_ged[index]).to(self.training_graphs[indices[index]].last_ged.device)
            if pred_ged[index] < best_ged[index]:
                new_solution += 1
                self.training_graphs[indices[index]].best_ged = (pred_ged[index]).to(self.training_graphs[indices[index]].best_ged.device)
                self.training_graphs[indices[index]].best_mapping_label = pred_solution[batch.batch[batch.edge_index_mapping[0]]==index].to(self.training_graphs[indices[index]].best_mapping_label.device)
        
        return losses.item(),D_loss.item(),pred_ged.sum().item(),batch.ged.sum().item(),best_ged.sum().item(),map_loss.item(),ged_loss.item(),new_solution
        
    def diffusion_ged_parallel(self,batch,test_k=100):
        # generate k node matching matrices
        start_time = time.time()
        num_parallel_sampling = test_k
        data = batch[0]
        new_batch = Batch().from_data_list([data for i in range(num_parallel_sampling)])
        gt_mapping_label = new_batch.edge_attr_mapping
        
        # sample random node matching matrix
        mapping_t = torch.randn_like(gt_mapping_label,device=self.device)
        mapping_t = (mapping_t>0).long()
       
        steps = self.args.inference_diffusion_steps
        time_schedule = InferenceSchedule(T=self.diffusion.T, inference_T=steps)

        # diffusion
        for s in range(steps):
            t1,t2 = time_schedule(s)
            t1 = np.array([t1]).astype(int)
            t2 = np.array([t2]).astype(int)
            mapping_t = self.categorical_denoise_step(new_batch,mapping_t,t1,t2)
        
       
        n1 = batch.n[0,0].item()
        n2 = batch.n[0,1].item()

        pred_matching_matrix = torch.zeros((num_parallel_sampling,n1,n2),device=self.device)
        mapping_edge_idx = new_batch.edge_index_mapping
        graph_edge_idx = new_batch.edge_index
        batch_mapping_edge_idx = mapping_edge_idx - new_batch.batch[mapping_edge_idx[0]] * (n1+n2)
        batch_mapping_edge_idx[1] -= n1

        pred_matching_matrix[new_batch.batch[mapping_edge_idx[0]],batch_mapping_edge_idx[0],batch_mapping_edge_idx[1]] = mapping_t.squeeze(-1)
        batch_idx = torch.arange(num_parallel_sampling,device=pred_matching_matrix.device)

        # extract node mappings
        greedy_mask = torch.zeros_like(pred_matching_matrix,dtype=torch.bool)
        solution = torch.zeros_like(pred_matching_matrix,dtype=torch.bool)

        for s in range(min(n1,n2)):
            pred_matching_matrix = pred_matching_matrix.view(num_parallel_sampling,-1)
            argmax_result = torch.argmax(pred_matching_matrix,dim=-1)
            rows = argmax_result // n2
            columns = argmax_result % n2
            
            solution[batch_idx,rows,columns] = True
            greedy_mask[batch_idx,rows,:] = True
            greedy_mask[batch_idx,:,columns] = True

            pred_matching_matrix = pred_matching_matrix.view(num_parallel_sampling,n1,n2)
            pred_matching_matrix[greedy_mask] = float('-inf')
        
        zeros_column = torch.where(~torch.any(solution == 1, dim=1))
        # if |V| < |V'|, add nodes to V with empty labels, and map each to an unmatched nodes in V'
        solution = torch.cat([solution,torch.zeros(num_parallel_sampling,n2-n1,n2,device=solution.device)],dim=1)
        solution[zeros_column[0],torch.arange(n1,n2,device=solution.device).repeat(num_parallel_sampling),zeros_column[1]] = 1
        extracted_mapping = torch.nonzero(solution)

        x1 = new_batch.x[(new_batch.x_indicator==0).squeeze(1)]
        x2 = new_batch.x[(new_batch.x_indicator==1).squeeze(1)]
        dense_x1 = x1.view(num_parallel_sampling,n1,-1)
        dense_x2 = x2.view(num_parallel_sampling,n2,-1)


        # permute G' according to the extracted mapping
        permuted_x2 = (dense_x2[extracted_mapping[:,0],extracted_mapping[:,2]]).view(num_parallel_sampling,n2,-1)
        dense_x1 = torch.cat([dense_x1,torch.zeros(num_parallel_sampling,n2-n1,dense_x1.shape[-1],device=dense_x1.device)],dim=1)
        edge1 = new_batch.edge_index[:,(new_batch.x_indicator[new_batch.edge_index[0]]==0).squeeze(1)]
        edge1 = remove_self_loops(edge1)[0]
        edge1_batch = new_batch.batch[edge1[0]]
        edge1 = edge1 - edge1_batch * n1 
        dense_adj_1 = to_dense_adj(edge_index=edge1,batch=new_batch.batch[(new_batch.x_indicator==1).squeeze(1)],max_num_nodes=n2)
        reversed_mapping = torch.tensor(sorted(extracted_mapping.tolist(),key=lambda x:(x[0],x[2])),device=extracted_mapping.device)
        edge2 = new_batch.edge_index[:,(new_batch.x_indicator[new_batch.edge_index[0]]==1).squeeze(1)]
        edge2_batch = new_batch.batch[edge2[0]]
        edge2 = edge2 - (edge2_batch+1) * n1
        reversed_mapping[:,2] += reversed_mapping[:,0] * n2
        reversed_mapping[:,1] += reversed_mapping[:,0] * n2
        edge2[0] = reversed_mapping[edge2[0],1]
        edge2[1] = reversed_mapping[edge2[1],1]
        dense_adj_2 = to_dense_adj(remove_self_loops(edge2)[0],batch=new_batch.batch[(new_batch.x_indicator==1).squeeze(1)],max_num_nodes=n2)

        # ged = difference in G and G' (after permuting G')
        adj_diff = torch.abs(dense_adj_1-dense_adj_2).view(num_parallel_sampling,-1).sum(dim=-1) // 2
        feat_diff = torch.sum(~torch.all(dense_x1 == permuted_x2,dim=-1),dim=-1)
        ged = adj_diff + feat_diff

        min_ged_idx = torch.argmin(ged)
        min_ged = ged[min_ged_idx].item()
        end_time = time.time()
        min_mapping = solution[min_ged == ged,:n1]
        return min_ged,min_mapping,end_time-start_time
    
    def diffusion_ged_sequential(self,batch,test_k=100,k_range=None,return_mapping=False):
        start_time = time.time()
        gt_mapping_idx,gt_mapping_label = batch.edge_index_mapping,batch.edge_attr_mapping
        mapping_t = torch.randn_like(gt_mapping_label,device=self.device)
        mapping_t = (mapping_t>0).long()
        steps = self.args.inference_diffusion_steps
        time_schedule = InferenceSchedule(T=self.diffusion.T, inference_T=steps)

        # diffusion
        for s in range(steps):
            t1,t2 = time_schedule(s)
            t1 = np.array([t1]).astype(int)
            t2 = np.array([t2]).astype(int)
            mapping_t = self.categorical_denoise_step(batch,mapping_t,t1,t2)
        
        mapping_t = pyg.utils.softmax(mapping_t.squeeze(-1), gt_mapping_idx[0]).unsqueeze(-1)
        mapping_t = (mapping_t * 1e9 + 1).round()

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
        pred_matching_matrix = torch.zeros((n1,n2),device=self.device)
        pred_matching_matrix[batch.edge_index_mapping[0],batch.edge_index_mapping[1]-n1] = mapping_t.squeeze(-1)
        
        # GEDGNN topk
        solver = KBestMSolver(pred_matching_matrix, g1, g2)
        if k_range == None:
            solver.get_matching(test_k)
            min_ged = solver.min_ged
            min_mappings = []
            end_time = time.time()
            if return_mapping:
                for sp in solver.subspaces:
                    if sp.ged == solver.min_ged:
                        min_mappings.append(sp.best_matching)
                    elif sp.ged2 == solver.min_ged:
                        min_mappings.append(sp.second_matching)
            return min_ged,min_mappings,end_time-start_time
        
        else:
            k_running_time = {}
            pre_geds = {}
            for k in k_range:
                solver.get_matching(k)
                min_ged = solver.min_ged
                end_time = time.time()
                k_running_time[k] = end_time-start_time
                pre_geds[k] = min_ged
            return pre_geds,None,k_running_time



    def categorical_denoise_step(self,data,mapping_t,t1,t2):      
        batch_size = torch.max(data.batch) + 1
        t1 = torch.from_numpy(t1).repeat(batch_size)

        # predict node matching matrix
        with torch.no_grad():
            pred_mapping_label = self.model(data,mapping_t,t1.float().to(self.device))
            
        prob_mapping = torch.nn.functional.sigmoid(pred_mapping_label)
        

        # compute posterior
        prob_mapping = torch.cat([1-prob_mapping,prob_mapping],dim=-1)
        mapping_t = self.categorical_posterior(t2,t1,prob_mapping,mapping_t,data.batch[data.edge_index_mapping[0]])
       
        return mapping_t
    
    def categorical_posterior(self, target_t, t, x0_pred_prob, xt, mapping_batch):
        diffusion = self.diffusion
        if target_t is None:
            target_t = t - 1
        else:
            target_t = torch.from_numpy(target_t).view(1)
        target_t = target_t.repeat(t.shape[0])

        Q_t = (np.linalg.inv(diffusion.Q_bar[target_t]) @ diffusion.Q_bar[t])
        Q_t = Q_t.reshape(t.shape[0],2,2)
        
        Q_t = torch.from_numpy(Q_t).float().to(x0_pred_prob.device)

        Q_bar_t_source = torch.from_numpy(diffusion.Q_bar[t]).float().to(x0_pred_prob.device).reshape(t.shape[0],2,2)
        Q_bar_t_target = torch.from_numpy(diffusion.Q_bar[target_t]).float().to(x0_pred_prob.device).reshape(t.shape[0],2,2)

        x0_pred_prob = x0_pred_prob.unsqueeze(1)
        xt = F.one_hot(xt.long(), num_classes=2).float()
        
        x_t_target_prob_part_1 = torch.matmul(xt, Q_t[mapping_batch].permute((0,2, 1)).contiguous())
        
        x_t_target_prob_part_2 = Q_bar_t_target[:,0]
        
        x_t_target_prob_part_3 = (Q_bar_t_source[:,0][mapping_batch].unsqueeze(1) * xt).sum(dim=-1,keepdim=True)
        x_t_target_prob = (x_t_target_prob_part_1 * x_t_target_prob_part_2[mapping_batch].unsqueeze(1)) / x_t_target_prob_part_3
        
        sum_x_t_target_prob = x_t_target_prob[..., 1] * x0_pred_prob[..., 0]

        x_t_target_prob_part_2_new = Q_bar_t_target[:,1]
        x_t_target_prob_part_3_new = (Q_bar_t_source[:,1][mapping_batch].unsqueeze(1) * xt).sum(dim=-1,keepdim=True)
        x_t_target_prob_new = (x_t_target_prob_part_1 * x_t_target_prob_part_2_new[mapping_batch].unsqueeze(1)) / x_t_target_prob_part_3_new
        sum_x_t_target_prob += x_t_target_prob_new[..., 1] * x0_pred_prob[..., 1]
        
        if target_t[0] > 0:
            xt = torch.bernoulli(sum_x_t_target_prob.clamp(0, 1))
        else:
            xt = sum_x_t_target_prob.clamp(min=0)
        return xt
    
    def save(self, epoch):
        torch.save(self.model.state_dict(),
                   self.args.abs_path + self.args.model_path + self.args.dataset + '_' + str(epoch) + '_' + self.args.model_name + '_' + self.args.unsupervised_approach + '_' + self.args.run_timestamp + '.pt')

    def load(self, epoch):
        self.model.load_state_dict(
            torch.load(self.args.abs_path + self.args.model_path + self.args.dataset + '_' + str(epoch) + '_' + self.args.model_name + '_' + self.args.unsupervised_approach + '_' + self.args.run_timestamp + '.pt'))

    @staticmethod
    def cal_pk(num, pre, gt):
        tmp = list(zip(gt, pre))
        tmp.sort()
        beta = []
        for i, p in enumerate(tmp):
            beta.append((p[1], p[0], i))
        beta.sort()
        limit = min(num, len(beta))
        if limit == 0:
            return 0.0
        ans = 0
        for i in range(limit):
            if beta[i][2] < num:
                ans += 1
        return ans / limit
    
    def score(self,testing_graph_set='test', test_k=100, top_k_approach='parallel'):
        assert test_k > 0
        if testing_graph_set == 'test':
            loader = self.testing_data_loader
        elif testing_graph_set == 'val':
            loader = self.validation_data_loader
        elif testing_graph_set == 'train':
            loader = DataLoader(self.training_graphs,batch_size=1,shuffle=False)
    
        print("\n\nEvalute DiffGED with {} topk {} on {} set.\n".format(top_k_approach,test_k,testing_graph_set))
        self.model.eval()
        num = 0  # total testing number
        time_usage = 0
        
        mae = []  # ged mae
        num_acc = 0  # the number of exact prediction (pre_ged == gt_ged)
        num_fea = 0  # the number of feasible prediction (pre_ged >= gt_ged)
        rho = []
        tau = []
        pk10 = []
        pk20 = []

        pres = {}
        gts = {}

        for batch in tqdm(loader,file=sys.stdout) :
            batch.to(self.device)
            gt_ged = batch.ged
            gt = gt_ged.item()
            if top_k_approach == 'parallel':
                model_out = self.diffusion_ged_parallel(batch,test_k)
            else:
                model_out = self.diffusion_ged_sequential(batch,test_k)
            
            pre_ged,running_time = model_out[0],model_out[2]
            num += 1
            time_usage += running_time

            
            i_j = batch.i_j
            i = i_j[0][0].item()
            
            if i in pres:
                pres[i].append(pre_ged)
                gts[i].append(gt)
            else:
                pres[i] = [pre_ged]
                gts[i] = [gt]
            mae.append(abs(pre_ged-gt))
            if pre_ged== gt:
                num_acc += 1
                num_fea += 1
            elif pre_ged> gt:
                num_fea += 1

        for i in pres:
            rho.append(spearmanr(pres[i],gts[i])[0])
            tau.append(kendalltau(pres[i],gts[i])[0])
            pk10.append(self.cal_pk(10, pres[i],gts[i]))
            pk20.append(self.cal_pk(20, pres[i],gts[i]))

        time_usage = round(time_usage / num, 5)
        mae = round(np.mean(mae), 3)
        acc = round(num_acc / num, 3)
        fea = round(num_fea / num, 3)
        rho = round(np.mean(rho), 3)
        tau = round(np.mean(tau), 3)
        pk10 = round(np.mean(pk10), 3)
        pk20 = round(np.mean(pk20), 3)

        self.results.append(('model_name', 'topk_approach' 'dataset', 'graph_set', '#testing_pairs', 'time_usage(s/p)', 'mae', 'acc',
                            'fea', 'rho', 'tau', 'pk10', 'pk20'))
        self.results.append((self.args.model_name, top_k_approach, self.args.dataset, testing_graph_set, num, time_usage, mae, acc,
                            fea, rho, tau, pk10, pk20))

        print(*self.results[-2], sep='\t')
        print(*self.results[-1], sep='\t')

        with open(self.args.abs_path + self.args.result_path + f'result_GEDRanker_{self.args.dataset}_{testing_graph_set}_{self.args.unsupervised_approach}_{self.args.run_timestamp}.json','w') as f:
            json.dump({'time':time_usage,'mae':mae,'acc':acc,'fea':fea,'rho':rho,'tau':tau,'pk10':pk10,'pk20':pk20},f)
    
    





           
