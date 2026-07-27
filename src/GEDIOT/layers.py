import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.pool import global_add_pool,global_mean_pool
from torch_geometric.utils import cumsum, dense_to_sparse,scatter
class GedMatrixModule(torch.nn.Module):
    def __init__(self, d, k):
        """
        :param args: Arguments object.
        """
        super(GedMatrixModule, self).__init__()

        self.d = d
        self.k = k
        self.init_weight_matrix()
        self.init_mlp()

    def init_weight_matrix(self):
        """
        Define and initilize a weight matrix of size (k, d, d).
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.k, self.d, self.d))
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def init_mlp(self):
        k = self.k
        layers = []

        layers.append(torch.nn.Linear(k, k * 2))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(k * 2, k))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(k, 1))
        # layers.append(torch.nn.Sigmoid())

        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similar matrix.
        """
        n1, d1 = embedding_1.shape
        n2, d2 = embedding_2.shape
        assert d1 == self.d == d2
        matrix = torch.bmm(torch.matmul(embedding_1, self.weight_matrix).permute(1,0,2),embedding_2.unsqueeze(-1)).squeeze(-1)
        matrix = self.mlp(matrix)

        return matrix

class AttentionModule(torch.nn.Module):
    """
    SimGNN Attention Module to make a pass on graph.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(AttentionModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.hidden_dim[-1],
                                                             self.args.hidden_dim[-1]))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding,batch):
        """
        Making a forward propagation pass to create a graph level representation.
        """
        global_context = global_mean_pool(torch.matmul(embedding, self.weight_matrix),batch)
        transformed_global = torch.tanh(global_context)
        sigmod_scores = torch.sigmoid((embedding * transformed_global[batch]).sum(dim=-1))
        representation = sigmod_scores.unsqueeze(-1) * embedding
       
        representation = global_add_pool(representation,batch)
        
        return representation

class TensorNetworkModule(torch.nn.Module):
    """
    SimGNN Tensor Network module to calculate similarity vector.
    """
    def __init__(self, args, input_dim=None):
        """
        :param args: Arguments object.
        """
        super(TensorNetworkModule, self).__init__()
        self.args = args
        self.input_dim = self.args.hidden_dim[-1] if (input_dim is None) else input_dim
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.input_dim,
                                                             self.input_dim,
                                                             self.args.tensor_neurons))

        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons,
                                                                   2*self.input_dim))
        self.bias = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        """
        batch_size = embedding_1.shape[0]
        scoring = torch.matmul(embedding_1, self.weight_matrix.view(self.input_dim, -1))
        scoring = scoring.view(batch_size, self.input_dim, -1).permute([0, 2, 1])
        scoring = torch.matmul(scoring, embedding_2.view(batch_size, self.input_dim, 1)).view(batch_size, -1)
        combined_representation = torch.cat((embedding_1, embedding_2), 1)
        block_scoring = torch.t(torch.mm(self.weight_matrix_block, torch.t(combined_representation)))
        scores = torch.relu(scoring + block_scoring + self.bias.view(-1))

        return scores

class CostMatrixModule(torch.nn.Module):
    def __init__(self, d):
        """
        :param args: Arguments object.
        """
        super(CostMatrixModule, self).__init__()

        self.d = d
        
        self.init_weight_matrix()
        

    def init_weight_matrix(self):
        
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.d, self.d))
        torch.nn.init.xavier_uniform_(self.weight_matrix)


    def forward(self, embedding_1, embedding_2):
        
        """
        Making a forward propagation pass to create a similar matrix.
        """
        n1, d1 = embedding_1.shape
        n2, d2 = embedding_2.shape
        assert d1 == self.d == d2

        matrix = (torch.matmul(embedding_1, self.weight_matrix) * embedding_2).sum(dim=-1).unsqueeze(-1)
        

        return matrix

class OTLayer(nn.Module):
    def __init__(self, max_iter: int=5):
        super(OTLayer, self).__init__()
        self.max_iter = max_iter
        self.epsilon = torch.nn.Parameter(torch.zeros(1))
        #self.epsilon = torch.zeros(1)
        self.sinkhorn = PSinkhorn()
    
    def forward(self, cost_matrix,edge_mapping_index,batch):
        K = -cost_matrix/(self.epsilon+0.05)
        match = self.sinkhorn(K,self.max_iter,edge_mapping_index,batch)
        return match

class PSinkhorn(nn.Module):
    def __init__(self):
        super().__init__()    
    def forward(self, corr,max_iter,edge_mapping_index,data,eps=1e-16):
        bs = torch.max(data.batch) + 1
        n1 = data.n[:,0]
        n2 = data.n[:,1]
        dummy_mar = torch.ones((bs),requires_grad=False,device=n1.device)
        dummy_g = (n1 < n2)

        cum_n = cumsum(n1+n2,dim=0)
        batch_edge_mapping_index = edge_mapping_index - cum_n[data.batch[edge_mapping_index[0]]]
        batch_edge_mapping_index[1] -= n1[data.batch[edge_mapping_index[0]]]
        matrix_mask = torch.zeros((bs,torch.max(n2),torch.max(n2)),dtype=bool)
        matrix_mask[data.batch[edge_mapping_index[0]],batch_edge_mapping_index[0],batch_edge_mapping_index[1]] = True
        matrix_mask[dummy_g,n1[dummy_g]]=True
        cols = torch.arange(torch.max(n2)).unsqueeze(0).to(n2.device)
        col_mask = cols >= n2.unsqueeze(1) 
        col_mask = col_mask.unsqueeze(1).expand(-1, torch.max(n2), -1) 
        matrix_mask[col_mask] = False
        
        matching_matrix = torch.zeros((bs,torch.max(n2),torch.max(n2)),device=corr.device)
        matching_matrix[data.batch[edge_mapping_index[0]],batch_edge_mapping_index[0],batch_edge_mapping_index[1]] = corr.squeeze(-1)
        
        dummy_corr = matching_matrix[matrix_mask]
        dummy_mar[dummy_g] = dummy_mar[dummy_g] + (n2[dummy_g]-n1[dummy_g]-1)
        
        dummy_n1 = n1.clone()
        dummy_n1[dummy_g] += 1
        log_prob1 = (torch.zeros(dummy_n1.sum(),device=n1.device))
        log_prob2 = (torch.zeros(n2.sum(),device=n1.device))
        log_prob1[cumsum(dummy_n1)[1:]-1] = log_prob1[cumsum(dummy_n1)[1:]-1] + torch.log(dummy_mar)
        
        log_uv = torch.zeros(dummy_n1.sum()+n2.sum(),device=n1.device)

        dummy_edge_mapping = torch.nonzero(matrix_mask).to(n1.device)
       
        dummy_edge_mapping[:,2] += dummy_n1[dummy_edge_mapping[:,0]]
        dummy_edge_mapping[:,1] += cumsum(dummy_n1+n2)[dummy_edge_mapping[:,0]]
        dummy_edge_mapping[:,2] += cumsum(dummy_n1+n2)[dummy_edge_mapping[:,0]]
        dummy_edge_mapping = dummy_edge_mapping[:,1:].transpose(0,1)
        
        row_idx = torch.unique(dummy_edge_mapping[0])
        col_idx = torch.unique(dummy_edge_mapping[1])
        
        for _ in range(max_iter):
            col = dummy_corr + log_uv[dummy_edge_mapping[0]]
            
            col_max = scatter(col,dummy_edge_mapping[1],reduce='max')
            
            col_safe = col - col_max[dummy_edge_mapping[1]]
            rse = col_max + torch.log(scatter(torch.exp(col_safe),dummy_edge_mapping[1],reduce='sum'))
            
            log_uv[col_idx] = log_prob2 - rse[col_idx]
            
            row = dummy_corr + log_uv[dummy_edge_mapping[1]]
            row_max = scatter(row,dummy_edge_mapping[0],reduce='max')
            row_safe = row - row_max[dummy_edge_mapping[0]]
            
            lse = row_max + torch.log(scatter(torch.exp(row_safe),dummy_edge_mapping[0],reduce='sum'))
            
           
            log_uv[row_idx] = log_prob1 - lse[row_idx]
        
        T = torch.exp(dummy_corr + log_uv[dummy_edge_mapping[0]] + log_uv[dummy_edge_mapping[1]])
        matching_matrix[matrix_mask] = T
        T = matching_matrix[data.batch[edge_mapping_index[0]],batch_edge_mapping_index[0],batch_edge_mapping_index[1]]
        return T.unsqueeze(-1)