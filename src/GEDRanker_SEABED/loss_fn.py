import torch
from torch_geometric.nn.pool import global_add_pool,global_mean_pool
from torch_geometric.utils import cumsum, to_dense_batch, remove_self_loops, to_dense_adj, scatter
from torch.distributions.categorical import Categorical

def mapping_loss(pred_mapping_label,data,mapping_label):
    mapping_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
   
    n1 = data.n[:,0:1]
    n2 = data.n[:,1:]
    mapping_batch = data.batch[data.edge_index_mapping[0]]
    epoch_percent = 0.5
    if epoch_percent >= 1.0:
        loss = mapping_loss(pred_mapping_label, mapping_label)
        reduce_loss = global_mean_pool(loss,mapping_batch).sum()
        return reduce_loss

    num_1 = global_add_pool(mapping_label,mapping_batch)
    
    num_0 = n1 * n2 - num_1
    
    mask_1 = (num_1 >= num_0)[mapping_batch]
    
    p_base = num_1 / num_0
    p = 1.0 - (p_base + epoch_percent * (1-p_base))
    
    mask_2 = (torch.rand_like(mapping_label).to(mapping_label.device) + mapping_label) > p[mapping_batch]

    loss_mask = (mask_1 | mask_2).squeeze(1)

    loss = mapping_loss(pred_mapping_label[loss_mask], mapping_label[loss_mask])
    reduce_loss = global_mean_pool(loss,mapping_batch[loss_mask]).sum()
    
    return reduce_loss

def roll_out_gumbel(pred_mapping_prob,batch,tau,iteration):
    bs = torch.max(batch.batch) + 1
    n1 = batch.n[:,0]
    n2 = batch.n[:,1]
    max_n1 = torch.max(n1)
    max_n2 = torch.max(n2)

    pred_matching_matrix = torch.full((bs,max_n1,max_n2),float('-inf'),device=pred_mapping_prob.device)
    mapping_edge_idx = batch.edge_index_mapping
    cum_n = cumsum(n1+n2,dim=0)
    batch_mapping_edge_idx = mapping_edge_idx - cum_n[batch.batch[mapping_edge_idx[0]]]
    batch_mapping_edge_idx[1] -= n1[batch.batch[mapping_edge_idx[0]]]
    
    pred_mapping_prob = pred_mapping_prob.squeeze(-1)
    
    gumbel_noise = torch.rand(pred_mapping_prob.shape,device=pred_mapping_prob.device)
    gumbel_noise = -torch.log(-torch.log(gumbel_noise + 1e-20) + 1e-20)
    sparse_log_alpha = (pred_mapping_prob + gumbel_noise)/tau
    
    
    for _ in range(iteration):
        row_max = scatter(sparse_log_alpha,mapping_edge_idx[0],reduce='max')
        sparse_log_alpha_safe = sparse_log_alpha - row_max[mapping_edge_idx[0]]
        lse = row_max + torch.log(scatter(torch.exp(sparse_log_alpha_safe),mapping_edge_idx[0],reduce='sum'))
        sparse_log_alpha = sparse_log_alpha - lse[mapping_edge_idx[0]]

        col_max = scatter(sparse_log_alpha,mapping_edge_idx[1],reduce='max')
        sparse_log_alpha_safe = sparse_log_alpha - col_max[mapping_edge_idx[1]]
        rse = col_max + torch.log(scatter(torch.exp(sparse_log_alpha_safe),mapping_edge_idx[1],reduce='sum'))
        sparse_log_alpha = sparse_log_alpha - rse[mapping_edge_idx[1]]
    
    pred_matching_matrix[batch.batch[mapping_edge_idx[0]],batch_mapping_edge_idx[0],batch_mapping_edge_idx[1]] = sparse_log_alpha.exp()
   
    batch_idx = torch.arange(bs,device=pred_matching_matrix.device)
    # extract node mappings
    greedy_mask = torch.zeros_like(pred_matching_matrix,dtype=torch.bool)
    solution = torch.zeros_like(pred_matching_matrix,dtype=torch.bool)
    terminates = (torch.sum(solution.view(bs,-1),dim=-1) != torch.min(batch.n,dim=1)[0])
    
    
    for s in range(min(max_n1,max_n2)):
        pred_matching_matrix = pred_matching_matrix.view(bs,-1)
        argmax_result = torch.argmax(pred_matching_matrix,dim=-1)
        rows = argmax_result // max_n2
        columns = argmax_result % max_n2
        solution[batch_idx[terminates],rows[terminates],columns[terminates]] = True

        greedy_mask[batch_idx[terminates],rows[terminates],:] = True
        greedy_mask[batch_idx[terminates],:,columns[terminates]] = True
        pred_matching_matrix = pred_matching_matrix.view(bs,max_n1,max_n2)
        
        pred_matching_matrix[greedy_mask] = float('-inf')
        terminates = (torch.sum(solution.view(bs,-1),dim=-1) != torch.min(batch.n,dim=1)[0])

    solution = torch.cat([solution,torch.zeros(bs,max_n2-max_n1,max_n2,device=solution.device)],dim=1)
    zeros_column = torch.where(~torch.any(solution == 1, dim=1))
    zeros_rows = torch.where(~torch.any(solution == 1, dim=-1))
    solution[zeros_column[0],zeros_rows[1],zeros_column[1]] = 1
    extracted_mapping = torch.nonzero(solution)

    mask = n2[extracted_mapping[:,0]]
    
    mapping_mask = ~ ((extracted_mapping[:,1] >= mask) | (extracted_mapping[:,2] >= mask))
    extracted_mapping_reduced = extracted_mapping[mapping_mask]

    x1 = batch.x[(batch.x_indicator==0).squeeze(1)]
    x2 = batch.x[(batch.x_indicator==1).squeeze(1)]
    dense_x1,x1_mask = to_dense_batch(x1,batch.batch[(batch.x_indicator==0).squeeze(1)],max_num_nodes=max_n2)
    dense_x2,x2_mask = to_dense_batch(x2,batch.batch[(batch.x_indicator==1).squeeze(1)],max_num_nodes=max_n2)
    permuted_x2,permuted_x2_mask = to_dense_batch((dense_x2[extracted_mapping_reduced[:,0],extracted_mapping_reduced[:,2]]),batch.batch[(batch.x_indicator==1).squeeze(1)],max_num_nodes=max_n2)
    
    edge1 = batch.edge_index[:,(batch.x_indicator[batch.edge_index[0]]==0).squeeze(1)]
    edge1 = remove_self_loops(edge1)[0]
    edge1_batch = batch.batch[edge1[0]]
    target_start_idx = edge1_batch * max_n2
    current_start_idx = cum_n[edge1_batch]
    edge1 = edge1 - current_start_idx + target_start_idx
    edge_dense_batch = torch.tensor([[i] * max_n2 for i in range(bs)],device=edge1.device).view(-1)
    dense_adj_1 = to_dense_adj(edge_index=edge1,batch=edge_dense_batch,max_num_nodes=max_n2)
    reversed_mapping = torch.tensor(sorted(extracted_mapping.tolist(),key=lambda x:(x[0],x[2])),device=extracted_mapping.device)

    edge2 = batch.edge_index[:,(batch.x_indicator[batch.edge_index[0]]==1).squeeze(1)]
    edge2_batch = batch.batch[edge2[0]]
    target_start_idx = edge2_batch * max_n2
    current_start_idx  = cum_n[edge2_batch] + n1[edge2_batch]
    edge2 = edge2 - current_start_idx + target_start_idx
    reversed_mapping[:,2] += reversed_mapping[:,0] * max_n2
    reversed_mapping[:,1] += reversed_mapping[:,0] * max_n2
    
    edge2[0] = reversed_mapping[edge2[0],1]
    edge2[1] = reversed_mapping[edge2[1],1]
   
    dense_adj_2 = to_dense_adj(remove_self_loops(edge2)[0],batch=edge_dense_batch,max_num_nodes=max_n2)
    # ged = difference in G and G' (after permuting G')
    adj_diff = torch.abs(dense_adj_1-dense_adj_2).view(bs,-1).sum(dim=-1) // 2
    feat_diff = torch.sum(~torch.all(dense_x1 == permuted_x2,dim=-1),dim=-1)
    ged = adj_diff + feat_diff
    solution_sparse = solution[batch.batch[mapping_edge_idx[0]],batch_mapping_edge_idx[0],batch_mapping_edge_idx[1]]
    
    return ged,solution_sparse.unsqueeze(-1),sparse_log_alpha.exp().unsqueeze(-1)#solution_sparse.unsqueeze(-1)

    
def roll_out(pred_mapping_prob,batch):
    
    bs = torch.max(batch.batch) + 1
    n1 = batch.n[:,0]
    n2 = batch.n[:,1]
    max_n1 = torch.max(n1)
    max_n2 = torch.max(n2)
    pred_matching_matrix = torch.full((bs,max_n1,max_n2),float('-inf'),device=pred_mapping_prob.device)
    mapping_edge_idx = batch.edge_index_mapping
    graph_edge_idx = batch.edge_index
    cum_n = cumsum(n1+n2,dim=0)
    batch_mapping_edge_idx = mapping_edge_idx - cum_n[batch.batch[mapping_edge_idx[0]]]
    batch_mapping_edge_idx[1] -= n1[batch.batch[mapping_edge_idx[0]]]
    
    pred_matching_matrix[batch.batch[mapping_edge_idx[0]],batch_mapping_edge_idx[0],batch_mapping_edge_idx[1]] = pred_mapping_prob.squeeze(-1)
    batch_idx = torch.arange(bs,device=pred_matching_matrix.device)
    # extract node mappings
    greedy_mask = torch.zeros_like(pred_matching_matrix,dtype=torch.bool)
    solution = torch.zeros_like(pred_matching_matrix,dtype=torch.bool)
    terminates = (torch.sum(solution.view(bs,-1),dim=-1) != torch.min(batch.n,dim=1)[0])
    
    
    for s in range(min(max_n1,max_n2)):
        pred_matching_matrix = pred_matching_matrix.view(bs,-1)
        argmax_result = torch.argmax(pred_matching_matrix,dim=-1)
        
    
        rows = argmax_result // max_n2
        columns = argmax_result % max_n2
       
        
        solution[batch_idx[terminates],rows[terminates],columns[terminates]] = True

        greedy_mask[batch_idx[terminates],rows[terminates],:] = True
        greedy_mask[batch_idx[terminates],:,columns[terminates]] = True
        pred_matching_matrix = pred_matching_matrix.view(bs,max_n1,max_n2)
        
        pred_matching_matrix[greedy_mask] = float('-inf')
        terminates = (torch.sum(solution.view(bs,-1),dim=-1) != torch.min(batch.n,dim=1)[0])

    solution = torch.cat([solution,torch.zeros(bs,max_n2-max_n1,max_n2,device=solution.device)],dim=1)
    zeros_column = torch.where(~torch.any(solution == 1, dim=1))
    zeros_rows = torch.where(~torch.any(solution == 1, dim=-1))
    solution[zeros_column[0],zeros_rows[1],zeros_column[1]] = 1
    extracted_mapping = torch.nonzero(solution)

    mask = n2[extracted_mapping[:,0]]
    
    mapping_mask = ~ ((extracted_mapping[:,1] >= mask) | (extracted_mapping[:,2] >= mask))
    extracted_mapping_reduced = extracted_mapping[mapping_mask]

    x1 = batch.x[(batch.x_indicator==0).squeeze(1)]
    x2 = batch.x[(batch.x_indicator==1).squeeze(1)]
    dense_x1,x1_mask = to_dense_batch(x1,batch.batch[(batch.x_indicator==0).squeeze(1)],max_num_nodes=max_n2)
    dense_x2,x2_mask = to_dense_batch(x2,batch.batch[(batch.x_indicator==1).squeeze(1)],max_num_nodes=max_n2)
    permuted_x2,permuted_x2_mask = to_dense_batch((dense_x2[extracted_mapping_reduced[:,0],extracted_mapping_reduced[:,2]]),batch.batch[(batch.x_indicator==1).squeeze(1)],max_num_nodes=max_n2)
    
    edge1 = batch.edge_index[:,(batch.x_indicator[batch.edge_index[0]]==0).squeeze(1)]
    edge1 = remove_self_loops(edge1)[0]
    edge1_batch = batch.batch[edge1[0]]
    target_start_idx = edge1_batch * max_n2
    current_start_idx = cum_n[edge1_batch]
    edge1 = edge1 - current_start_idx + target_start_idx
    edge_dense_batch = torch.tensor([[i] * max_n2 for i in range(bs)],device=edge1.device).view(-1)
    dense_adj_1 = to_dense_adj(edge_index=edge1,batch=edge_dense_batch,max_num_nodes=max_n2)
    reversed_mapping = torch.tensor(sorted(extracted_mapping.tolist(),key=lambda x:(x[0],x[2])),device=extracted_mapping.device)

    edge2 = batch.edge_index[:,(batch.x_indicator[batch.edge_index[0]]==1).squeeze(1)]
    edge2_batch = batch.batch[edge2[0]]
    target_start_idx = edge2_batch * max_n2
    current_start_idx  = cum_n[edge2_batch] + n1[edge2_batch]
    edge2 = edge2 - current_start_idx + target_start_idx
    reversed_mapping[:,2] += reversed_mapping[:,0] * max_n2
    reversed_mapping[:,1] += reversed_mapping[:,0] * max_n2
    
    edge2[0] = reversed_mapping[edge2[0],1]
    edge2[1] = reversed_mapping[edge2[1],1]
   
    dense_adj_2 = to_dense_adj(remove_self_loops(edge2)[0],batch=edge_dense_batch,max_num_nodes=max_n2)
    # ged = difference in G and G' (after permuting G')
    adj_diff = torch.abs(dense_adj_1-dense_adj_2).view(bs,-1).sum(dim=-1) // 2
    feat_diff = torch.sum(~torch.all(dense_x1 == permuted_x2,dim=-1),dim=-1)
    ged = adj_diff + feat_diff
    solution_sparse = solution[batch.batch[mapping_edge_idx[0]],batch_mapping_edge_idx[0],batch_mapping_edge_idx[1]]
    return ged,solution_sparse.unsqueeze(-1)

def bpr_loss(D_pred_ged_curr,D_pred_ged_best,D_pred_ged_last,normalize_curr_ged,normalize_best_ged,normalize_last_ged):
    ranking_loss1 = -(((D_pred_ged_curr - D_pred_ged_best)[normalize_curr_ged >= normalize_best_ged]).sigmoid()+1e-20).log().sum() - (((D_pred_ged_best - D_pred_ged_curr)[normalize_curr_ged <= normalize_best_ged]).sigmoid()+1e-20).log().sum()
    ranking_loss2 = -(((D_pred_ged_curr - D_pred_ged_last)[normalize_curr_ged >= normalize_last_ged]).sigmoid()+1e-20).log().sum() - (((D_pred_ged_last - D_pred_ged_curr)[normalize_curr_ged <= normalize_last_ged]).sigmoid()+1e-20).log().sum()
    D_loss = ranking_loss1 + ranking_loss2 
    return D_loss

def hinge_loss(D_pred_ged_curr,D_pred_ged_best,D_pred_ged_last,normalize_curr_ged,normalize_best_ged,normalize_last_ged):
    loss1 = torch.nn.MarginRankingLoss(reduction='sum',margin=1)
    loss2 = torch.nn.MarginRankingLoss(reduction='sum',margin=0)
    ranking_loss1 = loss1(D_pred_ged_curr[normalize_curr_ged > normalize_best_ged],D_pred_ged_best[normalize_curr_ged > normalize_best_ged],target = torch.ones(D_pred_ged_curr[normalize_curr_ged > normalize_best_ged].shape[0],device=D_pred_ged_curr.device)) + loss1(D_pred_ged_best[normalize_curr_ged < normalize_best_ged],D_pred_ged_curr[normalize_curr_ged < normalize_best_ged],target = torch.ones(D_pred_ged_curr[normalize_curr_ged < normalize_best_ged].shape[0],device=D_pred_ged_curr.device))
    ranking_loss2 = loss2(D_pred_ged_curr[normalize_curr_ged == normalize_best_ged],D_pred_ged_best[normalize_curr_ged == normalize_best_ged],target = torch.ones(D_pred_ged_curr[normalize_curr_ged == normalize_best_ged].shape[0],device=D_pred_ged_curr.device)) + loss2(D_pred_ged_best[normalize_curr_ged == normalize_best_ged],D_pred_ged_curr[normalize_curr_ged == normalize_best_ged],target = torch.ones(D_pred_ged_curr[normalize_curr_ged == normalize_best_ged].shape[0],device=D_pred_ged_curr.device))
    
    ranking_loss3 = loss1(D_pred_ged_curr[normalize_curr_ged > normalize_last_ged],D_pred_ged_last[normalize_curr_ged > normalize_last_ged],target = torch.ones(D_pred_ged_curr[normalize_curr_ged > normalize_last_ged].shape[0],device=D_pred_ged_curr.device)) + loss1(D_pred_ged_last[normalize_curr_ged < normalize_last_ged],D_pred_ged_curr[normalize_curr_ged < normalize_last_ged],target = torch.ones(D_pred_ged_curr[normalize_curr_ged < normalize_last_ged].shape[0],device=D_pred_ged_curr.device))
    ranking_loss4 = loss2(D_pred_ged_curr[normalize_curr_ged == normalize_last_ged],D_pred_ged_last[normalize_curr_ged == normalize_last_ged],target = torch.ones(D_pred_ged_curr[normalize_curr_ged == normalize_last_ged].shape[0],device=D_pred_ged_curr.device)) + loss2(D_pred_ged_last[normalize_curr_ged == normalize_last_ged],D_pred_ged_curr[normalize_curr_ged == normalize_last_ged],target = torch.ones(D_pred_ged_curr[normalize_curr_ged == normalize_last_ged].shape[0],device=D_pred_ged_curr.device))
    D_loss = ranking_loss1 + ranking_loss2 + ranking_loss3 + ranking_loss4
    return D_loss





    
