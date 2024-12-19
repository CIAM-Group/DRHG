import torch
import numpy as np
import math

def assemble_solution_for_sorted_problem_batch(destruction_mask, endpoint_mask, reduced_solution, new_problem_index_on_sorted_problem, padding_mask):
    batch_size = destruction_mask.size(0)
    problem_size = destruction_mask.size(1)
    reduced_problem_size = reduced_solution.size(1)

    destruction_mask_flattened = destruction_mask.flatten()
    endpoint_mask_flattened_no_padding = endpoint_mask[~padding_mask].flatten()
    reduced_problem_size_no_padding = (~padding_mask).sum(dim=1)
    padding_length = padding_mask.sum(dim=1)
    reduced_problem_size_total = endpoint_mask_flattened_no_padding.size(0)
    
    length_to_add = torch.cumsum(torch.concat([torch.tensor([0]), reduced_problem_size_no_padding[:-1]]), dim=0)
    reduced_solution_flattened_no_padding = (reduced_solution + length_to_add[:,None] - padding_length[:,None])[~padding_mask].flatten() #
    endpoint_mask_after_repair_flattened_no_padding = endpoint_mask[~padding_mask][reduced_solution_flattened_no_padding]  

    # length of segments before repair
    endpoints_ids_on_sorted_problem = torch.nonzero(destruction_mask_flattened==2).squeeze()
    
    left_endpoints = endpoints_ids_on_sorted_problem[::2]
    right_endpoints = endpoints_ids_on_sorted_problem[1::2]
    segment_length_before_repair = right_endpoints - left_endpoints + 1

    # segment order & length after repair
    endpoints_ids_after_repair_from_1 = \
        endpoint_mask_flattened_no_padding.cumsum(dim=0)[reduced_solution_flattened_no_padding][endpoint_mask_after_repair_flattened_no_padding] # 从1开始

    segment_order_after_repair = ((endpoints_ids_after_repair_from_1[::2]-0.5) / 2).floor().long()

    segment_length_after_repair = segment_length_before_repair.gather(index=segment_order_after_repair, dim=0) # 
    inner_len_aft_repair = segment_length_after_repair - 2

    # destruction_mask_after_repair
    destruction_mask_after_repair_flattened = torch.ones((batch_size*problem_size,), dtype=int)
    endpoints_ids_in_new_destruction_mask = torch.arange(reduced_problem_size_total)[endpoint_mask_after_repair_flattened_no_padding]
    inner_len_aft_repair_cum = inner_len_aft_repair.cumsum(dim=0)
    endpoints_ids_in_new_destruction_mask[2::2] += inner_len_aft_repair_cum[:-1]  # 左端点
    endpoints_ids_in_new_destruction_mask[1::2] += inner_len_aft_repair_cum  # 右端点
    destruction_mask_after_repair_flattened[endpoints_ids_in_new_destruction_mask] = 2
    # 中间点
    segment_num = segment_length_before_repair.size(0)
    # print(segment_length_before_repair.size())
    if segment_num == 0: # 没有中间点，直接返回 reduced solution
        destruction_mask_after_repair = destruction_mask.gather(1, reduced_solution)
        complete_solution_on_sorted_problem = reduced_solution
        print('no inner point')
        return(destruction_mask_after_repair, complete_solution_on_sorted_problem)
    else: 
        max_inner_len = segment_length_before_repair.max() - 2

    inner_point_ids = torch.arange(max_inner_len)[None,:].repeat((segment_num,1)) + endpoints_ids_in_new_destruction_mask[::2][:, None] + 1
    inner_mask = torch.arange(max_inner_len)[None,:] < inner_len_aft_repair[:,None]
    inner_point_ids = inner_point_ids[inner_mask]
    destruction_mask_after_repair_flattened[inner_point_ids] = 0
    # reshape 到 batch
    destruction_mask_after_repair = destruction_mask_after_repair_flattened.reshape((batch_size,-1))

    # 按照 destruction_mask_of_repaired_solution, 把 endpoint + isopoint 的地方填上 solution, 再把 inner point 的地方填上solution
    # 先构造 inner part
    complete_solution_on_sorted_problem_flattened = torch.zeros((batch_size*problem_size,), dtype=int)
    reversed_segment_mask = endpoints_ids_after_repair_from_1[::2] > endpoints_ids_after_repair_from_1[1::2]
    
    left_endpoints_ids_on_sorted_problem  = endpoints_ids_on_sorted_problem[::2]
    right_endpoints_ids_on_sorted_problem = endpoints_ids_on_sorted_problem[1::2]
    inner_part_ids_forward  = torch.arange(max_inner_len)[None,:].repeat((segment_num,1)) + left_endpoints_ids_on_sorted_problem[:, None] + 1
    inner_part_ids_backward = - torch.arange(max_inner_len)[None,:].repeat((segment_num,1)) + right_endpoints_ids_on_sorted_problem[:, None] - 1
    inner_part_after_repair = inner_part_ids_forward[segment_order_after_repair,:]
    inner_part_after_repair[reversed_segment_mask,:] = inner_part_ids_backward[segment_order_after_repair,:][reversed_segment_mask,:]
    inner_part_after_repair = inner_part_after_repair[inner_mask]
    # 组装
    new_problem_index_on_flattened_problem =  (new_problem_index_on_sorted_problem + torch.arange(batch_size)[:,None] * problem_size)[~padding_mask]
    reduced_solution_indexed_by_sorted_problem_flattened = new_problem_index_on_flattened_problem[reduced_solution_flattened_no_padding]
    complete_solution_on_sorted_problem_flattened[destruction_mask_after_repair_flattened>0] = reduced_solution_indexed_by_sorted_problem_flattened # 没有+batch_id*problem_size
    complete_solution_on_sorted_problem_flattened[destruction_mask_after_repair_flattened==0] = inner_part_after_repair
    complete_solution_on_sorted_problem = complete_solution_on_sorted_problem_flattened.reshape((batch_size,-1))
    complete_solution_on_sorted_problem -= torch.arange(batch_size)[:,None] * problem_size

    return(destruction_mask_after_repair, complete_solution_on_sorted_problem)


def get_edges(solution): 
    sol_len = len(solution)
    edges = [(solution[i], solution[i+1]) for i in range(sol_len-1)] + [(solution[i+1], solution[i]) for i in range(sol_len-1)]
    edges = edges + [(solution[0], solution[-1]), (solution[-1], solution[-0])]
    return edges

def get_edges_left_to_right(solution):
    sol_len = len(solution)
    edges = [(solution[i], solution[i+1]) for i in range(sol_len-1)] 
    edges = edges + [(solution[-1], solution[-0])]
    return edges


# def destroy_by_edge(problem, solution, solution_opt):
#     # TODO: 写成torch版本
#     # 没有batch维度
#     # return: the same as sampling_reduced_problem
#     edges = get_edges(solution)
#     edges_opt = get_edges(solution_opt)
#     # print(edges)
#     common_edges = set(edges) & set(edges_opt)

#     sorted_problems = problem[solution,:]
#     # solution = torch.cat([solution, solution[0]])    
#     solution = np.append(solution, solution[0])
#     n = len(solution)-1

#     # 先计算destruction mask: 0在中间, 2是端点, 1是散点
#     # 先默认solution[0]和solution[-1]是断开的
#     destruction_mask = np.arange(n) # 按解的顺序来index的
#     # destruction_mask[0] = 2
#     in_flag = False
#     for i in range(n-1):
#         if (solution[i], solution[i+1]) in common_edges:
#             if i == n-2: # 最后一个点是端点
#                 destruction_mask[i+1] = 2
#             if not in_flag:
#                 in_flag = True
#                 destruction_mask[i] = 2
#             else: 
#                 destruction_mask[i] = 0
#         else: 
#             if i == n-2: # 最后一个点是散点
#                 destruction_mask[i+1] = 1
#             if in_flag:
#                 in_flag = False
#                 destruction_mask[i] = 2
#             else:
#                 destruction_mask[i] = 1


#     # 根据 destruction mask 构造reduced_problem
#     reduced_problems = sorted_problems[destruction_mask>0, :]
#     endpoint_mask = (destruction_mask[destruction_mask>0] - 1).astype(bool)

#     # 端点的另一个端点；如果不是端点，赋值为-1; 
#     another_endpoint = np.ones(endpoint_mask.shape, dtype=int) * -1 
#     index = np.arange(endpoint_mask.shape[0], dtype=int)

#     cumulative_sum = np.cumsum(endpoint_mask.astype(int), axis=0) 
#     is_left_endpoint = np.zeros(endpoint_mask.shape[0], dtype=bool)
#     is_left_endpoint[(cumulative_sum % 2)==1] = 1
#     is_right_endpoint = np.roll(is_left_endpoint, shift=1)  

#     another_endpoint[is_left_endpoint]  = index[is_left_endpoint] + 1
#     another_endpoint[is_right_endpoint] = index[is_right_endpoint] - 1
#     point_couples = np.tile(index,(2,1)).T              
#     point_couples[is_left_endpoint, 1]  = another_endpoint[is_left_endpoint]  
#     point_couples[is_right_endpoint, 1] = another_endpoint[is_right_endpoint]      

#     return destruction_mask, reduced_problems, endpoint_mask, another_endpoint, point_couples

# def destroy_perfect_batch(problems, solution, solution_opt):
#     # retrun destruction_mask, sorted problems, shift_ist_by_ist

#     batch_size = solution.size(0)
#     problem_size = solution.size(1)

   
#     index = solution[:,:,None].repeat(1,1,2)
#     sorted_problems = torch.gather(problems, dim=1, index=index)

#     # 将 solution_opt 换到 solution 排序的 problem 下
#     idx = torch.argsort(solution, dim=1)
#     solution_opt_on_sorted = idx.gather(index=solution_opt, dim=1).cpu().numpy()
#     solution_on_sorted = np.broadcast_to(np.arange(problem_size),(batch_size, problem_size))

#     edges_l2r_list = [set(get_edges_left_to_right(solution_on_sorted[i])) for i in range(batch_size)]
#     edges_opt_list = [set(get_edges(solution_opt_on_sorted[i])) for i in range(batch_size)]

#     # destroy by edge
#     edges_to_destroy = [np.array(list(edges_l2r_list[i] - edges_opt_list[i])) 
#                         if len(edges_l2r_list[i] - edges_opt_list[i]) !=0 
#                         else np.array(list(edges_l2r_list[i])[0]).reshape((1,2)) # 两个解完全一样，随机 destroy 一条边
#                         for i in range(batch_size)] 
#     num_edges_to_destroy = [edges_to_destroy[i].shape[0] for i in range(batch_size)]
#     # print([(i, edges_to_destroy[i]) for i in range(batch_size) if len(edges_to_destroy[i]) < 5])
#     # print('bbbbbbbbbbbbbb')
#     # print(edges_to_destroy[0][0,0])
#     # print(edges_to_destroy[78][0,0])
#     # print([(i, np.array(list(edges_l2r_list[i] - edges_opt_list[i]))) for i in range(batch_size) if len(edges_l2r_list[i] - edges_opt_list[i])==0])
                        
#     shift_ist_by_ist = torch.tensor([edges_to_destroy[i][0,0]+1 for i in range(batch_size)])

#     shifted_index = torch.arange(problem_size)[None, :, None].repeat((batch_size, 1, 2)) + shift_ist_by_ist[:, None, None]
#     shifted_index = torch.where(shifted_index >= problem_size, shifted_index-problem_size, shifted_index)
#     sorted_problems = sorted_problems.gather(index=shifted_index, dim=1)

#     destroyed_edge_left_node  = [torch.concat([torch.ones((num_edges_to_destroy[i], 1)) * i, 
#                                                 (torch.tensor(edges_to_destroy[i][:, 0]) - shift_ist_by_ist[i]).unsqueeze(-1)], dim=1) # (-1) - 0 一定是断开的
#                                 for i in range(batch_size)] 
#     destroyed_edge_left_node  = torch.concat(destroyed_edge_left_node, dim=0).long()
#     destroyed_edge_left_node_mask  = torch.zeros((batch_size, problem_size), dtype=bool)
#     destroyed_edge_left_node_mask[destroyed_edge_left_node[:, 0], destroyed_edge_left_node[:, 1]] = 1
#     destroyed_edge_right_node_mask = torch.roll(destroyed_edge_left_node_mask, shifts=1, dims=1)

#     # mask 对应 shift 之后的 problem

#     destruction_mask = torch.zeros((batch_size, problem_size), dtype=int)  
#     destruction_mask[destroyed_edge_left_node_mask & destroyed_edge_right_node_mask] = 1 
#     destruction_mask[destroyed_edge_left_node_mask ^ destroyed_edge_right_node_mask] = 2 

#     return(destruction_mask, sorted_problems, shift_ist_by_ist)
    

# def augment_xy_data_by_8_fold(problems):
#     # problems.shape: (batch, problem, 2)

#     x = problems[:, :, [0]]
#     y = problems[:, :, [1]]
#     # x,y shape: (batch, problem, 1)

#     dat1 = torch.cat((x, y), dim=2)
#     dat2 = torch.cat((1 - x, y), dim=2)
#     dat3 = torch.cat((x, 1 - y), dim=2)
#     dat4 = torch.cat((1 - x, 1 - y), dim=2)
#     dat5 = torch.cat((y, x), dim=2)
#     dat6 = torch.cat((1 - y, x), dim=2)
#     dat7 = torch.cat((y, 1 - x), dim=2)
#     dat8 = torch.cat((1 - y, 1 - x), dim=2)

#     aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
#     # shape: (8*batch, problem, 2)

#     return aug_problems

def compare_to_optimal(solution_to_compare, optimal_solution):   
    batch_size = solution_to_compare.size(0)
    problem_size = solution_to_compare.size(1)
    index_0_compare = torch.argmin(solution_to_compare, dim=1)
    index_0_optimal = torch.argmin(optimal_solution, dim=1)
    shifted_index_compare = torch.arange(problem_size)[None,:].repeat((batch_size, 1)) + index_0_compare[:, None]
    shifted_index_compare = torch.where(shifted_index_compare>=problem_size, shifted_index_compare-problem_size, shifted_index_compare)
    shifted_index_optimal = torch.arange(problem_size)[None,:].repeat((batch_size, 1)) + index_0_optimal[:, None]
    shifted_index_optimal = torch.where(shifted_index_optimal>=problem_size, shifted_index_optimal-problem_size, shifted_index_optimal)

    shifted_solution_to_compare = solution_to_compare.gather(index=shifted_index_compare, dim=1)
    shifted_solution_optimal    = optimal_solution.gather(index=shifted_index_optimal, dim=1)
    shifted_solution_optimal_flipped = torch.roll(shifted_solution_optimal.flip(dims=(1,)), shifts=1, dims=1)

    equal_1 = (shifted_solution_optimal == shifted_solution_to_compare).all(dim=1)
    equal_2 = (shifted_solution_optimal_flipped == shifted_solution_to_compare).all(dim=1)
    is_optimal = equal_1 | equal_2

    return(is_optimal)

# def augment_by_rotation(problems):
#     batch_size = problems.shape[0]

#     problems = problems - 0.5
#     theta = torch.atan2(problems[:, :, 1], problems[:, :, 0])
#     rho = torch.linalg.norm(problems, dim=2)
#     rotation = torch.rand(batch_size) * 2 * math.pi
#     # rotation
#     theta = theta + rotation.unsqueeze(-1).expand_as(theta)

#     # symm
#     symmetry = torch.rand(batch_size).unsqueeze(-1).expand_as(theta) > 0.5
#     theta[symmetry] = -theta[symmetry]

#     # shrink
#     rho = rho * (torch.rand_like(problems[:, 0, 0])[:, None].expand_as(rho) * 0.6 + 0.7)

#     # recover
#     x = rho * torch.cos(theta) + 0.5
#     y = rho * torch.sin(theta) + 0.5
#     problems = torch.stack([x, y], dim=-1)

#     # # noise
#     # problems1 = problems.unsqueeze(1).expand(-1, problems.size(1), -1, -1)
#     # problems2 = problems.unsqueeze(2).expand(-1, -1, problems.size(1), -1)
#     # dist = torch.linalg.norm(problems1 - problems2, dim=3)
#     # # mask 0 in diagonal
#     # dist[dist == 0] = 500.
#     # noise_threshold = dist.min(-1)[0].min(-1)[0].unsqueeze(-1).expand_as(x)

#     # theta_ = torch.rand_like(x) * 2 * math.pi
#     # rho_ = torch.rand_like(x)
#     # x_noise = rho_ * torch.cos(theta_)
#     # y_noise = rho_ * torch.sin(theta_)

#     # problems = torch.stack([
#     #     x + x_noise * noise_threshold,
#     #     y + y_noise * noise_threshold,
#     # ], dim=-1)
#     return problems