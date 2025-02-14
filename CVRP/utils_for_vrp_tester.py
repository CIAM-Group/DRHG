import torch
def assemble_vrp_solution_for_sorted_problem_batch(destruction_mask, endpoint_mask, reduced_solution, new_problem_index_on_sorted_problem, padding_mask):
    # INPUT:
    #  destruction_mask, [batch, problem + 1]
    #  endpoint_mask, [batch, reduced_problem_size + 1]
    #  padding_mask,  [batch, reduced_problem_size + 1] padding depot
    #  new_problem_index_on_sorted_problem, [batch, reduced_problem_size + 1] # depot 0
    #  reduced_solution, [batch, reduced_problem_size, 2]

    batch_size = destruction_mask.size(0)
    problem_size = destruction_mask.size(1) - 1

    # get rid of depot
    reduced_solution_node  = reduced_solution[:, :, 0] - 1  
    reduced_solution_flag  = reduced_solution[:, :, 1]
    endpoint_mask_wo_depot = endpoint_mask[:, 1:]
    padding_mask_wo_depot  = padding_mask[:, :-1] # 
    destruction_mask_wo_depot = destruction_mask[:, 1:]

    destruction_mask_flattened = destruction_mask_wo_depot.flatten()
    endpoint_mask_flattened_no_padding = endpoint_mask_wo_depot[~padding_mask_wo_depot].flatten()
    reduced_problem_size_no_padding = (~padding_mask_wo_depot).sum(dim=1)
    padding_length = padding_mask_wo_depot.sum(dim=1)
    reduced_problem_size_total = endpoint_mask_flattened_no_padding.size(0)
    
    length_to_add = torch.cumsum(torch.concat([torch.tensor([0]), reduced_problem_size_no_padding[:-1]]), dim=0)
    reduced_solution_flattened_no_padding = (reduced_solution_node + length_to_add[:,None] - padding_length[:,None])[~padding_mask_wo_depot].flatten() 
    endpoint_mask_after_repair_flattened_no_padding = endpoint_mask_wo_depot[~padding_mask_wo_depot][reduced_solution_flattened_no_padding]  

    # length of segments before repair
    endpoints_ids_on_sorted_problem = torch.nonzero(destruction_mask_flattened==2).squeeze()
    
    left_endpoints = endpoints_ids_on_sorted_problem[::2]
    right_endpoints = endpoints_ids_on_sorted_problem[1::2]
    try: 
        segment_length_before_repair = right_endpoints - left_endpoints + 1
    except:
        print(destruction_mask[:,0].long())
        endpoints_in_destruction_mask = destruction_mask_wo_depot.gt(1.5)
        destruction_mask_sum = endpoints_in_destruction_mask.sum(dim=1)
        print(destruction_mask_sum)
        print(torch.nonzero(destruction_mask_sum%2 != 0))
        raise ValueError('wrong segment length')

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
    endpoints_ids_in_new_destruction_mask[2::2] += inner_len_aft_repair_cum[:-1]  # left endpoint
    endpoints_ids_in_new_destruction_mask[1::2] += inner_len_aft_repair_cum  # right endpoint
    destruction_mask_after_repair_flattened[endpoints_ids_in_new_destruction_mask] = 2

    # middle node
    segment_num = segment_length_before_repair.size(0)
    if segment_num == 0: # no middle node，return reduced solution
        destruction_mask_after_repair = destruction_mask_wo_depot.gather(1, reduced_solution_node)
        complete_solution_on_sorted_problem = reduced_solution_node + 1
        complete_flag_on_sorted_problem = reduced_solution_flag
        print('no inner point')
        return(destruction_mask_after_repair, complete_solution_on_sorted_problem, complete_flag_on_sorted_problem)
    else: 
        max_inner_len = segment_length_before_repair.max() - 2

    inner_point_ids = torch.arange(max_inner_len)[None,:].repeat((segment_num,1)) + endpoints_ids_in_new_destruction_mask[::2][:, None] + 1
    inner_mask = torch.arange(max_inner_len)[None,:] < inner_len_aft_repair[:,None]
    inner_point_ids = inner_point_ids[inner_mask]
    destruction_mask_after_repair_flattened[inner_point_ids] = 0
    destruction_mask_after_repair = destruction_mask_after_repair_flattened.reshape((batch_size,-1))

    # inner part
    reversed_segment_mask = endpoints_ids_after_repair_from_1[::2] > endpoints_ids_after_repair_from_1[1::2] 
    left_endpoints_ids_on_sorted_problem  = endpoints_ids_on_sorted_problem[::2]
    right_endpoints_ids_on_sorted_problem = endpoints_ids_on_sorted_problem[1::2]
    inner_part_ids_forward  = torch.arange(max_inner_len)[None,:].repeat((segment_num,1)) + left_endpoints_ids_on_sorted_problem[:, None] + 1
    inner_part_ids_backward = - torch.arange(max_inner_len)[None,:].repeat((segment_num,1)) + right_endpoints_ids_on_sorted_problem[:, None] - 1
    inner_part_after_repair = inner_part_ids_forward[segment_order_after_repair,:]
    inner_part_after_repair[reversed_segment_mask,:] = inner_part_ids_backward[segment_order_after_repair,:][reversed_segment_mask,:]
    inner_part_after_repair = inner_part_after_repair[inner_mask]

    # assemble solution_node
    new_problem_index_on_flattened_problem =  (new_problem_index_on_sorted_problem - 1 + torch.arange(batch_size)[:,None] * problem_size)[new_problem_index_on_sorted_problem>0]
    reduced_solution_indexed_by_sorted_problem_flattened = new_problem_index_on_flattened_problem[reduced_solution_flattened_no_padding]
    complete_solution_on_sorted_problem_flattened = torch.zeros((batch_size*problem_size,), dtype=int)
    complete_solution_on_sorted_problem_flattened[destruction_mask_after_repair_flattened>0] = reduced_solution_indexed_by_sorted_problem_flattened 
    complete_solution_on_sorted_problem_flattened[destruction_mask_after_repair_flattened==0] = inner_part_after_repair
    complete_solution_on_sorted_problem = complete_solution_on_sorted_problem_flattened.reshape((batch_size,-1)) + 1
    complete_solution_on_sorted_problem -= torch.arange(batch_size)[:,None] * problem_size

    # assemble solution_flag
    reduced_flag_indexed_by_sorted_problem_flattened = reduced_solution_flag[~padding_mask_wo_depot]
    complete_flag_on_sorted_problem_flattened = torch.zeros((batch_size*problem_size,), dtype=int)
    complete_flag_on_sorted_problem_flattened[destruction_mask_after_repair_flattened>0] = reduced_flag_indexed_by_sorted_problem_flattened
    complete_flag_on_sorted_problem = complete_flag_on_sorted_problem_flattened.reshape((batch_size,-1))

    return(destruction_mask_after_repair, complete_solution_on_sorted_problem,  complete_flag_on_sorted_problem)


def valid_partial_reduced_solution_ist(problem_demand, endpoint_mask, solution_node, solution_flag, capacity): 
    demand_order_by_solution = problem_demand[solution_node]
    endpoint_mask_order_by_solution = endpoint_mask[solution_node]
    second_endpoint_mask = torch.concat([torch.zeros([1]).bool(), (torch.cumsum(endpoint_mask_order_by_solution, dim=0) % 2) == 1], dim=0)[:-1]
    demand_order_by_solution[second_endpoint_mask] = 0
    id_return_to_depot = (torch.nonzero(solution_flag).squeeze() - 1)[1:]
    id_return_to_depot = torch.concat([id_return_to_depot, torch.tensor(solution_flag.size(0)).unsqueeze(0)-1])
    demand_cum_sum = torch.concat([torch.zeros([1]), demand_order_by_solution.cumsum(dim=0)[id_return_to_depot]], dim=0) 
    tour_demand = demand_cum_sum[1:] - demand_cum_sum[:-1]
    
    if (tour_demand > capacity).any():
        print(tour_demand)
        print(id_return_to_depot)
        print(demand_order_by_solution)
        raise ValueError('exceed capacity')
    
    return
