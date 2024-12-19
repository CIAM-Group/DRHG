# sampling reduced problem 时, 返回的是4维的point couple, 并将其设为了env.problem

from dataclasses import dataclass
import torch

import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# from utils_for_tester import destroy_perfect_batch



@dataclass
class Step_State:
    data: torch.Tensor

    first_node: torch.Tensor
    current_node: torch.Tensor



class TSPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = None
        self.data_path = env_params['data_path']
        self.load_way = env_params['load_way']
        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.problems = None  # shape: [B,V,2]
        self.original_problem = None
        self.original_solution = None
        self.first_node = None  # shape: [B,V]

        self.raw_data_nodes = []
        self.raw_data_tours = []
        # shape: (batch, node, node)

        self.selected_count = None
        self.current_node = None
        self.selected_node_list = None
        self.selected_student_list = None

        self.test_in_tsplib = env_params['test_in_tsplib']
        self.tsplib_path = env_params['tsplib_path']
        self.tsplib_cost = None
        self.tsplib_name = None
        self.tsplib_problems = None
        self.problem_max_min = None
        self.episode = None

    def load_problems(self, episode, batch_size):
        self.episode = episode

        self.batch_size = batch_size

        self.problems, self.solution = self.raw_data_nodes[episode:episode + batch_size], self.raw_data_tours[episode:episode + batch_size]

        if_inverse = True
        if_inverse_index = torch.randint(low=0, high=100, size=[1])[0]  # in [4,N]
        if if_inverse_index < 50:
            if_inverse = False

        if if_inverse:
            if not self.test_in_tsplib:
                self.solution = torch.flip(self.solution , dims=[1])

        # test and test in tsplib：
        if self.env_params['mode'] == 'test' and self.test_in_tsplib:
            self.tsplib_problems, self.tsplib_cost, self.tsplib_name = self.make_tsplib_data(self.tsplib_path)

            self.problems = torch.from_numpy(self.tsplib_problems[episode].reshape(1,-1,2)).cuda().float()
            self.problem_max_min = [torch.max(self.problems),torch.min(self.problems)]

            self.problems = (self.problems - self.problem_max_min[1])/(self.problem_max_min[0]-self.problem_max_min[1])
            self.solution = None
        
        self.problem_size = self.problems.shape[1]
    

    def load_reduced_problems(self, episode, batch_size, destroy_mode, destroy_params, norm_p=2, return_sorted_problem=False, if_return=True):

        self.episode = episode
        self.batch_size = batch_size

        # load complete problem
        self.problems, self.solution = self.raw_data_nodes[episode : episode + batch_size], self.raw_data_tours[episode : episode + batch_size]
            # shape: [B,V,2]  ;  shape: [B,V]


        if_inverse = True
        if_inverse_index = torch.randint(low=0, high=100, size=[1])[0]  
        if if_inverse_index < 50:
            if_inverse = False
        if if_inverse:
            self.solution = torch.flip( self.solution , dims=[1])

        self.problem_size = self.problems.shape[1]

        self.original_problem = self.problems.clone()
        self.original_solution = self.solution.clone()

        return(self.sampling_reduced_problems(destroy_mode, destroy_params, norm_p, return_sorted_problem, if_return))
        
    def sampling_reduced_problems(self, destroy_mode, destroy_params, norm_p, return_sorted_problem, if_return):
        # use destruction mask to get reduced_problems
        # destruction mask: iso-node：1，endpint：2，in the segment：0
        # point_couples: [(0,0), (1,2), (2,1)] 1-2 is a segment with endpoint 1 and 2
        
        if destroy_mode == 'knn-location':
            location = torch.tensor(destroy_params['center_location'])
            knn_k = torch.randint(destroy_params['knn_k'][0], destroy_params['knn_k'][1], (1,))
            return(self.sampling_reduced_problems_by_knn_location_batch(location, knn_k, norm_p, return_sorted_problem, if_return))
        
        elif destroy_mode == 'fixed_size':
            reduced_problem_size = torch.randint(destroy_params['reduced_problem_size'][0], destroy_params['reduced_problem_size'][1],  (1,))
            return(self.sampling_reduced_problem_for_fix_size(reduced_problem_size, norm_p, return_sorted_problem, if_return))
                
        else:
            raise NotImplementedError('{} not implemented'.format(destroy_mode))

    def sampling_reduced_problems_by_knn_location_batch(self, center_location, k=10, norm_p=2, return_sorted_problem=False, if_return=True):
        # location: [batch, 2]

     
        batch_size = self.problems.size(0)
        problem_size = self.problems.size(1)
        if len(center_location.size())==1:
            center_location = center_location.unsqueeze(0).repeat((batch_size,1))

        # re-arrange the problems，so that the solutioin is [0,1,2,...]
        index = self.solution[:,:,None].repeat(1,1,2)
        sorted_problems = torch.gather(self.problems, dim=1, index=index)

        # sample destroyed point
        first_k = self.get_nearest_neighbors(sorted_problems, center_location, k=k, norm_p=norm_p) #[batch, n, k]

        # instance by instance shift
        shift_ist_by_ist = first_k[:,0] # [batch_size]
        shifted_index = torch.arange(problem_size)[None, :, None].repeat((batch_size, 1, 2)) + shift_ist_by_ist[:, None, None]
        shifted_index[shifted_index >= problem_size] = shifted_index[shifted_index >= problem_size] - problem_size
        sorted_problems = sorted_problems.gather(index=shifted_index, dim=1)
        first_k = first_k - shift_ist_by_ist[:,None] 
        first_k[first_k<0] = first_k[first_k<0] + problem_size
        
        # sample destruction mask
        destruction_mask = torch.zeros(self.solution.size(), dtype=torch.long) # [batch, n, k], 
        destruction_mask = destruction_mask.scatter(dim=1, index=first_k, value=1) 

        # identify iso nodes
        iso_at_before = torch.cat([torch.ones((batch_size,1),dtype=bool), destruction_mask[:,:-1] > destruction_mask[:,1:]], dim=1)
        iso_at_after  = torch.cat([destruction_mask[:,1:] > destruction_mask[:,:-1], torch.ones((batch_size,1),dtype=bool)], dim=1)
        destruction_mask[iso_at_before & iso_at_after] = 1 
        destruction_mask[(iso_at_before ^ iso_at_after) & (~destruction_mask.bool())] = 2 

        return self.destroy_by_destruction_mask_batch(destruction_mask, sorted_problems, 0, shift_ist_by_ist, return_sorted_problem, if_return) 
 
        
    def sampling_reduced_problem_for_fix_size(self, reduced_problem_size, norm_p=2, return_sorted_problem=False, if_return=True):

        problem = self.problems
        solution = self.solution
        batch_size = solution.size(0)
        problem_size = solution.size(1) 
        sorted_problems = problem.gather(index=solution.unsqueeze(-1).repeat(1,1,2), dim=1)

        perm = torch.randperm(problem_size)
        shift = - perm[0].item()
        center_point = torch.zeros((batch_size,1)).long()
        sorted_problems = torch.roll(sorted_problems, shifts=shift, dims=1) # roll the cluster center to 0 

        dist = (sorted_problems[:, :, None, :] - sorted_problems[:, None, :, :]).norm(p=norm_p, dim=-1) #[batch,n,n]
        dist_to_centers_ordered_by_solution = dist.gather(index=center_point.unsqueeze(-1).repeat(1, 1, problem_size), dim=1).squeeze() # [batch, n]

        sorted_dist, sorts = torch.sort(dist_to_centers_ordered_by_solution, dim=1)

        # calculate the distance to neighbors and 2nd-order neighbors
        left_dist_to_centers  = torch.roll(dist_to_centers_ordered_by_solution, dims=1, shifts=1)
        right_dist_to_centers = torch.roll(dist_to_centers_ordered_by_solution, dims=1, shifts=-1)
        left_left_dist_to_centers   = torch.roll(dist_to_centers_ordered_by_solution, dims=1, shifts=2)
        right_right_dist_to_centers = torch.roll(dist_to_centers_ordered_by_solution, dims=1, shifts=-2)
        # calculate if the neighbors is connected before the destroy
        left_is_further  = left_dist_to_centers > dist_to_centers_ordered_by_solution
        right_is_further = right_dist_to_centers > dist_to_centers_ordered_by_solution
        # 2nd-order neighbors
        left_left_is_connected   = (left_left_dist_to_centers > dist_to_centers_ordered_by_solution) & left_is_further
        right_right_is_connected = (right_right_dist_to_centers > dist_to_centers_ordered_by_solution) & right_is_further

        # hyper_point increase: 
        hyper_point_increase_ordered_by_solution = left_is_further.long() + right_is_further.long() + left_left_is_connected.long() + right_right_is_connected.long() - 1
        hyper_point_increase_ordered_by_solution[hyper_point_increase_ordered_by_solution<0] = 0
        hyper_point_increase_ordered_by_dist_sorts = hyper_point_increase_ordered_by_solution.gather(index=sorts, dim=1) 
        hyper_point_cumulative_on_dist_sorts = torch.cumsum(hyper_point_increase_ordered_by_dist_sorts, dim=1)

        # sort the node and get the node to destroy
        destroy_on_dist_sorts = hyper_point_cumulative_on_dist_sorts <= reduced_problem_size 
        hyper_point_cumulative_on_dist_sorts[ hyper_point_cumulative_on_dist_sorts > reduced_problem_size ] = 0
        _, sorts_on_sorts = torch.sort(sorts, dim=1)
        destroy_on_solution_order = destroy_on_dist_sorts.gather(index=sorts_on_sorts, dim=1) 
        destruction_mask = destroy_on_solution_order.long()

        # iso node 
        iso_at_before = torch.cat([torch.ones((batch_size,1),dtype=bool), destruction_mask[:,:-1] > destruction_mask[:,1:]], dim=1)
        iso_at_after  = torch.cat([destruction_mask[:,1:] > destruction_mask[:,:-1], torch.ones((batch_size,1),dtype=bool)], dim=1)
        destruction_mask[iso_at_before & iso_at_after] = 1 
        destruction_mask[(iso_at_before ^ iso_at_after) & (~destruction_mask.bool())] = 2 

        # check the size of hyper-graph 
        isolate_plus_endpoint_num = torch.sum(destruction_mask>0, dim=1) 

        kept_instance = isolate_plus_endpoint_num == reduced_problem_size
        kept_instance_num = torch.sum(kept_instance) 
        sorted_problems = sorted_problems[kept_instance,:,:]
        reduced_problems = sorted_problems[destruction_mask[:,:,None].repeat((1,1,2))[kept_instance,:,:]>0].reshape(kept_instance_num,reduced_problem_size,2) 
        new_problem_index_on_sorted_problem = torch.arange(problem_size)[None,:].repeat((kept_instance_num, 1))[destruction_mask[kept_instance,:]>0].reshape(kept_instance_num, reduced_problem_size)

        # identify endpoint and point couples
        endpoint_mask = (destruction_mask[kept_instance,:][destruction_mask[kept_instance,:]>0] - 1).bool().reshape(kept_instance_num,reduced_problem_size) 
        another_endpoint = torch.ones((kept_instance_num, reduced_problem_size), dtype=torch.long) * -1 

        cumulative_sum = torch.cumsum(endpoint_mask.int(), dim=1) 
        is_left_endpoint = torch.zeros((kept_instance_num, reduced_problem_size), dtype=bool)
        is_left_endpoint[(cumulative_sum % 2)==1] = 1
        is_right_endpoint = torch.roll(is_left_endpoint, dims=1, shifts=1) 

        index = torch.arange(reduced_problem_size.item(), dtype=torch.long)[None,:].repeat((kept_instance_num, 1))
        another_endpoint[is_left_endpoint]  = index[is_left_endpoint] + 1
        another_endpoint[is_right_endpoint] = index[is_right_endpoint] - 1

        point_couples = index.unsqueeze(-1).repeat((1,1,2))       
        point_couples[:,:,1][is_left_endpoint]  = another_endpoint[is_left_endpoint]  
        point_couples[:,:,1][is_right_endpoint] = another_endpoint[is_right_endpoint]   

        padding_mask = torch.zeros((kept_instance_num, reduced_problem_size))

        # set env.problems, env.solution, etc.
        self.problems = reduced_problems
        self.solution = torch.arange(reduced_problems.size(1)).unsqueeze(0).repeat((kept_instance_num,1))
        self.endpoint_mask = endpoint_mask
        self.another_endpoint = another_endpoint
        self.point_couples = point_couples
        self.problem_size = reduced_problems.size(1)
        self.batch_size = kept_instance_num # special for fixed_size

        if return_sorted_problem:
            return destruction_mask, reduced_problems, endpoint_mask, another_endpoint, point_couples, padding_mask, new_problem_index_on_sorted_problem, sorted_problems, shift

        if if_return:       
            return destruction_mask, reduced_problems, endpoint_mask, another_endpoint, point_couples, padding_mask, new_problem_index_on_sorted_problem 

    def destroy_by_destruction_mask_batch(self, destruction_mask, sorted_problems, shift=None, shift_ist_by_ist=None, return_sorted_problem=False, if_return=True):
        # shift affects the sorted_problem and the destruction_mask
        # shift and sorted_problem are both given or both None(shift=0)
        # shift_ist_by_ist: especially for knn_location；otherwise, None

        batch_size = destruction_mask.size(0)
        problem_size = destruction_mask.size(1)
        if shift is None:
            shift = 0

        # node reserved
        reserved_mask = destruction_mask.clone()
        reserved_mask[reserved_mask>0] = 1 #[batch, n, k]
        len_per_instance = reserved_mask.sum(dim=1).long()
        max_len = torch.max(len_per_instance).item()
        min_len = torch.min(len_per_instance).item()
        reserved_mask = reserved_mask.bool()       

        if max_len == min_len: 
            # print('good luck !')
            reduced_problems = sorted_problems[reserved_mask.unsqueeze(-1).repeat((1,1,2))].reshape((batch_size,-1,2))
            endpoint_mask    = (destruction_mask[destruction_mask>0] - 1).bool().reshape((batch_size,-1)) 
            padding_mask     = torch.zeros(endpoint_mask.size()).bool()
            new_problem_index_on_sorted_problem = torch.arange(problem_size).unsqueeze(0).repeat((batch_size,1))[reserved_mask].reshape((batch_size,-1))

        else: 
            indices = torch.nonzero(destruction_mask >= 1).long()

            index_reverse = max_len - torch.arange(max_len)[None,:].repeat((batch_size, 1))
            padding_mask = torch.where(index_reverse>len_per_instance[:,None].repeat(1, max_len), 1, 0).bool()
            endpoint_mask = 1 - padding_mask.long() # padding 0 
            endpoint_mask[~padding_mask] = destruction_mask[destruction_mask>0] - 1 
            endpoint_mask = endpoint_mask.bool()

            first_node_id_a = torch.concat([torch.tensor([0]),len_per_instance.cumsum(dim=0)[:-1]], dim=0)
            first_node_id = indices[first_node_id_a,1]
            new_problem_index_on_sorted_problem = torch.where(index_reverse>len_per_instance[:,None].repeat(1, max_len), 
                                                              first_node_id[:,None].repeat(1,max_len), 
                                                              -1)
            new_problem_index_on_sorted_problem[new_problem_index_on_sorted_problem<0] = indices[:,1]

            reduced_problems = sorted_problems.gather(1, new_problem_index_on_sorted_problem[:,:,None].repeat((1,1,2)))
            

        # another_endpoint；if iso-node, then -1; 
        another_endpoint = torch.ones((batch_size, max_len), dtype=torch.long) * -1 
        index = torch.arange(max_len, dtype=torch.long).unsqueeze(0).repeat((batch_size, 1))

        cumulative_sum = torch.cumsum(endpoint_mask.int(), dim=1) 
        is_left_endpoint = torch.zeros((batch_size, max_len), dtype=bool)
        is_left_endpoint[(cumulative_sum % 2)==1] = 1
        is_right_endpoint = torch.roll(is_left_endpoint, dims=1, shifts=1) 

        another_endpoint[is_left_endpoint]  = index[is_left_endpoint] + 1
        another_endpoint[is_right_endpoint] = index[is_right_endpoint] - 1

        # point couples
        point_couples = index.unsqueeze(-1).repeat((1,1,2))       
        point_couples[:,:,1][is_left_endpoint]  = another_endpoint[is_left_endpoint]  
        point_couples[:,:,1][is_right_endpoint] = another_endpoint[is_right_endpoint]            


        # set env.problems and env.solution, etc.
        self.problems = reduced_problems
        self.solution = torch.arange(reduced_problems.size(1)).unsqueeze(0).repeat((batch_size,1))
        self.endpoint_mask = endpoint_mask
        self.another_endpoint = another_endpoint
        self.point_couples = point_couples
        self.problem_size = reduced_problems.size(1)

        
        if return_sorted_problem:
            if not (shift_ist_by_ist is None):
                shift = shift_ist_by_ist
            return destruction_mask, reduced_problems, endpoint_mask, another_endpoint, point_couples, padding_mask, new_problem_index_on_sorted_problem, sorted_problems, shift

        if if_return:       
            return destruction_mask, reduced_problems, endpoint_mask, another_endpoint, point_couples, padding_mask, new_problem_index_on_sorted_problem  



    # def destroy_solution(self, problem, complete_solution):
    #     self.problems, self.solution, first_node_index, length_of_subpath, double_solution = self.sampling_subpaths(
    #         problem, complete_solution, mode=self.env_params['mode'],repair=True)

    #     partial_solution_length = self._get_travel_distance_2(self.problems, self.solution,test_in_tsplib=self.env_params['test_in_tsplib'],
    #                                                                   need_optimal=False)
        
    #     return partial_solution_length,first_node_index,length_of_subpath,double_solution

    def destroy_REC_solution(self, REC_problem, REC_complete_solution, endpoint_mask):
        self.problems, self.solution, first_node_index, length_of_subpath, double_solution, self.endpoint_mask, self.another_endpoint, self.point_couples, special_last_point_mask =\
              self.sampling_REC_subpaths(REC_problem, REC_complete_solution, endpoint_mask)

        partial_solution_length = self._get_travel_distance_2(self.problems, self.solution, test_in_tsplib=self.env_params['test_in_tsplib'],
                                                                      need_optimal=False)

        return partial_solution_length, first_node_index, length_of_subpath, double_solution, self.endpoint_mask, self.another_endpoint,self.point_couples, special_last_point_mask 


    def shuffle_data(self):
        index = torch.randperm(len(self.raw_data_nodes)).long()
        self.raw_data_nodes = self.raw_data_nodes[index]
        self.raw_data_tours = self.raw_data_tours[index]


    def load_raw_data(self, episode, begin_index=0):

        print('load raw dataset begin!')

        self.raw_data_nodes = []
        self.raw_data_tours = []
        for line in tqdm(open(self.data_path, "r").readlines()[0+begin_index:episode+begin_index], ascii=True):
            line = line.split(" ")
            num_nodes = int(line.index('output') // 2)
            nodes = [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * num_nodes, 2)]

            self.raw_data_nodes.append(nodes)
            tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]]#[:-1]
            self.raw_data_tours.append(tour_nodes)

        self.raw_data_nodes = torch.tensor(self.raw_data_nodes,requires_grad=False)
        self.raw_data_tours = torch.tensor(self.raw_data_tours,requires_grad=False)

        print(f'load raw dataset done!', )  

    def load_pt_data(self, episode, data_path, device):
        data = torch.load(data_path, map_location=device)
        self.raw_data_nodes = data['data'][:episode]
        self.raw_data_tours = data['solution'][:episode]

    def load_pt_problem(self, episode, problem_path,  device):
        self.raw_data_nodes = torch.load(problem_path, map_location=device)[:episode]

    def load_pt_solution(self, episode, solution_path, device):
        self.raw_data_tours = torch.load(solution_path, map_location=device)[:episode]

    def set_raw_data_tours(self, raw_data_tours):
        self.raw_data_tours = raw_data_tours
        
    
    def make_tsplib_data(self, tsplib_data_path):
        tsplib_data = torch.load(tsplib_data_path) # .pt
        instance_data = tsplib_data[1]
        cost = tsplib_data[2]
        instance_name = tsplib_data[0]
        return(instance_data, cost, instance_name)

    def reset(self):
        self.selected_count = 0
        self.selected_node_list = torch.zeros((self.batch_size, 0), dtype=torch.long)
        self.selected_student_list = torch.zeros((self.batch_size, 0), dtype=torch.long)

        self.step_state = Step_State(data=self.problems, first_node=self.first_node,
                                     current_node=self.current_node)

        reward = None
        done = False
        return reward, done # Reset_State(self.problems)后面好像没有用到

    def pre_step(self):
        reward = None
        reward_student = None
        done = False
        return self.step_state, reward, reward_student, done

    def step(self, selected, selected_student):

        self.selected_count += 1

        gather_index = selected[:, None, None].expand((len(selected), 1, 2))
        # shape: test: [B*k,1,2], train: [B,1,2]

        self.current_node = self.problems.gather(index=gather_index, dim=1).squeeze(1)

        if self.selected_count == 1:
            self.first_node = self.current_node

        self.selected_node_list = torch.cat((self.selected_node_list, selected[:, None]), dim=1)  # shape: [B, current_step]

        self.selected_student_list = torch.cat((self.selected_student_list, selected_student[:, None]), dim=1)

        # UPDATE STEP STATE
        self.step_state.current_node = self.current_node[:, None, :]
        if self.selected_count == 1:
            self.step_state.first_node = self.step_state.current_node
        # current_node_shape =self.step_state.current_node.shape  # shape: [B, 1, 2]

        # returning values
        done = (self.selected_count == self.problems.shape[1])
        if done:
            reward, reward_student = self._get_travel_distance()  # note the minus sign!
        else:
            reward, reward_student = None, None

        return self.step_state, reward, reward_student, done

    def make_dir(self,path_destination):
        isExists = os.path.exists(path_destination)
        if not isExists:
            os.makedirs(path_destination)
        return


    def _get_travel_distance(self):

        if self.test_in_tsplib:
            travel_distances = self.tsplib_cost
            self.problems =  self.problems * (self.problem_max_min[0] - self.problem_max_min[1]) + self.problem_max_min[1]

        else:
            gathering_index = self.solution.unsqueeze(2).expand(self.batch_size, self.problems.shape[1], 2) # solution是teacher solution
            # shape: (batch, problem, 2)
            ordered_seq = self.problems.gather(dim=1, index=gathering_index)
            # shape: (batch, problem, 2)

            rolled_seq = ordered_seq.roll(dims=1, shifts=-1)
            segment_lengths = ((ordered_seq - rolled_seq) ** 2)
            segment_lengths = segment_lengths.sum(2).sqrt()
            # shape: (batch, problem)
            travel_distances = segment_lengths.sum(1)

        # trained model's distance
        gathering_index_student = self.selected_student_list.unsqueeze(2).expand(-1, self.problems.shape[1], 2)
        ordered_seq_student = self.problems.gather(dim=1, index=gathering_index_student)
        rolled_seq_student = ordered_seq_student.roll(dims=1, shifts=-1)
        segment_lengths_student = ((ordered_seq_student - rolled_seq_student) ** 2)
        segment_lengths_student = segment_lengths_student.sum(2).sqrt()
        # shape: (batch,problem)
        travel_distances_student = segment_lengths_student.sum(1)
        # shape: (batch)
        return -travel_distances, -travel_distances_student

    def _get_travel_distance_2(self, problems, solution, test_in_tsplib=False, need_optimal = False):

        if test_in_tsplib:
            if need_optimal:
                return self.tsplib_cost, self.tsplib_name
            else:
                problems_copy = problems.clone().detach() * (self.problem_max_min[0] - self.problem_max_min[1]) + \
                                self.problem_max_min[1]
                # print(problems[0],problems_copy[0],self.problem_max_min[0] , self.problem_max_min[1])

                gathering_index = solution.unsqueeze(2).expand(problems_copy.shape[0], problems_copy.shape[1], 2)
                seq_expanded = problems_copy
                ordered_seq = seq_expanded.gather(dim=1, index=gathering_index)
                rolled_seq = ordered_seq.roll(dims=1, shifts=-1)

                segment_lengths = ((ordered_seq - rolled_seq) ** 2)
                segment_lengths = segment_lengths.sum(2).sqrt()

                travel_distances = segment_lengths.sum(1)

                return travel_distances
        else:
            gathering_index = solution.unsqueeze(2).expand(problems.shape[0], problems.shape[1], 2)
            seq_expanded = problems
            ordered_seq = seq_expanded.gather(dim=1, index=gathering_index)
            rolled_seq = ordered_seq.roll(dims=1, shifts=-1)

            segment_lengths = ((ordered_seq - rolled_seq) ** 2)
            segment_lengths = segment_lengths.sum(2).sqrt()
            travel_distances = segment_lengths.sum(1)

        return travel_distances
    
    
    def coordinate_transform(self, coordinates):    
        # coordinate transformation
        max_x, indices_max_x = coordinates[:,:,0].max(dim=1)
        max_y, indices_max_y = coordinates[:,:,1].max(dim=1)
        min_x, indices_min_x = coordinates[:,:,0].min(dim=1)
        min_y, indices_min_y = coordinates[:,:,1].min(dim=1)
        # shapes: (batch_size, ); (batch_size, )
        
        diff_x = max_x - min_x
        diff_y = max_y - min_y
        xy_exchanged = diff_y > diff_x

        # shift to zero
        coordinates[:, :, 0] -= (min_x).unsqueeze(-1)
        coordinates[:, :, 1] -= (min_y).unsqueeze(-1)

        # exchange coordinates for those diff_y > diff_x
        coordinates[xy_exchanged, :, 0], coordinates[xy_exchanged, :, 1] =  coordinates[xy_exchanged, :, 1], coordinates[xy_exchanged, :, 0]
        
        # scale to (0, 1)
        scale_degree = torch.max(diff_x, diff_y)
        scale_degree = scale_degree.view(coordinates.shape[0], 1, 1)
        coordinates /= scale_degree

        return coordinates
    
    def get_nearest_neighbors(self, coords, center_location, k, norm_p):
        # coords [batch, problem, 2]
        distances = (coords - center_location[:, None, :]).norm(p=norm_p, dim=-1) #[batch, problem]
        _, sorts = torch.sort(distances, dim=-1)
        first_k = sorts[:,:k] #[batch, k]
        return first_k

        