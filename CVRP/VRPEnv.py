from dataclasses import dataclass
import torch

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import pickle

@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)


@dataclass
class Step_State:
    problems: torch.Tensor
    first_node: torch.Tensor
    current_node: torch.Tensor



class VRPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.device = env_params['device']
        self.problem_size = None
        self.data_path = env_params['data_path']
        self.load_way = env_params['load_way']
        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.problems = None  # shape: [B,V+1,4]
        self.first_node = None  # shape: [B]

        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        self.selected_student_list = None

        self.test_in_vrplib = env_params['test_in_vrplib']
        self.vrplib_cost = None
        self.vrplib_name = None
        self.vrplib_problems = None
        self.problem_max_min = None
        self.episode = None


    def load_problems(self, episode, batch_size, load_cvrplib=False):
        self.episode = episode
        self.batch_size = batch_size

        if not load_cvrplib:
            if self.load_way in ['txt','pt']:
                
                self.solution       = self.raw_data_node_flag[episode:episode + batch_size] # shape (B,V,2)
                self.problems_nodes = self.raw_data_nodes[episode:episode + batch_size] # shape (B,V+1,2)
                self.Batch_demand   = self.raw_data_demand[episode:episode + batch_size] # shape (B,V+1)
                self.Batch_capacity = self.raw_data_capacity[episode:episode + batch_size][:,None].repeat(1,self.solution.shape[1]+1) # shape (B,V+1)
                self.problems       = torch.cat((self.problems_nodes, self.Batch_demand[:,:,None],
                                           self.Batch_capacity[:,:,None]),dim=2)
                # shape (B,V+1,4)

            else: raise NotImplementedError()
        
        else: 
            self.solution       = self.raw_data_node_flag[episode] # shape (B,V,2) 
            self.problems_nodes = self.raw_data_nodes[episode].unsqueeze(0) # shape (B,V+1,2)
            self.Batch_demand   = self.raw_data_demand[episode].unsqueeze(0) # shape (B,V+1)
            self.Batch_capacity = self.raw_data_capacity[episode][None,None].repeat(1,self.solution.shape[1]+1) # shape (B,V+1)

            self.problems       = torch.cat((self.problems_nodes, 
                                             self.Batch_demand[:,:,None],
                                             self.Batch_capacity[:,:,None]),dim=2)

        self.problem_size = self.problems.shape[1]-1 # delete depot


    def vrp_whole_and_solution_subrandom_inverse(self, solution):

        clockwise_or_not = torch.rand(1)[0]
        if clockwise_or_not >= 0.5:
            solution = torch.flip(solution, dims=[1])

            index = torch.arange(solution.shape[1]).roll(shifts=1)
            solution[:, :, 1] = solution[:, index, 1]

        # 1.
        # find the num of subtours of instances
        # all_subtour_num: find the num of subtours of all instance     
        # max_subtour_length: max length of all subtours
        batch_size = solution.shape[0]
        problem_size = solution.shape[1]

        visit_depot_num = torch.sum(solution[:, :, 1], dim=1)
        all_subtour_num = torch.sum(visit_depot_num)

        fake_solution = torch.cat((solution[:, :, 1], torch.ones(batch_size)[:, None]), dim=1)

        start_from_depot = fake_solution.nonzero()
        start_from_depot_1 = start_from_depot[:, 1]
        start_from_depot_2 = torch.roll(start_from_depot_1, shifts=-1)

        sub_tours_length = start_from_depot_2 - start_from_depot_1
        max_subtour_length = torch.max(sub_tours_length)

        # 2
        # padding every subtour to length = max_subtour_length
        # padding every instance by subtours until max_subtour_num
        start_from_depot2 = solution[:, :, 1].nonzero()
        start_from_depot3 = solution[:, :, 1].roll(shifts=-1, dims=1).nonzero()

        repeat_solutions_node = solution[:, :, 0].repeat_interleave(visit_depot_num, dim=0)
        double_repeat_solution_node = repeat_solutions_node.repeat(1, 2)

        x1 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             >= start_from_depot2[:, 1][:, None]
        x2 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             <= start_from_depot3[:, 1][:, None]

        x3 = (x1 * x2).long()

        sub_tours = double_repeat_solution_node * x3

        x4 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             < (start_from_depot2[:, 1][:, None] + max_subtour_length)

        x5 = x1 * x4

        sub_tours_padding = sub_tours[x5].reshape(all_subtour_num, max_subtour_length)

        # 3.
        # clockwise_or_not
        clockwise_or_not = torch.rand(len(sub_tours_padding))
        clockwise_or_not_bool = clockwise_or_not.le(0.5)

        sub_tours_padding[clockwise_or_not_bool] = torch.flip(sub_tours_padding[clockwise_or_not_bool], dims=[1])

        # 4
        # restore subtours to complete solution
        sub_tours_back = sub_tours
        sub_tours_back[x5] = sub_tours_padding.ravel()

        solution_node_flip = sub_tours_back[sub_tours_back.gt(0.1)].reshape(batch_size, problem_size)
        solution_flip = torch.cat((solution_node_flip.unsqueeze(2), solution[:, :, 1].unsqueeze(2)), dim=2)

        return solution_flip


    def shuffle_data(self):
        index = torch.randperm(len(self.raw_data_nodes)).long()
        self.raw_data_nodes = self.raw_data_nodes[index]
        self.raw_data_capacity = self.raw_data_capacity[index]
        self.raw_data_demand = self.raw_data_demand[index]
        self.raw_data_cost = self.raw_data_cost[index]
        self.raw_data_node_flag = self.raw_data_node_flag[index]


    def load_raw_data(self, episode=1000000, from_pt=False, pt_path=None, cvrplib=False):

        def tow_col_nodeflag(node_flag):
            tow_col_node_flag = []
            V = int(len(node_flag) / 2)
            for i in range(V):
                tow_col_node_flag.append([node_flag[i], node_flag[V + i]])
            return tow_col_node_flag
        
        if from_pt:
            assert(not pt_path is None)
            if not cvrplib: 
                problem_solution_cost = torch.load(pt_path, map_location=self.device)
                problem  = problem_solution_cost['problem']
                solution = problem_solution_cost['solution']
                cost     = problem_solution_cost['cost']
                self.raw_data_nodes = problem[:, :, :2] # not for cvrplib
                self.raw_data_demand = problem[:, :, 2]
                self.raw_data_capacity = problem[:, 0, 3] 
                self.raw_data_node_flag = solution
                self.raw_data_cost = cost
                print('load raw dataset done!')
                return()
            else:
                problem_solution_cost = torch.load(pt_path, map_location=self.device)
                problem  = problem_solution_cost['problem']
                solution = problem_solution_cost['solution']
                cost     = problem_solution_cost['cost']
                n = len(cost)
                self.raw_data_nodes = [problem[i][:, :2] for i in range(n)] # cvrplib
                self.raw_data_demand = [problem[i][:, 2] for i in range(n)]
                self.raw_data_capacity = torch.tensor([problem[i][0, 3] for i in range(n)])
                self.raw_data_node_flag = solution
                self.raw_data_cost = cost
                self.raw_instance_name = problem_solution_cost['name']
                print('load cvrplib done!')
                return()

        # from txt
        if episode >=  500000:
            need_round_2 = True
            first = int(0.5 * episode)
        else: 
            first = episode
            need_round_2 = False

        if self.env_params['mode']=='train':

            self.raw_data_nodes_1 = []
            self.raw_data_capacity_1 = []
            self.raw_data_demand_1 = []
            self.raw_data_cost_1 = []
            self.raw_data_node_flag_1 = []
            for line in tqdm(open( self.data_path, "r").readlines()[0:first], ascii=True):
                line = line.split(",")

                depot_index = int(line.index('depot'))
                customer_index = int(line.index('customer'))
                capacity_index = int(line.index('capacity'))
                demand_index = int(line.index('demand'))
                cost_index = int(line.index('cost'))
                node_flag_index = int(line.index('node_flag'))

                depot = [[float(line[depot_index + 1]), float(line[depot_index + 2])]]
                customer = [[float(line[idx]), float(line[idx + 1])] for idx in range(customer_index + 1, capacity_index, 2)]
                loc = depot + customer

                capacity = int(float(line[capacity_index + 1]))
                demand = [int(line[idx]) for idx in range(demand_index + 1, cost_index)]

                cost = float(line[cost_index + 1])
                node_flag = [int(line[idx]) for idx in range(node_flag_index + 1, len(line))]
                node_flag = tow_col_nodeflag(node_flag)

                self.raw_data_nodes_1.append(loc)
                self.raw_data_capacity_1.append(capacity)
                self.raw_data_demand_1.append(demand)
                self.raw_data_cost_1.append(cost)
                self.raw_data_node_flag_1.append(node_flag)

            self.raw_data_nodes_1 = torch.tensor(self.raw_data_nodes_1, requires_grad=False) # shape (B, V+1, 2)  customer num + depot
            self.raw_data_capacity_1 = torch.tensor(self.raw_data_capacity_1, requires_grad=False) # shape (B )
            self.raw_data_demand_1 = torch.tensor(self.raw_data_demand_1, requires_grad=False) # shape (B, V+1) customer num + depot
            self.raw_data_cost_1 = torch.tensor(self.raw_data_cost_1, requires_grad=False) # shape (B )
            self.raw_data_node_flag_1 = torch.tensor(self.raw_data_node_flag_1, requires_grad=False) # shape (B, V, 2)

            self.raw_data_nodes     = self.raw_data_nodes_1
            self.raw_data_capacity  = self.raw_data_capacity_1
            self.raw_data_demand    = self.raw_data_demand_1
            self.raw_data_cost      = self.raw_data_cost_1
            self.raw_data_node_flag = self.raw_data_node_flag_1

            if need_round_2: 
                self.raw_data_nodes_2 = []
                self.raw_data_capacity_2 = []
                self.raw_data_demand_2 = []
                self.raw_data_cost_2 = []
                self.raw_data_node_flag_2 = []
                for line in tqdm(open(self.data_path, "r").readlines()[int(0.5 * episode):int(episode)], ascii=True):
                    line = line.split(",")

                    depot_index = int(line.index('depot'))
                    customer_index = int(line.index('customer'))
                    capacity_index = int(line.index('capacity'))
                    demand_index = int(line.index('demand'))
                    cost_index = int(line.index('cost'))
                    node_flag_index = int(line.index('node_flag'))

                    depot = [[float(line[depot_index + 1]), float(line[depot_index + 2])]]
                    customer = [[float(line[idx]), float(line[idx + 1])] for idx in range(customer_index + 1, capacity_index, 2)]
                    loc = depot + customer

                    capacity = int(float(line[capacity_index + 1]))
                    demand =  [int(line[idx]) for idx in range(demand_index + 1, cost_index)]
                    cost = float(line[cost_index + 1])
                    node_flag = [int(line[idx]) for idx in range(node_flag_index + 1, len(line))]
                    node_flag = tow_col_nodeflag(node_flag)
                   
                    self.raw_data_nodes_2.append(loc)
                    self.raw_data_capacity_2.append(capacity)
                    self.raw_data_demand_2.append(demand)
                    self.raw_data_cost_2.append(cost)
                    self.raw_data_node_flag_2.append(node_flag)

                self.raw_data_nodes_2 = torch.tensor(self.raw_data_nodes_2, requires_grad=False) # shape (B,V+1,2)  customer num + depot
                self.raw_data_capacity_2 = torch.tensor(self.raw_data_capacity_2, requires_grad=False) # shape (B )
                self.raw_data_demand_2 = torch.tensor(self.raw_data_demand_2, requires_grad=False) # shape (B,V+1) customer num + depot
                self.raw_data_cost_2 = torch.tensor(self.raw_data_cost_2, requires_grad=False) # shape (B )
                self.raw_data_node_flag_2 = torch.tensor(self.raw_data_node_flag_2, requires_grad=False) # shape (B,V,2)
                

                self.raw_data_nodes = torch.cat((self.raw_data_nodes_1,self.raw_data_nodes_2),dim=0)
                self.raw_data_capacity = torch.cat((self.raw_data_capacity_1, self.raw_data_capacity_2), dim=0)
                self.raw_data_demand = torch.cat((self.raw_data_demand_1, self.raw_data_demand_2), dim=0)
                self.raw_data_cost = torch.cat((self.raw_data_cost_1, self.raw_data_cost_2), dim=0)
                self.raw_data_node_flag = torch.cat((self.raw_data_node_flag_1, self.raw_data_node_flag_2), dim=0)

        if self.env_params['mode'] == 'test':

            self.raw_data_nodes = []
            self.raw_data_capacity = []
            self.raw_data_demand = []
            self.raw_data_cost = []
            self.raw_data_node_flag = []
            for line in tqdm(open(self.data_path, "r").readlines()[0:episode], ascii=True):
                line = line.split(",")

                depot_index = int(line.index('depot'))
                customer_index = int(line.index('customer'))
                capacity_index = int(line.index('capacity'))
                demand_index = int(line.index('demand'))
                cost_index = int(line.index('cost'))
                node_flag_index = int(line.index('node_flag'))

                depot = [[float(line[depot_index + 1]), float(line[depot_index + 2])]]
                customer = [[float(line[idx]), float(line[idx + 1])] for idx in range(customer_index + 1, capacity_index, 2)]
                loc = depot + customer

                capacity = int(float(line[capacity_index + 1]))
                demand = [0] + [int(line[idx]) for idx in range(demand_index + 1, cost_index)]
                cost = float(line[cost_index + 1])
                node_flag = [int(line[idx]) for idx in range(node_flag_index + 1, len(line))]
                node_flag = tow_col_nodeflag(node_flag)

                self.raw_data_nodes.append(loc)
                self.raw_data_capacity.append(capacity)
                self.raw_data_demand.append(demand)
                self.raw_data_cost.append(cost)
                self.raw_data_node_flag.append(node_flag)

            self.raw_data_nodes = torch.tensor(self.raw_data_nodes, requires_grad=False) # shape (B,V+1,2)  customer num + depot
            self.raw_data_capacity = torch.tensor(self.raw_data_capacity, requires_grad=False) # shape (B )
            self.raw_data_demand = torch.tensor(self.raw_data_demand, requires_grad=False) # shape (B,V+1) customer num + depot
            self.raw_data_cost = torch.tensor(self.raw_data_cost, requires_grad=False) # shape (B )
            self.raw_data_node_flag = torch.tensor(self.raw_data_node_flag, requires_grad=False) # shape (B,V,2)

        print(f'load raw dataset done!', ) 
        solution = self.raw_data_node_flag
        problem  = torch.concat([self.raw_data_nodes, self.raw_data_demand[:,:,None], self.raw_data_capacity[:,None,None].repeat(1, solution.shape[1]+1, 1)], dim=2)
        cost     = self.raw_data_cost



    def reset(self, mode):
        self.selected_count = 0

        if mode == 'train':
            self.selected_node_list = torch.zeros((self.batch_size, 0), dtype=torch.long)
            self.selected_teacher_flag = torch.zeros((self.batch_size, 0), dtype=torch.long)
            self.selected_student_list = torch.zeros((self.batch_size, 0), dtype=torch.long)
            self.selected_student_flag= torch.zeros((self.batch_size, 0), dtype=torch.long)
            self.current_node=self.problems[:,0,:]
            self.first_node = self.current_node
            self.step_state = Step_State(problems=self.problems, first_node=self.first_node[:, None, :],
                                         current_node=self.current_node[:, None, :])
        if mode == 'test':
            self.selected_node_list = torch.zeros((self.batch_size, 0), dtype=torch.long)
            self.selected_teacher_flag = torch.zeros((self.batch_size, 0), dtype=torch.long)
            self.selected_student_list = torch.zeros((self.batch_size, 0), dtype=torch.long)
            self.selected_student_flag = torch.zeros((self.batch_size, 0), dtype=torch.long)
            self.current_node = self.problems[:, 0, :]
            self.first_node = self.current_node
            self.step_state = Step_State(problems=self.problems, first_node=self.first_node[:, None, :],
                                            current_node=self.current_node[:, None, :])

        reward = None
        done = False
        return Reset_State(self.problems), reward, done

    def pre_step(self):
        reward = None
        reward_student = None
        done = False
        return self.step_state, reward, reward_student, done

    def step(self, selected, selected_student, selected_flag_teacher, selected_flag_student, is_another_endpoint=None, is_first_endpoint=None, raw_capacity=None):

        self.selected_count += 1
        gather_index = selected[:, None, None].expand((len(selected), 1, 4)) # shape [B,1,4]

        # 1. update capacity
        # if flag = 1，back to depot，refill capacity
        is_depot = selected_flag_teacher==1
        if not self.test_in_vrplib:
            self.problems[is_depot, :, 3] =  self.raw_data_capacity.ravel()[0].item()
        else:
            self.problems[is_depot, :, 3] = raw_capacity

        # if capacity < demand, back to depot，refill capacity，turn flag to 1
        self.current_node_temp = self.problems.gather(index=gather_index, dim=1).squeeze(1)
         
        if is_another_endpoint is None:
            demands = self.current_node_temp[:,2].clone()
        else: # if the last node is the first endpoint, demand has been counted, set it to 0
            demands = torch.where(is_another_endpoint, 0, self.current_node_temp[:,2])

        smaller_ = self.problems[:, 0, 3] < demands # capacity < demand
        selected_flag_teacher[smaller_] = 1
        if not self.test_in_vrplib:
            self.problems[smaller_, :, 3] = self.raw_data_capacity.ravel()[0].item()
        else:
            self.problems[smaller_, :, 3] = raw_capacity

        # 2. extract demand
        self.problems[:,:,3] =  self.problems[:,:,3]- demands[:,None]

        # 3. update state
        self.current_node = self.problems.gather(index=gather_index, dim=1).squeeze(1) 
        # shape [B,4]

        self.selected_node_list = torch.cat((self.selected_node_list, selected[:, None]), dim=1)
        self.selected_teacher_flag = torch.cat((self.selected_teacher_flag, selected_flag_teacher[:, None]), dim=1)

        self.selected_student_list = torch.cat((self.selected_student_list, selected_student[:, None]), dim=1)
        self.selected_student_flag = torch.cat((self.selected_student_flag, selected_flag_student[:, None]), dim=1)

        self.step_state.current_node = self.current_node[:, None, :]
        self.step_state.first_node[:, 0, 2]= 0
        self.step_state.current_node[:, 0, 2] = 0
        self.first_node[:, 2] = 0
        self.current_node[:, 2] = 0
        self.step_state.first_node[:, 0, 3] = self.problems[:,1,3].clone()
        self.step_state.current_node[:, 0, 3] = self.problems[:, 1, 3].clone()
        self.first_node[:, 3] = self.problems[:, 1, 3].clone()
        self.current_node[:, 3] = self.problems[:, 1, 3].clone()

        # returning values
        done = (self.selected_count == self.problems.shape[1] - 1)
        if done:
            reward, reward_student = self._get_travel_distance()  # note the minus sign!
        else:
            reward, reward_student = None, None 

        return self.step_state, reward, reward_student, done


    def make_dir(self, path_destination):
        isExists = os.path.exists(path_destination)
        if not isExists:
            os.makedirs(path_destination)
        return


    def cal_length(self, problems, order_node, order_flag):
        # problems:   [B,V+1,2]
        # order_node: [B,V]
        # order_flag: [B,V]
        problem_size = problems.shape[1] - 1
        order_node_ = order_node.clone()
        order_flag_ = order_flag.clone()

        index_small = torch.le(order_flag_, 0.5)
        index_bigger = torch.gt(order_flag_, 0.5)

        order_flag_[index_small] = order_node_[index_small] 
        order_flag_[index_bigger] = 0 

        roll_node = order_node_.roll(dims=1, shifts=1)
        
        order_gathering_index = order_node_.unsqueeze(2).expand(-1, problem_size, 2)
        order_loc = problems.gather(dim=1, index=order_gathering_index)

        roll_gathering_index = roll_node.unsqueeze(2).expand(-1, problem_size, 2)
        roll_loc = problems.gather(dim=1, index=roll_gathering_index)

        flag_gathering_index = order_flag_.unsqueeze(2).expand(-1, problem_size, 2)
        flag_loc = problems.gather(dim=1, index=flag_gathering_index)

        order_lengths = ((order_loc - flag_loc) ** 2)
        order_flag_[:,0]=0
        flag_gathering_index = order_flag_.unsqueeze(2).expand(-1, problem_size, 2)
        flag_loc = problems.gather(dim=1, index=flag_gathering_index)

        roll_lengths = ((roll_loc - flag_loc) ** 2)
        length = (order_lengths.sum(2).sqrt() + roll_lengths.sum(2).sqrt()).sum(1)

        return length

    def _get_travel_distance(self):
        
        # teacher
        problems = self.problems[:,:,[0,1]]
        order_node = self.solution[:,:,0]
        order_flag = self.solution[:,:,1]
        travel_distances = self.cal_length( problems, order_node, order_flag)

        # student
        problems = self.problems[:, :, [0, 1]]
        order_node = self.selected_student_list.clone()
        order_flag = self.selected_student_flag.clone()
        travel_distances_student = self.cal_length(problems, order_node, order_flag)

        return -travel_distances, -travel_distances_student


    def _get_travel_distance_2(self, problems_, solution_,test_in_vrplib = False,need_optimal =False ):

        if test_in_vrplib:
            if need_optimal:
                return self.vrplib_cost, self.vrplib_name
            else:
                problems = problems_[:, :, [0, 1]].clone() * (self.problem_max_min[0] - self.problem_max_min[1]) + self.problem_max_min[1]
                order_node = solution_[:, :, 0].clone()
                order_flag = solution_[:, :, 1].clone()
                travel_distances = self.cal_length(problems, order_node, order_flag)
        else:
            problems = problems_[:, :, [0, 1]].clone()
            order_node = solution_[:, :, 0].clone()
            order_flag = solution_[:, :, 1].clone()
            travel_distances = self.cal_length(problems, order_node, order_flag)


        return travel_distances

    
    def valida_solution_legal(self, problem, solution, capacity_=50):

        capacity = capacity_
        demand = problem[:, :, 2]

        order_node = solution[:, :, 0].clone()
        order_flag = solution[:, :, 1].clone()

        # 0. whether every node in solution node list is unique
        uniques = torch.unique(order_node[0])
        if len(uniques) != problem.shape[1] - 1:
            assert False, 'wrong node list!'

        # 1. whether demands of every subtour exceed the capacity

        # 1.1 find the subtours and their length
        batch_size = solution.shape[0]

        visit_depot_num = torch.sum(solution[:, :, 1], dim=1)
        all_subtour_num = torch.sum(visit_depot_num)

        fake_solution = torch.cat((solution[:, :, 1], torch.ones(batch_size)[:, None]), dim=1)

        start_from_depot = fake_solution.nonzero()
        start_from_depot_1 = start_from_depot[:, 1]
        start_from_depot_2 = torch.roll(start_from_depot_1, shifts=-1)

        sub_tours_length = start_from_depot_2 - start_from_depot_1
        max_subtour_length = torch.max(sub_tours_length)

        # 1.2 padding
        start_from_depot2 = solution[:, :, 1].nonzero()
        start_from_depot3 = solution[:, :, 1].roll(shifts=-1, dims=1).nonzero()

        repeat_solutions_node = solution[:, :, 0].repeat_interleave(visit_depot_num, dim=0)
        double_repeat_solution_node = repeat_solutions_node.repeat(1, 2)

        x1 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             >= start_from_depot2[:, 1][:, None]
        x2 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             <= start_from_depot3[:, 1][:, None]

        x3 = (x1 * x2).long()

        sub_tourss = double_repeat_solution_node * x3

        x4 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             < (start_from_depot2[:, 1][:, None] + max_subtour_length)

        x5 = x1 * x4

        sub_tours_padding = sub_tourss[x5].reshape(all_subtour_num, max_subtour_length)

        # 1.3 check constraint
        demands = torch.repeat_interleave(demand, repeats=visit_depot_num, dim=0)

        index = torch.arange(sub_tours_padding.shape[0])[:, None].repeat(1, sub_tours_padding.shape[1])
        sub_tours_demands = demands[index, sub_tours_padding].sum(dim=1)
        if_illegal = (sub_tours_demands > capacity)

        if if_illegal.any():
            illegal_tour = torch.nonzero(if_illegal)[0,0]
            cum_sum = torch.cumsum(visit_depot_num, dim=0)
            print(torch.nonzero(cum_sum>illegal_tour)[0,0])
            print(sub_tours_demands[if_illegal])
            print(demands[index, sub_tours_padding][if_illegal,:])
            assert False, 'wrong capacity!'

        return
    

    def load_reduced_problems(self, episode, batch_size, destroy_mode, destroy_params, return_sorted_problem):

        self.load_problems(episode, batch_size)

        return(self.sampling_reduced_problems(destroy_mode, destroy_params, return_sorted_problem))


    def sampling_reduced_problems(self, destroy_mode, destroy_params, return_sorted_problem):
        # it will update self.problems, self.solution, self.problem_size
        if destroy_mode == 'fixed_size':
            reduced_problem_size = torch.randint(destroy_params['reduced_problem_size'][0], destroy_params['reduced_problem_size'][1],  (1,))
            return(self.sampling_reduced_problems_for_fixed_size(reduced_problem_size, return_sorted_problem))
        elif destroy_mode == 'knn-location':
            location = destroy_params['center_location']
            if location is None:
                location = torch.rand([2]) 
            else: location = torch.tensor(location)
            knn_k = torch.randint(destroy_params['knn_k'][0], destroy_params['knn_k'][1], (1,))
            return(self.sampling_reduced_problems_by_knn_location(location, knn_k, return_sorted_problem))
        else:
            raise NotImplementedError('{} not implemented'.format(destroy_mode))

    def sampling_reduced_problems_by_knn_location(self, center_location, k, return_sorted_problem):
        def get_nearest_neighbors(coords, center_location, k):
            # coords [batch, problem, 2]
            distances = (coords - center_location[:, None, :]).norm(p=2, dim=-1) #[batch, problem]
            _, sorts = torch.sort(distances, dim=-1)
            first_k = sorts[:,:k] #[batch, k]
            return first_k
        
        problems_coord = self.problems[:,:,:2]
        problems_demand = self.problems[:,:,2]
        capacity = self.problems[:,:,3]
        solution_node = self.solution[:,:,0]
        solution_flag= self.solution[:,:,1]
     
        batch_size = problems_coord.size(0)
        problem_size = problems_demand.size(1) - 1
        if len(center_location.size())==1:
            center_location = center_location.unsqueeze(0).repeat((batch_size,1))

        # concat depot = 0
        solution_node_with_depot = torch.concat([torch.zeros((batch_size,1),dtype=int), solution_node], dim=1) 
        solution_flag_with_depot = torch.concat([torch.ones((batch_size,1),dtype=int), solution_flag], dim=1) 

        # rearrange the problems, so that the solutions are [0,1,2,...]
        problem_coords_sorted = problems_coord.gather(dim=1, index=solution_node_with_depot[:,:,None].repeat(1,1,2))
        demand_sorted = problems_demand.gather(dim=1, index=solution_node_with_depot[:,:])

        # sample destroyed point
        first_k = get_nearest_neighbors(problem_coords_sorted, center_location, k=k) # [batch, n, k]
        destroy_node_mask = torch.zeros(solution_node_with_depot.size(), dtype=bool) # [batch, n, k]
        destroy_node_mask = destroy_node_mask.scatter(dim=1, index=first_k, value=1) 

        return(self.destroy_by_destroyed_node(problem_coords_sorted,
                                              demand_sorted,
                                              capacity,
                                              solution_flag_with_depot, 
                                              destroy_node_mask,
                                              return_sorted_problem))

 
 
    def sampling_reduced_problems_for_fixed_size(self, reduced_problem_size, return_sorted_problem):
        problems_coord = self.problems[:,:,:2]
        problems_demand = self.problems[:,:,2]
        capacity = self.problems[:,:,3]
        solution_node = self.solution[:,:,0]
        solution_flag= self.solution[:,:,1].bool()
        batch_size = problems_coord.size(0)
        problem_size = problems_coord.size(1)

        perm = torch.randperm(problem_size)
        center_point = torch.ones((batch_size,1)).long() * perm[0]

        # concat depot = 0
        solution_node_with_depot = torch.concat([torch.zeros((batch_size,1),dtype=int), solution_node], dim=1) 
        solution_flag_with_depot = torch.concat([torch.ones((batch_size,1),dtype=int), solution_flag], dim=1) 
        
        # rearrange the problems, so that the solutions are [0,1,2,...]
        problem_coords_sorted = problems_coord.gather(dim=1, index=solution_node_with_depot[:,:,None].repeat(1,1,2))
        demand_sorted = problems_demand.gather(dim=1, index=solution_node_with_depot[:,:])
        
        # distance
        dist = (problem_coords_sorted[:, :, None, :] - problem_coords_sorted[:, None, :, :]).norm(p=2, dim=-1) #[batch,n+1,n+1]
        dist_to_centers_ordered_by_solution = dist.gather(index=center_point.unsqueeze(-1).repeat(1, 1, problem_size), dim=1).squeeze()[:, 1:] # [batch, n], 去掉depot
        sorted_dist, sorts = torch.sort(dist_to_centers_ordered_by_solution, dim=1)

        # neighbors' distance to centers 
        left_dist_to_centers  = torch.roll(dist_to_centers_ordered_by_solution, dims=1, shifts=1)
        right_dist_to_centers = torch.roll(dist_to_centers_ordered_by_solution, dims=1, shifts=-1)
        left_left_dist_to_centers   = torch.roll(dist_to_centers_ordered_by_solution, dims=1, shifts=2)
        right_right_dist_to_centers = torch.roll(dist_to_centers_ordered_by_solution, dims=1, shifts=-2)
        # neighbors is via depot or not
        left_is_depot = solution_flag
        right_is_depot = solution_flag.roll(dims=1, shifts=-1)
        right_right_is_depot = solution_flag.roll(dims=1, shifts=-2)
        left_left_is_depot = solution_flag.roll(dims=1, shifts=1)

        # whether 1st-order neighbors are connected to current node before destroy
        left_is_connected  = (left_dist_to_centers > dist_to_centers_ordered_by_solution) & (~left_is_depot)
        right_is_connected = (right_dist_to_centers > dist_to_centers_ordered_by_solution) & (~right_is_depot)
        # whether 2nd-order neighbors are connected to current node before destroy
        left_left_is_connected   = (left_left_dist_to_centers > dist_to_centers_ordered_by_solution) & left_is_connected & (~left_left_is_depot)
        right_right_is_connected = (right_right_dist_to_centers > dist_to_centers_ordered_by_solution) & right_is_connected & (~right_right_is_depot)
       
        # hyper_point increase
        original_plus = left_is_connected.long() + right_is_connected.long() + left_left_is_connected.long() + right_right_is_connected.long() - 1
        original_plus[original_plus<=0] = 0 
        
        plus = original_plus.clone()
        plus_at_depot = torch.sum(left_is_depot | right_is_depot, dim=1) + 1 # +1:depot

        hyper_point_increase_on_dist_sort = plus.gather(index=sorts, dim=1)      

        # decide the node to disconnet from its 1st neighbors
        hyper_point_cumulative_on_dist_sorts = torch.cumsum(hyper_point_increase_on_dist_sort, dim=1) + plus_at_depot.unsqueeze(1)
        destroy_on_dist_sorts = hyper_point_cumulative_on_dist_sorts <= reduced_problem_size 

        _, sorts_on_sorts = torch.sort(sorts, dim=1)
        destroy_on_solution_order = destroy_on_dist_sorts.gather(index=sorts_on_sorts, dim=1) 
        destroy_on_solution_order = torch.concat([torch.ones([batch_size,1]).bool(), destroy_on_solution_order], dim=1)

        # calculate destruction mask
        (destruction_mask, 
        reduced_problem_coords, reduced_problem_demand, reduced_problem_capacity,
        endpoint_mask, another_endpoint, point_couples,
        padding_mask, 
        new_problem_index_on_sorted_problem, 
        ) = \
            self.destroy_by_destroyed_node(problem_coords_sorted, demand_sorted, capacity, solution_flag_with_depot, destroy_on_solution_order, False)

        
        # check the size of hyper-graph 
        isolate_plus_endpoint_num = torch.sum(destruction_mask>0, dim=1) 
        kept_instance = isolate_plus_endpoint_num == reduced_problem_size # around 70%

        # keep valid instances
        reduced_problem_coords = reduced_problem_coords[kept_instance]
        reduced_problem_demand = reduced_problem_demand[kept_instance]
        reduced_problem_capacity = reduced_problem_capacity[kept_instance]
        endpoint_mask = endpoint_mask[kept_instance]
        another_endpoint = another_endpoint[kept_instance]
        point_couples = point_couples[kept_instance]
        padding_mask = padding_mask[kept_instance]
        new_problem_index_on_sorted_problem = new_problem_index_on_sorted_problem[kept_instance]
        problem_coords_sorted = problem_coords_sorted[kept_instance]
        demand_sorted = demand_sorted[kept_instance]

        # reset problem and solution
        self.problems = self.problems[kept_instance]
        self.solution = self.solution[kept_instance]
        self.batch_size = torch.sum(kept_instance)
        
        if return_sorted_problem:
            return (destruction_mask, 
                reduced_problem_coords, reduced_problem_demand, reduced_problem_capacity,
                endpoint_mask, another_endpoint, point_couples,
                padding_mask, 
                new_problem_index_on_sorted_problem, 
                problem_coords_sorted, 
                demand_sorted)
        
        return (destruction_mask, 
                reduced_problem_coords, reduced_problem_demand, reduced_problem_capacity,
                endpoint_mask, another_endpoint, point_couples,
                padding_mask, 
                new_problem_index_on_sorted_problem, 
                )
        
 
     
    def destroy_by_destroyed_node(self, problem_coords_sorted, demand_sorted, capacity, solution_flag_with_depot, destroy_node_mask, return_sorted_problem):
        '''
        # INPUT:
        # demand and capacity shape: [batch, problem + 1]
        # solution_flag_sorted shape: [batch, problem]
        # destroy_node_mask shape: [batch, problem + 1]
        #
        # RETURN
        #  destruction_mask, [batch, problem + 1]
        #  reduced_problems, [batch, problem + 1, 4]
        #  endpoint_mask, another_endpoint, point_couples, [batch, reduced_problem_size + 1, ~]
        #  padding_mask, [batch, reduced_problem_size + 1] padding depot
        #  new_problem_index_on_sorted_problem, [batch, reduced_problem_size + 1] 
        #
        #  the depot must be disconnected from other nodes
        '''

        batch_size = problem_coords_sorted.size(0)
        problem_size = problem_coords_sorted.size(1) - 1 #extract depot

        # 1. destruction mask: 2 for endpoint nodes, 0 for inner nodes, 1 for iso nodes
        left_edge_is_destroyed  = destroy_node_mask.bool() | destroy_node_mask.bool().roll(shifts=1, dims=1) | solution_flag_with_depot.bool() # destroy all edges from depot
        right_edge_is_destroyed = torch.roll(left_edge_is_destroyed, shifts=-1, dims=1)
        
        destruction_mask = torch.zeros((batch_size, problem_size+1), dtype=int)  
        destruction_mask[(left_edge_is_destroyed & right_edge_is_destroyed)] = 1 
        destruction_mask[(left_edge_is_destroyed ^ right_edge_is_destroyed)] = 2 

        # 2. reduced problems
        reserved_mask = destruction_mask.clone()
        reserved_mask[reserved_mask>0] = 1 #[batch, n, k]
        len_per_instance = reserved_mask.sum(dim=1).long()
        max_len = torch.max(len_per_instance).item()
        min_len = torch.min(len_per_instance).item()
        reserved_mask = reserved_mask.bool()   
        reduced_problem_with_depot_size = max_len 

        if max_len == min_len: # lucky, no padding
            endpoint_mask    = (destruction_mask[destruction_mask>0] - 1).bool().reshape((batch_size,-1))
            padding_mask     = torch.zeros(endpoint_mask.size()).bool()
            new_problem_index_on_sorted_problem = (torch.arange(problem_size+1).unsqueeze(0)).repeat((batch_size,1))[reserved_mask].reshape((batch_size,-1))

        else: 
            indices = torch.nonzero(destruction_mask >= 1).long()
            index_reverse = max_len - torch.arange(max_len)[None,:].repeat((batch_size, 1))
            padding_mask = torch.where(index_reverse>len_per_instance[:,None].repeat(1, max_len), 1, 0).bool()
           
            endpoint_mask = 1 - padding_mask.long() # padding 0 
            endpoint_mask[~padding_mask] = destruction_mask[destruction_mask>0]-1 
            endpoint_mask = endpoint_mask.bool()

            new_problem_index_on_sorted_problem = torch.where(index_reverse>len_per_instance[:,None].repeat(1, max_len), 0, -1) # -1: indicate the node should be kept
            new_problem_index_on_sorted_problem[new_problem_index_on_sorted_problem<0] = indices[:,1]

        reduced_problem_coords = problem_coords_sorted.gather(1, new_problem_index_on_sorted_problem[:,:,None].repeat((1,1,2)))
        node_demand_of_reduced_problem = demand_sorted.gather(1, new_problem_index_on_sorted_problem[:,:])

        # 3. point couples
        cumulative_sum = torch.cumsum(endpoint_mask.int(), dim=1) 
        is_left_endpoint = torch.zeros((batch_size, reduced_problem_with_depot_size), dtype=bool)
        is_left_endpoint[(cumulative_sum % 2)==1] = 1
        is_right_endpoint = torch.roll(is_left_endpoint, dims=1, shifts=1) 

        index = torch.arange(reduced_problem_with_depot_size, dtype=torch.long).unsqueeze(0).repeat((batch_size, 1))
        another_endpoint = torch.ones((batch_size, reduced_problem_with_depot_size), dtype=torch.long) * -1  # 端点的另一个端点；如果不是端点，赋值为-1;
        another_endpoint[is_left_endpoint]  = index[is_left_endpoint] + 1
        another_endpoint[is_right_endpoint] = index[is_right_endpoint] - 1

        point_couples = index.unsqueeze(-1).repeat((1,1,2))  
        point_couples[:,:,1] = torch.where(another_endpoint > (-0.5), another_endpoint, index)      

        # 4. segment demand:         
        endpoint_batch = torch.full_like(destruction_mask, -1).long()
        endpoint_id = torch.nonzero(destruction_mask == 2).long() # [2, 3]表示instance 2 的id_3
        endpoint_batch[endpoint_id[:, 0], endpoint_id[:, 1]] = endpoint_id[:, 1] # [batch, n], 如果不是endpoint, 为-1 
        left_endpoints = endpoint_id[::2]
        right_endpoints = endpoint_id[1::2]

        demand_cumsum = demand_sorted.cumsum(dim=1)
        demand_cumsum_left  = demand_cumsum[left_endpoints[:,0], left_endpoints[:,1]]
        demand_cumsum_right = demand_cumsum[right_endpoints[:,0], right_endpoints[:,1]]
        demand_left = demand_sorted[left_endpoints[:,0], left_endpoints[:,1]]

        segment_demand = demand_cumsum_right - demand_cumsum_left + demand_left 
        segment_demand = segment_demand.repeat_interleave(2, dim=0) # left and right endpoint

        reduced_problem_demand = torch.zeros((batch_size, reduced_problem_with_depot_size))
        reduced_problem_demand[~endpoint_mask] = node_demand_of_reduced_problem[~endpoint_mask] 
        reduced_problem_demand[endpoint_mask] = segment_demand

        # capacity
        reduced_problem_capacity = capacity[:, :reduced_problem_with_depot_size]

        # reduced solutions
        reduced_solution_flag = solution_flag_with_depot.gather(1, new_problem_index_on_sorted_problem[:, 1:]) # 去掉depot
        reduced_solution_node = torch.arange(reduced_problem_with_depot_size-1).unsqueeze(0).repeat(batch_size, 1) + 1

        # reset problem 和 solution
        self.problems = torch.concat([reduced_problem_coords, reduced_problem_demand[:,:,None], reduced_problem_capacity[:,:,None]], dim=2)
        self.problem_size = reduced_problem_with_depot_size - 1 # extract the depot
        self.solution = torch.concat([reduced_solution_node.unsqueeze(-1), reduced_solution_flag.unsqueeze(-1)], dim=2)

        
        if return_sorted_problem:
            return (destruction_mask, 
                reduced_problem_coords, reduced_problem_demand, reduced_problem_capacity,
                endpoint_mask, another_endpoint, point_couples,
                padding_mask, 
                new_problem_index_on_sorted_problem, 
                problem_coords_sorted, 
                demand_sorted)
        
        return (destruction_mask, 
                reduced_problem_coords, reduced_problem_demand, reduced_problem_capacity,
                endpoint_mask, another_endpoint, point_couples,
                padding_mask, 
                new_problem_index_on_sorted_problem, 
                )
    
    def coordinate_transform(self, coordinates):    
        # coordinate transformation
        max_x, indices_max_x = coordinates[:,:,1].max(dim=1)
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
    

    def Rearrange_solution_clockwise(self, problem, solution):

        problem_size = solution.shape[1]
        coor = problem[:, :, [0, 1]].clone()
        order_node = solution[:, :, 0]
        order_flag = solution[:, :, 1]

        # 1. find number of subtours and their length
        batch_size = solution.shape[0]
        visit_depot_num = torch.sum(order_flag, dim=1) # subtour num of each instances
        all_subtour_num = torch.sum(visit_depot_num)
        fake_solution = torch.cat((order_flag, torch.ones(batch_size)[:, None]), dim=1) # [batch, problem_size + 1]

        # sub tours length
        start_from_depot = fake_solution.nonzero() 
        start_from_depot_1 = start_from_depot[:, 1] 
        start_from_depot_2 = torch.roll(start_from_depot_1, shifts=-1)
        sub_tours_length = start_from_depot_2 - start_from_depot_1
        max_subtour_length = torch.max(sub_tours_length)

        # 2. get subtours, padding at two level: tour length and number of tour in each instance
        repeat_solutions_node = order_node.repeat_interleave(visit_depot_num, dim=0) 
        double_repeat_solution_node = repeat_solutions_node.repeat(1, 2)
        start_from_depot2 = order_flag.nonzero()
        start_from_depot3 = order_flag.roll(shifts=-1, dims=1).nonzero()
        x1 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
            >= start_from_depot2[:, 1][:, None] # mask, 
        x2 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
            <= start_from_depot3[:, 1][:, None] # mask
        x3 = (x1 * x2).long() # mask
        sub_tours = double_repeat_solution_node * x3 # [all_subtour_num, problem_size*2]

        x4 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
            < (start_from_depot2[:, 1][:, None] + max_subtour_length)
        x5 = x1 * x4
        sub_tours_padding = sub_tours[x5].reshape(all_subtour_num, max_subtour_length) # [all_subtour_num, max_subtour_length]
        subtour_lengths = (sub_tours_padding > 1).int().sum(1)

        # 3. centroid of each subtour
        repeated_coor = torch.repeat_interleave(coor, repeats=visit_depot_num, dim=0)
        depot_coor = repeated_coor[:, [0], :].clone()
        repeated_coor[:, 0, :] = 0
        subtours_coor = repeated_coor.gather(dim=1, index=sub_tours_padding[:, :, None].repeat(1, 1, 2))
        subtours_coor = torch.cat((subtours_coor, depot_coor), dim=1)
        subtours_coor_sum = torch.sum(subtours_coor, dim=1)
        subtours_centroid = subtours_coor_sum / (subtour_lengths + 1)[:, None]
        subtours_centroid_total_num = subtours_centroid.shape[0]

        # 4. calculate the centroids 
        temp_tri = np.triu(np.ones((len(visit_depot_num), len(visit_depot_num))), k=1)
        visit_depot_num_numpy = visit_depot_num.clone().cpu().numpy()
        temp_index = np.dot(visit_depot_num_numpy, temp_tri)

        # where is the first subtour in an isntance
        temp_index_1 = torch.from_numpy(temp_index).long().to(self.device)
        # where is the last subtour in an isntance
        temp_index_2 = visit_depot_num + temp_index_1

        x1 = torch.arange(subtours_centroid_total_num)[None, :].repeat(batch_size, 1) >= temp_index_1[:, None]
        x2 = torch.arange(subtours_centroid_total_num)[None, :].repeat(batch_size, 1) < temp_index_2[:, None]
        x3_ = (x1 * x2).int()
        x3 = x3_[:, :, None].repeat(1, 1, 2)

        subtours_centroid_repeat = subtours_centroid[None, :, :].repeat(batch_size, 1, 1)
        subtours_centroid_sperate = subtours_centroid_repeat * x3

        # decide a base centroid
        index2 = temp_index_1.clone().unsqueeze(1).unsqueeze(2).repeat(1, 1, 2)
        based_centroids = subtours_centroid_sperate.gather(dim=1, index=index2)

        # 5. calculate angle
        # centroid to depot
        single_depot_coor = coor[:, [0], :]
        repeated_depot_coor = coor[:, [0], :].repeat(1, all_subtour_num, 1)

        all_centroid_depot_vectors = subtours_centroid_sperate - repeated_depot_coor
        based_centroid_depot_vectors = based_centroids - single_depot_coor
        repeated_based_centroid_depot_vectors = based_centroid_depot_vectors.repeat(1, all_subtour_num, 1)

        # angle: centroid - depot - based centroid 
        x1_times_x2 = (repeated_based_centroid_depot_vectors * all_centroid_depot_vectors).sum(2)

        x1_module_length = torch.sqrt((repeated_based_centroid_depot_vectors ** 2).sum(2))
        x2_module_length = torch.sqrt((all_centroid_depot_vectors ** 2).sum(2))

        cos_value = x1_times_x2 / (x1_module_length * x2_module_length)
        cos_value[cos_value.ge(1)] = 1 - 1e-5
        cos_value[cos_value.le(-1)] = -1 + 1e-5 
        cross_value = np.cross(repeated_based_centroid_depot_vectors.cpu().numpy(),
                            all_centroid_depot_vectors.cpu().numpy())

        cross_value = torch.tensor(cross_value)
        negtivate_sign_2 = torch.ones(size=(cross_value.shape[0], cross_value.shape[1]))
        negtivate_sign_2[cross_value.lt(0)] = -1

        theta_value = torch.arccos(cos_value)  
        theta_value = torch.where(torch.isnan(theta_value), torch.full_like(theta_value, 2 * 3.1415926), theta_value)
        theta_value = negtivate_sign_2 * theta_value

        theta_value[theta_value.lt(0)] += 2 * 3.1415926

        #############################################################################
        # 6. sort subtours by angle
        theta_value[x3_.le(0)] = 6 * 3.1415926
        theta_value_sort_value, theta_value_sort_index = torch.sort(theta_value, dim=1) 
        repeated_sub_tours_padding = sub_tours_padding.unsqueeze(0).repeat(batch_size, 1, 1) #[batch_size, all_subtour_num, max_subtour_length]
        gather_theta_value_sort_index = theta_value_sort_index.unsqueeze(2).repeat(1, 1, max_subtour_length) 

        resort_repeated_sub_tours_padding = repeated_sub_tours_padding.gather(dim=1, index=gather_theta_value_sort_index) 

        x4 = torch.arange(all_subtour_num)[None, :].repeat(batch_size, 1)
        x5 = (x4 < visit_depot_num[:, None]).int()
        x6 = x5.unsqueeze(2).repeat(1, 1, max_subtour_length) #[batch_size, all_subtour_num, max_subtour_length]

        resort_repeated_sub_tours_padding = resort_repeated_sub_tours_padding * x6
        resort_repeated_sub_tours_padding = resort_repeated_sub_tours_padding.reshape(batch_size, -1) #[batch_size, all_subtour_num, max_subtour_length]

        resort_sub_tours = resort_repeated_sub_tours_padding[resort_repeated_sub_tours_padding.gt(0)].reshape(batch_size,
                                                                                                            -1)

        repeated_sub_tours_length = sub_tours_length[sub_tours_length.gt(0)].unsqueeze(0).repeat(batch_size, 1)

        resort_repeated_sub_tours_length = repeated_sub_tours_length.gather(dim=1, index=theta_value_sort_index)
        resort_repeated_sub_tours_length = resort_repeated_sub_tours_length * x5
        max_subtour_number = visit_depot_num.max()
        resort_repeated_sub_tours_length = resort_repeated_sub_tours_length[:, :max_subtour_number]

        # 2.5 transform the subtours to node-flag form
        temp_tri = np.triu(np.ones((batch_size, max_subtour_number.item(), max_subtour_number.item())), k=1)
        resort_repeated_sub_tours_length_numpy = resort_repeated_sub_tours_length.clone().cpu().numpy()
        temp_index = np.dot(resort_repeated_sub_tours_length_numpy, temp_tri)
        temp_index_1 = torch.from_numpy(temp_index).long().to(self.device)
        index1 = torch.arange(batch_size)
        temp_index_1 = temp_index_1[index1, index1]
        temp_index_1[temp_index_1.ge(problem_size)] = 0

        flag = torch.zeros(size=(batch_size, problem_size), dtype=torch.int)
        index1 = torch.arange(batch_size)[:, None].repeat(1, max_subtour_number)
        flag[index1, temp_index_1] = 1

        solution_ = torch.cat([resort_sub_tours.unsqueeze(2), flag.unsqueeze(2)], dim=2)
        return solution_
    
