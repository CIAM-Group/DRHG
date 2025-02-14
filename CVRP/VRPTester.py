
import torch

import os
from logging import getLogger

from CVRP.VRPEnv import VRPEnv as Env
from CVRP.VRPModel import VRPModel as Model
from utils.utils import *
from CVRP.utils_for_vrp_tester import assemble_vrp_solution_for_sorted_problem_batch

import random
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(1234)

class VRPTester():
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='tester')
        self.result_folder = get_result_folder()

        # set shortcuts
        self.iter_budget = tester_params['iter_budget']
        self.destroy_mode    = tester_params['destroy_mode']
        self.destroy_params  = tester_params['destroy_params']
        self.initial_solution_path = tester_params['initial_solution_path']

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device
        self.env_params['device'] = device

        if self.env_params['use_model'] == 'DRHG':
            self.model = Model(**self.model_params)
            self.env = Env(**self.env_params)
        else:
            raise NotImplementedError("{} not implemented".format(self.env_params['use_model']))

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset()

        if self.env_params['load_way']=='txt':
            self.env.load_raw_data(self.tester_params['test_episodes'] )

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()

        test_num_episode = self.tester_params['test_episodes']
        episode = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score, score_student_mean = self._test_one_batch(episode, batch_size, clock=self.time_estimator)
            
            score_AM.update(score, batch_size)
            score_student_AM.update(score_student_mean, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f},Score_studetnt: {:.4f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score,score_student_mean))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" NO-AUG SCORE student: {:.4f} ".format(score_student_AM.avg))
                self.logger.info(" Gap: {:.4f}%".format((score_student_AM.avg-score_AM.avg) / score_AM.avg * 100))
                gap_ = (score_student_AM.avg-score_AM.avg) / score_AM.avg * 100


        return score_AM.avg, score_student_AM.avg


    def _test_one_batch(self, episode, batch_size, clock=None):

        capacitys = {100: 50,
                     200: 80,
                     500: 100,
                     1000: 250}

        # Ready
        ###############################################
        self.model.eval()
        self.model.mode = 'test'

        with torch.no_grad():

            # load problems and solutions
            self.env.load_problems(episode, batch_size, )
            self.origin_problem = self.env.problems
            self.optimal_length = self.env._get_travel_distance_2(self.origin_problem, self.env.solution)
            self.optimal_solution = self.env.solution.clone()
            initial_solution = torch.load(self.initial_solution_path, map_location=self.device)[episode:episode + batch_size]

            # get ready
            reset_state, _, _ = self.env.reset(self.env_params['mode'])
            best_solution = initial_solution.clone()   #self.env.selected_node_list
            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_solution)
            escape_time, _ = clock.get_est_string(1, 1)

            self.logger.info("initial solution, gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}".format(
                ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100, escape_time,
            current_best_length.mean().item(), self.optimal_length.mean().item()))


            ################################ destroy and repair ###########################################
            current_solution = torch.zeros(best_solution.size(), dtype=int)
            
            if self.tester_params['rearrange_solution']:
                best_solution = self.env.Rearrange_solution_clockwise(self.env.problems, best_solution)
                self.logger.info('rearrange solution clockwise')

            # reset Env：
            self.env.problems = self.origin_problem
            self.env.problem_size = self.origin_problem.size(1)
            self.env.solution = best_solution

            destroy_mode = self.destroy_mode[0]
            destroy_params = self.destroy_params[destroy_mode]

            # new solution
            iter_budget = self.iter_budget
            for bbbb in range(iter_budget):
                
                # destroy
                destruction_mask, reduced_problem_coords, reduced_problem_demand, reduced_problem_capacity, endpoint_mask, another_endpoint, point_couples, padding_mask, new_problem_index_on_sorted_problem, \
                    problem_coords_sorted, demand_sorted= \
                            self.env.sampling_reduced_problems(destroy_mode, destroy_params,True)
                self.logger.info('test reduced problem size:{} '.format(self.env.problem_size))

                # coordinate transformation
                self.env.problems[:, :, :2] = self.env.coordinate_transform(self.env.problems[:, :, :2])
                self.logger.info('coordinate_transform imposed')

                state, _ , done = self.env.reset(mode='test')  
                state, _, _, done = self.env.pre_step()
                current_step = 0
                # repair
                while not done:
                    loss_node, selected_teacher, selected_student, selected_flag_teacher, selected_flag_student = \
                            self.model(state, 
                                       self.env.selected_node_list, 
                                       self.env.solution,
                                       current_step,
                                       raw_data_capacity=self.env.raw_data_capacity, 
                                       point_couples=point_couples, 
                                       endpoint_mask=endpoint_mask)  

                    this_is_padding = padding_mask[:, current_step]
                    if current_step == 0:
                        selected_flag_teacher = torch.ones(batch_size, dtype=torch.int)
                        selected_flag_student = selected_flag_teacher
                        selected_student[this_is_padding] = current_step + 1
                        selected_teacher = selected_student # in testing, no teacher; only for env.step()
                        last_selected = selected_student
                        last_is_second_endpoint = torch.zeros((batch_size), dtype=bool)
                        connect_to_another_endpoint = None
                        connect_to_another_endpoint = torch.zeros((batch_size), dtype=bool)
                        last_is_endpoint = torch.gather(endpoint_mask, dim=1, index=last_selected.unsqueeze(1)).squeeze()
                    else:
                        last_is_padding = padding_mask[:, current_step-1]
                        last_is_endpoint = torch.gather(endpoint_mask, dim=1, index=last_selected.unsqueeze(1)).squeeze()
                        connect_to_another_endpoint = last_is_endpoint & (~last_is_second_endpoint) # this step should be second point

                        selected_student[connect_to_another_endpoint] = \
                                another_endpoint.gather(index=last_selected.unsqueeze(1), dim=1).squeeze(1)[connect_to_another_endpoint]
                        selected_flag_student[connect_to_another_endpoint] = 0
                        
                        selected_student[this_is_padding] = current_step + 1
                        selected_flag_student[last_is_padding] = 1

                        selected_teacher = selected_student # in testing, no teacher; only for env.step()
                        selected_flag_teacher = selected_flag_student

                        last_selected    = selected_student
                        last_is_second_endpoint = connect_to_another_endpoint
                    
                    current_step += 1
                    state, reward, reward_student, done = \
                        self.env.step(selected_teacher, selected_student, selected_flag_teacher, selected_flag_student, \
                                      is_another_endpoint=connect_to_another_endpoint, is_first_endpoint=None)

                reduced_solution_after_repair = torch.cat((self.env.selected_student_list.unsqueeze(2),
                                                           self.env.selected_student_flag.unsqueeze(2)), dim=2)
                
                # restore solution of hyper-graph on original problem
                destruction_mask_after_repair, complete_solution_on_sorted_problem, complete_flag_on_sorted_problem = assemble_vrp_solution_for_sorted_problem_batch(destruction_mask, 
                                                                                        endpoint_mask, 
                                                                                        reduced_solution_after_repair,
                                                                                        new_problem_index_on_sorted_problem,
                                                                                        padding_mask)
                
                current_solution[:,:,0] = best_solution[:,:,0].gather(1, index=complete_solution_on_sorted_problem - 1) # 注意-1: solution是从1开始的
                current_solution[:,:,1] = complete_flag_on_sorted_problem
                
                # update if improve
                current_length = self.env._get_travel_distance_2(self.origin_problem, current_solution)
                is_better = current_length < current_best_length - 1e-7
                self.logger.info("improved: {}".format(torch.sum(is_better).item()))
                best_solution[is_better,:] = current_solution[is_better,:]
                current_best_length[is_better] = current_length[is_better]

                # reset Env
                if self.tester_params['rearrange_solution']:
                    best_solution = self.env.Rearrange_solution_clockwise(self.origin_problem, best_solution)
                    self.logger.info('rearrange solution clockwise')

                self.env.problems = self.origin_problem
                self.env.problem_size = self.origin_problem.size(1)
                self.env.solution = best_solution

                # logging
                escape_time,_ = clock.get_est_string(1, 1)
                self.logger.info("RECstep{},  gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}".format(
                   bbbb, ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100, escape_time,
                    current_best_length.mean().item(), self.optimal_length.mean().item()))
                
                self.env.valida_solution_legal(self.origin_problem, best_solution, capacity_=capacitys[self.env.problem_size - 1])
                
            # all budgets are run out
            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_solution)
            self.logger.info("-------------------------------------------------------------------------------")

            return self.optimal_length.mean().item(), current_best_length.mean().item()