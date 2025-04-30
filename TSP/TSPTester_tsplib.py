import torch

import os
from logging import getLogger

from TSP.TSPEnv import TSPEnv as Env
from TSP.TSPModel_DRHG import TSPModel as Model_DRHG
from TSP.TSPModel_DRHG_aug import TSPModel as Model_DRHG_rp
from utils.utils import *
from utils_for_tester import assemble_solution_for_sorted_problem_batch

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

class TSPTester():
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params,
                 ):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params
        self.model_params = model_params
        self.iter_budget = tester_params['iter_budget']
        self.destroy_mode    = tester_params['destroy_mode']
        self.destroy_params  = tester_params['destroy_params']
        self.initial_solution_path = tester_params['initial_solution_path']
        # result folder, logger
        self.logger = getLogger(name='tester')
        self.result_folder = get_result_folder()
        self.knn_k_high = int(self.destroy_params[self.destroy_mode[0]]['knn_k'][1])
        self.recordings = {'tsp_name':[],'solution':[]}

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            # torch.set_default_tensor_type('torch.cuda.DoubleTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        if self.env_params['use_model'] == 'DRHG':
            self.model = Model_DRHG(**self.model_params)
            self.env = Env(**self.env_params)
        elif self.env_params['use_model'] == 'DRHG_rp':
            self.model = Model_DRHG_rp(**self.model_params)
            self.env = Env(**self.env_params)
        else: raise NotImplementedError("{} not implemented".format(self.env_params['use_model']))

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        torch.set_printoptions(precision=20)

        # utility
        self.time_estimator = TimeEstimator()
        self.time_estimator_2 =  TimeEstimator()

    def run(self):
        self.time_estimator.reset()
        self.time_estimator_2.reset()

        # if self.env_params['load_way']=='allin':
        #     self.env.load_raw_data(self.tester_params['test_episodes'] )

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()
        aug_score_AM = AverageMeter()
        gap_AM = AverageMeter()

        test_num_episode = self.tester_params['test_episodes']
        episode = 0
        gap_log_all = []

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score, score_student_mean, aug_score, problems_size, gap, gap_log = self._test_one_batch(episode, batch_size, clock=self.time_estimator_2)

            gap_log_all.extend(gap_log)
            score_AM.update(score, batch_size)
            score_student_AM.update(score_student_mean, batch_size)
            aug_score_AM.update(aug_score, batch_size)
            gap_AM.update(gap, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f},Score_studetnt: {:.4f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score,score_student_mean, aug_score))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" NO-AUG SCORE student: {:.4f} ".format(score_student_AM.avg))
                self.logger.info(" Gap: {:.4f}%".format((score_student_AM.avg-score_AM.avg) / score_AM.avg * 100))
                gap_ = (score_student_AM.avg-score_AM.avg) / score_AM.avg * 100
                torch.save(self.recordings, 'tsplib_result_b{}.pt'.format(self.iter_budget))
                self.logger.info('Result saved.')

        return score_AM.avg, score_student_AM.avg, gap_, gap_log_all

    def _test_one_batch(self, episode, batch_size, clock=None):

        # Ready
        ###############################################
        self.model.eval()

        with torch.no_grad():

            # load problem and optimal solution
            self.env.load_problems(episode, batch_size)
            self.origin_problem = self.env.problems
            reward, done = self.env.reset()
            self.optimal_length, tsplib_name = self.env._get_travel_distance_2(self.origin_problem, self.env.solution, test_in_tsplib=True, need_optimal=True)
            self.optimal_length, tsplib_name = self.optimal_length[episode], tsplib_name[episode]

            # load initial solution
            RI_solutions = torch.load(self.initial_solution_path, map_location=self.device)[1]
            self.env.selected_node_list = torch.tensor(RI_solutions[episode:episode + batch_size][0]).unsqueeze(0)
            best_solution = self.env.selected_node_list.clone().long()   #self.env.selected_node_list
            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_solution, test_in_tsplib=True)

            escape_time, _ = clock.get_est_string(1, 1)
            self.logger.info("initial solution, gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}".format(
                ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100, escape_time,
            current_best_length.mean().item(), self.optimal_length.mean().item()))

            B_V = batch_size * 1
            ########################################## destroy and repair ########################################
            current_solution = torch.zeros(best_solution.size(), dtype=int)
            # set env
            self.env.problems = self.origin_problem
            self.env.problem_size = self.origin_problem.size(1)
            self.env.solution = best_solution
            self.solution_1 = self.env.solution.clone()
            self.logger.info('problem size: {}'.format(self.env.problem_size))

            destroy_mode = self.destroy_mode[0]
            destroy_params = self.destroy_params[destroy_mode] # 
            destroy_params['knn_k'][1] = min(int(self.env.problem_size * 0.75), self.knn_k_high)

            
            # pre step of destruction
            iter_budget = self.iter_budget
            num_interval = torch.sqrt(torch.tensor(iter_budget)).long()
            center_x = (torch.arange(num_interval) + 0.5)/ num_interval
            center_y = (torch.arange(num_interval) + 0.5) / num_interval
            gap_log = []
            for bbbb in range(iter_budget):
                if destroy_params['center_type'] == "equally":
                    destroy_params['center_location'] = (random.choice(center_x), random.choice(center_y))
                elif destroy_params['center_type'] == "random":
                    destroy_params['center_location'] = (random.uniform(0, 1), random.uniform(0, 1))
                else: 
                    raise NotImplementedError("center_type {} not implemented".format(destroy_params['center_type']))

                # sampling reduced_problem      
                destroy_params['center_location'] = (random.choice(center_x), random.choice(center_y))
                destruction_mask, reduced_problems, endpoint_mask, another_endpoint, point_couples, padding_mask, new_problem_index_on_sorted_problem, sorted_problems, shift = \
                                            self.env.sampling_reduced_problems(destroy_mode, destroy_params, return_sorted_problem=True, if_return=True, norm_p=2 )


                # pre step of reconstruction
                reward, done = self.env.reset() 
                selected_teacher_all = torch.ones(size=(B_V,  0),dtype=torch.int)
                selected_student_all = torch.ones(size=(B_V,  0),dtype=torch.int)               
                state, reward, reward_student, done = self.env.pre_step()  
                if self.tester_params['coordinate_transform']:
                    state.data = self.env.coordinate_transform(state.data.clone())
                    self.logger.info('coordinate_transform imposed.')

                # get solution on reduced problem
                current_step = 0
                while not done:
                    if current_step == 0:
                        selected_teacher= torch.zeros((batch_size),dtype=torch.int64)  # B_V = 1
                        selected_student = selected_teacher
                        last_selected = selected_student
                        last_is_second_endpoint = torch.zeros((batch_size),dtype=bool)

                    else:
                        last_is_padding = padding_mask[:,current_step-1]
                        last_is_endpoint = torch.gather(endpoint_mask, dim=1, index=last_selected.unsqueeze(1)).squeeze()
                        connect_to_another_endpoint = last_is_endpoint & (~last_is_second_endpoint)
                        selected_teacher, _, _, selected_student = self.model(state, 
                                                                              self.env.selected_node_list, 
                                                                              self.env.solution,
                                                                              current_step,
                                                                              point_couples=point_couples, 
                                                                              endpoint_mask=endpoint_mask)
                        
                        selected_student[connect_to_another_endpoint] = \
                                another_endpoint.gather(index=last_selected.unsqueeze(1), dim=1).squeeze(1)[connect_to_another_endpoint]
                        selected_student[last_is_padding] = current_step
                        selected_teacher = selected_student
                        last_selected    = selected_student
                        last_is_second_endpoint = connect_to_another_endpoint

                    current_step += 1

                    selected_teacher_all  = torch.cat((selected_teacher_all, selected_teacher[:,  None]), dim=1)
                    selected_student_all = torch.cat((selected_student_all, selected_student[:, None]), dim=1)
                    
                    state, reward, reward_student, done = self.env.step(selected_teacher, selected_student) 
                
                           
                reduced_solution_indexed_by_sorted_problem = new_problem_index_on_sorted_problem.gather(dim=1, index=self.env.selected_node_list)
                if destroy_mode != 'knn-location':
                    best_solution = torch.roll(best_solution, shifts=shift, dims=1)
                else: 
                    problem_size = best_solution.size(1)
                    shift_ist_by_ist = shift.unsqueeze(-1)
                    shifted_index = torch.arange(problem_size).unsqueeze(0).repeat((batch_size,1)) + shift_ist_by_ist
                    shifted_index[shifted_index >= problem_size] = shifted_index[shifted_index >= problem_size] - problem_size
                    best_solution = best_solution.gather(index=shifted_index, dim=1)

                _, complete_solution_on_sorted_problem_batch = assemble_solution_for_sorted_problem_batch(destruction_mask, 
                                                                                    endpoint_mask, 
                                                                                    self.env.selected_node_list, 
                                                                                    new_problem_index_on_sorted_problem, 
                                                                                    padding_mask)
                
                current_solution = best_solution.gather(1, index=complete_solution_on_sorted_problem_batch)
                
                current_length = self.env._get_travel_distance_2(self.origin_problem, current_solution, test_in_tsplib=True)
                is_better = current_length < current_best_length - 1e-6

                self.logger.info("improved: {}".format(torch.sum(is_better).item()))

                best_solution[is_better,:] = current_solution[is_better,:]
                current_best_length[is_better] = current_length[is_better]

                # Reset env
                self.env.problems = self.origin_problem
                self.env.problem_size = self.origin_problem.size(1)
                self.env.solution = best_solution

                escape_time,_ = clock.get_est_string(1, 1)

                self.logger.info("repair step{},  gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}".format(
                   bbbb, ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100, escape_time,
                    current_best_length.mean().item(), self.optimal_length.mean().item()))

            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_solution, test_in_tsplib=True)

            self.logger.info("-------------------------------------------------------------------------------")
            gap = ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100
            gap_log.append(tsplib_name + '\n')
            gap_log.append("repair step {},  gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f} \n".format(
                                                                bbbb,
                                                                gap,
                                                                escape_time,
                                                                current_best_length.mean().item(),
                                                                self.optimal_length.mean().item()))
            
            self.recordings['tsp_name'].append(tsplib_name)
            self.recordings['solution'].append(best_solution.squeeze().unsqueeze(0))


            ####################################### END repair #########################################


            return self.optimal_length.mean().item(), current_best_length.mean().item(), current_best_length.mean().item(), self.env.problem_size, gap, gap_log
