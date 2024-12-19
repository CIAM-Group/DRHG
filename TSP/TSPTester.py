
import torch

import os


from logging import getLogger

from TSP.TSPEnv import TSPEnv as Env
from TSPModel_DRHG import TSPModel as Model_REC_v5_2
from TSPModel_DRHG_aug import TSPModel as Model_REC_v5_2_rp
from utils.utils import *
from utils_for_tester import assemble_solution_for_sorted_problem_batch, compare_to_optimal

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
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # set shortcuts
        self.iter_budget = tester_params['iter_budget']
        self.destroy_mode    = tester_params['destroy_mode']
        self.destroy_params  = tester_params['destroy_params']
        self.initial_solution_path = tester_params['initial_solution_path']

        # result folder, logger
        self.logger = getLogger(name='tester')
        self.result_folder = get_result_folder()

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

        # ENV and MODEL
        if self.env_params['use_model'] == 'REC_v5_2':
            self.model = Model_REC_v5_2(**self.model_params)
            self.env = Env(**self.env_params)
        elif self.env_params['use_model'] == 'REC_v5_2_rp':
            self.model = Model_REC_v5_2_rp(**self.model_params)
            self.env = Env(**self.env_params)
        else: 
            raise NotImplementedError("{} not implemented".format(self.env_params['use_model']))

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

        if self.env_params['load_way']=='allin':
            self.env.load_raw_data(self.tester_params['test_episodes'] )

        # k_nearest = self.env_params['k_nearest']
        # beam_width = self.env_params['beam_width']
        # decode_method = self.env_params['decode_method']

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        test_num_episode = self.tester_params['test_episodes']
        episode = 0
        problems_100 = []
        problems_100_200 = []
        problems_200_500 = []
        problems_500_1000 = []
        problems_1000 = []
        gap_log_all = []
        solution = []
        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score, score_student_mean, aug_score, problems_size, gap_log, batch_solution = self._test_one_batch(episode,batch_size, clock=self.time_estimator_2)
            gap_log_all.extend(gap_log)
           
            score_AM.update(score, batch_size)
            score_student_AM.update(score_student_mean, batch_size)
            aug_score_AM.update(aug_score, batch_size)

            solution.extend(batch_solution)
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

                self.test_solution = torch.concat(solution, dim=0)


        return score_AM.avg, score_student_AM.avg, gap_, gap_log_all

    def _test_one_batch(self, episode, batch_size, clock=None):

        # Ready
        ###############################################
        self.model.eval()

        with torch.no_grad():

            self.env.load_problems(episode, batch_size)
            self.origin_problem = self.env.problems
            reward, done = self.env.reset()
            self.optimal_length = self.env._get_travel_distance_2(self.origin_problem, self.env.solution)
            self.optimal_solution = self.env.solution.clone()

            # load initial solution
            self.env.selected_node_list = torch.load(self.initial_solution_path, map_location=self.device)[episode:episode + batch_size]
            best_solution = self.env.selected_node_list.clone().long()   
            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_solution)

            escape_time, _ = clock.get_est_string(1, 1)
            self.logger.info("initial solution, gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}".format(
                ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100, escape_time,
            current_best_length.mean().item(), self.optimal_length.mean().item()))

            # torch.save(best_solution, '{}/initial_tsp{}.pt'.format(
            #                 self.result_folder, self.env.problem_size,))
            # torch.save(self.origin_problem, '{}/original_problem{}.pt'.format(
            #                 self.result_folder, self.env.problem_size,))

            B_V = batch_size * 1
            ########################################## Destroy and repair ########################################
            current_solution = torch.zeros(best_solution.size(), dtype=int)
            # 先重置Env：
            self.env.problems = self.origin_problem
            self.env.problem_size = self.origin_problem.size(1)
            self.env.solution = best_solution

            destroy_mode = self.destroy_mode[0]
            destroy_params = self.destroy_params[destroy_mode]

            
            # 生成新解
            iter_budget = self.iter_budget
            num_interval = torch.sqrt(torch.tensor(iter_budget)).long()
            center_x = (torch.arange(num_interval) + 0.5)/ num_interval
            center_y = (torch.arange(num_interval) + 0.5) / num_interval
            gap_log = []
            optimal_num_list = []
            for bbbb in range(iter_budget):

                # sampling reduced_problem      
                # 重置了env.problems, env.problem_size, env.solution
                if destroy_mode == "knn-location" and destroy_params:
                    destroy_params['center_location'] = (random.choice(center_x), random.choice(center_y))
                elif destroy_mode == "knn-random":
                    destroy_params['center_location'] = (torch.rand(2,1))
                    destroy_mode = "knn-location"
                    print("!!!!!!!!!!!!!!!!!!!!!!!")
                # print(destroy_params)
                destruction_mask, reduced_problems, endpoint_mask, another_endpoint, point_couples, padding_mask, new_problem_index_on_sorted_problem, sorted_problems, shift = \
                                            self.env.sampling_reduced_problems(destroy_mode, destroy_params, norm_p=None, return_sorted_problem=True, if_return=True )


                # pre step of reconstruction
                reward, done = self.env.reset() # 重置env.selected_node_list 为空; env.first_node 和 env.last_node似乎没什么用
                selected_teacher_all = torch.ones(size=(B_V,  0),dtype=torch.int)
                selected_student_all = torch.ones(size=(B_V,  0),dtype=torch.int)               
                state, reward, reward_student, done = self.env.pre_step()  # state: data, first_node = current_node
                self.logger.info('test reduced problem size:{} '.format(self.env.problem_size))
                if self.tester_params['coordinate_transform']:
                    state.data = self.env.coordinate_transform(state.data.clone())
                    self.logger.info('coordinate_transform imposed.')

                # get solution on reduced problem
                current_step = 0
                while not done:
                # 如果是 test mode, 则 selected_teacher = selected_student, 都是 model 生成的 solution
                    if current_step == 0:
                        selected_teacher= torch.zeros((batch_size),dtype=torch.int64)  # B_V = 1
                        selected_student = selected_teacher
                        last_selected = selected_student
                        last_is_second_endpoint = torch.zeros((batch_size),dtype=bool)

                    else:
                        # 先判断是否last_selected是端点且另一个端点没被选过(不是第二个端点)
                        # 如果是，则直接选择另一个端点
                        # 如果上一个是padding, 则选择的index应该为current_step
                        last_is_padding = padding_mask[:,current_step-1]
                        last_is_endpoint = torch.gather(endpoint_mask, dim=1, index=last_selected.unsqueeze(1)).squeeze()
                        connect_to_another_endpoint = last_is_endpoint & (~last_is_second_endpoint)
                        selected_teacher, _, _, selected_student = self.model(state, 
                                                                                  self.env.selected_node_list, 
                                                                                  self.env.solution, 
                                                                                  current_step, 
                                                                                  point_couples=point_couples, 
                                                                                  endpoint_mask=endpoint_mask)
                #         state, selected_node_list, solution, current_step, point_couples=None, endpoint_mask=None
                        
                        selected_student[connect_to_another_endpoint] = \
                                another_endpoint.gather(index=last_selected.unsqueeze(1), dim=1).squeeze(1)[connect_to_another_endpoint]
                        selected_student[last_is_padding] = current_step
                        selected_teacher = selected_student
                        last_selected    = selected_student
                        last_is_second_endpoint = connect_to_another_endpoint
                        # print('select by constraint: ', selected_student )
                        # raise ValueError('stop')
                    current_step += 1

                    selected_teacher_all  = torch.cat((selected_teacher_all, selected_teacher[:,  None]), dim=1)
                    selected_student_all = torch.cat((selected_student_all, selected_student[:, None]), dim=1)
                    
                    state, reward, reward_student, done = self.env.step(selected_teacher, selected_student) # 里面调用了get_travel_distance(),需要知道env.solution
                    # reward是reduced problem没有重构时的reward, reward_student是重构后的；但都没有用到
                    # print(self.env.selected_node_list)
                
                    

                # 把reduced后的问题的解复原到原问题上;        
                # reduced_solution_indexed_by_sorted_problem = new_problem_index_on_sorted_problem.gather(dim=1, index=self.env.selected_node_list)
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
                
                # current_solution_v2 = best_solution.gather(1, index=complete_solution_on_sorted_problem_batch)
                current_solution = best_solution.gather(1, index=complete_solution_on_sorted_problem_batch)
                current_length = self.env._get_travel_distance_2(self.origin_problem, current_solution)

                is_better = current_length < current_best_length - 1e-6
                self.logger.info("improved: {}".format(torch.sum(is_better).item()))

                best_solution[is_better,:] = current_solution[is_better,:]
                current_best_length[is_better] = current_length[is_better]


                # 计算完整个batch, 重置Env
                self.env.problems = self.origin_problem
                self.env.problem_size = self.origin_problem.size(1)
                self.env.solution = best_solution

                escape_time,_ = clock.get_est_string(1, 1)

                self.logger.info("repair step{},  gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}".format(
                   bbbb, ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100, escape_time,
                    current_best_length.mean().item(), self.optimal_length.mean().item()))

                
                is_optimal = compare_to_optimal(best_solution, self.optimal_solution)
                optimal_num = torch.sum(is_optimal).item()
                # optimal_num_list.append([bbbb, optimal_num])
                # np.savetxt(self.result_folder+'/optimal_num_ep_{}_to_{}.txt'.format(episode,episode+batch_size), 
                #             optimal_num_list, delimiter=',' , fmt='%d')
                self.logger.info('optimal_num: {}'.format(optimal_num))


            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_solution)

            print(f'current_best_length', ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100, '%')

            self.logger.info("-------------------------------------------------------------------------------")

            ####################################### END destroy and repair #########################################


            return self.optimal_length.mean().item(), current_best_length.mean().item(), current_best_length.mean().item(), self.env.problem_size, gap_log, best_solution
