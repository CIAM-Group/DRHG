import torch
from logging import getLogger

from CVRP.VRPEnv import VRPEnv as Env
from CVRP.VRPModel import VRPModel as Model
from CVRP.test_VRP_for_training import test_in_train

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
import numpy as np
from utils.utils import *
from torch import nn
import os
import random
import numpy as np
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(1234)

class VRPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params 
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params   = trainer_params
        self.destroy_mode     = trainer_params['destroy_mode']
        self.destroy_params   = trainer_params['destroy_params']

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()
        
        self.load_way = env_params['load_way']
        if self.load_way == 'pt':
            self.pt_load = True
            print('hereeeeeeeeeeeee')
        else:
            self.pt_load = False

        # cuda
        USE_CUDA = self.trainer_params['use_cuda'] # True
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num'] # 0
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device
        self.env_params['device'] = device

        # Main Components
        if self.env_params['use_model'] == 'DRHG':
            self.model = Model(**self.model_params)
            self.env = Env(**self.env_params)
        else:
            raise NotImplementedError("{} not implemented".format(self.env_params['use_model']))

        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])
        self.env.device=device

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):

        self.time_estimator.reset(self.start_epoch)

        self.env.load_raw_data(self.trainer_params['train_episodes'], from_pt=self.pt_load, pt_path=self.env_params['data_path'])

        save_gap = []
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')
            self.env.shuffle_data()
            # Train
            train_score, train_student_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_student_score', epoch, train_student_score)
            self.result_log.append('train_loss', epoch, train_loss)
            # LR Decay
            self.scheduler.step()

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            if epoch > 1:  # save latest images, every epoch
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],self.result_log, labels=['train_loss'])

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

                copy_src = (epoch == self.start_epoch)
                for destroy in self.destroy_mode:
                    overwrite_tester_params = {
                        'use_cuda':self.trainer_params['use_cuda'],
                        'cuda_device_num': self.trainer_params['cuda_device_num'],
                        'model_load':{
                            'path': self.result_folder,
                            'epoch': epoch
                        },
                        'save_solution': False,
                        'rearrange_solution':self.trainer_params['rearrange_solution']
                    }
                    if self.destroy_mode[0] != 'fixed_size':
                        overwrite_tester_params['destroy_mode'] = [destroy]
                        overwrite_tester_params['destroy_params'] = self.destroy_params
                    overwrite_env_params = {
                        'use_model': self.env_params['use_model']
                    }
                    overwrite_model_params = self.model_params
                    score_optimal, score_student = test_in_train(
                            overwrite_tester_params, overwrite_env_params, overwrite_model_params, copy_src)
                    gap = (score_student - score_optimal) / score_optimal  * 100

                self.result_log.append('test_score', epoch, score_student)
                save_gap.append([score_optimal, score_student, gap])
                np.savetxt(self.result_folder+'/gap.txt',save_gap,delimiter=',',fmt='%s')

            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],self.result_log, labels=['train_loss'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],self.result_log, labels=['test_score'])

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes'] # 100000
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score,score_student_mean, avg_loss = self._train_one_batch(episode, batch_size)

            score_AM.update(avg_score, batch_size)
            score_student_AM.update(score_student_mean, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size


            loop_cnt += 1
            self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f}, Score_studetnt: {:.4f},  Loss: {:.4f}'
                             .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode, score_AM.avg, score_student_AM.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f}, Score_studetnt: {:.4f}, Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,score_AM.avg, score_student_AM.avg, loss_AM.avg))

        return score_AM.avg, score_student_AM.avg, loss_AM.avg

    def _train_one_batch(self, episode, batch_size):

        # Prep
        ###############################################
        self.model.train() 

        destroy_mode = random.choice(self.destroy_mode)
        destroy_params = self.destroy_params[destroy_mode]

        self.env.load_problems(episode, batch_size)
        if self.trainer_params['rearrange_solution']:
            self.env.solution = self.env.Rearrange_solution_clockwise(self.env.problems, self.env.solution)
            self.logger.info('rearrange solution clockwise')

        destruction_mask, \
        reduced_problem_coords, reduced_problem_demand, reduced_problem_capacity,\
        endpoint_mask, another_endpoint, point_couples,\
        padding_mask, \
        new_problem_index_on_sorted_problem \
            = self.env.sampling_reduced_problems(destroy_mode, destroy_params, False) 
        
        old_batch_size = batch_size + 0
        batch_size = self.env.solution.size(0)
        self.logger.info('true batch size: {}'.format(batch_size))
    
        if destroy_mode != 'by_angle':
            self.env.problems[:, :, :2] = self.env.coordinate_transform(self.env.problems[:, :, :2])
            self.logger.info('coordinate_transform imposed')
        else: 
            self.logger.info('by angle')
        self.logger.info('reduced_problem_size: {}'.format(self.env.problem_size))

        reset_state, _, _ = self.env.reset(self.env_params['mode'])
        loss_list = []

        ###############################################
        state, reward, reward_student, done = self.env.pre_step()
        
        current_step=0

        while not done: 

            if current_step == 0:
                selected_teacher = self.env.solution[:, 0, 0] 
                selected_flag_teacher = self.env.solution[:, 0, 1]
                selected_student = selected_teacher
                selected_flag_student = selected_flag_teacher
                loss_mean = torch.tensor(0)
                loss_not_count = torch.tensor(0)

                last_selected = selected_student
                this_is_another_endpoint = torch.zeros([batch_size,], dtype=bool)

            else: 
                loss_node, selected_teacher, selected_student, selected_flag_teacher, selected_flag_student = \
                    self.model(state, self.env.selected_node_list, self.env.solution, current_step,
                               raw_data_capacity=self.env.raw_data_capacity, point_couples=point_couples, endpoint_mask=endpoint_mask) 

                last_is_endpoint = torch.gather(endpoint_mask, dim=1, index=last_selected.unsqueeze(1)).squeeze()
                this_is_endpoint = torch.gather(endpoint_mask, dim=1, index=selected_teacher.unsqueeze(1)).squeeze()
                this_is_another_endpoint = this_is_endpoint & last_is_endpoint # 
                last_selected    = selected_teacher


                selected_student = selected_teacher
                selected_flag_student = selected_flag_teacher

                should_not_be_counted = this_is_another_endpoint | padding_mask[:, current_step]

                loss_mean = loss_node[~should_not_be_counted].mean()
                loss_mean = loss_mean.nan_to_num(0) 

                self.model.zero_grad()
                loss_mean.backward()
                self.optimizer.step()
            
            if torch.isnan(loss_mean):
                print(should_not_be_counted)
                raise ValueError('loss is nan')

            current_step+=1
            state, reward, reward_student, done = \
                self.env.step(selected_teacher, selected_student, selected_flag_teacher, selected_flag_student, this_is_another_endpoint)  

            loss_list.append(loss_mean.item())

        loss_batch = torch.tensor(loss_list).mean().item()

        return 0,0, loss_batch