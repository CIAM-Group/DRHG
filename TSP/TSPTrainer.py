# 会被 constraint 选择的部分不传梯度

import os
import sys
import random
# os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

import torch
from logging import getLogger

from TSP.TSPEnv import TSPEnv as Env
from TSP.TSPModel_DRHG import TSPModel as Model_REC_v5_2
from TSP.TSPModel_DRHG_aug import TSPModel as Model_REC_v5_2_rp

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from TSP.test_REC_for_training_CT import test_in_train
from utils.utils import *
from torch import nn
import warnings




class TSPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params       = env_params # {'problem_size': 100, 'pomo_size': 100}
        self.model_params     = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params   = trainer_params
        self.destroy_mode     = trainer_params['destroy_mode']
        self.destroy_params   = trainer_params['destroy_params']

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda'] # True
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num'] # 0
            torch.cuda.set_device(cuda_device_num)
            self.device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        random_seed = 123
        torch.manual_seed(random_seed)
        
        # Main Components
        if self.env_params['use_model'] == 'REC_v5_2':
            self.model = Model_REC_v5_2(**self.model_params)
            self.env = Env(**self.env_params)
        elif self.env_params['use_model'] == 'REC_v5_2_rp':
            self.model = Model_REC_v5_2_rp(**self.model_params)
            self.env = Env(**self.env_params)
        else:
            raise NotImplementedError("{} not implemented".format(self.env_params['use_model']))

        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        # print(model_load)
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            print(self.optimizer)
            print(self.scheduler)
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):

        total = sum([param.nelement() for param in self.model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))
        self.time_estimator.reset(self.start_epoch)

        if self.env_params['load_way']=='allin':
            self.env.load_raw_data(self.trainer_params['train_episodes'] )
        elif self.env_params['load_way']=='pt':
            self.env.load_pt_data(self.trainer_params['train_episodes'], self.env_params['pt_data_path'], self.device)
        total = sum([param.nelement() for param in self.model.parameters()])

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
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            if epoch > 1:  # save latest images, every epoch
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

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

                # test the model 
                for destroy in self.destroy_mode:
                    overwrite_tester_params = {
                        'cuda_device_num': self.trainer_params['cuda_device_num'],
                        'model_load':{
                            'path': self.result_folder,
                            'epoch': epoch
                        },
                    }
                    overwrite_env_params = {
                        'use_model': self.env_params['use_model']
                    }
                    overwrite_model_params = self.model_params
                    score_optimal, score_student ,gap = test_in_train(
                            overwrite_tester_params, overwrite_env_params, overwrite_model_params)

                save_gap.append([score_optimal, score_student,gap])
                np.savetxt(self.result_folder+'/gap.txt',save_gap,delimiter=',',fmt='%s')

            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

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

            avg_score,score_student_mean, avg_loss, truely_bs = self._train_one_batch(episode,batch_size)

            score_AM.update(avg_score, truely_bs)
            score_student_AM.update(score_student_mean, truely_bs)
            loss_AM.update(avg_loss, truely_bs)

            episode += batch_size


            loop_cnt += 1
            # if loop_cnt <= 10:
            self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f}, Score_studetnt: {:.4f},  Loss: {:.4f}'
                             .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                     score_AM.avg, score_student_AM.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f}, Score_studetnt: {:.4f}, Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, score_student_AM.avg, loss_AM.avg))

        return score_AM.avg, score_student_AM.avg, loss_AM.avg

    def _train_one_batch(self, episode, batch_size):

        # Prep
        ###############################################
        self.model.train() 

        destroy_mode = random.choice(self.destroy_mode)
        destroy_params = self.destroy_params[destroy_mode]

        destruction_mask, reduced_problems, endpoint_mask, another_endpoint, point_couples, padding_mask, new_problem_index_on_sorted_problem, = \
                                            self.env.load_reduced_problems(episode, batch_size, destroy_mode, destroy_params)
        
        reward, done = self.env.reset()
        truely_bs = reduced_problems.size(0)

        # train 
        ###############################################
        state, reward, reward_student, done = self.env.pre_step()
        self.logger.info('coordinate_transform imposed')
        state.data = self.env.coordinate_transform(state.data.clone())

        current_step=0
        self.logger.info('reduced problem size: {}, batch size: {} '.format(self.env.problem_size, truely_bs))

        loss_list = []
        while not done:

            if current_step == 0:
                selected_teacher = self.env.solution[:, -1] # last node, is given
                selected_student = self.env.solution[:, -1]
                prob = torch.ones(self.env.solution.shape[0], 1)

                last_selected = selected_student

            elif current_step == 1:
                selected_teacher = self.env.solution[:, 0] # current node
                selected_student = self.env.solution[:, 0]
                prob = torch.ones(self.env.solution.shape[0], 1)

                last_selected = selected_teacher

            else:
                selected_teacher, prob, _, selected_student = self.model(state, 
                                                                         self.env.selected_node_list, 
                                                                         self.env.solution, 
                                                                         current_step, 
                                                                         point_couples=point_couples, 
                                                                         endpoint_mask=endpoint_mask)

                last_is_endpoint = torch.gather(endpoint_mask, dim=1, index=last_selected.unsqueeze(1)).squeeze()
                this_is_endpoint = torch.gather(endpoint_mask, dim=1, index=selected_teacher.unsqueeze(1)).squeeze()
                this_is_constraint = this_is_endpoint & last_is_endpoint
                last_selected    = selected_teacher

                loss_mean = -prob.type(torch.float64).log()[~this_is_constraint].mean()
                self.model.zero_grad()
                loss_mean.backward()
                self.optimizer.step()

                loss_list.append(loss_mean.item())

            current_step += 1
            
            state, reward, reward_student, done = self.env.step(selected_teacher, selected_student) 
       
            
        loss_batch = torch.tensor(loss_list).mean().item()
        return 0,0, loss_batch, truely_bs
    
    
