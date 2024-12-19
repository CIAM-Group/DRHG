
##########################################################################################
# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
DEBUG_MODE = True
CUDA_DEVICE_NUM = 0

##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from TSP.TSPTrainer import TSPTrainer as Trainer

##########################################################################################   
# parameters

data_path = 'data'

use_model = 'REC_v5_2'
reduced_problem_size = [20,80]
repeats = [1, 32]
norm_p = 1

if use_model == 'REC_v5_2_rp':
    logger_params = {
        'log_file': {
            'desc': 'ddddebug_train_P1_FS_{}_{}_rp_{}_{}_CT_mask_{}'.format(
                                    reduced_problem_size[0], reduced_problem_size[1], repeats[0], repeats[1], use_model),
            'filename': 'log.txt'
        }
    }
else:
   logger_params = {
        'log_file': {
            'desc': 'ddddebug_train_P1_FS_{}_{}_CT_mask_{}'.format(
                                    reduced_problem_size[0], reduced_problem_size[1], use_model),
            'filename': 'log.txt'
        }
    } 

def _set_debug_mode():
    global trainer_params

    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 16
    trainer_params['train_batch_size'] = 8

env_params = {
    'test_in_tsplib': False,
    'tsplib_path': None,
    'data_path':os.path.join(data_path,"re_generate_test_TSP100_0423_n1w.txt"),
    'mode': 'train',
    'load_way':'allin',
    'use_model': use_model,
}

model_params = {
    'mode': 'train',
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num':6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 1,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
    'repeated_first_node': repeats[0],
    'repeated_last_node': repeats[1],
}



optimizer_params = {
    'optimizer': {
        'lr': 5e-5,
        'weight_decay': 1e-6
                 },
    'scheduler': {
        'milestones': [1 * i for i in range(1, 150)],
        'gamma': 0.96
                 }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 100,
    'train_episodes': 1000000,
    'train_batch_size': 1024,
    'logging': {
        'model_save_interval': 1,
        'img_save_interval': 3000,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_tsp_100.json'
               },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
               },
               },
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        'path': '',  # directory path of pre-trained model and log files saved.
        'epoch': 100,  # epoch version of pre-trained model to laod.
                  },
    'destroy_mode': ['fixed_size'], # knn, segment
    'destroy_params': { 
        'fixed_size':{
            'reduced_problem_size': reduced_problem_size,
            },
        }, 
    'coordinate_transform': True,  
    }



##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()



def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()
