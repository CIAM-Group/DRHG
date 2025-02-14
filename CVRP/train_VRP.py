##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
DEBUG_MODE = True
# USE_CUDA = False
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
import numpy as np
from utils.utils import create_logger, copy_all_src

from CVRP.VRPTrainer import VRPTrainer as Trainer

##########################################################################################
# parameters

txt_path = os.path.abspath(r"D:\OneDrive - City University of Hong Kong - Student\Documents\Python Scripts\data\CVRP_HGS_train\vrp100_hgs_train_100w.txt")

pt_folder = os.path.abspath(r"D:\OneDrive - City University of Hong Kong - Student\Documents\Python Scripts\data\CVRP")

pt_path = pt_folder + '/CVRP100_100w_train_problem_solution_cost.pt'

model_load = {
        'enable': False,  # enable loading pre-trained model
        'path': None,  # directory path of pre-trained model and log files saved.
        'epoch': None,  # epoch version of pre-trained model to laod.
                  }

use_model = 'DRHG'
rearrange = True
destroy_mode = 'fixed_size'
fixed_size = [20, 80]

def _set_debug_mode():
    global trainer_params

    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 32
    trainer_params['train_batch_size'] = 16

logger_params = {
    'log_file': {
        'desc': 'train',
        'filename': 'log.txt'
    }
}

env_params = {
    'problem_size': 100,
    'pomo_size': 1,
    'k_nearest': 1,
    'beam_width': 16,
    'decode_method': 'greedy',
    'test_in_vrplib': False,
    'vrplib_path': None,
    'data_path' : txt_path,
    'mode': 'train',
    'load_way':'txt', 
    'sub_path': False, 
    'use_model': use_model,
}

model_params = {
    'mode': 'train',
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'decoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
                 },
    'scheduler': {
        'milestones': [1 * i for i in range(1, 100)],
        'gamma': 0.97
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
        'img_save_interval': 10000,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_cvrp.json'
               },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
               },
               },
    'model_load': model_load,
    'destroy_mode': [destroy_mode], 
    'destroy_params': { 
        'fixed_size':{
            'reduced_problem_size': fixed_size,
        },
        'knn-location':{
            'center_location': None, # None means random
            'knn_k': [10,70],
            },
        'by_angle':{
            'theta_a': None, # None means random
            'angle_range': [0.1*np.pi, 1*np.pi],
            },
        },
    
    'rearrange_solution': True,
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

