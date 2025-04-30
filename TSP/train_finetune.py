
##########################################################################################
# Machine Environment Config
DEBUG_MODE = True
USE_CUDA = True
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

use_model = 'DRHG' # DRHG; DRHG_rp
                   # DRHG: default model
                   # DRHG_rp: you can set the number of repeated first node and last node
model_load = {
        'enable': True,  # enable loading pre-trained model
        'path': './result/no_finetune',  # directory path of pre-trained model and log files saved.
        'epoch': 100,  # epoch version of pre-trained model to laod.
                  }

data_load_way = 'pt' # pt, txt
                     # if pt, load data from .pt file, you should set the 'pt_data_path' in env_params
                     # if txt, load data from .txt file, you should set the 'data_path' in env_params 
pt_data_path = os.path.join("./data/data_TSP1000_finetune.pt")

reduced_problem_size = [20, 800]
ending_epochs = 102

episode_and_batch = [100000, 512]


def _set_debug_mode():
    global trainer_params, reduced_problem_size

    trainer_params['epochs'] = 102
    trainer_params['train_episodes'] = 16
    trainer_params['train_batch_size'] = 8
    reduced_problem_size = [20, 80]

env_params = {
    'mode': 'train',
    'use_model': use_model,
    'load_way':'pt',
    'data_path': None,
    'pt_data_path': pt_data_path,
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
    'epochs': ending_epochs,
    'train_episodes': episode_and_batch[0],
    'train_batch_size': episode_and_batch[1],
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
    'model_load': model_load,
    'destroy_mode': ['fixed_size'], # 'fixed_size' for training
    'destroy_params': { 
        'fixed_size':{
            'reduced_problem_size': reduced_problem_size,
            },
        }, 
    'coordinate_transform': True, # please don't change this
    }

logger_params = {
    'log_file': {
        'desc': 'debug_train_finetune',
        'filename': 'log.txt'
    }
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
