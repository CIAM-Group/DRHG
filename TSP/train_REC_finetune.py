
##########################################################################################
# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
# DEBUG_MODE = True
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


data_path = os.path.abspath(r"/public/home/lik/project/data")


model_load_REC = {
        'enable': True,  # enable loading pre-trained model
        'path': './result/20240711_171934_train_finetune_FS2080_1k_20_800_tsp1000_20_800_CT_mask_REC_v5_2',  # directory path of pre-trained model and log files saved.
        'epoch': 149,  # epoch version of pre-trained model to laod.
                  }
prefix = "finetune_FS2080_1k_20_800"
reduced_problem_size = [20, 800]
ending_epochs = 150
# repeats = [8, 8]
tsp_size = 1000
episode_and_batch = {100:[1000000, 1024],
                     1000:[100000, 512],
                     10000:[5000, 16]}


def _set_debug_mode():
    global trainer_params

    trainer_params['epochs'] = 102
    trainer_params['train_episodes'] = 16
    trainer_params['train_batch_size'] = 8

env_params = {
    'problem_size': 100,
    'pomo_size': 1,
    'k_nearest': 1,
    'beam_width': 16,
    'decode_method': 'greedy',
    'test_in_tsplib': False,
    'tsplib_path': None,
    'data_path':os.path.join(data_path,"TSP/re_generate_train_TSP100_2.txt"),
    'mode': 'train',
    'load_way':'pt',
    'sub_path': True,
    'use_model': 'REC_v5_2',
    'pt_data_path': os.path.join(data_path,"TSP/pt_data/data_TSP{}_new.pt".format(tsp_size)),

}
# 'load_way':'separate' or 'allin'

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
    'train_episodes': episode_and_batch[tsp_size][0],
    'train_batch_size': episode_and_batch[tsp_size][1],
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
    'model_load_REC': model_load_REC,
    'destroy_mode': ['fixed_size'], # knn, segment
    'destroy_params': { 
        'fixed_size':{
            'reduced_problem_size': reduced_problem_size,
            },
        }, 
    'coordinate_transform': True,
    }

logger_params = {
    'log_file': {
        'desc': 'train_{}_tsp{}_{}_{}_CT_mask_{}'.format(prefix,
                                tsp_size, reduced_problem_size[0], reduced_problem_size[1], env_params['use_model']),
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
