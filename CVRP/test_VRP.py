##########################################################################################
# Machine Environment Config

DEBUG_MODE = True
USE_CUDA = True
CUDA_DEVICE_NUM = 0

##########################################################################################
# Path Config

import os
import sys
import pytz
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
import time
import numpy as np
from utils.utils import create_logger, copy_all_src

from CVRP.VRPTester import VRPTester as Tester


##########################################################################################
# test settings
use_model = 'DRHG' # Unlike TSP, this is the only model
model_load = {
        'path' : 'result/vrp_pretrained',  # directory path of pre-trained model and log files saved.
        'epoch': 100,  # epoch version of pre-trained model to laod.
    }
data_folder = 'data'
data_load_way = 'txt' # pt, txt
                     # if pt, load data from .pt file, you should set the 'pt_data_path' in env_params
                     # if txt, load data from .txt file, you should set the 'data_path' in env_params 

problem_size = 100
BUDGET = 5
knn_k = [20, 100]

test_paras = {
   # problem_size: [filename, episode, batch]
    100: [ 'vrp100_test_lkh.txt',10000,1250],
    200: ['vrp200_test_lkh.txt', 128, 128],
    500: ['vrp500_test_lkh.txt', 128, 128],
    1000: ['vrp1000_test_lkh.txt', 128, 128],
}



##########################################################################################
# params

env_params = {
    'mode': 'test',
    'use_model': use_model,
    'load_way': 'txt',   
    'data_path': os.path.join(data_folder,test_paras[problem_size][0]),
    'test_in_vrplib':False,
    'vrplib_path': None,
}

model_params = {
    'mode': 'test',
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'decoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': model_load,
    'test_episodes': test_paras[problem_size][1],   
    'test_batch_size': test_paras[problem_size][2],
    'initial_solution_path': "./sweep_solution/sweep_solution_{}.pt".format(problem_size),

    'destroy_mode': ['knn-location'],
    'destroy_params': { 
        'knn-location':{
            'center_type': 'random', # random or equally; 
                                      # equally: precompute centers depending on budget, then sample from them
                                      # random: sample centers from the whole space 
            'knn_k': knn_k,
            },
        },  
    'iter_budget': BUDGET,
    'rearrange_solution':True, # please do not change this
}


process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))
logger_params = {
    'log_file': {
        'desc': 'test_cvrp_{}_iter_{}'.format(problem_size, tester_params['iter_budget']),
        'filename': 'log.txt',
        'filepath': './result_test/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'
    }
}



##########################################################################################
# main

def _set_debug_mode():
    global tester_params, iter_budget
    tester_params['test_episodes'] = 8
    tester_params['test_batch_size'] = 4
    tester_params['iter_budget'] = 5

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()
    logger = _print_config()

    begin = time.time()
    tester = Tester(env_params=env_params,
                model_params=model_params,
                tester_params=tester_params)

    copy_all_src(tester.result_folder)

    score_optimal, score_student = tester.run()
    end = time.time()
    print('total time: {}s'.format(int(end-begin)))

    return score_optimal, score_student

def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":

    main()
