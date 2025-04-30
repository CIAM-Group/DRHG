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

import pytz
from datetime import datetime
import time

##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from TSP.TSPTester_tsplib import TSPTester as Tester


##########################################################################################
# parameters

data_load_way = 'tsplib' # especially for tsplib
tsplib_data_path = './data/tsplib_data.pt' # you should set the 'tsplib_path' in env_params 
                                           # set the 'test_in_tsplib' = True as following

use_model = 'DRHG' # DRHG; DRHG_rp
                   # DRHG: default model
                   # DRHG_rp: you can set the number of repeated first node and last node

model_load =  {'path': './result/finetuned',
                    'epoch': 120}

test_paras = {
    'tsplib': ['', 91, 1],
}
knn_k = [20, 100]

problem_size = 'tsplib'
BUDGET = 3

env_params = {
    'mode': 'test',
    'use_model': use_model,
    'load_way': data_load_way,
    'test_in_tsplib':True,
    'tsplib_path': tsplib_data_path,
    'data_path': None,
}    



model_params = {
    'mode': 'test',
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num':6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 1,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}



# # for tsplib test_batch_size = 1
tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': model_load,
    'initial_solution_path': 'RI_solution/RI_tsplib.pt',
    'test_episodes': test_paras[problem_size][1],   
    'test_batch_size': test_paras[problem_size][2],

    'destroy_mode': ['knn-location'], 
    'destroy_params': {    
        'knn-location':{    
        'center_type': 'equally', # random or equally; 
                                      # equally: precompute centers depending on budget, then sample from them
                                      # random: sample centers from the whole space 
            'knn_k': knn_k,
        },
    },

    'iter_budget': BUDGET,
    'coordinate_transform': True,

}


process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))

logger_params = {
    'log_file': {
        'desc': 'test_tsp_{}_iter{}'.format(problem_size, tester_params['iter_budget']),
        'filename': 'log.txt',
        'filepath': './result_test/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'
    }
}


##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    logger = _print_config()
    begin = time.time()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    copy_all_src(tester.result_folder)

    score_optimal, score_student, gap, _ = tester.run()

    end = time.time()
    logger.info('total time: {}s'.format(int(end-begin)))
    return score_optimal, score_student,gap



def _set_debug_mode():
    global tester_params   
    tester_params['test_episodes'] = 4
    tester_params['test_batch_size'] = 1
    tester_params['iter_budget'] = 5

def _print_config():
    logger = logging.getLogger('root')
    logger.info('use model {}, epoch {}'.format(tester_params['model_load']['path'], tester_params['model_load']['epoch']))
    logger.info('initial solution: {}'.format(tester_params['initial_solution_path']))
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

    return logger



##########################################################################################

if __name__ == "__main__":

    main()

