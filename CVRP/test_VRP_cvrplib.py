##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
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

from CVRP.VRPTester_cvrplib import VRPTester as Tester

##########################################################################################
# test settings
b = os.path.abspath(".").replace('\\', '/')
use_model = 'DRHG'
model_load_path  = 'result/vrp_pretrained'
model_load_epoch = 100

problem_size = 'cvrplib'
iter_budget = 3
knn_k = [20, 200]
load_way = 'pt'
pt_load = {'enable': True,
            'path': 'data/cvrp_problem_solution_cost.pt',
            }

mode = 'test'
destroy_mode = 'knn-location'

middle_name = '/lib_k_{}_{}/'.format(knn_k[0],knn_k[1])
process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))


test_paras = {
   'cvrplib': [None, 5, 1], # 193 in total; None means using pt_path
}

def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 4



##########################################################################################
# params

env_params = {
    'mode': 'test',
    'test_in_vrplib':True,
    'load_way': load_way,
    'sub_path': False,
    'use_model': use_model,
    'data_path': None, 
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
    'model_load': {
        'path' : model_load_path,  # directory path of pre-trained model and log files saved.
        'epoch': model_load_epoch,  # epoch version of pre-trained model to laod.
    },
    'test_episodes': test_paras[problem_size][1],   # 65
    'test_batch_size': test_paras[problem_size][2],
    'initial_solution_path': "./sweep_solution/sweep_solution_{}.pt".format(problem_size),
    'iter_budget': iter_budget,
    'destroy_mode': [destroy_mode],
    'destroy_params': { 
        'knn-location':{
            'center_location': None, # None means random
            'knn_k': knn_k,
            },
        },
    'rearrange_solution':True,
    'pt_load':pt_load,
    
}

logger_params = {
        'log_file': {
            'desc': 'cvrplib_iter_{}'.format(iter_budget),
            'sub_folder': model_load_path.split('/')[-1] + '_ep' + str(model_load_epoch) + middle_name,
            'filename': 'log.txt',
            'filepath': './test_by_model/' + '{sub_folder}' + process_start_time.strftime(
                "%Y%m%d_%H%M%S") + '{desc}'
        }
    }




##########################################################################################
# main
def main():

    if DEBUG_MODE:
        _set_debug_mode()

    if destroy_mode == 'knn-location':
        tester_params['destroy_params']['knn-location']['knn_k'] = knn_k

    create_logger(**logger_params)
    logger = _print_config()
    begin = time.time()

    tester = Tester(env_params=env_params,
                model_params=model_params,
                tester_params=tester_params)

    copy_all_src(tester.result_folder)

    score_optimal, score_student = tester.run()    
    end = time.time()
    tester.logger.info('total time: {}s'.format(int(end - begin)))
    return score_optimal, score_student


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

##########################################################################################

if __name__ == "__main__":

    main()


