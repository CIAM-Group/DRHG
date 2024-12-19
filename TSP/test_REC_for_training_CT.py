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

import pytz
from datetime import datetime

##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from TSP.TSPTester import TSPTester as Tester


##########################################################################################
# parameters

data_path = "./data"

test_paras = {
    # problem_size: [filename, episode, batch]
    100: ['re_generate_test_TSP100_0423_n1w.txt', 10000, 2000],
    200: ['re_generate_test_TSP200_0423_n128.txt', 128, 128],
    500: ['re_generate_test_TSP500_0423_n128.txt', 128, 128],
    1000: ['re_generate_test_TSP1000_0423_n128.txt', 128, 128],
    10000:['test_LKH3_pop_TSP10000_n16.txt',16,16],
}

problem_size = 100
BUDGET = 5

env_params = {
    'decode_method': 'greedy',
    'mode': 'test',
    'test_in_tsplib':False,
    'tsplib_path': None,
    'data_path': os.path.join(data_path,test_paras[problem_size][0]),
    'load_way': 'allin',
    # 'sub_path': False,
    'use_model': 'REC', # sciv, common, CT
}


model_params = {
    'mode': 'test',
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}


# 如果是tsplib，'test_episodes': 91, test_batch_size = 1
tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': None,
    'initial_solution_path': "./RI_solution/RI_{}.pt".format(problem_size),
    'test_episodes': test_paras[problem_size][1],   # 65
    'test_batch_size': test_paras[problem_size][2],
    'destroy_mode': ['knn-location'], # knn, segment
    'destroy_params': {
        'knn-location':{
            'center_location': None,
            'knn_k': [20,80],
            },
        },
    'iter_budget': BUDGET,
    'coordinate_transform': True,

}


process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))

logger_params = {
    'log_file': {
        'desc': 'test_for_train',
        'filename': 'log.txt',
        'filepath': './result_test/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'
    }
}

#######################################################################################
# util function
def _set_debug_mode():
    global tester_params   
    tester_params['test_episodes'] = 4
    tester_params['test_batch_size'] = 2


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

##########################################################################################
# for test in training 

def test_in_train(overwrite_tester_params, overwrite_env_params, overwrite_model_params):
    if DEBUG_MODE:
        _set_debug_mode()

    # create_logger(**logger_params)
    _print_config()

    # overwrite tester params
    for key in overwrite_tester_params.keys():
        tester_params[key] = overwrite_tester_params[key]

    for key in overwrite_env_params.keys():
        env_params[key] = overwrite_env_params[key]
    
    for key in overwrite_model_params.keys():
        model_params[key] = overwrite_model_params[key]

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params,
                    )

    copy_all_src(tester.result_folder)

    score_optimal, score_student, gap, _ = tester.run()
    return score_optimal, score_student,gap

