##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
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
from CVRP.VRPTester import VRPTester as Tester

##########################################################################################
# test settings
data_dir = os.path.abspath(".").replace('\\', '/') + "/data/"
model_load_path  = 'result/'
model_load_epoch = 0

problem_size = 100
iter_budget = 5
mode = 'test'

test_paras = {
   # problem_size: [filename, episode, batch]
    100: [ 'vrp100_test_lkh.txt',128,128],
    200: ['vrp200_test_lkh.txt', 128, 128],
    500: ['vrp500_test_lkh.txt', 128, 128],
    1000: ['vrp1000_test_lkh.txt', 128, 128],
}

##########################################################################################
# params

env_params = {
    'mode': 'test',
    'test_in_vrplib':False,
    'vrplib_path': None,
    'data_path': data_dir + f"{test_paras[problem_size][0]}",
    'load_way': 'txt',
    'sub_path': False,
    'use_model': 'DRHG',
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
    'destroy_mode': ['knn-location'],
    'destroy_params': { 
        'knn-location':{
            'center_location': None, # None means random
            'knn_k': [10,70],
            },
        },
    'save_solution': True,
    'case_check': False,

}


logger_params = {
    'log_file': {
        'desc': 'test_for_train',
        'filename': 'log.txt'
    }
}

def test_in_train(overwrite_tester_params, overwrite_env_params, overwrite_model_params, copy_src=True):
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    # overwrite tester params
    for key in overwrite_tester_params.keys():
        tester_params[key] = overwrite_tester_params[key]

    for key in overwrite_env_params.keys():
        env_params[key] = overwrite_env_params[key]
    
    for key in overwrite_model_params.keys():
        model_params[key] = overwrite_model_params[key]

    print(tester_params)

    tester = Tester(env_params=env_params,
                model_params=model_params,
                tester_params=tester_params)
    
    if copy_src:
        copy_all_src(tester.result_folder)

    score_optimal, score_student = tester.run()
    return score_optimal, score_student

def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 20


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

