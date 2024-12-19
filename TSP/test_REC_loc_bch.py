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
import time

##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from TSP.TSPTester import TSPTester as Tester


##########################################################################################
# parameters
# b = os.path.abspath(".").replace('\\', '/')

# data_path = os.path.abspath(r"/public/home/lik/project/data/TSP")
# data_path = os.path.abspath(r"D:\OneDrive - City University of Hong Kong - Student\Documents\Python Scripts\data\TSP")
# data_path = os.path.abspath("../../../data/TSP")
data_path = 'data'
model_load =  {'path': './result/20240606_180333_train_FS_20_80_CT_mask_REC_v5_2',
                    'epoch': 100}

use_model = 'REC_v5_2' #REC_v5_2_rp
# repeats = [1, 16]
destroy = [20, 50]
problem_size = 100 # 
BUDGET = 10
test_ps_list = [100]


test_paras = {
    # problem_size: [filename, episode, batch]
    # 100: ['re_generate_test_TSP100_0423_n1w.txt', 10000, 10000],
    100: ['re_generate_test_TSP100_0423_n1w.txt', 100, 100],
    200: ['re_generate_test_TSP200_0423_n128.txt', 128, 128],
    500: ['re_generate_test_TSP500_0423_n128.txt', 128, 128],
    1000: ['re_generate_test_TSP1000_0423_n128.txt', 128, 128],
    # 1000: ['re_generate_test_TSP1000_0423_n128.txt', 128, 128],
    2000: ['test_LKH3_TSP2000_n128_seed123.txt',128,128],
    3000: ['test_LKH3_TSP3000_n128_seed123.txt',128,128],
    4000: ['test_LKH3_TSP4000_n128_seed123.txt',128,128],
    5000: ['test_LKH3_TSP5000_n128_seed123.txt',128,128],
    6000: ['test_LKH3_TSP6000_n128_seed123.txt',128,128],
    7000: ['test_LKH3_TSP7000_n128_seed123.txt',128,128],
    8000: ['test_LKH3_TSP8000_n128_seed123.txt',128,128],
    9000: ['test_LKH3_TSP9000_n128_seed123.txt',128,128],
    10000:['test_LKH3_pop_TSP10000_n16.txt',16,16],
    50000:['test_LKH3_TSP50000_n16_seed123.txt',16,16],
    100000:['test_LKH3_TSP100000_n32_seed123.txt',16,16]
}


env_params = {
    'mode': 'test',
    'test_in_tsplib':False,
    'tsplib_path': None,
    'data_path': os.path.join(data_path,test_paras[problem_size][0]),
    'load_way': 'allin',
    'use_model': use_model, # sciv, common, CT
    # 'use_RI': False
}
# 'load_way':'separate' or 'allin'



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
    # 'repeated_first_node': repeats[0],
    # 'repeated_last_node': repeats[1],
}





# 如果是tsplib，'test_episodes': 91, test_batch_size = 1
tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': model_load,
    'initial_solution_path': "./RI_solution/RI_{}.pt".format(problem_size),
    'test_episodes': test_paras[problem_size][1],   # 65
    'test_batch_size': test_paras[problem_size][2],

    'destroy_mode': ['knn-location'],
    'destroy_params': { 
        'knn-location':{
            'center_type': 'equally', # equally, random, assign
            'center_location': [0.5, 0.5], # for center_type == assign
            'knn_k': destroy,
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
    tester_params['test_episodes'] = 8
    tester_params['test_batch_size'] = 4
    tester_params['iter_budget'] = 2

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


