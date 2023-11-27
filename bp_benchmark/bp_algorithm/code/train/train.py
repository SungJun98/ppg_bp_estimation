import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import argparse
import random
import numpy as np
import torch
import pytorch_lightning as pl

# Load modules
from core.solver_s2s import Solver as solver_s2s
from core.solver_s2l import SolverS2l as solver_s2l
from core.solver_f2l import SolverF2l as solver_f2l
from core.utils import log_params_mlflow, init_mlflow

from omegaconf import OmegaConf
from time import time, ctime
import mlflow as mf
from shutil import rmtree
from pathlib import Path

import coloredlogs, logging

#############################
import pdb
from utils_ import *
#############################
coloredlogs.install()
logger = logging.getLogger(__name__)  


def main(args):        

    SEED = args.seed
    torch.cuda.manual_seed(SEED) 
    torch.cuda.manual_seed_all(SEED)
    pl.utilities.seed.seed_everything(seed=SEED)

    if os.path.exists(args.config_file) == False:         
        raise RuntimeError("config_file {} does not exist".format(args.config_file))

    time_start = time()


    # Set Config
    config = OmegaConf.load(args.config_file)
    config = merge_config_parser(config, args)

    # Hijacking Config
    config.hijack = {"reversed_total_group_count": []}

    #--- get the solver
    if config.exp.model_type in ['unet1d', 'ppgiabp', 'vnet']:
        solver = solver_s2s(config)
    
    ## Our Interest
    elif config.exp.model_type in ['resnet1d','spectroresnet','mlpbp', 'convtr']:
        torch.use_deterministic_algorithms(True)
        solver = solver_s2l(config)

    else:
        solver = solver_f2l(config)
    
    #--- training and logging into mlflow
    init_mlflow(config)
    with mf.start_run(run_name=f"{config.exp.N_fold}fold_CV_Results") as run:
        log_params_mlflow(config)
        cv_metrics = solver.evaluate() # Final Output #dict
        logger.info(cv_metrics)
        mf.log_metrics(cv_metrics) 
   
    time_now = time()
    logger.warning(f"Time Used: {ctime(time_now-time_start)}")
    
    if not config.no_result_save:
        result_path = f"{config.exp.model_type}/{config.method}"

        filtered_metrics = {k: v for k, v in cv_metrics.items() if not k.startswith('nv')}
        filtered_metrics["name"] = config.exp.exp_detail
        filtered_metrics = rename_metric(filtered_metrics, config)
        
        save_result(filtered_metrics, path=f'./{result_path}/reseults_detail.csv')

        filtered_metrics = {k: v for k, v in filtered_metrics.items() if not k.endswith('_std')}
        filtered_metrics = {k: v for k, v in filtered_metrics.items() if not k.endswith('_me')}
        save_result(filtered_metrics, path=f'./{result_path}/reseults.csv')

if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())

