#%%
import os
import argparse

from time import time, ctime
from omegaconf import OmegaConf
from core.solver_s2s import Solver as solver_s2s
from core.solver_s2l import SolverS2l as solver_s2l

import coloredlogs, logging

####
from utils_ import *
####

coloredlogs.install()
logger = logging.getLogger(__name__)  

def main(args):        
    if os.path.exists(args.config_file) == False:         
        raise RuntimeError("config_file {} does not exist".format(args.config_file))

    time_start = time()
    config = OmegaConf.load(args.config_file)
    config = merge_config_parser(config, args)

    if config.exp.model_type in ['unet1d', 'ppgiabp', 'vnet']:
        solver = solver_s2s(config)
    elif config.exp.model_type in ['resnet1d','spectroresnet','mlpbp', 'convtr']:
        solver = solver_s2l(config)

    cv_metrics = solver.test()
    time_now = time()
    logger.warning(f"Time Used: {ctime(time_now-time_start)}")

    # =============================================================================
    # output
    # =============================================================================

    if not config.no_result_save:
        result_path = f"{config.exp.model_type}/{config.method}/test/)"
        os.makedirs(f'./{result_path}', exist_ok=True)

        filtered_metrics = {k: v for k, v in cv_metrics.items() if not k.startswith('nv')}
        filtered_metrics["name"] = config.exp.exp_detail
        filtered_metrics = rename_metric(filtered_metrics, config)
        
        save_result(filtered_metrics, path=f'./{result_path}/reseults_detail.csv')

        filtered_metrics = {k: v for k, v in filtered_metrics.items() if not k.endswith('_std')}
        filtered_metrics = {k: v for k, v in filtered_metrics.items() if not k.endswith('_me')}
        save_result(filtered_metrics, path=f'./{result_path}/reseults.csv')

#%%
if __name__=='__main__':
    parser = get_parser()
    main(parser.parse_args())