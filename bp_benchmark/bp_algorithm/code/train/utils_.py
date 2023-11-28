from omegaconf import OmegaConf
import argparse
import pandas as pd 
import os

def get_parser():
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument("--config_file", type=str, help="Path for the config file") 

    ## Common For DL modl
    parser.add_argument("--save_model", action="store_true", help="Save model in a directory named model-$MODEL_NAME")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--wd", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--remove_outlier", action="store_true")
    
    ## Controllable for ConvTrasformer ##
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--feature_size", type=int, default=None)
    parser.add_argument("--d_input", type=int, default=None)
    parser.add_argument("--num_filters", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--dropout", type=int, default=None)
    parser.add_argument("--num_layer", type=int, default=None)
    parser.add_argument("--d_output", type=int, default=None)
    parser.add_argument("--method", type=str, default="erm", choices=["erm", "vrex", "crex", "drex", "cdrex", "cdrex_time"])
    parser.add_argument("--no_result_save", action="store_true")

    ## Variance Penalty
    parser.add_argument("--sbp_beta", type=float, default=0, help="variance penalty for sbp")
    parser.add_argument("--dbp_beta", type=float, default=0, help="variance penalty for dpb")

    ## CREx
    parser.add_argument("--C1", type=float, default=0, help="reversed_group_count_penalty")

    ## DREx
    parser.add_argument("--tukey", action='store_true')
    parser.add_argument("--beta", default=0, type=float, help="For Tukey")
    parser.add_argument("--C21", default=0, type=float)
    parser.add_argument("--C22", default=0, type=float)



    #### Not use ####
    parser.add_argument("--pl_log", type=bool, default=False, help="Create lr Logs")
    return parser

def merge_config_parser(config,args):
    args_dict = vars(args)

    merged_config = OmegaConf.merge(config, args_dict)

    if args.max_epochs:
        merged_config.param_trainer.max_epochs = args.max_epochs
    if args.lr:
        merged_config.param_model.lr = args.lr
    if args.wd:
        merged_config.param_model.wd = args.wd
    if args.seed:
        merged_config.param_model.wd = args.seed
    # Manual Setting for Sweep
    if merged_config.exp.model_type == "convtr":
        if args.batch_size:
            merged_config.param_model.batch_size = args.batch_size
        if args.feature_size:
            merged_config.param_model.feature_size = args.feature_size
        if args.d_input:
            merged_config.param_model.d_input = args.d_input
        if args.num_filters:
            merged_config.param_model.num_filters = args.num_filters
        if args.num_heads:
            merged_config.param_model.num_heads = args.num_heads
        if args.d_model:
            merged_config.param_model.d_model = args.d_model
        if args.dropout:
            merged_config.param_model.dropout = args.dropout
        if args.num_layer:
            merged_config.param_model.num_layer = args.num_layer
        if args.d_output:
            merged_config.param_model.d_output = args.d_output
    return merged_config

def save_result(metric, path):
    metric = pd.DataFrame([metric])
    if not os.path.isfile(path):
        metric.to_csv(path, header=True, index=False)
    else:
        metric.to_csv(path, mode='a', header=False, index=False)

def rename_metric(metric, config):
    if config.method == "erm":
        return metric
    else:
        metric['name'] = f"sbp_beta-{config.sbp_beta}_dbp_beta-{config.dbp_beta}_" + metric['name']

