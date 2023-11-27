from omegaconf import OmegaConf
import argparse

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