"""
Code modified from:
https://github.com/ignavierng/golem/blob/main/src/utils/config.py
"""
import argparse
import sys

import yaml


def load_yaml_config(path):
    """Load the config file in yaml format.

    Args:
        path (str): Path to load the config file.

    Returns:
        dict: config.
    """
    with open(path, 'r') as infile:
        return yaml.safe_load(infile)


def save_yaml_config(config, path):
    """Load the config file in yaml format.

    Args:
        config (dict object): Config.
        path (str): Path to save the config.
    """
    with open(path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def get_args():
    """Add arguments for parser.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()

    add_dataset_args(parser)
    add_glasso_args(parser)
    add_search_args(parser)
    add_other_args(parser)

    return parser.parse_args(args=sys.argv[1:])


def add_dataset_args(parser):
    """Add dataset arguments for parser.

    Args:
        parser (argparse.ArgumentParser): Parser.
    """
    parser.add_argument('--n',
                    type=int,
                    default=1000,
                    help="Number of samples.")

    parser.add_argument('--d',
                        type=int,
                        default=5,
                        help="Number of nodes.")

    parser.add_argument('--degree',
                        type=int,
                        default=2,
                        help="Degree of graph.")

    parser.add_argument('--noise_type',
                        type=str,
                        default='gaussian',
                        help="Type of noise ['gaussian', 'exponential', 'gumbel'].")


def add_glasso_args(parser):
    """Add model arguments for parser.

    Args:
        parser (argparse.ArgumentParser): Parser.
    """
    parser.add_argument('--glasso_l1',
                        type=float,
                        default=0.2,
                        help="Coefficient of L1 penalty for GLasso.")

    parser.add_argument('--glasso_iter',
                        type=int,
                        default=5000,
                        help="Number of maximum iterations for GLasso.")

    parser.add_argument('--glasso_thres',
                        type=float,
                        default=0.0,
                        help="Thresholding for GLasso.")


def add_search_args(parser):
    """Add training arguments for parser.

    Args:
        parser (argparse.ArgumentParser): Parser.
    """
    parser.add_argument('--super_graph_method',
                        type=str,
                        default=None,
                        help="Method of super-structure to restrict search space. Set to None to disable.")

    parser.add_argument('--search_strategy',
                        type=str,
                        default='global',
                        help="Strategy of DAG search.")

    parser.add_argument('--search_method',
                        type=str,
                        default='dp',
                        help="Method of of exact search.")

    parser.add_argument('--n_jobs',
                        type=int,
                        default=1,
                        help="Number of jobs for parallel computing when performing the search.")

    parser.add_argument('--use_path_extension',
                        dest='use_path_extension',
                        action='store_true',
                        help="Whether to use optimal path extension for order graph.")

    parser.add_argument('--use_k_cycle_heuristic',
                        dest='use_k_cycle_heuristic',
                        action='store_true',
                        help="Whether to use k-cycle conflict heuristic for astar.")

    parser.add_argument('--k',
                        type=int,
                        default=3,
                        help="Parameter used by k-cycle conflict heuristic for astar.")

    parser.add_argument('--verbose',
                        dest='verbose',
                        action='store_true',
                        help="Whether to log messages related to search procedure.")


def add_other_args(parser):
    """Add other arguments for parser.

    Args:
        parser (argparse.ArgumentParser): Parser.
    """
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help="Random seed.")
