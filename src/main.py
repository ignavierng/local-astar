"""
Code modified from:
https://github.com/ignavierng/golem/blob/main/src/main.py

Each run creates a directory based on current datetime to save:
- log file of training process
- experiment configurations
- observational data and ground truth
- final estimated solution
- visualization of final estimated solution
"""

import logging

import numpy as np

from data_loader import SyntheticDataset
from search.exact_search import exact_search
from search.local_search import local_search
from utils.config import save_yaml_config, get_args
from utils.dag import is_dag, get_cpdag
from utils.dir import create_dir, get_datetime_str
from utils.glasso import glasso
from utils.logging import setup_logger, get_system_info
from utils.utils import set_seed, checkpoint_after_search


def main():
    # Get arguments parsed
    args = get_args()

    # Setup for logging
    output_dir = 'output/{}'.format(get_datetime_str())
    create_dir(output_dir)    # Create directory to save log files and outputs
    setup_logger(log_path='{}/training.log'.format(output_dir), level='INFO')
    _logger = logging.getLogger(__name__)
    _logger.info("Finished setting up the logger.")

    # Get and save system info
    system_info = get_system_info()
    if system_info is not None:
        save_yaml_config(system_info, path='{}/system_info.yaml'.format(output_dir))

    # Save configs
    save_yaml_config(vars(args), path='{}/config.yaml'.format(output_dir))

    # Reproducibility
    set_seed(args.seed)

    # Load dataset
    dataset = SyntheticDataset(args.n, args.d, args.degree, args.noise_type)
    dataset.X = dataset.X - dataset.X.mean(axis=0, keepdims=True)    # Center the data
    _logger.info("Finished loading the dataset.")

    # Super-structure
    if args.super_graph_method is None:
        # Without super-structure
        super_graph = np.ones((args.d, args.d))
    elif args.super_graph_method == 'glasso':
        inv_cov_est = glasso(dataset.X, args.glasso_l1, args.glasso_iter)
        inv_cov_est[np.abs(inv_cov_est) < args.glasso_thres] = 0
        super_graph = (inv_cov_est != 0).astype(int)
    else:
        raise ValueError("Unknown super graph method.")
    # Diagonals must be zeros
    super_graph[np.diag_indices_from(super_graph)] = 0
    _logger.info("Finished computing the super-structure.")

    # Structure search
    if args.search_strategy == 'global':
        dag_est, search_stats = exact_search(dataset.X, super_graph, args.search_method,
                                             args.use_path_extension, args.use_k_cycle_heuristic,
                                             args.k, args.verbose)
        assert is_dag(dag_est)
        cpdag_est = get_cpdag(dag_est)    # Convert estimated DAG to CPDAG
    elif args.search_strategy == 'local':
        dag_est = None    # Local search only estimates a CPDAG, so no DAG is returned
        cpdag_est, search_stats = local_search(dataset.X, super_graph, args.search_method,
                                               args.local_with_super_graph, args.use_path_extension,
                                               args.use_k_cycle_heuristic, args.k, args.verbose,
                                               args.n_jobs, output_dir)
    else:
        raise ValueError("Unknown search strategy.")
    _logger.info("Finished the search procedure.")

    # Checkpoint
    checkpoint_after_search(output_dir, dataset.X, dataset.B, dataset.cpdag, dataset.skeleton,
                            super_graph, dag_est, cpdag_est, search_stats, _logger.info)


if __name__ == '__main__':
    main()
