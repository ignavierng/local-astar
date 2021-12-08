"""
Code modified from:
https://github.com/ignavier/golem/blob/main/src/utils/utils.py
"""
import random

import matplotlib.pyplot as plt
import numpy as np

from utils.dag import compute_und_accuracy, compute_cpdag_accuracy


def set_seed(seed):
    """Set random seed for reproducibility.

    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)


def checkpoint_after_search(output_dir, X, B_true, cpdag_true, skeleton_true,
                            super_graph, dag_est, cpdag_est, search_stats, print_func):
    """Checkpointing after the training ends.
    Args:
        output_dir (str): Output directory to save training outputs.
        X (numpy.ndarray): [n, d] data matrix.
        B_true (numpy.ndarray): [d, d] weighted matrix of ground truth.
        cpdag_true (numpy.ndarray): [d, d] binary adjacency matrix.
        skeleton_true (numpy.ndarray): [d, d] binary adjacency matrix.
        super_graph (numpy.ndarray): [d, d] binary adjacency matrix.
        dag_est (numpy.ndarray): [d, d] estimated DAG.
        cpdag_est (numpy.ndarray): [d, d] estimaged CPDAG.
        search_stats (dict): Some statistics related to the seach procedure.
        print_func (function): Printing function.
    """
    # Visualization
    plot_solution(B_true, super_graph, cpdag_true, cpdag_est,
                  save_name='{}/plot_solution.pdf'.format(output_dir))
    print_func("Finished plotting solution.")

    # Results for super-structure
    results_super_graph = compute_und_accuracy(skeleton_true, super_graph)
    print_func("Results of super-structure compared to true skeleton: {}.".format(results_super_graph))

    # Results for estimated CPDAG
    results_cpdag = compute_cpdag_accuracy(cpdag_true, cpdag_est)
    print_func("Results of estimated CPDAG compared to true CPDAG: {}.".format(results_cpdag))
    # print_func("Some statistics related to the seach procedure: {}.".format(search_stats))

    # Save training outputs
    np.save('{}/X.npy'.format(output_dir), X)
    np.save('{}/B_true.npy'.format(output_dir), B_true)
    np.save('{}/cpdag_true.npy'.format(output_dir), cpdag_true)
    np.save('{}/super_graph.npy'.format(output_dir), super_graph)
    if dag_est is not None:
        np.save('{}/dag_est.npy'.format(output_dir), dag_est)
    np.save('{}/cpdag_est.npy'.format(output_dir), cpdag_est)
    np.save('{}/search_stats.npy'.format(output_dir), search_stats)
    print_func("Finished saving search outputs at {}.".format(output_dir))


def plot_solution(B_true, super_graph, cpdag_true, cpdag_est, save_name=None):
    """Checkpointing after the training ends.
    Args:
        B_true (numpy.ndarray): [d, d] weighted matrix of ground truth.
        super_graph (numpy.ndarray): [d, d] binary adjacency matrix.
        cpdag_true (numpy.ndarray): [d, d] binary adjacency matrix.
        cpdag_est (numpy.ndarray): [d, d] estimaged CPDAG.
        save_name (str or None): Filename to solve the plot. Set to None
            to disable. Default: None.
    """
    fig, axes = plt.subplots(figsize=(10, 3), ncols=4)

    # Plot true DAG
    im = axes[0].imshow(B_true, cmap='RdBu', interpolation='none',
                        vmin=-2.25, vmax=2.25)
    axes[0].set_title("True DAG", fontsize=13)
    axes[0].tick_params(labelsize=13)

    # Plot estimated super-structure
    im = axes[1].imshow(super_graph, cmap='RdBu', interpolation='none',
                        vmin=-2.25, vmax=2.25)
    axes[1].set_title("Estimated super graph", fontsize=13)
    axes[1].set_yticklabels([])    # Remove yticks
    axes[1].tick_params(labelsize=13)

    # Plot true CPDAG
    im = axes[2].imshow(cpdag_true, cmap='RdBu', interpolation='none',
                        vmin=-2.25, vmax=2.25)
    axes[2].set_title("True CPDAG", fontsize=13)
    axes[2].set_yticklabels([])    # Remove yticks
    axes[2].tick_params(labelsize=13)

    # Plot estimated CPDAG
    im = axes[3].imshow(cpdag_est, cmap='RdBu', interpolation='none',
                        vmin=-2.25, vmax=2.25)
    axes[3].set_title("Estimated CPDAG", fontsize=13)
    axes[3].set_yticklabels([])    # Remove yticks
    axes[3].tick_params(labelsize=13)

    # Adjust space between subplots
    fig.subplots_adjust(wspace=0.1)

    # Colorbar (with abit of hard-coding)
    im_ratio = 3 / 10
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.05*im_ratio, pad=0.035)
    cbar.ax.tick_params(labelsize=13)
    # plt.show()

    if save_name is not None:
        fig.savefig(save_name, bbox_inches='tight')
