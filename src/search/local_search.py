from joblib import dump, load, delayed, Parallel
import logging
import os

import numpy as np

from search.exact_search import exact_search
from utils.dag import get_k_steps_neighbors, get_vstructures, get_neighbors, add_to_pdag, \
                        get_cpdag_from_pdag, get_dag_from_pdag, get_local_info
from utils.dir import create_dir


_logger = logging.getLogger(__name__)


def get_include_graph(local_dag_searched):
    include_graph = np.zeros_like(local_dag_searched)
    vstructures_searched, edges_searched = get_local_info(local_dag_searched, 0)
    for i, j in edges_searched:
        include_graph[i, j] = 1

    for i, j, k in vstructures_searched:
        include_graph[i, j] = 1
        include_graph[k, j] = 1
    return include_graph


def local_search(X, adj_und, search_method='astar', local_with_super_graph=True, use_path_extension=True,
                 use_k_cycle_heuristic=False, k=3, verbose=False, n_jobs=1, output_dir=None):
    # adj_und is a binary adjacency matrix of an undirected graph
    assert(adj_und == adj_und.T).all()    # adj_und must be symmetric
    assert ((adj_und == 0) | (adj_und == 1)).all()    # adj_und must be binary
    d = X.shape[1]

    if output_dir is None:
        memmap_dir = './joblib_memmap'
    else:
        memmap_dir = os.path.join(output_dir, 'X_memmap')
    create_dir(memmap_dir)

    # Convert X to memmap to reduce overhead for multiprocessing
    X_filename_memmap = os.path.join(memmap_dir, 'X_memmap')
    dump(X, X_filename_memmap)
    X = load(X_filename_memmap, mmap_mode='r')

    # Create pdag_trusted as writable memmap
    pdag_trusted_filename_memmap = os.path.join(memmap_dir, 'pdag_trusted_memmap')
    pdag_trusted = np.memmap(pdag_trusted_filename_memmap, dtype=np.int,
                             shape=(d, d), mode='w+')

    # Sort nodes based on number of two step neighbors
    n_two_steps_neighbors = [(node, len(get_k_steps_neighbors(adj_und, node, k=2)))
                             for node in range(d)]
    n_two_steps_neighbors.sort(key=lambda tup: tup[1])
    sorted_nodes = [tup[0] for tup in n_two_steps_neighbors]

    # With parallel computing
    # results is a list of tuples of the form (vstructures, arcs, search_stats)
    all_search_stats = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
        delayed(two_steps_search)(X, adj_und, pdag_trusted, node, search_method,
                                  local_with_super_graph, use_path_extension,
                                  use_k_cycle_heuristic, k, verbose)
        for node in sorted_nodes)

    # Without parallel computing
    # results is a list of tuples of the form (vstructures, arcs, search_stats)
#     results = [two_steps_search(X, adj_und, pdag_trusted, node, search_method,
#                                 use_path_extension, use_k_cycle_heuristic, k, verbose)
#                for node in sorted_nodes]

    # Convert memmap to numpy array
    pdag_trusted = np.array(pdag_trusted.tolist())

    # Convert PDAG to CPDAG
    cpdag_est = get_cpdag_from_pdag(pdag_trusted)
    return cpdag_est, all_search_stats


def two_steps_search(X, adj_und, pdag_trusted, node, search_method='astar', local_with_super_graph=True,
                     use_path_extension=True, use_k_cycle_heuristic=False, k=3, verbose=False, n_jobs=1):
    # To store statistics related to current seach procedure
    search_stats = {}

    # Get dag_trusted from pdag_trusted
    dag_trusted = get_dag_from_pdag(np.array(pdag_trusted.tolist()))

    # Get neighbors within two steps from current node
    two_steps_neighbors = get_k_steps_neighbors(adj_und, node, k=2)
    search_stats['two_steps_neighbors'] = two_steps_neighbors
    search_stats['num_two_steps_neighbors'] = len(two_steps_neighbors)
    if verbose:
        _logger.info('{} two-steps neighbors for {}th-node: {}'.format(len(two_steps_neighbors),
                                                                       node, two_steps_neighbors))

    if len(two_steps_neighbors) == 0:
        # Do not perform any search if the node is disjoint from others
        return set(), set(), search_stats

    neighbors_trusted = set(get_neighbors(dag_trusted, node))
    neighbors_to_search = set(get_k_steps_neighbors(adj_und, node, k=1))
    if neighbors_to_search.issubset(neighbors_trusted):
        # Do not search again if all neighbors in the super-structure have already been searched
        search_stats['num_edges_search'] = 0
        return set(), set(), search_stats

    nodes = [node] + list(two_steps_neighbors)  # First entry is the target node
    index_to_node = dict(zip(range(len(nodes)), nodes))    # Setting the node's index tracker
    X_nodes = X[:, nodes]
    local_dag_searched = dag_trusted[np.ix_(nodes, nodes)]
    include_graph = get_include_graph(local_dag_searched)

    # Search two-step neighbors with super-structure (more efficient)
    if local_with_super_graph:
        super_graph = adj_und[np.ix_(nodes, nodes)]
        search_stats['num_edges_search'] = np.abs(super_graph - include_graph).sum()
        dag_est, search_stats_ = exact_search(X_nodes, super_graph, search_method, use_path_extension,
                                              use_k_cycle_heuristic, k, verbose, include_graph)
    else:
        # Search two-step neighbors without super-structure (less efficient)
        dag_est, search_stats_ = exact_search(X_nodes, None, search_method, use_path_extension,
                                              use_k_cycle_heuristic, k, verbose, include_graph)
    search_stats.update(search_stats_)

    # Get all v-structures involved in the subgraph of nodes_curr
    vstructures = get_vstructures(dag_est)
    # Only trust the v-structures where node i is part of
    vstructures_trusted = {vstructure for vstructure in vstructures if 0 in vstructure}

    # Get the neighbors connected to node i
    neighbors = get_neighbors(dag_est, 0)
    # Get the nodes involved in v-structrures
    nodes_in_vstructures = set([i for vstructure in vstructures_trusted for i in vstructure])
    # Get arcs connecting to node i
    neighbors_not_in_vstructures = neighbors - nodes_in_vstructures
    arcs_trusted = {(0, neighbor) for neighbor in neighbors_not_in_vstructures}

    # Store results
    mapped_vstructures_trusted = {(index_to_node[i], index_to_node[j], index_to_node[k])
                                       for i, j, k in vstructures_trusted}
    mapped_arcs_trusted = {(index_to_node[i], index_to_node[j]) for i, j in arcs_trusted}

    # Add trusted arcs and v-structures to PDAG
    add_to_pdag(pdag_trusted, mapped_vstructures_trusted, mapped_arcs_trusted)
    return search_stats