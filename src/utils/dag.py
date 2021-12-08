import causaldag as cd
import networkx as nx
import numpy as np


def is_dag(B):
    """Check whether B corresponds to a DAG.

    Args:
        B (numpy.ndarray): [d, d] binary or weighted matrix.
    """
    return nx.is_directed_acyclic_graph(nx.DiGraph(B))


def get_skeleton(B):
    B_bin = (B != 0).astype(int)
    return ((B_bin + B_bin.T) != 0).astype(int)


def get_moral_graph(B):
    assert is_dag(B)
    return cd.DAG.from_amat(B).moral_graph().to_amat()


def get_cpdag(B):
    assert is_dag(B)
    return cd.DAG.from_amat(B).cpdag().to_amat()[0]


def get_cpdag_from_pdag(B_pdag):
    G = cd.PDAG.from_amat(B_pdag)
    G.to_complete_pdag()
    return G.to_amat()[0]


def get_dag_from_pdag(B_pdag):
    # The function G.to_dag().to_amat() from causaldag package
    # does not preserve the shape of B
    # So we need to manually preserve the shape
    B_dag = np.zeros_like(B_pdag)
    if np.all(B_pdag == 0):
        # All entries in B_pdag are zeros
        return B_dag
    else:
        G = cd.PDAG.from_amat(B_pdag)
        B_sub_dag, nodes = G.to_dag().to_amat()
        B_dag[np.ix_(nodes, nodes)] = B_sub_dag
        return B_dag


def get_vstructures(B):
    assert is_dag(B)
    return cd.DAG.from_amat(B).vstructures()


def get_neighbors(B, target_node):
    # B can be either a DAG or a PDAG
    if is_dag(B):    # DAG
        return cd.DAG.from_amat(B).neighbors_of(target_node)
    else:    # PDAG
        return cd.PDAG.from_amat(B).neighbors_of(target_node)


def get_local_info(B, node):
    # Get vstructures and edges that the target node is part of
    assert is_dag(B)
    # Get all v-structures involved in the subgraph of target node
    vstructures = get_vstructures(B)
    # Only trust the v-structures where the target node is part of
    vstructures_local = {vstructure for vstructure in vstructures if node in vstructure}

    # Get the neighbors conencted to target node
    neighbors = get_neighbors(B, node)
    # Get the nodes involved in v-structrures
    nodes_in_vstructures = set([i for vstructure in vstructures_local for i in vstructure])
    # Get arcs connecting to target node
    neighbors_not_in_vstructures = neighbors - nodes_in_vstructures
    edges_local = {(node, neighbor) if B[node, neighbor] != 0 else (neighbor, node)
                   for neighbor in neighbors_not_in_vstructures}
    return vstructures_local, edges_local


def add_to_pdag(pdag, vstructures, arcs):
    for i, j in arcs:
        pdag[i, j] = 1
        pdag[j, i] = 1

    for i, j, k in vstructures:
        pdag[i, j] = 1
        pdag[k, j] = 1


def get_arrows_from_cpdag(cpdag):
    # Get directed edges from cpdag
    skeleton = get_skeleton(cpdag)
    return (skeleton - cpdag).T


def compute_shd_cpdag(cpdag_true, cpdag_est):
    return cd.PDAG.from_amat(cpdag_true).shd(cd.PDAG.from_amat(cpdag_est))


def compute_und_accuracy(adj_true, adj_est):
    """
    Code modified from:
    https://github.com/xunzheng/notears/blob/ba61337bd0e5410c04cc708be57affc191a8c424/notears/utils.py#L201
    """
    d = len(adj_true)
    adj_triu_true = adj_true[np.triu_indices(d, k=1)]
    adj_triu_est = adj_est[np.triu_indices(d, k=1)]
    pred = np.flatnonzero(adj_triu_est)
    cond = np.flatnonzero(adj_triu_true)

    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # false pos
    false_pos = np.setdiff1d(pred, cond, assume_unique=True)
    # false neg
    false_neg = np.setdiff1d(cond, true_pos, assume_unique=True)
    # compute ratio
    nnz = len(pred)
    cond_neg_size = len(adj_triu_true) - len(cond)
    fdr = float(len(false_pos)) / max(nnz, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(false_pos)) / max(cond_neg_size, 1)
    try:
        f1 = len(true_pos) / (len(true_pos) + 0.5 * (len(false_pos) + len(false_neg)))
    except:
        f1 = None

    # structural hamming distance
    extra_lower = np.setdiff1d(pred, cond, assume_unique=True)
    missing_lower = np.setdiff1d(cond, pred, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower)
    return {'f1': f1, 'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': nnz}


def count_arrows_accuracy(arrows_true, arrows_est):
    """
    Code modified from:
    https://github.com/xunzheng/notears/blob/ba61337bd0e5410c04cc708be57affc191a8c424/notears/utils.py#L201
    """
    d = arrows_true.shape[0]
    # linear index of nonzeros
    pred = np.flatnonzero(arrows_est)
    cond = np.flatnonzero(arrows_true)
    cond_reversed = np.flatnonzero(arrows_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])

    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # false pos
    false_pos = np.setdiff1d(pred, cond, assume_unique=True)
    # false neg
    false_neg = np.setdiff1d(cond, true_pos, assume_unique=True)
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    nnz = len(pred)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(nnz, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    try:
        f1 = len(true_pos) / (len(true_pos) + 0.5 * (len(false_pos) + len(false_neg)))
    except:
        f1 = None
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(arrows_est + arrows_est.T))
    cond_lower = np.flatnonzero(np.tril(arrows_true + arrows_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'f1': f1, 'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': nnz}


def compute_cpdag_accuracy(cpdag_true, cpdag_est):
    d = len(cpdag_true)
    shd_cpdag = compute_shd_cpdag(cpdag_true, cpdag_est)
    nnz = (cpdag_est != 0).sum()

    # Compute F1 score of skeleton
    skeleton_true = get_skeleton(cpdag_true)
    skeleton_est = get_skeleton(cpdag_est)
    results_skeleton = compute_und_accuracy(skeleton_true, skeleton_est)
    for metric in list(results_skeleton.keys()):
        results_skeleton['{}_skeleton'.format(metric)] = results_skeleton.pop(metric)

    # Compute F1 score of arrows
    arrows_true = get_arrows_from_cpdag(cpdag_true)
    arrows_est = get_arrows_from_cpdag(cpdag_est)
    results_arrows = count_arrows_accuracy(arrows_true, arrows_est)
    for metric in list(results_arrows.keys()):
        results_arrows['{}_arrows'.format(metric)] = results_arrows.pop(metric)

    # All results
    results = {'nnz': nnz, 'shd_cpdag': shd_cpdag}
    results.update(results_skeleton)
    results.update(results_arrows)
    return results


def get_k_steps_neighbors(adj_und, target_node, k=2):
    """
    Get k-steps neighbors of a target node in the undirected graph.
    (i.e., including 1-step, 2-step, ..., k-step neighbors)
    Parameters
    ----------
    adj_und : numpy.ndarray, shape=(d, d)
        Binary adjacency matrix of an undirected graph
    target_node : int
        Target node to get the k-step neighbors of.
    k : int
        Steps of neigh bors to query.
    Returns
    -------
    k_steps_neighbors : set
        K-step neighbors of the target node.
    """
    assert(adj_und == adj_und.T).all()    # adj_und must be symmetric
    assert ((adj_und == 0) | (adj_und == 1)).all()    # adj_und must be binary
    G_und = nx.from_numpy_array(adj_und)
    length = nx.single_source_shortest_path_length(G_und, target_node)
    k_step_neighbors = {node for node, distance in length.items() if distance <= k and node != target_node}
    return k_step_neighbors
