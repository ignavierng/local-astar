"""
Code modified from:
- https://github.com/ignavier/golem/blob/main/src/data_loader/synthetic_dataset.py
- https://github.com/xunzheng/notears/blob/master/notears/utils.py
"""
import logging

import networkx as nx
import numpy as np

from utils.dag import get_cpdag, get_skeleton
from utils.dag import is_dag


class SyntheticDataset:
    """Generate synthetic data.

    Key instance variables:
        X (numpy.ndarray): [n, d] data matrix.
        B (numpy.ndarray): [d, d] weighted adjacency matrix of DAG.
        B_bin (numpy.ndarray): [d, d] binary adjacency matrix of DAG.
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, n, d, degree, noise_type):
        """Initialize self.

        Args:
            n (int): Number of samples.
            d (int): Number of nodes.
            degree (int): Degree of graph.
            noise_type ('gaussian', 'exponential', 'gumbel'): Type of noise.
        """
        self.n = n
        self.d = d
        self.degree = degree
        self.noise_type = noise_type
        self.B_ranges = ((-0.8, -0.2), (0.2, 0.8))

        self._setup()
        self._logger.debug("Finished setting up dataset class.")

    def _setup(self):
        """Generate B_bin, B and X."""
        self.B_bin = SyntheticDataset.simulate_random_dag(self.d, self.degree)
        self.B = SyntheticDataset.simulate_weight(self.B_bin, self.B_ranges)
        self.Omega = SyntheticDataset.simulate_Omega(self.d)
        self.X = SyntheticDataset.simulate_linear_sem(self.B, self.Omega, self.n, self.noise_type)
        assert is_dag(self.B)

    @staticmethod
    def simulate_Omega(d):
        """Simulate noise covariance matrix.

        Args:
            d (int): Number of nodes.

        Returns:
            numpy.ndarray: [d, d] noise covariance matrix.
        """
        return np.diag(np.random.uniform(low=1.0, high=2.0, size=d))

    @staticmethod
    def simulate_er_dag(d, degree):
        """Simulate ER DAG using NetworkX package.

        Args:
            d (int): Number of nodes.
            degree (int): Degree of graph.

        Returns:
            numpy.ndarray: [d, d] binary adjacency matrix of DAG.
        """
        def _get_acyclic_graph(B_und):
            return np.tril(B_und, k=-1)

        def _graph_to_adjmat(G):
            return nx.to_numpy_matrix(G)

        p = float(degree) / (d - 1)
        G_und = nx.generators.erdos_renyi_graph(n=d, p=p)
        B_und_bin = _graph_to_adjmat(G_und)    # Undirected
        B_bin = _get_acyclic_graph(B_und_bin)
        return B_bin

    @staticmethod
    def simulate_random_dag(d, degree):
        """Simulate random DAG.

        Args:
            d (int): Number of nodes.
            degree (int): Degree of graph.

        Returns:
            numpy.ndarray: [d, d] binary adjacency matrix of DAG.
        """
        def _random_permutation(B_bin):
            # np.random.permutation permutes first axis only
            P = np.random.permutation(np.eye(B_bin.shape[0]))
            return P.T @ B_bin @ P

        B_bin = SyntheticDataset.simulate_er_dag(d, degree)
        return _random_permutation(B_bin)

    @staticmethod
    def simulate_weight(B_bin, B_ranges):
        """Simulate the weights of B_bin.

        Args:
            B_bin (numpy.ndarray): [d, d] binary adjacency matrix of DAG.
            B_ranges (tuple): Disjoint weight ranges.

        Returns:
            numpy.ndarray: [d, d] weighted adjacency matrix of DAG.
        """
        B = np.zeros(B_bin.shape)
        S = np.random.randint(len(B_ranges), size=B.shape)  # Which range
        for i, (low, high) in enumerate(B_ranges):
            U = np.random.uniform(low=low, high=high, size=B.shape)
            B += B_bin * (S == i) * U
        return B

    @staticmethod
    def simulate_linear_sem(B, Omega, n, noise_type):
        """Simulate samples from linear SEM with specified type of noise.

        Args:
            B (numpy.ndarray): [d, d] weighted adjacency matrix of DAG.
            Omega (numpy.ndarray)): [d, d] noise covariance matrix.
            n (int): Number of samples.
            noise_type ('gaussian', 'exponential', 'gumbel'): Type of noise.

        Returns:
            numpy.ndarray: [n, d] data matrix.
        """
        def _simulate_single_equation(X, B_i, sigma_i):
            """Simulate samples from linear SEM for the i-th node.

            Args:
                X (numpy.ndarray): [n, number of parents] data matrix.
                B_i (numpy.ndarray): [d,] weighted vector for the i-th node.
                sigma_i (float): noise standard deviation for the i-th node.

            Returns:
                numpy.ndarray: [n,] data matrix.
            """
            if noise_type == 'gaussian':
                N_i = np.random.normal(scale=sigma_i, size=n)
            elif noise_type == 'exponential':
                # Exponential noise
                N_i = np.random.exponential(scale=sigma_i, size=n)
            elif noise_type == 'gumbel':
                # Gumbel noise
                N_i = np.random.gumbel(scale=sigma_i, size=n)
            else:
                raise ValueError("Unknown noise type.")
            return X @ B_i + N_i

        d = B.shape[0]
        X = np.zeros([n, d])
        G = nx.DiGraph(B)
        ordered_vertices = list(nx.topological_sort(G))
        assert len(ordered_vertices) == d
        for i in ordered_vertices:
            parents = list(G.predecessors(i))
            X[:, i] = _simulate_single_equation(X[:, parents], B[parents, i], np.sqrt(Omega[i, i]))

        return X

    @property
    def Theta(self):
        I = np.eye(self.d)
        return (I - self.B) @ np.linalg.inv(self.Omega) @ (I - self.B).T

    @property
    def cpdag(self):
        return get_cpdag(self.B)

    @property
    def skeleton(self):
        return get_skeleton(self.B)
