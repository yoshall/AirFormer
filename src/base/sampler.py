# coding=utf-8
import numpy as np
import scipy.sparse as sp

import src.utils.graph_algo as graph_algo

# data augmentation for air quality data
# We don't use it actually in our experiments
class RandomSampler:
    """Sampling the input graph data."""

    def __init__(self, adj_mat, filter_type):
        self._adj_mat = adj_mat
        self._filter_type = filter_type

    def sample(self, percent):
        """
        Randomly drop edge and preserve percent% edges.
        """
        if percent >= 1.0:
            raise ValueError

        adj_sp = sp.coo_matrix(self._adj_mat)

        nnz = adj_sp.nnz
        perm = np.random.permutation(nnz)
        preserve_nnz = int(nnz * percent)
        perm = perm[:preserve_nnz]
        r_adj = sp.coo_matrix((adj_sp.data[perm],
                               (adj_sp.row[perm],
                                adj_sp.col[perm])),
                              shape=adj_sp.shape)
        return r_adj.todense()


class CutEdgeSampler:
    """Sampling the input graph data."""

    def __init__(self, adj_mat, filter_type, m=200):
        self._adj_mat = adj_mat.copy()
        self._filter_type = filter_type

        new_adj = adj_mat + np.eye(adj_mat.shape[0])
        rw_adj = graph_algo.calculate_random_walk_matrix(new_adj).todense()

        # undirected_adj = adj_mat.copy()
        # undirected_adj[undirected_adj > 0] = 1
        # out_degree = np.array(adj_mat.sum(1)).flatten()
        square_adj = np.power(rw_adj, 2)
        out_degree = np.array(square_adj.sum(1)).flatten()

        adj_sp = sp.coo_matrix(rw_adj)

        out_degree_sum = 0
        for i in range(adj_sp.nnz):
            out_degree_sum += out_degree[adj_sp.row[i]] + out_degree[adj_sp.col[i]]

        p = np.zeros(adj_sp.nnz)
        for i in range(adj_sp.nnz):
            p[i] = (out_degree[adj_sp.row[i]] +
                    out_degree[adj_sp.col[i]]) / out_degree_sum * m
        self.droprate = sp.coo_matrix((p, (adj_sp.row, adj_sp.col)), shape=adj_sp.shape).todense()


    def sample(self, m=200):
        num_nodes = self._adj_mat.shape[0]
        prob = np.random.rand(num_nodes, num_nodes)
        drop_mask = (prob > self.droprate).astype(int) # preserve rate
        return self._adj_mat
