"""Dataset for networks."""

import random

import numpy as np
from maxent_word2vec.samplers.node_sampler import (
    ConfigModelNodeSampler,
    RandomWalkSampler,
)
from numba import njit
from torch.utils.data import Dataset


class GraphDataset(Dataset):
    """Dataset for training word2vec with negative sampling."""

    def __init__(
        self,
        adjmat,
        num_walks,
        window_length,
        noise_sampler=None,
        walk_length=40,
        p=1.0,
        q=1.0,
        buffer_size=100000,
    ):
        """Dataset for training word2vec with negative sampling.

        :param adjmat: Adjacency matrix of the graph.
        :type adjmat: scipy sparse matrix format (csr).
        :param num_walks: Number of random walkers per node
        :type num_walks: int
        :param window_length: length of the context window
        :type window_length: int
        :param noise_sampler: Noise sampler
        :type noise_sampler: NodeSampler
        :param padding_id: Index of the padding node
        :type padding_id: int
        :param walk_length: length per walk, defaults to 40
        :type walk_length: int, optional
        :param p: node2vec parameter p (1/p is the weights of the edge to previously visited node), defaults to 1
        :type p: float, optional
        :param q: node2vec parameter q (1/q) is the weights of the edges to nodes that are not directly connected to the previously visted node, defaults to 1
        :type q: float, optional
        :param buffer_size: Buffer size for sampled center and context pairs, defaults to 10000
        :type buffer_size: int, optional
        """
        self.adjmat = adjmat
        self.num_walks = num_walks
        self.window_length = window_length
        self.walk_length = walk_length
        self.rw_sampler = RandomWalkSampler(adjmat, walk_length=walk_length, p=p, q=q)
        self.node_order = np.random.choice(
            adjmat.shape[0], adjmat.shape[0], replace=False
        )
        self.n_nodes = adjmat.shape[0]

        self.ave_deg = adjmat.sum() / adjmat.shape[0]

        # Counter and Memory
        self.n_sampled = 0
        self.sample_id = 0
        self.scanned_node_id = 0
        self.buffer_size = buffer_size
        self.contexts = None
        self.centers = None
        self.random_contexts = None
        self.n_words = adjmat.shape[0]

        if noise_sampler is None:
            self.noise_sampler = ConfigModelNodeSampler().fit(adjmat)
        else:
            self.noise_sampler = noise_sampler

        # Initialize
        self._generate_samples()

    def __len__(self):
        return self.n_nodes * self.num_walks * self.walk_length

    def __getitem__(self, idx):
        if self.sample_id == self.n_sampled:
            self._generate_samples()

        center = self.centers[self.sample_id]
        cont = self.contexts[self.sample_id, :].astype(np.int64)
        rand_cont = self.random_contexts[self.sample_id, :].astype(np.int64)

        self.sample_id += 1

        return center, cont, rand_cont

    def _generate_samples(self):
        next_scanned_node_id = np.minimum(
            self.scanned_node_id + self.buffer_size, self.n_nodes
        )
        walks = self.rw_sampler.sampling(
            self.node_order[self.scanned_node_id : next_scanned_node_id]
        )
        self.centers, self.contexts = _get_center_context(
            walks,
            walks.shape[0],
            walks.shape[1],
            self.window_length,
            padding_id=self.n_words,
        )
        self.random_contexts = self.noise_sampler.sampling(
            self.centers, self.contexts.shape[1]
        )
        self.n_sampled = len(self.centers)
        self.scanned_node_id = next_scanned_node_id % self.n_nodes
        self.sample_id = 0


@njit(nogil=True)
def _get_center_context(walks, n_walks, walk_len, window_length, padding_id):
    centers = np.zeros(n_walks * walk_len, dtype=np.int64)
    contexts = np.zeros((n_walks * walk_len, 2 * window_length), dtype=np.int64)
    for t_walk in range(walk_len):
        start, end = n_walks * t_walk, n_walks * (t_walk + 1)
        centers[start:end] = walks[:, t_walk]
        contexts[start:end, :] = _get_context(
            walks, n_walks, walk_len, t_walk, window_length, padding_id
        )
    order = np.arange(walk_len * n_walks)
    random.shuffle(order)
    return centers[order], contexts[order, :]


@njit(nogil=True)
def _get_context(walks, n_walks, walk_len, t_walk, window_length, padding_id):
    retval = padding_id * np.ones((n_walks, 2 * window_length), dtype=np.int64)
    for i in range(window_length):
        if t_walk - 1 - i < 0:
            break
        retval[:, window_length - 1 - i] = walks[:, t_walk - 1 - i]

    for i in range(window_length):
        if t_walk + 1 + i >= walk_len:
            break
        retval[:, window_length + i] = walks[:, t_walk + 1 + i]
    return retval
