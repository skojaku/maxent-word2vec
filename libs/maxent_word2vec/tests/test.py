import unittest

import maxent_word2vec as mwvec
import networkx as nx
import numpy as np
from maxent_word2vec.datasets.GraphDataset import GraphDataset
from maxent_word2vec.models.SemAxisOffsetModel import SemAxisOffsetModel
from maxent_word2vec.models.word2vec import SGNSWord2Vec


class TestResidual2Vec(unittest.TestCase):
    def setUp(self):
        self.G = nx.karate_club_graph()
        self.A = nx.adjacency_matrix(self.G)

    def test_word2vec_network(self):
        dataset = mwvec.GraphDataset(
            self.A, num_walks=10, window_length=10, walk_length=50
        )
        emb = SGNSWord2Vec().fit(dataset).transform(dim=2)

        left = np.array([32, 33])
        right = np.array([1, 0, 2])
        X_train = emb[np.concatenate([left, right]), :]
        Y_train = np.concatenate([np.zeros_like(left), np.ones_like(right)])
        offset_model = SemAxisOffsetModel().fit(X_train, Y_train, emb)
        emb = SGNSWord2Vec(offset_model=offset_model).fit(dataset).transform(dim=2)


if __name__ == "__main__":
    unittest.main()
