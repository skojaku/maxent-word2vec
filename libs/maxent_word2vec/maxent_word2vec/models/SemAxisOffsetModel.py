import numpy as np
import torch
from scipy import sparse


class SemAxisOffsetModel:
    def __init__(self):
        pass

    def fit(self, X_train, Y_train, X):
        labels, label_ids = np.unique(Y_train, return_inverse=True)
        n_samples = len(label_ids)
        n_classes = len(labels)
        if n_classes != 2:
            raise ValueError("SemAxisOffsetModel requires 2 labels")

        # Calculate the membership matrix
        U = sparse.csr_matrix(
            (np.ones(n_samples), (label_ids, np.arange(n_samples))),
            shape=(n_classes, n_samples),
        )

        # Calc cluster centroids
        mu = (
            sparse.diags(1 / np.maximum(1, np.array(U.sum(axis=1)).reshape(-1))) @ U
        ) @ X_train
        sem_axis = np.array(mu[1, :] - mu[0, :]).reshape(-1)
        sem_axis /= np.linalg.norm(sem_axis)

        # Calc the rank of embeddings
        self.offset_score = torch.from_numpy(
            np.append(np.array(X @ sem_axis.reshape((-1, 1))).reshape(-1), 0)
            # append for the case of unknown word
        )
        self.sem_axis = torch.from_numpy(sem_axis)
        print(self.offset_score)
        return self

    def forward(self, iword, owords):
        return self.offset_score[owords] * self.offset_score[iword, None]
