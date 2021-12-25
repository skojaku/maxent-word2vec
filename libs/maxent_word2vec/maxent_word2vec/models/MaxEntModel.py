import numpy as np
import torch
from scipy import sparse


class MaxEntModel:
    def __init__(self, cuda=False):
        self.cuda = cuda
        pass

    def fit(self, X_train, Y_train, X):
        labels, label_ids = np.unique(Y_train, return_inverse=True)
        n_samples = len(label_ids)
        n_classes = len(labels)

        # Calculate the membership matrix
        U = sparse.csr_matrix(
            (np.ones(n_samples), (label_ids, np.arange(n_samples))),
            shape=(n_classes, n_samples),
        )

        # Calculate the spectrum centers
        Z = (
            sparse.diags(1 / np.maximum(1, np.array(U.sum(axis=1)).reshape(-1))) @ U
        ) @ X_train

        X_rand = X @ (Z.T @ np.linalg.inv(Z @ Z.T) @ Z)

        # Calc the rank of embeddings
        self.X_rand = torch.from_numpy(
            np.vstack([X_rand, np.zeros((1, X_rand.shape[1]))])
            # append for the case of unknown word
        )
        self.Z = torch.from_numpy(Z)
        if self.cuda:
            self.X_rand = self.X_rand.cuda()
        return self

    def get_semaxis(self):
        return self.Z.copy().cpu().detach().numpy()

    def forward(self, iword, owords):
        ivectors = self.X_rand[iword, :].unsqueeze(2)
        ovectors = self.X_rand[owords, :]
        return torch.bmm(ovectors, ivectors).squeeze()
