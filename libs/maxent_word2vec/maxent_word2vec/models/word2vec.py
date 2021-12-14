"""SGNS Word2Vec model."""
import numpy as np
import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm


class SGNSWord2Vec:
    """Torch implementation of the skip-gram Word2Vec.

    .. highlight:: python .. code-block:: python
    """

    def __init__(
        self,
        batch_size=256,
        cuda=False,
        buffer_size=100000,
        miniters=200,
        offset_model=None,
    ):
        """
        :param batch_size: Number of batches for the SGD, defaults to 4
        :type batch_size: int
        :param buffer_size: Buffer size for sampled center and context pairs, defaults to 10000
        :type buffer_size: int, optional
        :param miniter: Minimum number of iterations, defaults to 200
        :type miniter: int, optional
        """
        self.cuda = cuda
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.miniters = miniters
        self.offset_model = offset_model

    def fit(self, dataset):
        """Learn the graph structure to generate the node embeddings.

        :param adjmat: Adjacency matrix of the graph.
        :type adjmat: numpy.ndarray or scipy sparse matrix format (csr).
        :return: self
        :rtype: self
        """
        # Set up the Training dataset
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )
        self.n_words = dataset.n_words

        return self

    def transform(self, dim):
        """Generate embedding vectors.

        :param dim: Dimension
        :type dim: int
        :return: Embedding vectors
        :rtype: numpy.ndarray of shape (num_nodes, dim), where num_nodes is the number of nodes.
          Each ith row in the array corresponds to the embedding of the ith node.
        """

        # Set up the embedding model
        PADDING_IDX = self.n_words
        model = _Word2Vec(
            vocab_size=self.n_words + 1, embedding_size=dim, padding_idx=PADDING_IDX
        )
        neg_sampling = NegativeSampling(embedding=model, offset_model=self.offset_model)
        if self.cuda:
            model = model.cuda()

        # Training
        optim = Adam(model.parameters(), lr=0.003)
        # scaler = torch.cuda.amp.GradScaler()
        pbar = tqdm(self.dataloader, miniters=100)
        for iword, owords, nwords in pbar:
            # optim.zero_grad()
            for param in model.parameters():
                param.grad = None
            # with torch.cuda.amp.autocast():
            loss = neg_sampling(iword, owords, nwords)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optim.step()
            pbar.set_postfix(loss=loss.item())

        self.in_vec = model.ivectors.weight.data.cpu().numpy()[:PADDING_IDX, :]
        self.out_vec = model.ovectors.weight.data.cpu().numpy()[:PADDING_IDX, :]
        return self.in_vec


class _Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx):
        super(_Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.ivectors = nn.Embedding(
            self.vocab_size, self.embedding_size, padding_idx=padding_idx
        )
        self.ovectors = nn.Embedding(
            self.vocab_size, self.embedding_size, padding_idx=padding_idx
        )
        self.ivectors.weight = nn.Parameter(
            torch.cat(
                [
                    torch.zeros(1, self.embedding_size),
                    FloatTensor(self.vocab_size, self.embedding_size).uniform_(
                        -0.5 / self.embedding_size, 0.5 / self.embedding_size
                    ),
                ]
            )
        )
        self.ovectors.weight = nn.Parameter(
            torch.cat(
                [
                    torch.zeros(1, self.embedding_size),
                    FloatTensor(self.vocab_size, self.embedding_size).uniform_(
                        -0.5 / self.embedding_size, 0.5 / self.embedding_size
                    ),
                ]
            )
        )
        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True

    def forward(self, data):
        return self.forward_i(data)

    def forward_i(self, data):
        v = LongTensor(data)
        v = v.cuda() if self.ivectors.weight.is_cuda else v
        return self.ivectors(v)

    def forward_o(self, data):
        v = LongTensor(data)
        v = v.cuda() if self.ovectors.weight.is_cuda else v
        return self.ovectors(v)


class NegativeSampling(nn.Module):
    def __init__(self, embedding, offset_model=None):
        super(NegativeSampling, self).__init__()
        self.embedding = embedding
        self.weights = None
        self.offset_model = offset_model

    def forward(self, iword, owords, nwords):
        ivectors = self.embedding.forward_i(iword).unsqueeze(2)
        ovectors = self.embedding.forward_o(owords)
        nvectors = self.embedding.forward_o(nwords).neg()
        o_dot = torch.bmm(ovectors, ivectors).squeeze()
        n_dot = torch.bmm(nvectors, ivectors).squeeze()
        if self.offset_model is not None:
            o_dot += self.offset_model.forward(iword, owords)
            n_dot -= self.offset_model.forward(iword, nwords)

        oloss = o_dot.sigmoid().clamp(1e-12, 1).log().mean(1)
        nloss = n_dot.sigmoid().clamp(1e-12, 1).log().mean(1)
        return -(oloss + nloss).mean()
