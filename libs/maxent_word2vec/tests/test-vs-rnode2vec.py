# %%
%load_ext autoreload
%autoreload 2
import numpy as np
import pandas as pd

import residual2vec as rv

import seaborn as sns
from scipy import sparse
from sklearn.decomposition import PCA
#from residual_node2vec import Residual2Vec, Residual2VecTruncated
#from residual_node2vec.samplers import ResidualMatrixSampler
#from residual_node2vec import utils
import networkx as nx
from residual2vec.residual2vec import _truncated_residual_matrix

#
# Load
#
G = nx.karate_club_graph()
A = nx.adjacency_matrix(G)

group_ids = np.unique([d[1]["club"] for d in G.nodes(data = True)], return_inverse=True)[1]
deg = np.array(A.sum(axis = 1)).reshape(-1)
# %%
#model = Residual2VecTruncated(num_blocks = 5)
#model.fit(A)
#emb = model.transform(dim = 8)
#from sklearn.decomposition import PCA
#xy = PCA(n_components=2).fit_transform(emb)
#sns.scatterplot(x=xy[:,0], y=xy[:,1], hue = np.arange(A.shape[0]))
#noise_sampler = node_samplers.ErdosRenyiNodeSampler()
noise_sampler = rv.ConfigModelNodeSampler()
#noise_sampler = node_samplers.SBMNodeSampler(group_membership = np.arange(A.shape[0]), window_length = 10)
#noise_sampler = node_samplers.SBMNodeSampler(group_membership = np.arange(A.shape[0]))
model = rv.residual2vec_sgd(noise_sampler, batch_size = 128, window_length = 10, q = 1, p = 1, cuda = True, miniters = 200)
model.fit(A)
emb = model.transform(dim = 8)
from sklearn.decomposition import PCA
xy = PCA(n_components=2).fit_transform(emb)
sns.scatterplot(x=xy[:,0], y=xy[:,1], hue = group_ids, s = 10 * deg)

#sns.heatmap(Qs)
# %%
model = residual2vec(num_blocks=5)
model.fit(A)
emb = model.transform(dim = 8)
from sklearn.decomposition import PCA
xy = PCA(n_components=2).fit_transform(emb)
sns.scatterplot(x=xy[:,0], y=xy[:,1], hue = np.arange(A.shape[0]))

# %%
#num_blocks = A.shape[]
#group_ids = np.random.choice(num_blocks, A.shape[0], replace=True).astype(int)
group_ids = np.arange(A.shape[0])
group_ids_null = np.zeros(A.shape[0]).astype(int)
window_length = 10
Qr = _truncated_residual_matrix(A, group_ids, group_ids_null, window_length)
sns.heatmap(Qr.toarray())
# %%
Qr.toarray()
