"""Test by visual inspection of the generated embeddings."""
# %%
import seaborn as sns
import matplotlib.pyplot as plt
import maxent_word2vec as mwvec
import networkx as nx
import numpy as np
from maxent_word2vec.datasets.GraphDataset import GraphDataset
from maxent_word2vec.models.SemAxisOffsetModel import SemAxisOffsetModel
from maxent_word2vec.models.MaxEntModel import MaxEntModel
from maxent_word2vec.models.word2vec import SGNSWord2Vec
import pandas as pd

#
# Data
#
G = nx.karate_club_graph()
A = nx.adjacency_matrix(G)
class_labels = np.unique(
    [d[1]["club"] for d in G.nodes(data=True)], return_inverse=True
)[1]
# %%
dataset = mwvec.GraphDataset(A, num_walks=10, window_length=5, walk_length=50)
emb = SGNSWord2Vec().fit(dataset).transform(dim=32)

left = np.array([32, 33])
middle = np.array([12, 13])
right = np.array([1, 0, 2])
X_train = emb[np.concatenate([left, right, middle]), :]
Y_train = np.concatenate(
    [np.zeros_like(left), np.ones_like(right), np.ones_like(middle) * 2]
)
offset_model = MaxEntModel().fit(X_train, Y_train, emb)
# offset_model = SemAxisOffsetModel().fit(X_train, Y_train, emb)
emb2 = SGNSWord2Vec(offset_model=offset_model).fit(dataset).transform(dim=2)
# %%
offset_model.Z
# %%
#
# Prep plot data
#
from sklearn.decomposition import PCA

xy = PCA(n_components=2).fit_transform(emb)
xy2 = PCA(n_components=2).fit_transform(emb2)

df = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "group": class_labels})
df2 = pd.DataFrame({"x": xy2[:, 0], "y": xy2[:, 1], "group": class_labels})
# %%
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(7, 5), ncols=2)

sns.scatterplot(data=df, x="x", y="y", hue="group", ax=ax[0])
sns.scatterplot(data=df2, x="x", y="y", hue="group", ax=ax[1])
# fig.savefig(output_file, bbox_extra_artists=(lgd,), bbox_inches="tight", dpi=300)
sns.scatterplot()

# %%
