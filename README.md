# maxent-word2vec
Bias-aware word2vec based on the maximum entropy principle


```python
from maxent_word2vec.datasets.GraphDataset import GraphDataset
from maxent_word2vec.models.SemAxisOffsetModel import SemAxisOffsetModel
from maxent_word2vec.models.word2vec import SGNSWord2Vec
import netowrkx as nx
import numpy as np

#
# Input
#
A = nx.adjacency_matrix(nx.karate_club_graph()) # adjacency matrix for the network
left, right = np.array([0,1]), np.array([32, 33]) # Left and right node groups for the semi-axis


# 
# Create an embedding using the node2vec
#
dataset = GraphDataset(A, num_walks=10, window_length=3, walk_length=80)
emb = SGNSWord2Vec(cuda=True).fit(dataset).transform(dim=3)

#
# Fit the Semi-Axis
#
X_train = emb[np.concatenate([left, right]), :]
Y_train = np.concatenate([np.zeros_like(left), np.ones_like(right)])
offset_model = SemAxisOffsetModel().fit(X_train, Y_train, emb)

# 
# Generate a debiased embedding.
#
emb_debiased = SGNSWord2Vec(offset_model=offset_model, cuda=False).fit(dataset).transform(dim=3)
```
