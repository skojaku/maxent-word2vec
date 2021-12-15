# %%
import numpy as np
import pandas as pd
from scipy import sparse
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from maxent_word2vec.datasets.GraphDataset import GraphDataset
from maxent_word2vec.models.SemAxisOffsetModel import SemAxisOffsetModel
from maxent_word2vec.models.word2vec import SGNSWord2Vec
import pandas as pd
import maxent_word2vec as mwvec

node_table_file = "node_table.csv"
citation_net_file = "net.npz"

node_table = pd.read_csv(node_table_file)
net = sparse.load_npz(citation_net_file)
A = net + net.T
A.eliminate_zeros()


# %%
# Create an embedding using the node2vec
#
dataset = mwvec.GraphDataset(A, num_walks=10, window_length=3, walk_length=80)
emb = SGNSWord2Vec(cuda=True).fit(dataset).transform(dim=3)


# %%
#
# Plot the original embeddings
#
# project an embedding to 2D for visualization
#
from sklearn.decomposition import PCA


def to_2d(emb):
    xy = PCA(n_components=2).fit_transform(emb)
    plot_data = node_table.copy()
    plot_data["x"] = xy[:, 0]
    plot_data["y"] = xy[:, 1]
    plot_data["deg"] = np.array(net.sum(axis=0)).reshape(-1)
    return plot_data


def plot(emb):
    sns.set_style("white")
    sns.set(font_scale=1.2)
    sns.set_style("ticks")
    fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

    cmap = sns.color_palette().as_hex()
    df = to_2d(emb)
    sns.scatterplot(
        x="x",
        y="y",
        hue="journal_code",
        cmap=cmap,
        data=df,
        size="deg",
        sizes=(10, 100),
        alpha=1.0,
        edgecolor="k",
        ax=axes[0],
    )
    ax = axes[0]
    ax.axis("off")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.1, 1.1), ncol=1)

    sns.scatterplot(
        x="x",
        y="y",
        hue="year",
        palette="YlGnBu_r",
        data=df,
        size="deg",
        sizes=(10, 100),
        alpha=1.0,
        edgecolor="k",
        ax=axes[1],
    )
    ax = axes[1]
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.1, 1.1), ncol=1)
    ax.set_title("Original")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.axis("off")
    plt.subplots_adjust(wspace=1)
    return fig, axes


plot(emb)

# %%
#
# Train the offset models
#
topk = 30
node_ids = node_table.sort_values(by="year")["year_journal_id"].values
left = node_ids[:topk]
right = node_ids[-topk:]
X_train = emb[np.concatenate([left, right]), :]
Y_train = np.concatenate([np.zeros_like(left), np.ones_like(right)])
offset_model = SemAxisOffsetModel().fit(X_train, Y_train, emb)
emb2 = SGNSWord2Vec(offset_model=offset_model, cuda=False).fit(dataset).transform(dim=3)

# %%
#
# Visualize the debiased embeddings
#
plot(emb2)

# %%
#
# Test the class predictions
#
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

labels = np.unique(node_table["journal_code"].values, return_inverse=True)[1]
# clf = KNeighborsClassifier(n_neighbors=10)
clf = LogisticRegression(random_state=0, multi_class="ovr")

scores = cross_val_score(clf, emb, labels, cv=10, scoring="f1_macro")
scores2 = cross_val_score(clf, emb2, labels, cv=10, scoring="f1_macro")
df = pd.concat(
    [
        pd.DataFrame({"model": "original", "score": scores}),
        pd.DataFrame({"model": "debiased", "score": scores2}),
    ]
)
ax = sns.barplot(x="model", y="score", data=df)
sns.despine()
ax.set_ylabel("Macro F1-score (Journal prediction)")

# %%
