# %%
import json
import sys
from math import e

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as metrics
from maxent_word2vec.models.SemAxisOffsetModel import SemAxisOffsetModel
from scipy import sparse, stats
from scipy.spatial import distance

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    output_file = snakemake.output["output_file"]
else:
    token_table_file = "../../data/wiki1b/preprocessed/token_table.csv"
    left_right_word_file = "../../data/wiki1b/preprocessed/gender_left_right_words.json"
    neutral_word_file = "../../data/wiki1b/preprocessed/occupations.json"
    emb_file = "../../data/wiki1b/embeddings/word2vec_wl=10.npz"
    deb_emb_file = "../../data/wiki1b/embeddings/debiased-word2vec_wl=10.npz"
    # emb_file = "./emb.npz"
#
# Load
#
emb = np.load(emb_file)["emb"]
deb_emb = np.load(deb_emb_file)["emb"]
token_table = pd.read_csv(token_table_file)
nemb = np.einsum("ij,i->ij", emb, 1 / np.linalg.norm(emb, axis=1))

with open(neutral_word_file, "r") as f:
    neutral_words = json.load(f)["occupation"]

with open(left_right_word_file, "r") as f:
    left_right_words = json.load(f)
    left = left_right_words["left"]
    right = left_right_words["right"]

# %%
def get_word_ids(words, token_table):
    if isinstance(words, str):
        words = [words]

    id_dict = (
        token_table[token_table["words"].isin(words)].set_index("words")["id"].to_dict()
    )
    return np.array([id_dict[w] for w in words if w in id_dict])


def calc_bias(emb, token_table, left, right, neutral_words, metric="cosine"):
    left_ids = get_word_ids(left, token_table)
    right_ids = get_word_ids(right, token_table)
    neutral_ids = get_word_ids(neutral_words, token_table)
    L = emb[left_ids, :]
    R = emb[right_ids, :]
    N = emb[neutral_ids, :]

    if metric == "cosine":
        L = np.einsum("ij,i->ij", L, 1 / np.linalg.norm(L, axis=1))
        R = np.einsum("ij,i->ij", R, 1 / np.linalg.norm(R, axis=1))
        N = np.einsum("ij,i->ij", N, 1 / np.linalg.norm(N, axis=1))

        L = np.mean(L, axis=0)
        R = np.mean(R, axis=0)
        bias = N @ L - N @ R
        return bias
    elif metric == "euclidean":
        dL = distance.pdist(np.vstack([L, N]))
        dL = distance.squareform(dL)[: L.shape[0], L.shape[0] :]
        dR = distance.pdist(np.vstack([R, N]))
        dR = distance.squareform(dR)[: R.shape[0], R.shape[0] :]
        return np.mean(dL, axis=0) - np.mean(dR, axis=0)


metric = "cosine"
bias_original = calc_bias(emb, token_table, left, right, neutral_words, metric=metric)
bias_deb = calc_bias(deb_emb, token_table, left, right, neutral_words, metric=metric)

# %%
plot_data = pd.DataFrame(
    {
        "Man-woman bias": np.concatenate([bias_original, bias_deb]),
        "Embedding": ["original"] * len(bias_original) + ["debiased"] * len(bias_deb),
    }
)

# %%
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4.5, 5))
sns.boxplot(data=plot_data, x="Embedding", y="Man-woman bias", ax=ax)
ax = sns.stripplot(data=plot_data, x="Embedding", y="Man-woman bias", color=".3", ax=ax)
sns.despine()

# %%
fig, ax = plt.subplots(figsize=(4.5, 5))
ax = sns.swarmplot()
