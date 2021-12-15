# %%
import numpy as np
import pandas as pd
from scipy import sparse

paper_table_file = (
    "/data/sg/skojaku/authordynamics/data/aps/preprocessed/paper_table.csv"
)
citation_net_file = (
    "/data/sg/skojaku/authordynamics/data/aps/preprocessed/citation_net.npz"
)

#
# Load
#
paper_table = pd.read_csv(paper_table_file)
citation_net = sparse.load_npz(citation_net_file)
min_year = 0  # 1980
focal_journals = ["PRA", "PRB", "PRC", "PRD", "PRE", "PRL"]

# %%
paper_table = paper_table[paper_table["year"] >= min_year]
paper_table = paper_table[paper_table["journal_code"].isin(focal_journals)]
year_journal_table = paper_table[["journal_code", "year"]].drop_duplicates()
year_journal_table["year_journal_id"] = np.arange(year_journal_table.shape[0])
paper_table = pd.merge(paper_table, year_journal_table, on=["journal_code", "year"])
node_ids = paper_table["year_journal_id"].values
n_nodes = year_journal_table.shape[0]


# %%
#
# Convert to
#
U = sparse.csr_matrix(
    (
        np.ones(paper_table.shape[0]),
        (paper_table["year_journal_id"], paper_table["paper_id"]),
    ),
    shape=(n_nodes, citation_net.shape[0]),
)

# %%
citation_net = U @ citation_net @ U.T

#
# Save
#
year_journal_table.to_csv("node_table.csv", index=False)
sparse.save_npz("net.npz", citation_net)
