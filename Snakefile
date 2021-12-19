from os.path import join as j

configfile: "workflow/config.yaml"

DATA_DIR = config["data_dir"]

PAPER_DIR = config["paper_dir"]
PAPER_SRC, SUPP_SRC = [j(PAPER_DIR, f) for f in ("main.tex", "supp.tex")]
PAPER, SUPP = [j(PAPER_DIR, f) for f in ("main.pdf", "supp.pdf")]

rule paper:
    input:
        PAPER_SRC, SUPP_SRC
    params:
        paper_dir = PAPER_DIR
    output:
        PAPER, SUPP
    shell:
        "cd {params.paper_dir}; make"

# ==================
# Preprocessed files
# ==================
WIKI1B_DATA_SRC_CLEANED = j(DATA_DIR, "wiki1b", "preprocessed", "supp", "cleaned_enwik9.txt")
WIKI1B_TOKEN_TABLE = j(DATA_DIR, "wiki1b", "preprocessed", "token_table.csv")
WIKI1B_TOKENED_CORPUS = j(DATA_DIR, "wiki1b", "preprocessed", "tokenized_corpus.txt")
GENDER_WORD_FILE = j(DATA_DIR, "wiki1b", "preprocessed", "gender_left_right_words.json")
# =================
# Embeddings
# =================
EMB_FILE = j(DATA_DIR, "{data}", "embeddings", "word2vec_wl={wl}.npz")
DEB_EMB_FILE = j(DATA_DIR, "{data}", "embeddings", "debiased-word2vec_wl={wl}.npz")


rule wiki1b_embedding:
    input:
        input_file = WIKI1B_TOKENED_CORPUS
    output:
        output_file = EMB_FILE
    params:
        window_length = lambda wildcards:wildcards.wl
    wildcard_constraints:
        data = "wiki1b"
    script:
        "workflow/process/embedding-doc.py"

rule wiki1b_debiased:
    input:
        input_file = WIKI1B_TOKENED_CORPUS,
        token_table_file = WIKI1B_TOKEN_TABLE,
        emb_file = EMB_FILE,
        left_right_word_file = GENDER_WORD_FILE,
    params:
        window_length = lambda wildcards:wildcards.wl
    output:
        output_file = DEB_EMB_FILE
    wildcard_constraints:
        data = "wiki1b"
    script:
        "workflow/process/debiased-embedding-doc.py"

rule all:
    input:
        expand(DEB_EMB_FILE, data = "wiki1b", wl = [10])

# rule some_data_processing:
    # input:
        # "data/some_data.csv"
    # output:
        # "data/derived/some_derived_data.csv"
    # script:
        # "workflow/scripts/process_some_data.py"
