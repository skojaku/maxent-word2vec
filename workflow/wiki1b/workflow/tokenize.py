import sys

import pandas as pd
from gensim import corpora
from tqdm import tqdm

if "snakemake" in sys.modules:
    filename = snakemake.input["input_file"]
    tokenized_corpus = snakemake.output["tokenized_corpus"]
    token2id_table = snakemake.output["token_table"]
else:
    filename = sys.argv[1]
    tokenized_corpus = sys.argv[2]
    token2id_table = sys.argv[3]


def tokenize(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        sentences = [line.strip() for line in lines if len(line) > 2]
        line = " ".join(sentences)

        gensim_dictionary = corpora.Dictionary([line.split(" ")])

        df = pd.DataFrame(
            {
                "words": gensim_dictionary.token2id.keys(),
                "id": gensim_dictionary.token2id.values(),
            }
        )
        return df, sentences, gensim_dictionary


df, sentences, gensim_dictionary = tokenize(filename)
df.to_csv(token2id_table)

with open(tokenized_corpus, "w") as f:
    first = True
    for sent in tqdm(sentences):
        words = sent.split(" ")
        token_ids = ["%d" % gensim_dictionary.token2id[w] for w in words]
        tokens = " ".join(token_ids)
        if first:
            f.write(tokens)
            first = False
        else:
            f.write("\n" + tokens)
