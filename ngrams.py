import pandas as pd
from collections import defaultdict
from nltk.util import ngrams 
import multiprocessing
from tqdm import tqdm
import numpy as np
import pickle

SUMMARIES_PATH = "MovieSummaries/plot_summaries.txt"
N = 3 # N as is N-Gram
RESULTS_PATH = f"{N}gram_results.pkl"


def main():
    df_summaries = pd.read_csv(SUMMARIES_PATH, sep="\t", header=None)
    df_summaries.columns = ["ID", "Summary"]
    df_summaries["ID"] = df_summaries["ID"].astype(int)

    summaries_and_ids = df_summaries.values

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        results: list[list] = list(
            tqdm(
                pool.imap(ngram_computation, summaries_and_ids), 
                desc="computing ngrams", total=len(summaries_and_ids)
            )
        )

    with open(RESULTS_PATH, "wb") as file:
        pickle.dump(results, file)

def tokenize(text: str) -> list[str]:
    # Basic tokenization (you could use nltk or spaCy for more complex tokenization)
    return text.lower().split()

def ngram_computation(summary_and_id: np.ndarray) -> list:
    summary: str = summary_and_id[1]
    id: int = summary_and_id[0]

    tokens = tokenize(summary)
    n_grams: list[tuple[int]] = ngrams(tokens, N)

    return [id, n_grams]


if __name__ == "__main__":
    main()