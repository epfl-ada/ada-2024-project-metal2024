import pandas as pd
from nltk.util import ngrams 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import multiprocessing
from tqdm import tqdm_gui as tqdm # remove _gui if you dont like the GUI and want CLI
import numpy as np
import pickle

SUMMARIES_PATH = "data/processed_plot_summaries.csv"
N = 3 # N as is N-Gram
RESULTS_PATH = f"src/ngrams/results/morethan100MB/{N}gram_results.pkl"

NLTK_TOKENIZING = False
if NLTK_TOKENIZING:
    nltk.download("stopwords")
    nltk.download("punkt")

CUSTOM_STOPWORDS = [ # words that will be banned from our Ngram
    ",", ";", ".", "",  
    "and", "or", "but", "nor", "for", "yet", "the", "a", "an", "of", "in", "to", "on", "with", "by", "at", "from", "about", "as",
    "he", "she", "it", "they", "we", "I", "you", "is", "are", "was", "were", "be", "have", "has", "her", "his", "had", "do", "does", "will", "would",
    "that", "which", "what", "when", "who", "whom", "how", "where",
    "him", "finds","tells", "out", "ends", "up", "so", "can",
    "none", "other", "than", "named", "their", "then", "while"]


def main():
    df_summaries = pd.read_csv(SUMMARIES_PATH)
    # renaming columns
    df_summaries.columns = ["ID", "Summary", "Date"]
    # removing the date column
    df_summaries = df_summaries[["ID", "Summary"]]
    # converting ID to int
    df_summaries["ID"] = df_summaries["ID"].astype(int)
    # converting to numpy array
    summaries_and_ids = df_summaries.values

    # computing ngrams
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        results: list[list] = list(
            tqdm(
                pool.imap(ngram_computation, summaries_and_ids), 
                desc=f"computing {N}grams", total=len(summaries_and_ids)
            )
        )

    # saving the results
    with open(RESULTS_PATH, "wb") as file:
        pickle.dump(results, file)

def tokenize(text: str) -> list[str]:
    if NLTK_TOKENIZING:
        stop_words = set(stopwords.words("english"))
        tokens = word_tokenize(text.lower())  
        filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        return filtered_tokens
    # tokenizing the text
    tokens = text.lower().split()
    # removing our custom stopwords
    return [token for token in tokens if token.isalpha() and token not in CUSTOM_STOPWORDS] 

def ngram_computation(summary_and_id: np.ndarray) -> list:
    """
    computes ngrams of a summary
    """

    summary: str = summary_and_id[1]
    id: int = summary_and_id[0]

    tokens = tokenize(summary)
    n_grams: list[tuple[int]] = ngrams(tokens, N)

    return [id, n_grams]


if __name__ == "__main__":
    main()
