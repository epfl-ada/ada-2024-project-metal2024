import pandas as pd
import os
import pickle
from tqdm import tqdm_gui
from collections import Counter, defaultdict
import sys
sys.path.append(".")
from src.utils import periods_map_inverse

N = 1 # N as in Ngram
TOP_NGRAM = 10 # TOP N ngram per year or decade
DATASET_PATH = "DATA/"
GROUPBY = ["year", "decade", "period"][2]
# please run ngrams.py before to generate the ngram below
NGRAM_PATH = f"src/ngrams/results/morethan100MB/{N}gram_results.pkl"
OUTPUT_PATH = f"src/ngrams/results/{N}grams_per_{GROUPBY}.csv"

def main():
    movies_ids, ngrams = inputs()

    ngrams_dict = dict(ngrams)

    movies_ids = movies_ids.groupby("Movie release date").agg(list).reset_index()
    movies_ids["Number of movies"] = movies_ids["Wikipedia movie ID"].apply(lambda x: len(x))

    movies_ids[f"top {TOP_NGRAM} {N}grams"] = movies_ids["Wikipedia movie ID"].apply(lambda x: count_ngrams(x, ngrams_dict))
    #movies_ids[f"most 5 common {N}grams"] = movies_ids[f"{N}grams"].apply(lambda x: x.most_common(5))

    movies_ids[["Movie release date", "Number of movies", f"top {TOP_NGRAM} {N}grams"]].to_csv(OUTPUT_PATH, index=False)



def inputs():
    """
    Returns: movies_ids and ngrams
    """
    #movies_df = pd.read_csv(os.path.join(DATASET_PATH,"movie.metadata.tsv"), delimiter="\t", header=None)
    #movies_df.columns = ["Wikipedia movie ID", "Freebase movie ID", "Movie name", "Movie release date", "Movie box office revenue", "Movie runtime", "Movie languages (Freebase ID:name tuples)", "Movie countries (Freebase ID:name tuples)", "Movie genres (Freebase ID:name tuples)"]

    # importing the already preprocessing movie dataset
    movies_df = pd.read_csv(os.path.join(DATASET_PATH,"processed_movies.csv"))

    def int_or_empty(string):
        try:
            year = int(string)
            # if GROUPBY is on decade, then round per decade
            if GROUPBY == "decade":
                year -= year % 10
            return year
        except:
            return pd.NA
        
    def get_period(year):
        try:
            return periods_map_inverse[int(year)]
        except:
            return pd.NA  

    if GROUPBY == "period":
        movies_df["Movie release date"] = movies_df["Movie release date"].apply(get_period)
    else:
        movies_df["Movie release date"] = movies_df["Movie release date"].astype(int).apply(int_or_empty)  #pd.to_datetime(movies_df['Movie release date'], errors='coerce').dt.year.apply(int_or_empty)

    with open(NGRAM_PATH, "rb") as file:
        ngrams = pickle.load(file)

    return movies_df[["Wikipedia movie ID", 'Movie release date']], ngrams


def count_ngrams(ids: list, ngrams_dict):
    res = Counter()
    for id in ids:
        try:
            #print(list(ngrams_dict[id]))
            res.update(
                list(
                    ngrams_dict[id]
                )
            )
        except: continue
    return res.most_common(TOP_NGRAM)


if __name__ == "__main__":
    main()