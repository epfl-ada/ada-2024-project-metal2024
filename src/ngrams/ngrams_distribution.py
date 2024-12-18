import pandas as pd
import os
import pickle
from collections import Counter
import sys
sys.path.append(".")
from src.utils import periods_map_inverse

N = 3 # N as in Ngram
TOP_NGRAM = 20 # TOP N ngram per year or decade
DATASET_PATH = "data/"
# GROUPBY can be "year", "decade" or "period"
GROUPBY = ["year", "decade", "period"][2]
# please run ngrams.py before to generate the ngram below
NGRAM_PATH = f"src/ngrams/results/morethan100MB/{N}gram_results.pkl"
OUTPUT_PATH = f"src/ngrams/results/{N}grams_per_{GROUPBY}.csv"

def main():
    """
    take ngrams from the pkl made by ngrams.py and group them by year, decade or period, outputting a csv file"""

    movies_ids, ngrams = inputs()

    # converting ngrams to dict for faster access
    ngrams_dict = dict(ngrams)

    # group by year and aggregate by list the other columns
    movies_ids = movies_ids.groupby("Movie release date").agg(list).reset_index()

    # count the number of movies per year
    movies_ids["Number of movies"] = movies_ids["Wikipedia movie ID"].apply(lambda x: len(x))

    # count the ngrams for each grouped column
    movies_ids[f"top {TOP_NGRAM} {N}grams"] = movies_ids["Wikipedia movie ID"].apply(lambda x: count_ngrams(x, ngrams_dict))

    # save the results
    movies_ids[["Movie release date", "Number of movies", f"top {TOP_NGRAM} {N}grams"]].to_csv(OUTPUT_PATH, index=False)



def inputs():
    """
    Returns: movies_ids and ngrams
    """

    # importing the already preprocessing movie dataset
    movies_df = pd.read_csv(os.path.join(DATASET_PATH,"processed_movies.csv"))

    def int_or_empty(string):
        """converts the string to int or pd.NA if it fails
        and if GROUPBY is on decade, then round per decade
        """
        try:
            year = int(string)
            # if GROUPBY is on decade, then round per decade
            if GROUPBY == "decade":
                year -= year % 10
            return year
        except:
            return pd.NA
        
    def get_period(year):
        """converts the year to period or pd.NA if it fails
        """
        try:
            return periods_map_inverse[int(year)]
        except:
            return pd.NA  

    if GROUPBY == "period":
        movies_df["Movie release date"] = movies_df["Movie release date"].apply(get_period)
    else:
        movies_df["Movie release date"] = movies_df["Movie release date"].astype(int).apply(int_or_empty) 

    with open(NGRAM_PATH, "rb") as file:
        ngrams = pickle.load(file)

    return movies_df[["Wikipedia movie ID", 'Movie release date']], ngrams


def count_ngrams(ids: list, ngrams_dict):
    res = Counter()
    for id in ids:
        try:
            res.update(
                list(
                    ngrams_dict[id]
                )
            )
        except: continue
    # return the most common ngrams
    return res.most_common(TOP_NGRAM)

if __name__ == "__main__":
    main()