import pandas as pd
import os
import pickle
from tqdm import tqdm_gui
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
sys.path.append(".")
from src.utils import periods_map_inverse

TOP_N = 10 # TOP N ngram per year or decade
NGRAM_RANGE = (1,3)
DATASET_PATH = "DATA/"
GROUPBY = ["year", "decade", "period"][1]
OUTPUT_PATH = f"src/ngrams/results/{NGRAM_RANGE[0]-NGRAM_RANGE[1]}grams_tfidf_per_{GROUPBY}.csv"

def main():
    movies_ids, movie_plots_df = inputs()

    movies_ids = movies_ids.groupby("Movie release date").agg(list).reset_index()
    movies_ids["Number of movies"] = movies_ids["Wikipedia movie ID"].apply(lambda x: len(x))

    movies_ids[f"Ngrams and score"] = movies_ids["Wikipedia movie ID"].apply(lambda x: tfidf_ranking(x, NGRAM_RANGE, TOP_N, movie_plots_df))
    #movies_ids[f"most 5 common {N}grams"] = movies_ids[f"{N}grams"].apply(lambda x: x.most_common(5))

    movies_ids[["Movie release date", "Number of movies", "Ngrams and score"]].to_csv(OUTPUT_PATH, index=False)



def inputs():
    """
    Returns: movies_ids and ngrams
    """
    movies_df = pd.read_csv(os.path.join(DATASET_PATH,"processed_movies.csv"))

    movie_plots_df = pd.read_csv(os.path.join(DATASET_PATH, "processed_plot_summaries.csv"))
    movie_plots_df.columns = ["Wikipedia movie ID", "summary", "date"]
    movie_plots_df = movie_plots_df[["Wikipedia movie ID", "summary"]]
    movie_plots_df = movie_plots_df.set_index("Wikipedia movie ID")

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
        movies_df["Movie release date"] = pd.to_datetime(movies_df['Movie release date'], errors='coerce').dt.year.apply(int_or_empty)

    return movies_df[["Wikipedia movie ID", 'Movie release date']], movie_plots_df


def tfidf_ranking(ids: list, ngram_range, top_n, movie_plots_df):

    documents = []
    for id in ids:
        try:
            documents.append(
                movie_plots_df.loc[id]["summary"]
            )
        except:
            continue
    
    if not documents: return pd.NA

    vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words='english')
    
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray().sum(axis=0)
    
    tfidf_scores = pd.DataFrame({'ngram': feature_names, 'tfidf': scores})
    tfidf_scores = tfidf_scores.sort_values(by='tfidf', ascending=False)
    
    return tfidf_scores.head(top_n).values


if __name__ == "__main__":
    main()