import pandas as pd
import os
import pickle
from tqdm import tqdm_gui
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
sys.path.append(".")
from src.utils import periods_map_inverse

TOP_N = 20 # TOP N ngram per year or decade
NGRAM_RANGE = (1,3)
DATASET_PATH = "data/"
GROUPBY = ["year", "decade", "period"][1]
OUTPUT_PATH = f"src/ngrams/results/{NGRAM_RANGE[0]}-{NGRAM_RANGE[1]}grams_tfidf_per_{GROUPBY}.csv"

def main():
    movies_ids, movie_plots_df = inputs()

    # group by year and aggregate by list the other columns
    movies_ids = movies_ids.groupby("Movie release date").agg(list).reset_index()
    # count the number of movies per year
    movies_ids["Number of movies"] = movies_ids["Wikipedia movie ID"].apply(lambda x: len(x))
    # count the ngrams for each grouped column
    movies_ids["Ngrams and score"] = movies_ids["Wikipedia movie ID"].apply(lambda x: tfidf_ranking(x, NGRAM_RANGE, TOP_N, movie_plots_df))
    # save the results
    movies_ids[["Movie release date", "Number of movies", "Ngrams and score"]].to_csv(OUTPUT_PATH, index=False)


def inputs():
    """
    Returns: movies_ids and ngrams
    """
    movies_df = pd.read_csv(os.path.join(DATASET_PATH,"processed_movies.csv"))

    movie_plots_df = pd.read_csv(os.path.join(DATASET_PATH, "processed_plot_summaries.csv"))
    # renaming columns
    movie_plots_df.columns = ["Wikipedia movie ID", "summary", "date"]
    # removing the date column
    movie_plots_df = movie_plots_df[["Wikipedia movie ID", "summary"]]
    # setting the index as the Wikipedia movie ID
    movie_plots_df = movie_plots_df.set_index("Wikipedia movie ID")

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
        """converts the year to period
        """
        try:
            return periods_map_inverse[int(year)]
        except:
            return pd.NA  

    if GROUPBY == "period":
        movies_df["Movie release date"] = movies_df["Movie release date"].apply(get_period)
    else:
        movies_df["Movie release date"] = movies_df["Movie release date"].astype(int).apply(int_or_empty) #pd.to_datetime(movies_df['Movie release date'], errors='coerce').dt.year.apply(int_or_empty)

    # converts the movie_plots_df to a Series
    movie_plots_df = movie_plots_df.squeeze()

    return movies_df, movie_plots_df #[["Wikipedia movie ID", 'Movie release date']], movie_plots_df


def tfidf_ranking(ids: list, ngram_range, top_n, movie_plots_df):

    documents = []
    # get the documents' summaries
    for id in ids:
        try:
            documents.append(
                movie_plots_df.get(id)
            )
        except:
            continue
    # remove empty summaries
    documents = [doc for doc in documents if doc]
    # if there are no documents then return pd.NA
    if not documents or documents is None: return pd.NA

    # compute the tfidf scores
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray().sum(axis=0)
    
    tfidf_scores = pd.DataFrame({'ngram': feature_names, 'tfidf': scores})
    tfidf_scores = tfidf_scores.sort_values(by='tfidf', ascending=False)
    # memory management
    del documents, feature_names, vectorizer, scores
    # return the top n ngrams
    return tfidf_scores.head(top_n).values


if __name__ == "__main__":
    main()