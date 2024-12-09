import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
sys.path.append(".")
from src.utils import movies_groupby_and_plots, constants

TOP_N = 20 # TOP N ngram per year or decade
NGRAM_RANGE = (1,3)
DATASET_PATH = constants.DATASET_PATH.value
GROUPBY = ["year", "decade", "period"][1]
OUTPUT_PATH = f"src/ngrams/results/{NGRAM_RANGE[0]}-{NGRAM_RANGE[1]}grams_tfidf_per_{GROUPBY}.csv"

def main():
    movies_ids, movie_plots_df = movies_groupby_and_plots(DATASET_PATH, GROUPBY)

    # count the ngrams for each grouped column
    movies_ids["Ngrams and score"] = movies_ids["Wikipedia movie ID"].apply(lambda x: tfidf_ranking(x, NGRAM_RANGE, TOP_N, movie_plots_df))
    # save the results
    movies_ids[["Movie release date", "Number of movies", "Ngrams and score"]].to_csv(OUTPUT_PATH, index=False)




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