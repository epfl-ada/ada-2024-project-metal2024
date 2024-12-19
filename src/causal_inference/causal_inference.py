import pandas as pd
import sys
sys.path.append(".")
from src.utils import movies_groupby_and_plots, constants, correct_locations, correct_money
import statsmodels.formula.api as smf
import numpy as np
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
import ast
import matplotlib.cm as cm
import re
import time

DATASET_PATH = constants.DATASET_PATH.value
NAMED_ENTITIES_PATH = constants.NAMED_ENTITIES_PATH.value
TF_IDF_NGRAMS_PATH = constants.NGRAMS_RESULTS_PATH.value + "1-3grams_tfidf.csv"
# two periods to compare
# the periods are the keys of the dictionary in src/utils.py
PERIOD0 = "The Great Depression (1929-1939)"
PERIOD1 = "The Civil Rights Movement (1960-1970)"
# column for named_entities/named_entities.csv to compare
COMPARED = ["DATE","PERSON","LOCATION","NUMBER","TIME","ORGANIZATION","MISC","DURATION","SET","ORDINAL","MONEY","PERCENT", "TF-IDF_NGRAMS"][5]
# dictionnary mapping the named entities to a preprocessing function if needed
NAMED_ENTITIES_PREPROCESSING = {"LOCATION": correct_locations, "MONEY": correct_money}
# the number of genres to keep for the logistic regression for the propensity score
# the more genres, the more accurate the results
TOP_N_GENRES = 30 #30
# be careful, the more samples, the longer the computation
# but the more accurate the results
# nx max_weight_matching is O(n^3)
SAMPLES = 1400 #1400

def main():
    
    # getting the movies 
    df_movies, _ = movies_groupby_and_plots(DATASET_PATH, "period")

    df_movies = df_movies.set_index("Movie release date")
    df_movies = df_movies.loc[[PERIOD0, PERIOD1]]
    df_movies = df_movies.reset_index()
    df_movies = df_movies.rename(columns={"Movie release date": "period"})

    # converting the period to 1 or 0
    df_movies["period"] = df_movies["period"].apply(lambda x: 1 if x == PERIOD1 else 0)
    
    # getting the propensity score
    df_propensity_score= logistic_regression(df_movies)
    df_propensity_score = df_propensity_score.sample(SAMPLES)

    stopwatch = time.time()
    matchings = propensity_score_matching(df_propensity_score)
    print(f"Time to compute the matching with {SAMPLES} samples: {time.time() - stopwatch:.2f} seconds")

    # getting the matchings
    matchings_period0 = [i[0] for i in matchings]
    matchings_period1 = [i[1] for i in matchings]

    if COMPARED == "TF-IDF_NGRAMS":
        df_compared = pd.read_csv(TF_IDF_NGRAMS_PATH)
        df_compared = df_compared.rename(columns={"Wikipedia movie ID": "ID", "Ngrams and score": COMPARED})
    else:
        df_compared = pd.read_csv(NAMED_ENTITIES_PATH)

    df_matched_period0 = df_compared.loc[df_compared["ID"].isin(matchings_period0)]
    df_matched_period1 = df_compared.loc[df_compared["ID"].isin(matchings_period1)]

    tf_idf = True if COMPARED == "TF-IDF_NGRAMS" else False
    df_matched_period0, df_matched_period1 = preprocessing(df_matched_period0, df_matched_period1, tf_idf=tf_idf)

    plots(df_matched_period0, df_matched_period1)


def preprocessing(df_matched_period0, df_matched_period1, tf_idf = False) -> tuple[pd.Series, pd.Series]:
    """converts strings to lists and preprocesses the data, returns value counts"""

    def evaluate_col(col) -> pd.Series:
        res = []
        for items in col:
            try:
                if tf_idf: 
                    # we have to use re to extract the words from the string
                    words = re.findall(r"'([^']*)'", items)
                    res.extend(words)
                else: res.extend(ast.literal_eval(items))
            except:
                continue


        return pd.Series(res)
    
    # preprocessing function
    preprocessing_function = NAMED_ENTITIES_PREPROCESSING.get(COMPARED, lambda x: x)
        
    location_period0 = preprocessing_function(evaluate_col(df_matched_period0[COMPARED])).value_counts()
    location_period1 = preprocessing_function(evaluate_col(df_matched_period1[COMPARED])).value_counts()

    return location_period0, location_period1


def plots(df_matched_period0: pd.Series, df_matched_period1: pd.Series, top_n=25):
    """plots the data"""

    _, ax = plt.subplots(2, 1, figsize=(14, 5))
    # make the color vary with the value
    df_matched_period0.head(top_n).plot(kind="bar", ax=ax[0], title=f"Top {top_n} {COMPARED} in {PERIOD0} for {df_matched_period0.shape[0]} movies", rot=45, color= cm.viridis(df_matched_period0.values/df_matched_period0.values.max()), ylabel="Count", logy=False)
    df_matched_period1.head(top_n).plot(kind="bar", ax=ax[1], title=f"Top {top_n} {COMPARED} in {PERIOD1} for {df_matched_period1.shape[0]} movies", rot=45, color= cm.viridis(df_matched_period1.values/df_matched_period1.values.max()), ylabel="Count", logy=False)
    plt.tight_layout()
    plt.savefig(f"src/causal_inference/results/{COMPARED}.svg")
    plt.show()



def logistic_regression(df: pd.DataFrame) -> pd.DataFrame:
    """
    returns the logistic regression model
    """
    # let's standardize the continuous features
    def standardize(series: pd.Series) -> pd.Series:
        return (series - series.mean())/series.std()

    rows_to_standardize = ["averageRating", "numVotes"]
    
    for row in rows_to_standardize:
        df[row] = standardize(df[row])

    # taking the first genre    
    import ast
    def get_first_genre(x):
        try:
            return ast.literal_eval(x)[0]
        except:
            return pd.NA 
    df["Movie genres"] = df["Movie genres"].apply(get_first_genre)

    # renaming columns to remove spaces
    df = df.rename(columns={"Movie genres": "Movie_genres", "Movie name": "Movie_name"})


    def keep_n_first_genres(df: pd.DataFrame, n: int) -> pd.DataFrame:
        """
        keeps the n first genres
        """
        counts = df["Movie_genres"].value_counts().sort_values(ascending=False)
        keep = counts.head(n).index
        return df[df["Movie_genres"].isin(keep)]

    # keep the top n genres
    df = keep_n_first_genres(df, TOP_N_GENRES)

    
    # one hot encoding
    df = pd.get_dummies(df, columns=["Movie_genres"], drop_first=True, dtype=int)

    # remove spaces frmo column names
    df.columns = df.columns.str.replace(" ", "_").str.replace("'", '').str.replace(" - ", "_").str.replace("and", "_").str.replace("/", "_").str.replace("-", "_")

    # drop rows with missing values
    df = df.dropna()

    # drop the movie name, we don't need it, too many unique values
    df = df.drop(columns=["Movie_name"])

    col_names = [f"C({col})" for col in df.columns if "Movie_genres" in col]
    col_names = " + ".join(col_names)

    mod = smf.logit(formula="period ~ averageRating + numVotes + " + col_names, data=df)


    res = mod.fit()

    # Extract the estimated propensity scores
    df['Propensity_score'] = res.predict()


    print(res.summary())

    return df



def propensity_score_matching(df: pd.DataFrame) -> pd.DataFrame:
    """
    returns the propensity score matching
    """
    def get_similarity(propensity_score1, propensity_score2):
        '''Calculate similarity for instances with given propensity scores'''
        return 1-np.abs(propensity_score1-propensity_score2)
    
    # Separate the treatment and control groups
    period1_df = df[df['period'] == 1]
    period0_df = df[df['period'] == 0]

    # Create an empty undirected graph
    G = nx.Graph()

    # Loop through all the pairs of instances
    for _, period0_row in tqdm(period0_df.iterrows(), desc='Building the graph', total=len(period0_df)):
        for _, period1_row in period1_df.iterrows():

            # Calculate the similarity 
            similarity = get_similarity(period0_row['Propensity_score'],
                                        period1_row['Propensity_score'])

            # Add an edge between the two instances weighted by the similarity between them
            G.add_weighted_edges_from([(period0_row["Wikipedia_movie_ID"], period1_row["Wikipedia_movie_ID"], similarity)])


    # Generate and return the maximum weight matching on the generated graph
    print("Computing the maximum weight matching, could take some time...")
    return nx.max_weight_matching(G, maxcardinality=True)
    




if __name__ == "__main__":
    main()