import pandas as pd
import sys
sys.path.append(".")
from src.utils import movies_groupby_and_plots, constants, correct_locations
import statsmodels.formula.api as smf
import numpy as np
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
import ast

DATASET_PATH = constants.DATASET_PATH.value
NAMED_ENTITIES_PATH = constants.NAMED_ENTITIES_PATH.value
# two periods to compare
# the periods are the keys of the dictionary in src/utils.py
PERIOD0 = "The Great Depression (1929-1939)"
PERIOD1 = "The Cold War and McCarthyism (1947-1991)"
# the number of genres to keep for the logistic regression for the propensity score
# the more genres, the more accurate the results
TOP_N_GENRES = 30
# be careful, the more samples, the longer the computation
# but the more accurate the results
# nx max_weight_matching is O(n^3)
SAMPLES = 200

def main():
    
    #locations(df_named_entities, PERIOD0, PERIOD1)
    df_movies, _ = movies_groupby_and_plots(DATASET_PATH, "period")

    df_movies = df_movies.set_index("Movie release date")
    df_movies = df_movies.loc[[PERIOD0, PERIOD1]]
    df_movies = df_movies.reset_index()
    df_movies = df_movies.rename(columns={"Movie release date": "period"})

    # converting the period to 1 or 0
    df_movies["period"] = df_movies["period"].apply(lambda x: 1 if x == PERIOD1 else 0)
    

    df_propensity_score= logistic_regression(df_movies)
    df_propensity_score = df_propensity_score.sample(SAMPLES)
    df_propensity_score.to_csv("src/causal_inference/matched.csv", index=False)

    matchings = propensity_score_matching(df_propensity_score)

    matchings_period0 = [i[0] for i in matchings]
    matchings_period1 = [i[1] for i in matchings]

    df_named_entities = pd.read_csv(NAMED_ENTITIES_PATH)

    df_matched_period0 = df_named_entities.loc[df_named_entities["ID"].isin(matchings_period0)]
    df_matched_period1 = df_named_entities.loc[df_named_entities["ID"].isin(matchings_period1)]

    #df_matched = pd.concat([df_matched_period0, df_matched_period1])
    #df_matched.to_csv("src/causal_inference/matched.csv", index=False)

    plots(df_matched_period0, df_matched_period1)

def plots(df_matched_period0, df_matched_period1, top_n = 15):

    def evaluate_col(col) -> pd.Series:
        res = []
        for items in col:
            try:
                res.extend(ast.literal_eval(items))
            except:
                continue

        return pd.Series(res)
        

    location_period0 = correct_locations(evaluate_col(df_matched_period0["LOCATION"])).value_counts().head(top_n)
    location_period1 = correct_locations(evaluate_col(df_matched_period1["LOCATION"])).value_counts().head(top_n)

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    location_period0.plot(kind="bar", ax=ax[0], title=f"Top {top_n} locations in {PERIOD0} for {df_matched_period0.shape[0]} movies")
    location_period1.plot(kind="bar", ax=ax[1], title=f"Top {top_n} locations in {PERIOD1} for {df_matched_period1.shape[0]} movies")
    plt.tight_layout()
    plt.show()



def logistic_regression(df: pd.DataFrame) -> pd.DataFrame:
    """
    returns the logistic regression model
    """
    # let's standardize the continuous features
    def standardize(series: pd.Series) -> pd.Series:
        return (series - series.mean())/series.std()

    rows_to_standardize = ["averageRating", "numVotes", ]
    
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
    return nx.max_weight_matching(G, maxcardinality=True)
    




if __name__ == "__main__":
    main()