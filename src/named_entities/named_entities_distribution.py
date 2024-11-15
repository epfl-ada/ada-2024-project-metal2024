import pandas as pd
import os
import pickle
from tqdm import tqdm_gui
from collections import Counter, defaultdict
import ast
import sys
sys.path.append(".")
from src.utils import periods_map_inverse

TOP_NAMED_ENTITIES = 10 # TOP NAMED_ENTITIES
DATASET_PATH = "DATA/"
GROUPBY = ["year", "decade", "period"][-1]
# please run named_identities.py before to generate the csv below
NE_CSV_PATH = "src/named_entities/named_entities.csv"
OUTPUT_PATH = f"src/named_entities/distributions/named_entities_per_{GROUPBY}.csv"


def main():
    movies_ids, named_entities_df = inputs()

    id_date_map = movies_ids[["Wikipedia movie ID", "Movie release date"]].set_index("Wikipedia movie ID")

    named_entities_df["Movie release date"] = named_entities_df["ID"].apply(
        lambda x: get_date(x, id_date_map))

    # applies our function that combines list to every column
    # except for ID, who simply will be concatenated in a list
    cols_without_id = [col for col in named_entities_df.columns if col != "ID"]
    agg_methods = {col: agg_strategy for col in cols_without_id}
    agg_methods.update({"ID": list})

    named_entities_df = named_entities_df.groupby("Movie release date").agg(agg_methods)#.reset_index(drop=True)

    named_entities_df['Number of movies'] = named_entities_df["ID"].apply(lambda x: len(x))

    # if GROUPBY is True, we remove the ID col, otherwise we move Number of movies to the first cols
    #moved_cols = ["Number of movies"] if not (GROUPBY=="decade") else ["Number of movies", "ID"]
    moved_cols = ["Number of movies", "ID", "Movie release date"]
    named_entities_othercols = [col for col in named_entities_df.columns if col not in moved_cols]
    named_entities_cols = ["Number of movies"] + named_entities_othercols
    
    named_entities_df[named_entities_cols].to_csv(OUTPUT_PATH, index=True)

def get_date(id, id_date_map):
    try:
        return id_date_map.loc[id]["Movie release date"]
    except:
        return pd.NA


def agg_strategy(cols: pd.Series) -> Counter:
    """when groupby, merge rows by evaluating strings to list and combining them
    """
    res = []
    for col in cols:
        try:
            res.extend(ast.literal_eval(col))
        except:
            continue
    return Counter(res).most_common(TOP_NAMED_ENTITIES)


def inputs():
    """
    Returns: movies_ids and named identities
    """
    movies_df = pd.read_csv(os.path.join(DATASET_PATH,"movie.metadata.tsv"), delimiter="\t", header=None)
    movies_df.columns = ["Wikipedia movie ID", "Freebase movie ID", "Movie name", "Movie release date", "Movie box office revenue", "Movie runtime", "Movie languages (Freebase ID:name tuples)", "Movie countries (Freebase ID:name tuples)", "Movie genres (Freebase ID:name tuples)"]

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

    named_entities_df = pd.read_csv(NE_CSV_PATH)

    return movies_df[["Wikipedia movie ID", 'Movie release date']], named_entities_df


if __name__ == "__main__":
    main()