import pandas as pd
import os
from collections import Counter
import ast
import sys
sys.path.append(".")
from src.utils import periods_map_inverse, constants, correct_locations

TOP_NAMED_ENTITIES = 20 # TOP NAMED_ENTITIES
DATASET_PATH = constants.DATASET_PATH.value
# GROUPBY can be "year", "decade" or "period"
GROUPBY = ["year", "decade", "period"][-1]
# please run named_identities.py before to generate the csv below
NE_CSV_PATH = "src/named_entities/named_entities.csv"
OUTPUT_PATH = f"src/named_entities/results/named_entities_per_{GROUPBY}.csv"


def main():
    """
    take named entities from the csv made by named_identities.py and group them by year, decade or period, outputting a csv file"""


    movies_ids, named_entities_df = inputs()

    # creating a map to get the date of a movie by its ID
    id_date_map = movies_ids[["Wikipedia movie ID", "Movie release date"]].set_index("Wikipedia movie ID")

    # get the date of the movie by its ID
    named_entities_df["Movie release date"] = named_entities_df["ID"].apply(
        lambda x: get_date(x, id_date_map))

    # applies our function that combines list to every column
    # except for ID, who simply will be concatenated in a list
    cols_without_id = [col for col in named_entities_df.columns if col != "ID"]
    agg_methods = {col: agg_strategy for col in cols_without_id}
    agg_methods.update({"ID": list})

    named_entities_df = named_entities_df.groupby("Movie release date").agg(agg_methods)
    # count the number of movies per year
    named_entities_df['Number of movies'] = named_entities_df["ID"].apply(lambda x: len(x))

    # reordering the columns
    moved_cols = ["Number of movies", "ID", "Movie release date"]
    named_entities_othercols = [col for col in named_entities_df.columns if col not in moved_cols]
    named_entities_cols = ["Number of movies"] + named_entities_othercols
    # save the results
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

    res = correct_locations(pd.Series(res)).to_list()
    return Counter(res).most_common(TOP_NAMED_ENTITIES)


def inputs():
    """
    Returns: movies_ids and named identities
    """
    movies_df = pd.read_csv(os.path.join(DATASET_PATH,"movie.metadata.tsv"), delimiter="\t", header=None)
    movies_df.columns = ["Wikipedia movie ID", "Freebase movie ID", "Movie name", "Movie release date", "Movie box office revenue", "Movie runtime", "Movie languages (Freebase ID:name tuples)", "Movie countries (Freebase ID:name tuples)", "Movie genres (Freebase ID:name tuples)"]

    def int_or_empty(string):
        """try to convert the string to int or pd.NA if it fails
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
        """try to convert the year to period or pd.NA if it fails
        """
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