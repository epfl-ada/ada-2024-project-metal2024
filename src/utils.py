from enum import Enum

class constants(Enum):
    DATASET_PATH = "data/"
    NAMED_ENTITIES_PATH = "src/named_entities/named_entities.csv"
    NGRAMS_RESULTS_PATH = "src/ngrams/results/"

periods_map_ = { 
    "The Belle Époque (1900-1914)": {"start_year": 1900, "end_year": 1913}, 
    "World War I (1914-1918)": {"start_year": 1914, "end_year": 1919}, 
    "The Roaring Twenties (1920-1929)": {"start_year": 1920, "end_year": 1928}, 
    "The Great Depression (1929-1939)": {"start_year": 1929, "end_year": 1939}, 
    "World War II (1939-1945)": {"start_year": 1940, "end_year": 1945}, 
    "Early Cold War (1946-1960)": {"start_year": 1946, "end_year": 1960}, 
    "The Civil Rights Movement (1960-1970)": {"start_year": 1961, "end_year": 1970}, 
    "Late Cold War (1971-1991)": {"start_year": 1971, "end_year": 1991}, 
    "Post-Cold War and the New World Order (1992-2001)": {"start_year": 1992, "end_year": 2000}, 
    "War on Terrorism (2001-present)": {"start_year": 2001, "end_year": 2024}, 
}

def make_periods_map(periods_map: dict):
    """converting periods map to list with every year in it"""
    for key, item in periods_map.items():
        periods_map[key] = list(range(
            item["start_year"], # start
            item["end_year"] + 1 # stop
        ))
    return periods_map

periods_map = make_periods_map(periods_map_)

# !!! each year currently can have only one period and not multiple
def inverse_dict():
    """inversing the dictionnary"""
    periods_map_inverse_ = {}
    for key, value_list in periods_map.items():
        for item in value_list:
            periods_map_inverse_[item] = key
    return periods_map_inverse_

periods_map_inverse = inverse_dict()


import pandas as pd
import os

def movies_groupby_and_plots(DATASET_PATH, GROUPBY, perform_groupby = False):
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

    if perform_groupby:
        # group by year and aggregate by list the other columns
        movies_df = movies_df.groupby("Movie release date").agg(list).reset_index()
        # count the number of movies per year
        movies_df["Number of movies"] = movies_df["Wikipedia movie ID"].apply(lambda x: len(x))

    return movies_df, movie_plots_df #[["Wikipedia movie ID", 'Movie release date']], movie_plots_df


def correct_locations(series: pd.Series) -> pd.Series:
    """group the locations from the named entities
    without counting the words multiple times
    """
    corrections = {
        "New": "New York City",
        "York": pd.NA,
        "City": pd.NA,
        "Los": "Los Angeles",
        "Angeles": pd.NA,
        "San": "San Francisco",
        "Francisco": pd.NA,
        "United": "United States",
        "States": pd.NA,
        "US": "United States",
        "Salt": "Salt Lake",
        "Lake": pd.NA,
        "Las": "Las Vegas",
        "Vegas": pd.NA
    }
    series = series.apply(lambda x: corrections[x] if x in corrections else x)
    return series.dropna()

def correct_money(series: pd.Series) -> pd.Series:
    """group the locations from the named entities
    without counting the words multiple times
    """
    corrections = {
        "$": "dollars",
        "USD": "dollars",
        "dollar": "dollars"
    }
    series = series.apply(lambda x: corrections[x] if x in corrections else x)
    return series.dropna()