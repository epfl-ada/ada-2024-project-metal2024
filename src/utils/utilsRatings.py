import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
from scipy.ndimage import convolve1d
from scipy.stats import spearmanr


def RatingsNbrOfMovies(rat, no_rat, end=107):
    # Group by release year and count movies with ratings
    
    rat_per_year = rat.groupby('Movie release date')['averageRating'].count().iloc[:end]

    # Group by release year and count movies without ratings
    no_rat_per_year = no_rat.groupby('Movie release date').size().iloc[:end-5]

    # Create a Plotly figure
    fig = go.Figure()

    # Add trace for movies with ratings
    fig.add_trace(go.Scatter(
        x=rat_per_year.index,
        y=rat_per_year.values,
        mode='lines+markers',
        name='With Rating',
    ))

    # Add trace for movies without ratings
    fig.add_trace(go.Scatter(
        x=no_rat_per_year.index,
        y=no_rat_per_year.values,
        mode='lines+markers',
        name='Without Rating',
    ))

    # Update layout
    fig.update_layout(
        title='Number of Movies With and Without Ratings by Release Year',
        xaxis_title='Release Year',
        yaxis_title='Number of Movies',
        template='plotly_white',
        showlegend=True,
    )

    # Display the figure in the Jupyter Notebook
    fig.show()

    # Optionally save the figure to an HTML file
    pio.write_html(fig, file='../ada-METAL-website/_includes/RatingsNbrOfMovies.html', auto_open=False)

# Smoothen the curve using a moving average
def smoothen_curve(values, window_size=4):
    mov_mean_weights = np.ones(window_size) / window_size
    return convolve1d(values, weights=mov_mean_weights)
def RatingsVsYearsAllPlotly(rat, end=107):
    # Group by release year and calculate metrics
    votes_per_year = rat.groupby('Movie release date')['numVotes'].sum()
    votes_per_year_mean = rat.groupby('Movie release date')['numVotes'].mean()

    # Smoothened values
    smoothened_values = smoothen_curve(votes_per_year.values, window_size=4)[:end]
    smoothened_values_mean = smoothen_curve(votes_per_year_mean.values, window_size=4)[:end]

    # Create a Plotly figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add trace for total votes (left y-axis)
    fig.add_trace(go.Scatter(
        x=votes_per_year.index[:end],
        y=smoothened_values,
        mode='lines',
        name='Total Votes (Log Scale)'
    ), secondary_y=False)

    # Add trace for mean votes per movie (right y-axis)
    fig.add_trace(go.Scatter(
        x=votes_per_year_mean.index[:end],
        y=smoothened_values_mean[:end],
        mode='lines',
        name='Mean Votes per Movie'
    ), secondary_y=True)

    # Update layout for axes and title
    fig.update_layout(
        title='Total and Mean Votes by Release Year',
        xaxis_title='Release Year',
        yaxis_title='Total Votes (Log Scale)',
        yaxis=dict(type='log'),  # Logarithmic scale for the left y-axis
        yaxis2_title='Mean Votes per Movie',  # Title for the right y-axis
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Show the figure in the notebook
    fig.show()

    # Save the figure to an HTML file
    pio.write_html(fig, file='../ada-METAL-website/_includes/RatingsNbrVsTime.html', auto_open=False)


def RatingsWithErrorBarsPlotly(rat, end=107):
    # Group by release year and calculate mean and standard deviation
    rating_stats = rat.groupby('Movie release date')['averageRating'].agg(['mean', 'std']).dropna().iloc[:end]

    # Extract years, means, and standard deviations
    years = rating_stats.index
    means = rating_stats['mean']
    stds = rating_stats['std']  # Halve the standard deviation

    # Multiply rating by the number of votes for each movie
    rat['rating_votes'] = rat['averageRating'] * rat['numVotes']

    # Group by movie release year and calculate the weighted average rating
    average_rating_votes_per_year = (
        rat.groupby('Movie release date')['rating_votes'].sum() /
        rat.groupby('Movie release date')['numVotes'].sum()
    ).iloc[:end]

    # Create the Plotly figure
    fig = go.Figure()

    # Add trace with error bars for mean rating
    fig.add_trace(go.Scatter(
        x=years,
        y=means,
        mode='markers+lines',
        error_y=dict(
            type='data',
            array=stds,   # Set the error values
            visible=True,
            color='red',
            thickness=1.5,
            width=3
        ),
        name='Mean Rating'
    ))

    # Add trace for weighted average rating
    fig.add_trace(go.Scatter(
        x=average_rating_votes_per_year.index,
        y=smoothen_curve(average_rating_votes_per_year.values),
        mode='lines+markers',
        line=dict(color='green'),
        name='Weighted Rating'
    ))

    # Update layout
    fig.update_layout(
        title='Movie Ratings by Release Year',
        xaxis_title='Release Year',
        yaxis_title='Rating',
        template='plotly_white',
        showlegend=True,
    )

    # Show the plot in the notebook
    fig.show()

    # Save the plot to an HTML file
    pio.write_html(fig, file='../ada-METAL-website/_includes/RatingsWithErrorBars.html', auto_open=False)
def get_movies_from_year(rat, year=1973):
    # Filter movies from the specified year
    movies_year = rat[rat['Movie release date'] == year]

    # Sort by descending order of number of votes
    movies_by_votes = movies_year.sort_values(by='numVotes', ascending=False)
    movies_by_votes_a = movies_year.sort_values(by='numVotes', ascending=True)


    # Sort by descending order of averageRating
    movies_by_rating = movies_year.sort_values(by='averageRating', ascending=False)
    movies_by_rating_a = movies_year.sort_values(by='averageRating', ascending=True)

    # Print results
    print(f"Movies from {year} sorted by number of votes (descending):")
    print(movies_by_votes[['Movie name', 'averageRating', 'numVotes']].head())
    print(f"Movies from {year} sorted by number of votes (ascending):")
    print(movies_by_votes_a[['Movie name', 'averageRating', 'numVotes']].head())
    print("\nMovies from {year} sorted by average rating (descending):")
    print(movies_by_rating[['Movie name', 'averageRating', 'numVotes']].head())
    print("\nMovies from {year} sorted by average rating (ascending):")
    print(movies_by_rating_a[['Movie name', 'averageRating', 'numVotes']].head())

def computePearsonSpearmanCorr(rat, year=None):

    # Filter for movies from 1973
    if year != None:
        movies_f = rat[rat['Movie release date'] == year]
    else:
        movies_f = rat
    # Pearson correlation coefficient for the year 1973
    pearson_corr = np.corrcoef(movies_f['numVotes'], movies_f['averageRating'])[0, 1]

    # Spearman correlation for the year 1973
    spearman_corr, _ = spearmanr(movies_f['numVotes'], movies_f['averageRating'])

    return pearson_corr, spearman_corr



# Function to plot the Pearson and Spearman correlations with customized colors
def plot_correlations(pearson_corr_all, spearman_corr_all, pearson_corr_1973, spearman_corr_1973, max_p, max_s, max_y,min_p, min_s, min_y):
    # Define the correlation values and labels
    correlations = [pearson_corr_all, spearman_corr_all,pearson_corr_1973, spearman_corr_1973, max_p, max_s,min_p, min_s]
    labels = [
        'Pearson (All Data)', 
        'Spearman (All Data)', 
        'Pearson (1973)', 
        'Spearman (1973)', 
        f'Maximum Pearson ({max_y})', 
        f'Maximum Spearman ({max_y})',
        f'Mininum Pearson ({min_y})', 
        f'Mininum Spearman ({min_y})'
    
    ]
    
    # Define the color list where Pearson is green and Spearman is blue
    colors = ['green', 'blue', 'green', 'blue', 'green', 'blue', 'green', 'blue']

    # Create a bar plot with custom colors
    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=correlations,
        marker=dict(color=colors)
    )])
    fig.update_xaxes(tickangle=45)


    # Update layout
    fig.update_layout(
        title='Pearson and Spearman Correlations',
        xaxis_title='Correlation Type',
        yaxis_title='Correlation Coefficient',
        template='plotly_white',
        showlegend=False,
        height=600
    )

    # Show the plot
    fig.show()
    pio.write_html(fig, file='../ada-METAL-website/_includes/RatingsCorr.html', auto_open=False)


def RatingsvsVotesScatterLog(rat, frac, nolog=False):
    # Filter out rows where numVotes <= 0
    sampled_rat = rat.sample(frac=frac, random_state=42)  # Sample 20% of the data for better visualization

    # Create the Plotly figure
    fig = go.Figure()

    # Add scatter plot for the sampled data
    fig.add_trace(go.Scatter(
        x=sampled_rat['numVotes'],
        y=sampled_rat['averageRating'],
        mode='markers',
        name="Sampled Data",
        marker=dict(color='rgba(135, 206, 250, 0.5)'),
        text=sampled_rat['Movie name'],  # Set hover text to movie titles
        hovertext=sampled_rat['Movie name']  # Show movie title on hover
    ))
    x_axis_dict = dict(
            title='Number of Votes' ,
        ) if (nolog) else dict(
            title='Number of Votes (log scale)',
            type='log' 
        )
    # Update layout to include log scale and labels
    fig.update_layout(
        title=f'{int(frac*100)}% Sampled Scatter Plot of Movie Ratings vs. Number of Votes',
        xaxis=x_axis_dict,
        yaxis=dict(
            title='Average Rating'
        ),
        template='plotly_white',
    )

    # Show the plot
    fig.show()
    pio.write_html(fig, file='../ada-METAL-website/_includes/RatingsScatterPlotUgly.html' if nolog else '../ada-METAL-website/_includes/RatingsScatterPlot.html', auto_open=False)


import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

def RatingsvsVotesScatterYears(rat, frac, years=None):
    # If years is provided, filter the data for the specified years
    if years is not None:
        # removed the battle of the apes because annoying for the analysis lol
        rat = rat[rat['Movie release date'].isin(years) &  ~rat["Movie name"].str.strip().eq("Battle for the Planet of the Apes")]

    # Filter out rows where numVotes <= 0
    sampled_rat = rat.sample(frac=frac, random_state=42)  # Sample the data for better visualization

    # Create the Plotly figure
    fig = go.Figure()

    # Get a color scale from Plotly Express (e.g., 'Set3' for distinct colors)
    color_scale = px.colors.qualitative.Light24
    
    # Iterate through each year in the list and plot data with a unique color for each year
    for i, year in enumerate(years):
        year_data = sampled_rat[sampled_rat['Movie release date'] == year]
        
        fig.add_trace(go.Scatter(
            x=year_data['numVotes'],
            y=year_data['averageRating'],
            mode='markers',
            name=str(year),
            marker=dict(color=color_scale[i % len(color_scale)],opacity=0.5  # Set opacity to 0.5
                        ),
            text=year_data['Movie name'],  # Set hover text to movie titles
            hovertext=year_data['Movie name']  # Show movie title on hover
        ))

    # Update layout to include log scale and labels
    fig.update_layout(
        title=f'Scatter Plot of Movie Ratings vs. Number of Votes for specific years',
        xaxis=dict(
            title='Number of Votes (log scale)',
            type='log'
        ),
        yaxis=dict(
            title='Average Rating'
        ),
        template='plotly_white',
        showlegend=True
    )

    # Show the plot
    fig.show()

    # Save the plot to an HTML file
    pio.write_html(fig, file='../ada-METAL-website/_includes/RatingsScatterPlotYears.html', auto_open=False)

# Compute the average rating, weighted average rating, and standard deviation per period
def compute_average_ratings_per_period(rat):
    # Compute the average rating per period
    avg_rating_per_period = rat.groupby('Period')['averageRating'].mean()

    # Compute the standard deviation of the average rating per period
    std_rating_per_period = rat.groupby('Period')['averageRating'].std()

    # Compute the weighted average rating per period
    weighted_avg_rating_per_period = (
        rat.groupby('Period').apply(
            lambda x: (x['numVotes'] * x['averageRating']).sum() / x['numVotes'].sum()
        )
    )

    return avg_rating_per_period, std_rating_per_period, weighted_avg_rating_per_period

# Plot the average ratings and weighted average ratings with error bars
def plot_ratings_per_period(avg_rating_per_period, std_rating_per_period, weighted_avg_rating_per_period, periods_map):
    fig = go.Figure()
    ordered_periods = list(periods_map.keys())
    
    # Reorder the data to match the period order
    avg_rating_per_period = avg_rating_per_period.reindex(ordered_periods)
    std_rating_per_period = std_rating_per_period.reindex(ordered_periods)
    weighted_avg_rating_per_period = weighted_avg_rating_per_period.reindex(ordered_periods)

    # Add the average rating line with error bars
    fig.add_trace(go.Scatter(
        x=avg_rating_per_period.index,
        y=avg_rating_per_period.values,
        mode='lines+markers',
        name='Average Rating',
        line=dict(color='blue'),
        marker=dict(size=6),
        error_y=dict(
            type='data',
            array=std_rating_per_period.values,  # Standard deviation as error
            visible=True
        )
    ))

    # Add the weighted average rating line
    fig.add_trace(go.Scatter(
        x=weighted_avg_rating_per_period.index,
        y=weighted_avg_rating_per_period.values,
        mode='lines+markers',
        name='Weighted Average Rating',
        line=dict(color='orange'),
        marker=dict(size=6)
    ))

    # Update layout
    fig.update_layout(
        title='Average and Weighted Average Ratings per Period with Error Bars',
        xaxis=dict(title='Period'),
        yaxis=dict(title='Rating'),
        template='plotly_white',
        legend=dict(title='Metrics'),
    )

    # Show the plot
    fig.show()
    pio.write_html(fig, file='../ada-METAL-website/_includes/RatingsVsPeriods.html', auto_open=False)


def RatingsvsVotesScatter(rat, frac, showByDefault, periods_map):

    # Filter out rows where numVotes <= 0
    sampled_rat = rat.sample(frac=frac, random_state=42)  # Sample the data for better visualization

    # Create the Plotly figure
    fig = go.Figure()

    # Get a color scale from Plotly Express (e.g., 'Set3' for distinct colors)
    color_scale = px.colors.qualitative.Light24
    
    # Iterate through each period and plot data with a unique color for each period
    for i, period in enumerate(periods_map.keys()):
        period_data = sampled_rat[sampled_rat['Period'] == period]
        
        fig.add_trace(go.Scatter(
            x=period_data['numVotes'],
            y=period_data['averageRating'],
            mode='markers',
            name=period,  # Use the period as the trace name
            marker=dict(color=color_scale[i % len(color_scale)], opacity=0.5),  # Set opacity to 0.5
            text=period_data['Movie name'],  # Set hover text to movie titles
            hovertext=period_data['Movie name'],  # Show movie title on hover
            visible='legendonly' if period not in showByDefault else None # Hide this period by default, only show in the legend
        ))

    # Update layout to include log scale and labels
    fig.update_layout(
        title=f'Scatter Plot of Movie Ratings vs. Number of Votes for periods',
        xaxis=dict(
            title='Number of Votes (log scale)',
            type='log'
        ),
        yaxis=dict(
            title='Average Rating'
        ),
        template='plotly_white',
        showlegend=True
    )

    # Show the plot
    fig.show()

    # Save the plot to an HTML file
    pio.write_html(fig, file='../ada-METAL-website/_includes/RatingsScatterPlotPeriods.html', auto_open=False)
# Function to map genres to themes

# Transform string into a list of strings (word = genre)
def clean_genres(x):
    # If the value is a string (which contains the genres), clean it
    if isinstance(x, str):
        str_to_list = [genre.strip("'") for genre in x.strip("[]").split(", ")]
        #str_to_list = [genre for genre in str_to_list if genre not in unwanted_genres]
        return str_to_list if str_to_list else None
    # If it's already a list, return it as is
    elif isinstance(x, list):
        return x
    else:
        genres = []
    # Remove the row if the list is empty
    return genres if genres else None

def map_genres_to_themes(genres, mapping):
    if isinstance(genres, list):
        themes = set()
        for genre in genres:
            if genre in mapping:
                themes.add(mapping[genre])
            else:
                themes.add('Other')
        return themes
    elif isinstance(genres, str):
        if genres in mapping:
            return {mapping[genres]}
        else:
            return {'Other'}
    return None

def compute_weighted_ratings_per_theme_period(rat, periods_map):
    """
    Compute the weighted average ratings per theme and period.
    
    Parameters:
    - rat: DataFrame containing movie data, including 'Themes', 'Period', 'numVotes', and 'averageRating'.
    - periods_map: Dictionary mapping periods to their ordered labels.
    
    Returns:
    - weighted_avg_ratings: DataFrame indexed by Theme and Period, containing weighted average ratings.
    """
    # Explode the 'Themes' column to handle multiple themes per movie
    exploded_rat = rat.explode('Themes')

    # Order periods explicitly according to the mapping
    ordered_periods = list(periods_map.keys())
    exploded_rat['Period'] = pd.Categorical(
        exploded_rat['Period'], categories=ordered_periods, ordered=True
    )

    # Compute the weighted average rating grouped by Theme and Period
    weighted_avg_ratings = exploded_rat.groupby(['Themes', 'Period']).apply(
        lambda x: (x['numVotes'] * x['averageRating']).sum() / x['numVotes'].sum()
    )

    # Convert to a DataFrame for easier handling
    weighted_avg_ratings = weighted_avg_ratings.reset_index(name='Weighted Average Rating')
    
    return weighted_avg_ratings


def plot_weighted_ratings_by_theme(weighted_avg_ratings, weighted_avg_rating_per_period,periods_map,old, showByDefault):
    """
    Plot the weighted average ratings per theme across periods using a line plot.
    
    Parameters:
    - weighted_avg_ratings: DataFrame containing weighted average ratings per theme and period.
    - periods_map: Dictionary mapping periods to their ordered labels.
    """
    fig = go.Figure()

    # Order periods explicitly according to the mapping
    ordered_periods = list(periods_map.keys())
    weighted_avg_ratings['Period'] = pd.Categorical(
        weighted_avg_ratings['Period'], categories=ordered_periods, ordered=True
    )

    # Get unique themes for plotting
    unique_themes = weighted_avg_ratings['Themes'].unique()

    # Assign colors to themes for consistent visualization
    color_scale = px.colors.qualitative.Light24
    theme_colors = {theme: color_scale[i % len(color_scale)] for i, theme in enumerate(unique_themes)}

    # Plot a line for each theme
    for theme, color in theme_colors.items():
        theme_data = weighted_avg_ratings[weighted_avg_ratings['Themes'] == theme]
        fig.add_trace(go.Scatter(
            x=theme_data['Period'],
            y=theme_data['Weighted Average Rating'],
            mode='lines+markers',
            name=theme,
            line=dict(color=color, width=2),
            marker=dict(size=6),
            visible='legendonly' if theme not in showByDefault else None # Hide this period by default, only show in the legend

        ))
        

    weighted_avg_rating_per_period = weighted_avg_rating_per_period.reindex(ordered_periods)
    # Add the weighted average rating line
    fig.add_trace(go.Scatter(
        x=weighted_avg_rating_per_period.index,
        y=weighted_avg_rating_per_period.values,
        mode='lines+markers',
        name='General Weighted Average Rating',
        line=dict(color='black'),
        marker=dict(size=6)
    ))

    # Update layout
    fig.update_layout(
        title='Weighted Average Ratings per Theme Across Periods vs General average rating',
        xaxis=dict(title='Period'),
        yaxis=dict(title='Weighted Average Rating'),
        template='plotly_white',
        legend=dict(title='Themes'),
        xaxis_tickangle=-45,
        height=620  # Adjust the height in pixels
    )

    # Show the plot
    fig.show()
    pio.write_html(fig, file='../ada-METAL-website/_includes/RatingsThemesVsPeriod'+ ('Old' if old else 'New')+'.html', auto_open=False)

def RatingsvsVotesByThemePeriod(rat, selected_pairs,old, frac=1.0):
    """
    Plot the average rating vs. the number of votes on log scales for selected Theme-Period pairs using Plotly.

    Parameters:
    - rat: DataFrame containing movie data.
    - selected_pairs: List of (Theme, Period) tuples to filter and plot.
    - frac: Fraction of data to sample for plotting (for visualization clarity).
    """
    # Sample the data for better visualization if a fraction is specified
    sampled_rat = rat.sample(frac=frac, random_state=42).explode('Themes')

    # Create a Plotly figure
    fig = go.Figure()

    # Get a color scale from Plotly Express (e.g., 'Light24' for distinct colors)
    color_scale = px.colors.qualitative.Light24

    # Generate colors for unique Theme-Period pairs
    unique_pairs = set(selected_pairs)
    pair_colors = {pair: color_scale[i % len(color_scale)] for i, pair in enumerate(unique_pairs)}

    # Plot each selected Theme-Period pair
    for pair, color in pair_colors.items():
        theme, period = pair
        pair_data = sampled_rat[
            (sampled_rat['Period'] == period) & 
            (sampled_rat['Themes'] == theme)
        ]
        fig.add_trace(go.Scatter(
            x=pair_data['numVotes'],
            y=pair_data['averageRating'],
            mode='markers',
            name=f"{theme} ({period})",
            marker=dict(
                color=color,
                size=6,
                opacity=0.4
            ),
            text=pair_data['Movie name'],  # Set hover text to movie titles
            hovertext=pair_data['Movie name']  # Show movie title on hover
        ))

    # Update layout to include log scales and labels
    fig.update_layout(
        title='Movie Ratings vs. Number of Votes by Theme-Period',
        xaxis=dict(
            title='Number of Votes (log scale)',
            type='log'
        ),
        yaxis=dict(
            title='Average Rating',
            type='linear'  # Linear for ratings
        ),
        template='plotly_white',
        showlegend=True,
        legend=dict(title='Theme-Period Pairs')
    )

    # Show the plot
    fig.show()
    pio.write_html(fig, file='../ada-METAL-website/_includes/RatingsThemesPeriodsPair'+ ('Old' if old else 'New')+'.html', auto_open=False)

def RatingsvsVotesByTheme(rat, theme_mapping, showByDefault, frac=1.0):
    """
    Plot the average rating vs. the number of votes on log scales for each theme using Plotly.

    Parameters:
    - rat: DataFrame containing movie data.
    - theme_mapping: Dictionary mapping themes to genres (not used directly in this function, assuming 'Theme' is already assigned).
    - frac: Fraction of data to sample for plotting (for visualization clarity).
    """
    # Sample the data for better visualization if a fraction is specified
    sampled_rat = rat.sample(frac=frac, random_state=42).explode('Themes')

    # Create a Plotly figure
    fig = go.Figure()

    # Get a color scale from Plotly Express (e.g., 'Light24' for distinct colors)
    color_scale = px.colors.qualitative.Light24

    # Get unique themes and assign colors
    unique_themes = theme_mapping.keys()
    theme_colors = {theme: color_scale[i % len(color_scale)] for i, theme in enumerate(unique_themes)}

    # Plot each theme
    for theme, color in theme_colors.items():
        theme_data = sampled_rat[sampled_rat['Themes'] == theme]
        fig.add_trace(go.Scatter(
            x=theme_data['numVotes'],
            y=theme_data['averageRating'],
            mode='markers',
            name=theme,
            marker=dict(
                color=color,
                size=6,
                opacity=0.7
            ),
            visible='legendonly' if theme not in showByDefault else None, # Hide this period by default, only show in the legend
            text=theme_data['Movie name'],  # Set hover text to movie titles
            hovertext=theme_data['Movie name']  # Show movie title on hover
        ))

    # Update layout to include log scales and labels
    fig.update_layout(
        title='Movie Ratings vs. Number of Votes by Theme',
        xaxis=dict(
            title='Number of Votes (log scale)',
            type='log'
        ),
        yaxis=dict(
            title='Average Rating',
            type='linear'  # Linear for ratings
        ),
        template='plotly_white',
        showlegend=True,
        legend=dict(title='Themes')
    )

    # Show the plot
    fig.show()
