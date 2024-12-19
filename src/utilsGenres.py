import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

periods_map = { 
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

def map_year_to_periods(year, periods):
    matching_periods = []
    for period, years in periods.items():
        if years['start_year'] <= year <= years['end_year']:
            matching_periods.append(period)
    if matching_periods:
        return matching_periods
    else:
        return ["Year not in any defined period"]
    
# Sort the periods based on their start year
sorted_periods = sorted(periods_map.items(), key=lambda x: x[1]['start_year'])

# Create a mapping from period name to its position in the timeline
period_order_map_dict = {period[0]: idx for idx, period in enumerate(sorted_periods)}


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

# Clearer to put it this way: theme and its corresponding genres
theme_mapping = {
    'Action/Adventure/Thriller': ['action', 'thriller', 'adventure', 'fantasy adventure', 'action comedy', 'action thrillers', 'spy', 'martial arts film', 'epic', 'road movie', 'action/adventure', 'adventure comedy', 'war film', 'military drama', 'combat film', 'anti-war film', 'political thriller', 'war film', 'military drama', 'combat film', 'western'],
    'Comedy': ['comedy', 'romantic comedy', 'black comedy', 'slapstick', 'comedy-drama', 'comedy film', 'adventure comedy', 'screwball comedy', 'fantasy comedy', 'domestic comedy', 'comedy of manners', 'sex comedy', 'comedy of errors'],
    'Drama/Mystery': ['drama', 'family drama', 'melodrama', 'crime drama', 'political drama', 'comedy-drama', 'marriage drama', 'courtroom drama', 'historical fiction', 'political cinema', 'social issues', 'crime comedy', 'mystery', 'detective', 'detective fiction'],
    'Romance': ['romance film', 'romantic comedy', 'romantic drama', 'romantic fantasy'],
    'Horror/Crime': ['crime fiction', 'crime thriller', 'gangster film', 'detective', 'detective fiction', 'crime drama', 'b-movie', 'heist', 'horror', 'slasher', 'psychological thriller', 'zombie film', 'horror comedy', 'monster movie', 'natural horror films', 'suspense'],
    'Science Fiction/Fantasy': ['science fiction', 'fantasy', 'superhero movie', 'sci-fi horror', 'superhero', 'fantasy comedy'],
    'Animation/Family': ['animation', 'animated cartoon', 'computer animation', 'animated musical', 'family', 'teen', '"children\'s/family"', '"children\'s"', 'family-oriented adventure', "children\'s fantasy"],
    'Historical/Biographical/Documentary': ['documentary', 'docudrama', 'rockumentary', 'concert film', 'mockumentary', 'period piece', 'historical drama', 'biography', 'history', 'biographical film', 'historical fiction'],
    'Short/Silent': ['short film', 'silent film'],
    'Black-and-White': ['black-and-white'], 
    'Indie/Experimental/LGBT': ['indie', 'experimental film', 'lgbt', 'gay', 'gay themed', 'gay interest'],
    'Musical': ['musical', 'musical drama', 'musical comedy'],
    'Other': []
}

# Inverting the theme_mapping to map genres to themes
genre_to_theme_mapping = {
    genre: theme
    for theme, genres in theme_mapping.items()
    for genre in genres
}

# Function to map genres to themes
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

# Define a color palette for the themes so they are assigned the same color for all plots
custom_colors = [
    "#e41a1c",  # Red
    "#377eb8",  # Blue
    "#4daf4a",  # Green
    "#ff7f00",  # Orange
    "#984ea3",  # Purple
    "#ffff33",  # Yellow
    "lime",
    "#000000",  # Black
    "#999999",  # Gray
    "pink",  # 
    "#c2c2f0",  # Lavender
    "blue",
    "#00ced1",  # Dark Turquoise
]


def plot_theme_over_years(theme_years_counts_pivot, themes, theme_colors):
    # Create an empty figure for the interactive plot
    fig = go.Figure()

    # Add each theme as a trace
    for theme in themes:
        if theme in theme_years_counts_pivot.columns:
            fig.add_trace(go.Scatter(
                x=theme_years_counts_pivot.index,  # x-axis is the release year
                y=theme_years_counts_pivot[theme],  # y-axis is the number of movies for each theme
                mode='lines',  # Use lines for the plot
                name=theme,  # Set the theme name for the legend
                line=dict(color=theme_colors.get(theme, 'gray')),  # Use color mapping for each theme
                hovertemplate='Year: %{x}<br>Number of Movies: %{y}<br>'  # Custom hover text
            ))

    # Add buttons for hide/show functionality
    visibility = [[True if i == idx else False for i in range(len(themes))]
                  for idx in range(len(themes))]

    # Define "Select Theme" button
    select_theme_button = dict(
        label="Select Theme",  # Displayed as a label in the dropdown
        method=None,  # Non-interactive option
        args=[]  # No action associated with this
    )

    theme_buttons = [
        dict(
            label=theme,
            method="update",
            args=[
                {"visible": vis},  # Update visibility
                {"title": f"Evolution of {theme} Movies Over the Years"}  # Update title
            ]
        )
        for theme, vis in zip(themes, visibility)
    ]

    # Combine "Select Theme" and theme-specific buttons
    buttons = [select_theme_button] + theme_buttons

    # Dropdown menu for individual themes
    dropdown_menu = dict(
        type="dropdown",
        buttons=buttons,
        x=1.214,
        xanchor="center",
        y=1.11,
        yanchor="top",
    )

    # Button for "Show All" placed on the side
    side_button = dict(
        type="buttons",
        buttons=[
            dict(
                label="Show All",
                method="update",
                args=[
                    {"visible": [True] * len(themes)},
                    {"title": "Evolution of Movie Themes Over the Years"}
                ]
            )
        ],
        x=1.005,  # Adjust position to place on the side
        xanchor="left",
        y=1.17,
        yanchor="middle",
    )

    # Update layout for the interactive plot
    fig.update_layout(
        updatemenus=[dropdown_menu, side_button],
        title={
            'text': 'Evolution of Movie Themes Over the Years',
            'x': 0.5,  # Center the title
            'xanchor': 'right'
        },
        title_x=0.5,
        barmode='stack',
        xaxis_title='Release Year',
        yaxis_title='Number of Movies',
        legend_title="Themes",
        height=500,
        width=1000,
    )

    # Save the interactive plot as an HTML file
    fig.write_html('/Users/lilly-flore/Desktop/line_theme_years_plot.html')

    # Create the static plot
    _, ax = plt.subplots(figsize=(12, 6))

    # Add each theme as a line on the static plot
    for theme in themes:
        if theme in theme_years_counts_pivot.columns:
            ax.plot(
                theme_years_counts_pivot.index,  # Years on the x-axis
                theme_years_counts_pivot[theme],  # Number of movies on the y-axis
                label=theme,  # Theme name for the legend
                color=theme_colors.get(theme, 'gray')  # Theme color
            )

    # Add a title and axis labels for the static plot
    ax.set_title('Evolution of Movie Themes Over the Years', fontsize=14)
    ax.set_xlabel('Release Year', fontsize=12)
    ax.set_ylabel('Number of Movies', fontsize=12)

    # Add a legend for the static plot
    ax.legend(title="Themes", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust margins to properly display the legend
    plt.tight_layout()

    # Show the static plot
    plt.show()

def plot_top3(theme_years_counts_pivot, themes, theme_colors, output_path_interactive, title):
    # 1. Create an interactive plot with Plotly
    fig_interactive = go.Figure()

    for theme in themes:
        if theme in theme_years_counts_pivot.columns:
            # Get genres associated with the theme
            associated_genres = theme_mapping.get(theme, [])
            if len(associated_genres) > 5:
                genres_text = ', '.join(associated_genres[:5]) + ', ...'  # Truncate and add ellipsis
            else:
                genres_text = ', '.join(associated_genres)

            fig_interactive.add_trace(go.Scatter(
                x=theme_years_counts_pivot.index,
                y=theme_years_counts_pivot[theme],
                mode='lines',
                name=theme,
                line=dict(color=theme_colors.get(theme, 'gray')),
                hovertemplate=f'<b>Genres:</b> {genres_text}<br>'
                              f'<b>Year:</b> %{{x}}<br><b>Number of Movies:</b> %{{y}}<br>'
            ))

    fig_interactive.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Release Year',
        yaxis_title='Number of Movies',
        legend_title="Themes",
        height=400,
        width=600,
    )

    # Save the interactive plot
    fig_interactive.write_html(output_path_interactive)

    # 2. Create a static plot with Matplotlib
    _, ax = plt.subplots(figsize=(12, 6))

    for theme in themes:
        if theme in theme_years_counts_pivot.columns:
            # Plot each theme with a line
            ax.plot(
                theme_years_counts_pivot.index,  # Years on the x-axis
                theme_years_counts_pivot[theme],  # Number of movies on the y-axis
                label=theme,  # Theme name for the legend
                color=theme_colors.get(theme, 'gray')  # Theme color
            )

    # Add title and axis labels for the static plot
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Release Year', fontsize=12)
    ax.set_ylabel('Number of Movies', fontsize=12)

    # Add legend for the static plot
    ax.legend(title="Themes", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust margins to display the legend correctly
    plt.tight_layout()


def plot_theme_over_periods(normalized_theme_periods_counts, themes, theme_colors):
    # Create an empty figure for the interactive plot
    fig = go.Figure()

    # Add each theme as a trace
    for theme in themes:
        if theme in normalized_theme_periods_counts.columns:
            fig.add_trace(go.Scatter(
                x=normalized_theme_periods_counts.index,  # x-axis is the release year
                y=normalized_theme_periods_counts[theme],  # y-axis is the number of movies for each theme
                mode='lines',  # Use lines for the plot
                name=theme,  # Set the theme name for the legend
                line=dict(color=theme_colors.get(theme, 'gray')),  # Use color mapping for each theme
                hovertemplate='Year: %{x}<br>Number of Movies: %{y}<br>'  # Custom hover text
            ))

    # Add buttons for hide/show functionality
    visibility = [[True if i == idx else False for i in range(len(themes))]
                  for idx in range(len(themes))]

    # Define "Select Theme" button
    select_theme_button = dict(
        label="Select Theme",  # Displayed as a label in the dropdown
        method=None,  # Non-interactive option
        args=[]  # No action associated with this
    )

    theme_buttons = [
        dict(
            label=theme,
            method="update",
            args=[
                {"visible": vis},  # Update visibility
                {"title": f"Evolution of {theme} Movies Over the Periods"}  # Update title
            ]
        )
        for theme, vis in zip(themes, visibility)
    ]

    # Combine "Select Theme" and theme-specific buttons
    buttons = [select_theme_button] + theme_buttons

    # Dropdown menu for individual themes
    dropdown_menu = dict(
        type="dropdown",
        buttons=buttons,
        x=1.214,
        xanchor="center",
        y=1.12,
        yanchor="top",
    )

    # Button for "Show All" placed on the side
    side_button = dict(
        type="buttons",
        buttons=[
            dict(
                label="Show All",
                method="update",
                args=[
                    {"visible": [True] * len(themes)},
                    {"title": "Evolution of Movie Themes Over the Periods"}
                ]
            )
        ],
        x=1.005,  # Adjust position to place on the side
        xanchor="left",
        y=1.185,
        yanchor="middle",
    )

    # Update layout for the interactive plot
    fig.update_layout(
        updatemenus=[dropdown_menu, side_button],
        title={
            'text': 'Evolution of Movie Themes Over the Periods',
            'x': 0.5,  # Center the title
            'xanchor': 'right'
        },
        title_x=0.5,
        barmode='stack',
        xaxis_title='Release Year',
        yaxis_title='Proportion of Movies',
        legend_title="Themes",
        height=600,
        width=1000,
    )

    # Save the interactive plot as an HTML file
    fig.write_html('/Users/lilly-flore/Desktop/line_theme_periods_plot.html')

    # Create the static plot
    _, ax = plt.subplots(figsize=(12, 8))

    # Plot each theme as a line on the static plot
    for theme in normalized_theme_periods_counts.columns:
        ax.plot(
            normalized_theme_periods_counts.index,  # Periods on the x-axis
            normalized_theme_periods_counts[theme],  # Normalized counts on the y-axis
            label=theme,  # Theme name for the legend
            color=theme_colors.get(theme, 'gray')  # Theme color
        )

    # Add title and labels for the static plot
    ax.set_title("Normalized Theme Counts Across Periods", fontsize=14)
    ax.set_xlabel('Periods', fontsize=12)
    ax.set_ylabel('Evolution of the Proprotion of Themes ', fontsize=12)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))

    # Add a legend for the static plot
    ax.legend(title="Themes", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xticks(rotation=80)

    # Adjust layout to display the legend properly
    plt.tight_layout()

    # Show the static plot
    plt.show()


def plot_bar_theme_periods(normalized_theme_periods_counts, theme_colors):
    """
    Generates an interactive Plotly bar chart and a static Matplotlib bar chart 
    to visualize the evolution of movie themes over periods.
    
    Parameters:
    - normalized_theme_periods_counts: DataFrame containing normalized theme counts per period.
    - theme_colors: Dictionary mapping themes to their respective colors.
    - output_path: Path to save the Plotly interactive plot as an HTML file.
    """
    themes = normalized_theme_periods_counts.columns.tolist()
    periods = normalized_theme_periods_counts.index.tolist()
    
    # Create the interactive Plotly bar chart
    fig = go.Figure()

    # Add each theme as a trace
    for theme in themes:
        fig.add_trace(go.Bar(
            x=periods,
            y=normalized_theme_periods_counts[theme],
            name=theme,
            marker_color=theme_colors.get(theme, 'gray'),
            hovertemplate='Period: %{x}<br>' + 'Theme proportion: %{y}<br>',
        ))
    
    # Configure dropdown menu for individual themes
    visibility = [[True if i == idx else False for i in range(len(themes))]
                  for idx in range(len(themes))]

    theme_buttons = [
        dict(
            label=theme,
            method="update",
            args=[
                {"visible": vis},  # Update visibility
                {"title": f"Evolution of {theme} Movies Over the Periods"}  # Update title
            ]
        )
        for theme, vis in zip(themes, visibility)
    ]

    dropdown_menu = dict(
        type="dropdown",
        buttons=[dict(label="Select Theme", method=None, args=[])] + theme_buttons,
        x=1.214,
        xanchor="center",
        y=1.12,
        yanchor="top",
    )

    # Add "Show All" button
    side_button = dict(
        type="buttons",
        buttons=[
            dict(
                label="Show All",
                method="update",
                args=[
                    {"visible": [True] * len(themes)},
                    {"title": "Evolution of Movie Themes Over the Periods"}
                ]
            )
        ],
        x=1.005,
        xanchor="left",
        y=1.185,
        yanchor="middle",
    )

    # Update Plotly layout
    fig.update_layout(
        updatemenus=[dropdown_menu, side_button],
        title={
            'text': 'Evolution of Movie Themes Over the Periods',
            'x': 0.5,
            'xanchor': 'right'
        },
        barmode='stack',
        xaxis_title="Period",
        yaxis_title="Proportion of Movies",
        legend_title="Themes",
        height=600,
        width=1000,
    )

    # Save the interactive plot
    fig.write_html('/Users/lilly-flore/Desktop/bar_theme_period.html')

    # Create the static Matplotlib bar chart
    _, ax = plt.subplots(figsize=(12, 8))

    # Plot each theme as a bar
    normalized_theme_periods_counts.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        color=[theme_colors.get(theme, 'gray') for theme in themes]
    )

    # Add a title and labels
    ax.set_title('Evolution of Movie Themes Over the Periods', fontsize=14)
    ax.set_xlabel('Periods', fontsize=12)
    ax.set_ylabel('Proportion of Movies', fontsize=12)

    # Rotate x-axis labels by 80 degrees
    plt.xticks(rotation=80)

    # Add a legend
    ax.legend(title="Themes", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout to make space for the legend
    plt.tight_layout()

    # Show the static plot
    plt.show()

def plot_correlation_matrix(normalized_theme_periods_counts):
    """
    Generates both an interactive Plotly heatmap and a static Seaborn heatmap to 
    visualize the correlation matrix between periods.

    Parameters:
    - normalized_theme_periods_counts: DataFrame of normalized theme counts for periods.
    - output_path: Path to save the Plotly interactive heatmap as an HTML file.
    """
    # Transpose the data for correlation calculation
    periods_themes = normalized_theme_periods_counts.T

    # Calculate the correlation matrix
    correlation_matrix = periods_themes.corr()

    # Generate hover text for interactive heatmap
    hover_text = []
    for i in range(len(correlation_matrix)):
        hover_row = []
        for j in range(len(correlation_matrix.columns)):
            top_themes = periods_themes.iloc[:, i].sort_values(ascending=False).index[:3]
            hover_row.append(f"Top 3 themes: {', '.join(top_themes)}")
        hover_text.append(hover_row)

    # Interactive Plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='Viridis',
        text=hover_text,
        hoverinfo='text',
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title={
            'text': "Interactive Correlation Matrix Between Periods",
            'x': 0.5,
            'xanchor': 'center'
        },
        height=800,
        width=1000,
        xaxis_tickangle=45
    )

    # Save and show interactive plot
    fig.write_html('/Users/lilly-flore/Desktop/period_corr_matrix.html')

    # Static Seaborn heatmap
    _, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix,
        annot=True,  # Annotate cells with the correlation value
        fmt=".2f",  # Display correlation value with 2 decimals
        cmap='viridis_r',
        cbar_kws={'label': 'Correlation'},
        xticklabels=correlation_matrix.columns,
        yticklabels=correlation_matrix.index,
        ax=ax,
        annot_kws={'size': 10},  # Adjust font size for annotations
        square=True  # Keep the heatmap square
    )

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=80)

    # Set the title and labels
    ax.set_title('Correlation Matrix Between Periods', fontsize=16)
    ax.set_xlabel('Periods', fontsize=12)
    ax.set_ylabel('Periods', fontsize=12)

    # Adjust layout to fit the labels and title
    plt.tight_layout()

    # Show the static plot
    plt.show()