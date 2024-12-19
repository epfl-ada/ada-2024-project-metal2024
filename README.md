## ADA Project for group METAL2024


### Films the mirror of a century ?

This project analyze the transformation of film genres over time, analyzing how cultural, political, and economic shifts have influenced cinematic trends and how they are represented during these periods.

We will explore how significant global events, such as wars, economic depressions, or technological advancements, have driven changes in audience tastes and genre popularity. By examining patterns, we aim to understand how specific genres, like action-packed thrillers or heartfelt dramas, emerge and tells us how they relect the periods of the last century.

The website for our project can be consulted at: https://apanoush.github.io/ada-METAL-website/report  

Our datastory follows the following order: 

1. Introduction and periods selection and presentation
2. Genres analysis 
3. Periods Wordcloud
4. Causal inference
5. Ratings analysis

### Research Questions 

1. How genres emerged through the years and defined periods ? Can we see some known events appear from this analyis

2. From the summaries is it possible to classify film directly to a period from their summaries ? Does this classification make sense in a temporal perspective.

3. Are some periods correlated in their way of approaching themes? Can we see common or diverse subject emerge ? Do the period really matter or are the production and themes concerned random?

4. How did we rate films through the year, what do we like is their anny preferences emerging ?

### Dataset:

We used the provided CMU dataset for this project, partially enriched using IMDB to complete missing data and add ratings and number of votes for each movie.
The whole data preprocessing pipeline can be found in src/datapreprocessing. 


### Methods : Implementation of the research questions :

### Followed timeline  
- 18.10 Start Data exploration and Preprocessing 
- 1.11 Start P2 Deliverable Sections
- 15.11 Finish P2 Deliverable 
- 22.11 Finish Data Preprocessing
- 29.11 Final decision of global aborded themes/periods
- 6.12 First glimpse of showable results, final decision on data story platform
- 13.12 Start of P3 Deliverable Sections
- 20.12 Final P3 Deliverable

### Organization within the team: List of work done by each member
- Timo: General Datastory writing and website Introduction (Website Part 1)
- Lilly-Flore: Themes grouping and exploration (Website Part 2)
- Mathis: General Datastory writing and Word cloud (Website Part 3)
- Anoush: Ngrams,tf-idf/named_entities and causal inference (Website Part 4)
- Emilien: Ratings analysis and ReadMe report (Website Part 5)

### Project structure: 

Where to find the work done: 
- `results.ipynb` contains the plots and ideas we had when analyzing our dataset. 
It follows the same order as the website and explains the website's ideas and how we got there in greater detail. 
- `src/datapreprocessing.ipynb` contains the preprocessing steps done for the data to be used directly in `results.ipynb`
- `./data/` contains the originial dataset, and the datasets resulting from the preprocessing steps
- `./tests/` contains multiple different tests done on plots processing including ngrams and named entities. 
- `./src/` contains most of the code. `utils.py` and it's derivates `utilsGenres.py`, `utilsRatings.py` contain the code used for their respective parts.  
Folders `causal_inference`, `named_entities` and `ngrams` contain the code needed for these parts.
As they produce some files in their respective `results` folder, it was preferable to create subfolders. 
