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

## Historical Periods

1. **The Progressive Era (1900-1914)**  
2. **World War I (1914-1918)**  
3. **The Roaring Twenties (1920-1929)**  
4. **The Great Depression (1929-1939)**  
5. **World War II (1939-1945)**  
6. **The Early Cold War (1947-1960)**  
7. **The Civil Rights Movement (1961-1970)**  
8. **The Late Cold War (1971-1991)**  
9. **The Post-Cold War and the New World Order (1992-2001)** 
10. **War on terror (2001)** 
### Research Questions 

1. How genres emerged through the years and defined periods ? Can we see some known events appear from this analyis

2. From the summaries is it possible to classify film directly to a period from their summaries ? Does this classification make sense in a temporal perspective.

3. Are some periods correlated in their way of approaching themes? Can we see common or diverse subject emerge ? Do the period really matter or are the production and themes concerned random?

4. How did we rate films through the year, what do we like is their anny preferences emerging ?

### Dataset:

We used the provided CMU dataset for this project, partially enriched using IMDB to complete missing data and add ratings and number of votes for each movie.
The whole data preprocessing pipeline can be found in src/datapreprocessing. 


### Methods : Implementation of the research questions :

To analyze the transformation of film genres over time, we employed standard data analysis tools and methodologies seen in class. The process began with with basic exploration of the raw data and it's missing values and later followed by preprocessing which removed incomlete data or completed it using the IMDB dataset, especially to add ratings information. Theme evolution and Rating analysis were mostly done using standard Data vizualization techniques and dynamic graphs for the website were generated using plotly. 

Text processing techniques, such as tokenization, stop-word removal, and lemmatization, were applied to film summaries to generate word clouds for each historical period, highlighting the dominant themes and keywords. For the causal inference analysis, we utilized statistical models to explore relationships between key variables derived from plots, deriving named entities such as Organization or Location.

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

├── README.md  
├── data (folder)  
├── results.ipynb  
├── src  
│   ├── causal_inference (folder)  
│   ├── named_entities (folder)  
│   ├── ngrams (folder)  
│   ├── datapreprocessing.ipynb  
│   ├── utils.py  
│   ├── utilsGenres.py  
│   ├── utilsRatings.py  
│   └── utilsWordcloud.py  
├── tests  
│   ├── All_Genres_Top_Periods.csv  
│   ├── named_entities_tests.ipynb   
│   ├── ngram_test.csv  
│   ├── ngrams_tf_idf_tests.ipynb  
│   ├── ngrams_viz.ipynb  
│   ├── plot-LLM.ipynb  
│   └── test.json  


Where to find the work done: 
- `results.ipynb` contains the plots and ideas we had when analyzing our dataset. 
It follows the same order as the website and explains the website's ideas and how we got there in greater detail. 
- `./data/` contains the originial dataset, and the datasets resulting from the preprocessing steps
- `src/datapreprocessing.ipynb` contains the preprocessing steps done for the data to be used directly in `results.ipynb`
- `./tests/` contains multiple different tests done on plots processing including ngrams and named entities. 
- `./src/` contains most of the code. `utils.py` and it's derivates `utilsGenres.py`, `utilsRatings.py` and `utilsWordcloud.py ` contain the code used for their respective parts.  
Src subfolders `causal_inference`, `named_entities` and `ngrams` contain the code and results needed for these parts as they produce some files in their respective `results` folder.
