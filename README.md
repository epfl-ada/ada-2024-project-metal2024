## ADA Project for group METAL2024

In this analysis, we will investigate how pivotal historical events shaped the cinema industry focusing on the United States, examining shifts in film production, character representation, and thematic content. We will focus on how historic events and periods such as World Wars impacted the types and tones of movies produced. Additionally, we explore the rise of women's and ethnic minority representation, considering how social movements like U.S. Civil Rights and women's suffrage may have influenced not just the number of roles but also the nature of these roles. 
Moreover, events like the space race and the advent of the internet sparked an expansion in science fiction, reflecting society's imagination of future technology and outer space. The financial crises, like the Great Depression may also have influenced cinema, possibly reducing production or favoring themes of survival and resilience.


Research Questions: A list of research questions you would like to address during the project.  
- How did the genre distribution over the years change in US produced cinema ? Did it follow some specific trajectories that we could associate to historical periods ? 
- How did women representation evolve in movies, in terms of number of actresses, but also the type of roles ? Is there any special trend and can we compare the different periods ?
- Can a sentiment analyzer find trends within years of given periods ? Are the movies produced during the great depression significally darker themed, or opposedly lighter to overcome the difficult time ? 
- How is Russia, or the USSR portrayed in American movies before, during and after the cold war ? Is there a significant bias in the movie plots against Communist countries ? Are the movies less translated and published in these countries ?
- Is there a link between scifi movies explosion and the Space Race ? Do the plots mention this event, or focus on imaginary events ? 

Proposed additional datasets (if any): List the additional dataset(s) you want to use (if any), and some ideas on how you expect to get, manage, process, and enrich it/them. Show us that you’ve read the docs and some examples, and that you have a clear idea on what to expect. Discuss data size and format if relevant. It is your responsibility to check that what you propose is feasible.  

We will use the given CMU dataset, and focus only on USA produced or published movies. We took this decision as it is easier to focus on the industry and timeline of a single country and the US is the most represented country in the dataset. From characters.metadat we will drop unused columns, and try to enrich some columns such as The gender columns which we tried to infer for non-given gender values using the actor's first name and a python library.   
In movies.metadata we will drop unused columns, and filter out all movies that did not get produced/published in the United States of America. Since many information is missing from this datase, we tried enriching it using IMBD data. As the box_office revenue is a quite volatile data depending on multiple outside factors, we decided to also query movie ratings, which might give more information about the public's reception of the movies.  
For the Historic events timeline, we decided to create it manually instead of using a predifined one we found online, so that we can focus on the themes we decided, and not be too restrained. However, we cross-refenced multiple sources to create a timeline of major historical events in the USA, with multiple periods that had an important impact on the society. 

Methods  

We will use multiple different methods to get useful insight of our data 
- Various Data vizualization with methods seen in class to get appealing views of our data and show it in our final report
- Usage of a sentiment analyzer to get the general mood of a plot summary
- N-gram and named_entities to extract the most used terms in a plot, useful to extract themes of movies and link it to the period


Proposed timeline  

- 18.10 Start Data exploration and Preprocessing 
- 1.11 Start P2 Deliverable Sections
- 15.11 Finish P2 Deliverable 
- 22.11 Finish Data Preprocessing
- 29.11 Final decision of global aborded themes/periods
- 6.12 First glimpse of showable results, final decision on data story platform
- 13.12 Start of P3 Deliverable Sections
- 20.12 Final P3 Deliverable

Organization within the team: A list of internal milestones up until project Milestone P3.  

- Lilly-Flore: Theme validation and Data exploration notebook  
- Mathis: Abstract and Theme validation. Timeline events selection
- Timo: Data preprocessing and merging with IMDB sources
- Anoush: Sentiment analyzer and Ngrams,tf-idf/named_entities
- Emilien: Data preprocessing and filtering, ReadMe report

Where to find the work done: 
- `results.ipynb` contains the first glimpse at the different vizualizations done 
- `src/datapreprocessing.ipynb` contains the preprocessing steps done until now, and merging steps with IMDB data. 
- `named_entities`, `ngrams`, `ngrams_tf_idf` are in `src/`, in their respective folders. They are computed with their `.py` script, which produces an intermediate result, that is serialized. Finally the results grouped by `year`,`decade` or `period` are computed with their `_distribution.py` script. The results are stored in their `results/` folder.

