import pandas as pd
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
import seaborn as sns
import numpy as np

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words.update(["two", "one", "find"])

# Function for classification
def classify_summary(summary, keyword_dict, synonyms=None, threshold=5):
    summary_lower = summary.lower()
    # Replace synonyms if provided
    if synonyms:
        for syn, replacement in synonyms.items():
            summary_lower = summary_lower.replace(syn, replacement)
    
    score = 0
    for kw, weight in keyword_dict.items():
        count = summary_lower.count(kw)
        if count > 0:
            score += weight * count
    return "Yes" if score >= threshold else "No"

# Function for text cleaning
def clean_text(text, stop_words=None):
    # Remove non-alphanumeric characters and lower text
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords if provided
    if stop_words:
        words = [w for w in text.split() if w not in stop_words]
        return ' '.join(words)
    return text

def analyze_dataset(name, file_path, keyword_dict, synonyms=None, threshold=5):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Classify summaries
    df["Classification"] = df["Summary"].apply(lambda x: classify_summary(x, keyword_dict, synonyms, threshold))
    
    # Filter summaries
    Yes_df = df[df["Classification"] == "Yes"]

    # Save only classified as Yes dataset
    yes_output_path = file_path.replace(".csv", "_classified_yes.csv")
    Yes_df.to_csv(yes_output_path, index=False)
    print(f"Filtered dataset with 'Yes' classifications saved to: {yes_output_path}")

    count = (df["Classification"] == "Yes").sum()
    print("Number of movies:", count)
    
    # Combine summaries into one text
    all_Yes_text = " ".join(Yes_df["Summary"].tolist())
    
    # Clean text
    all_YEs_text_clean = clean_text(all_Yes_text, stop_words)
    
    # Create word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_YEs_text_clean)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bicubic')
    plt.axis("off")
    plt.title(f"Word Cloud of {name} Summaries")
    plt.show()
    
    # TF-IDF transformation
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    X_tfidf = vectorizer.fit_transform(Yes_df["Summary"].apply(lambda x: clean_text(x, stop_words)))
    
    # LDA for topic modeling
    n_topics = 10
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X_tfidf)
    
    # Print topics
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, component in enumerate(lda.components_):
        top_keywords = [feature_names[i] for i in component.argsort()[:-11:-1]]
        print(f"Topic {topic_idx + 1}: {', '.join(top_keywords)}")
    
    # Save classified dataset
    output_path = file_path.replace(".csv", "_classified.csv")
    df.to_csv(output_path, index=False)
    print(f"Classified dataset saved to: {output_path}")

def analyze_time_distribution(file_path):
    # Load the filtered dataset
    df = pd.read_csv(file_path)
    
   
    # Convert Year column to numeric
    df["Year"] = pd.to_numeric(df["Movie release date"], errors='coerce')
    
    # Drop rows with invalid or missing years
    df = df.dropna(subset=["Year"])
    df["Year"] = df["Year"].astype(int)
    
    # Plot the distribution of films across time
    year_distribution = df.groupby("Year").size().reset_index(name="Count")
    
    # Plot the line plot using Seaborn
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=year_distribution, x="Year", y="Count", marker="o")
    plt.title("Distribution of Films Across Time")
    plt.xlabel("Year")
    plt.ylabel("Number of Films")
    
    # Adjust y-axis to show only integer values
    y_max = year_distribution["Count"].max() + 1
    plt.yticks(np.arange(0, y_max + 1, step=1))
    
    plt.grid(axis='both', linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()
    # Print a summary of the distribution
    print("Summary statistics for film years:")
    print(df["Year"].describe())

ww1_keywords = {
    "somme": 3,
    "verdun": 5,
    "trench": 4,
    "ww1": 5,
    "armistice": 2,
    "kaiser": 3,
    "allied": 2,
    "central": 3,
    "frontline": 4,
    "no man's land": 3,
    "mustard_gas": 5,
    "treaty of versailles": 5,
    "mobilization": 2,
    "balkans": 2,
    "zeppelin": 3,
    "shellshock": 4,
    "gallipoli": 4,
    "western front": 5,
    "lusitania": 4
}

synonyms = {
    "first world war": "world war i"
}

ww2_keywords = {
    "allies": 3, "allied": 3, "auschwitz": 5
}

W991_keywords = {
    "september 11 attacks": 5,
    "twin towers": 5,
    "world trade center": 5,
    "wtc": 4,
    "terrorist": 5,
    "al-qaeda": 5,
    "hijackers": 4,
    "boeing 767": 3,
    "destruction": 4,
    "osama bin laden": 5,
    "afghanistan": 3,
    "invasion": 2,
    "irak": 2,
    "marines": 3
}
rights_keywords = {
    "segregation": 7,
    "civil rights": 6,
    "Martin Luther King": 5,
    "protest": 4,
    "racial": 5,
    "integration": 5,
    "discrimination": 6,
    "abolition": 6,
    "boycott": 4,
    "Rosa Parks": 5,
    "sit-in": 5,
    "nonviolence": 4,
    "Jim Crow": 7,
    "oppression": 5,
    "voting rights": 6,
    "demonstration": 4,
    "march on Washington": 7,
    "Montgomery": 5,
    "Selma": 6,
    "Black Power": 5,
    "NAACP": 6,
    "activist": 4,
    "white supremacy": 7,
    "emancipation": 6,
    "equality act": 5,
    "racial harmony": 4,
    "abolitionist": 5,
    "fair housing": 4,
    "education rights": 4,
    "equal opportunity": 5,
    "freedom riders": 6,
    "lynching": 7,
    "abolition movement": 6,

}
Progress = {
    "reform": 5,
    "progressive": 4,
    "industrial": 3,
    "suffrage": 4,
    "labor": 3,
    "union": 4,
    "factory": 3,
    "child": 4,  # Related to child labor reform
    "education": 3,
    "worker": 3,
    "strike": 4,
    "social": 3,
    "temperance": 3,  # Related to anti-alcohol movements
    "prohibition": 4,
    "women vote": 4,  # Related to suffrage and labor movements
    "equality": 3,
    "tenement": 3,  # Related to urban housing reform
    "corruption": 4,  # Related to political reform
    "muckraker": 4,  # Journalists exposing corruption
    "roosevelt": 4,  # Referring to President Theodore Roosevelt
    "justice": 4,
    "equity": 3,
    "change": 3,  # General term for reform movements
    "poverty": 4,  # Highlighting poor urban conditions
}
Roaring = {
    "jazz": 8,
    "flapper": 6,
    "prohibition": 5,
    "speakeasy": 5,
    "roaring": 4,
    "gatsby": 4,
    "bootlegger": 5,
    "art deco": 3,
    "charleston": 4,
    "stock market": 5,
    "prohibitionist": 3,
    "mafia": 4,
    "automobile": 4,
    "consumerism": 3,
}
GreatDepression = {
    "depression": 8,
    "poverty": 7,
    "unemployment": 7,
    "dust bowl": 6,
    "hooverville": 5,
    "bankruptcy": 6,
    "soup kitchen": 5,
    "new deal": 7,
    "recession": 5,
    "foreclosure": 4,
    "stock market crash": 7,
    "migrant": 4,
    "relief": 5,
    "hardship": 4,
    "recovery": 4,
}
EarlyColdWar = {
    "communism": 8,
    "atomic": 7,
    "space": 6,
    "red scare": 5,
    "sputnik": 5,
    "mccarthyism": 5,
    "nuclear": 6,
    "arms race": 6,
    "iron curtain": 5,
    "truman": 4,
    "containment": 4,
    "berlin airlift": 5,
    "proxy war": 4,
    "espionnage":5
}
LateColdWar = {
    "vietnam": 8,
    "nuclear": 7,
    "detente": 5,
    "berlin": 6,
    "protest": 6,
    "afghanistan": 5,
    "cold war": 6,
    "reagan": 6,
    "glasnost": 5,
    "sandinista": 4,
    "proxy conflict": 4,
    "peace talks": 4,
    "iran contra": 4,
    "protest":2
}
PostColdWar = {
    "globalization": 8,
    "internet": 7,
    "y2k": 6,
    "iraq": 6,
    "genocide": 6,
    "clinton": 5,
    "bosnia": 5,
    "somalia": 5,
    "technology": 5,
    "dotcom": 4,
    "terrorism": 6,
    "peacekeeping": 4,
    "trade": 4,
    "multilateralism": 3,
    "economy": 3,
}