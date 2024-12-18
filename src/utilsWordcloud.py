import pandas as pd
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords

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