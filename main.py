import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import string
import spacy
from pytube import YouTube

# Load a spaCy language model
nlp = spacy.load("en_core_web_sm")
data = pd.read_csv("youtube.csv")
data = data[["title", "category"]]

# Preprocess text function
def preprocess_text(text):
    text = text.lower()
    doc = nlp(text)
    words = [token.text for token in doc if token.text not in string.punctuation]
    lemmatized_words = [token.lemma_ for token in doc]
    preprocessed_text = ' '.join(lemmatized_words)
    return preprocessed_text

data['title'] = data['title'].apply(preprocess_text)

# Split the data, vectorize, and train the model
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(data['title'], data['category'], test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_tfidf, y_train)

# Streamlit UI
st.title("YouTube Video Category Classifier")

# User input for YouTube URL
video_url = st.text_input("Enter a YouTube video URL:")

if video_url:
    try:
        # Download the video's metadata using PyTube
        st.write("Getting video information...")
        yt = YouTube(video_url)

        # Extract video title, short description, and thumbnail URL
        video_title = yt.title
        video_description = yt.description
        thumbnail_url = yt.thumbnail_url

        # Preprocess video title
        preprocessed_text = preprocess_text(video_title)

        # Vectorize the preprocessed text
        X_test_tfidf = tfidf_vectorizer.transform([preprocessed_text])

        # Predict the video category
        predicted_category = naive_bayes.predict(X_test_tfidf)[0]

        # Display results
        st.subheader("Video Title")
        st.write(f"{video_title}")
        st.subheader("Predicted Category")
        st.write(predicted_category.upper())
        # Display the video thumbnail
        st.image(thumbnail_url, caption='Video Thumbnail', use_column_width=True)




    except Exception as e:
        st.error(f"An error occurred {e}. Please check the provided YouTube URL and try again.")
        st.write(e)
