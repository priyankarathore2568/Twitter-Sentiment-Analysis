import streamlit as st
import joblib
from src.preprocessing.preprocess import clean_text

model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

st.title("X| Twitter Sentiment Analyzer")

text = st.text_area("Enter Tweet")

if st.button("Analyze"):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]

    if pred == 1:
        st.success("Positive 😊")
    else:
        st.error("Negative 😠")