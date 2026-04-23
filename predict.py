import joblib
from src.preprocessing.preprocess import clean_text

# load saved model and vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


def predict_sentiment(text):

    cleaned = clean_text(text)

    vec = vectorizer.transform([cleaned])

    prediction = model.predict(vec)

    return "Positive" if prediction[0] == 1 else "Negative"


print(predict_sentiment("This is a great product!"))
print(predict_sentiment("I love this product!"))
print(predict_sentiment("This is the worst experience ever"))