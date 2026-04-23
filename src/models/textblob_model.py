from textblob import TextBlob
import pandas as pd

def get_sentiment(text):
    """
    Returns:
    1 -> Positive
    0 -> Negative
    """

    analysis = TextBlob(str(text))
    polarity = analysis.sentiment.polarity

    if polarity > 0:
        return 1
    else:
        return 0


def apply_textblob(df):
    """
    df must contain column: 'clean_text'
    """

    df["textblob_pred"] = df["clean_text"].apply(get_sentiment)

    return df