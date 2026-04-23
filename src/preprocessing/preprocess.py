
import re

# text cleaning function
def clean_text(text):
    text=str(text).lower()  # lowercase
    text=re.sub(r'http\S+|www\S+|https\S+', '', text)  # remove URLs
    text=re.sub(r'@\w+', '', text)  # remove mentions
    text=re.sub(r'#\w+', '', text)  # remove hashtags
    text=re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text=re.sub(r'\d+', '', text)  # remove numbers
    text=re.sub(r'\s+', ' ', text).strip()  # remove extra whitespace
    return text
