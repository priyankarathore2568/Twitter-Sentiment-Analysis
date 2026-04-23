import pandas as pd

# load Apple_crowdflower dataset
"""df = pd.read_csv('data/Apple_crowdflower.csv',encoding='latin-1')
print("Shape:",df.shape)
print("\nRaw sample (before naming columns):")
print(df.sample(3).to_string(index=False))
print("\nFirst row (raw):")
print(df.iloc[0])
"""

def load_apple(path="data/Apple_crowdflower.csv"):
    df = pd.read_csv(path, encoding="latin1")

    # keep only important columns
    df = df[["text", "sentiment"]]
    df = df.dropna()
    # map sentiment if needed
    mapping = {1: 0, 3: 1, 5: 2}  # negative, neutral, positive (adjust if needed)
    df["sentiment"] = df["sentiment"].map(mapping)

    df["source"] = "apple"
    
    return df