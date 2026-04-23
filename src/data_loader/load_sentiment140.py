import pandas as pd

# Display settings (for full visibility)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)

# load sentiment140 dataset
def load_sentiment140(path="data/sentiment140.csv"):
  df = pd.read_csv('data/sentiment140.csv',encoding='latin-1',header=None)
  """# Dataset pattern exploration
     #print(number of rows, number of columns)If shape is wrong → file not loaded properly.
     print("Shape:",df.shape)
     print("\nRaw sample (before naming columns):")
     print(df.sample(3).to_string(index=False))
     print("\nFirst row (raw):")
     print(df.iloc[0]) 
  """
  # Set Column names for better readability
  df.columns = ["sentiment", "id", "date", "query", "user", "text"]
  
  # convert labels 4-positive, 0-negative to 1-positive, 0-negative for better ML compatibility
  df["sentiment"] = df["sentiment"].replace(4, 1)
  # keep only required columns
  df = df[["text", "sentiment"]]
  df["source"] = "sentiment140"
  return df


"""# Basic dataset info
print("\nColumn names:", list(df.columns))
print("\nClean dataset preview:")
print(df.head(5).to_string(index=False))

# Random samples
print("\nRandom sample (clean data):")
print(df.sample(3).to_string(index=False))

print("\nDataset info:")
print(df.info())

# Inspect single rows to understand columns
print("\nFirst row (clean):")
print(df.iloc[0])

# Column-wise preview (transpose view)
print("\nColumn-wise preview:")
print(df.head(1).T)
"""

