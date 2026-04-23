import pandas as pd
from .load_sentiment140 import load_sentiment140
from .load_apple_crowdflower import load_apple
#from .load_tweets import load_tweets

def combine_all():

    df1 = load_sentiment140()
    df2 = load_apple()
    #df3 = load_tweets()

    # unify format
    df1 = df1[["text","sentiment"]]
    df2 = df2[["text","sentiment"]]

    # tweets.csv has no label → optional inference or drop
    #df3 = df3[["text"]]
    #df3["sentiment"] = None
    #df3["sentiment"] = 0

    final_df = pd.concat([df1, df2], ignore_index=True)

    return final_df
