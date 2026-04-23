

import tensorflow as tf
Tokenizer = tf.keras.preprocessing.text.Tokenizer
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
from sklearn.model_selection import train_test_split
def train_lstm(df):

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(df["clean_text"])

    X_seq = tokenizer.texts_to_sequences(df["clean_text"])
    X_pad = pad_sequences(X_seq, maxlen=50)

    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_pad, y, test_size=0.2, random_state=42
    )

    model = build_model(vocab_size=5000)
    history = model.fit(X_train, y_train, epochs=3,batch_size=32, validation_split=0.2)

    y_pred = (model.predict(X_test) > 0.5).astype(int)
   
    return model, history, y_test, y_pred

def build_model(vocab_size=5000):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 128),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


