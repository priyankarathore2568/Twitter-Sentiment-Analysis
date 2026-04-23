import pandas as pd
import joblib

from sklearn.model_selection import cross_val_score
from src.data_loader.combine_data import combine_all
from src.preprocessing.preprocess import clean_text
from src.models.train_ml import prepare_data, get_models
from src.visualization.plots import plot_accuracy, plot_all_confusion, plot_lstm_history,plot_confusion

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from src.models.textblob_model import apply_textblob

from src.models.train_deep_model import train_lstm


# Load data
df = combine_all().sample(90000, random_state=42)

# apply cleaning /Clean text
df["clean_text"] = df["text"].apply(clean_text)
# Remove missing values
df = df.dropna(subset=["sentiment"])
 
## TextBlob predictions (for comparison)
text_blob_df = apply_textblob(df.sample(5000))
textblob_acc = accuracy_score(text_blob_df["sentiment"], text_blob_df["textblob_pred"])
print("TextBlob Accuracy:", textblob_acc)

# Prepare ML data
print("Starting ML training...")
X_train_tfidf_vec, X_test_tfidf_vec, y_train, y_test, vectorizer = prepare_data(df)

#  Models
models = get_models()
best_model = None
best_acc = 0
best_model_name = ""
results = {}
all_predictions = {}

# Train + evaluate
for name, model in models.items():

    model.fit(X_train_tfidf_vec, y_train)

    y_pred_ml = model.predict(X_test_tfidf_vec)
    
    acc = accuracy_score(y_test, y_pred_ml)
    
    print(f"\n{name} Classification Report:\n")
    print(classification_report(y_test, y_pred_ml))
    
    # store predictions correctly
    results[name] = acc
    all_predictions[name] = (y_test, y_pred_ml)
    plot_confusion(y_test, y_pred_ml, name)
    cv_scores = cross_val_score(model, X_train_tfidf_vec, y_train, cv=5)
    
    print(f"{name}: {acc:.4f}")
    print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
     # store best model
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_model_name = name
        
print("\n🏆 BEST MODEL:")
print(f"{best_model_name} with Accuracy: {best_acc:.4f}")

#LSTM model predictions
print("Training LSTM...")
lstm_model, history, y_true_lstm, y_pred_lstm = train_lstm(df)
lstm_acc = accuracy_score(y_true_lstm, y_pred_lstm)
results["LSTM"] = lstm_acc
print("LSTM Accuracy:", lstm_acc)
print("\nLSTM Classification Report:\n")
print(classification_report(y_true_lstm, y_pred_lstm))
all_predictions["LSTM"] = (y_true_lstm, y_pred_lstm)

#Text_blob accuracy
results["TextBlob"] = textblob_acc
all_predictions["TextBlob"] = (
    text_blob_df["sentiment"],
    text_blob_df["textblob_pred"]
)

# SAVE MODEL + VECTORIZER (IMPORTANT STEP)
joblib.dump(best_model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\n Model saved successfully")

# Visualization
plot_accuracy(results)

plot_lstm_history(history)

plot_confusion(
    text_blob_df["sentiment"],
    text_blob_df["textblob_pred"],
    "TextBlob"
)
plot_confusion(y_true_lstm, y_pred_lstm, "LSTM")

plot_all_confusion(all_predictions)
