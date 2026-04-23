
## Tf-idf conversion import
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


##Model Training imports
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


# load preprocessed dataset
def prepare_data(df):
   # Split data into features and labels
    final_df= df[["clean_text", "sentiment"]]
    X = final_df["clean_text"]
    y = final_df["sentiment"]
    """
     print(final_df.head(10).to_string(index=False))
     print("\nSentiment distribution:")
     print(final_df["sentiment"].value_counts())
  """
   # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
    #TF-IDF vectorization
    vectorizer= TfidfVectorizer(max_features=5000,ngram_range=(1, 2), stop_words="english")
    X_train_tfidf_vec= vectorizer.fit_transform(X_train)
    X_test_tfidf_vec = vectorizer.transform(X_test)
    return X_train_tfidf_vec, X_test_tfidf_vec, y_train, y_test, vectorizer


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC( C=0.1,
            max_iter=5000,
            dual="auto"
        )
    }



"""# Logistic_regression_model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf_vec, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf_vec)
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_lr))

#Naive_bayes_model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf_vec, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf_vec)
print("\nNaive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Naive Bayes Classification Report:\n", classification_report(y_test, y_pred_nb))

#Linear_SVM_model
svm_model = LinearSVC()
svm_model.fit(X_train_tfidf_vec, y_train)
y_pred_svm = svm_model.predict(X_test_tfidf_vec)
print("\nLinear SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Linear SVM Classification Report:\n", classification_report(y_test, y_pred_svm))

"""

