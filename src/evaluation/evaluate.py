from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(name, model, X_test_tfidf_vec, y_test):

    y_pred = model.predict(X_test_tfidf_vec)
    acc = accuracy_score(y_test, y_pred)

    print(name, ":", acc)
    print("\n====", name, "====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return y_pred

