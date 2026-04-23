import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_accuracy(results):
    model_names = list(results.keys())
    accuracies = list(results.values())
    plt.figure(figsize=(10,6))
    plt.bar(model_names, accuracies)
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=45)
    plt.ylim(0,1)
    for i, v in enumerate(accuracies):
      plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    plt.tight_layout()
    plt.savefig("visuals/model_accuracy_comparison.png",bbox_inches='tight')
    plt.show()


def plot_confusion(y_true, y_pred, name):
    import os
    os.makedirs("visuals", exist_ok=True)
    plt.figure(figsize=(6,4))
    cm = confusion_matrix(y_true, y_pred)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",xticklabels=["Neg", "Pos"],
                yticklabels=["Neg", "Pos"])
    plt.title(name + " Confusion Matrix")
    plt.savefig(f"visuals/{name.lower()}_confusion_matrix.png")
    plt.show()
    
def plot_lstm_history(history):

    # Accuracy plot
    plt.figure(figsize=(10,6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title("LSTM Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("visuals/lstm_accuracy.png",bbox_inches='tight')
    plt.show()
    
    # Loss plot
    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("LSTM Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("visuals/lstm_loss.png",bbox_inches='tight')
    plt.show()
    
def plot_all_confusion(matrices):
    import matplotlib.pyplot as plt
    import seaborn as sns

    n = len(matrices)
    plt.figure(figsize=(12,8))

    for i, (name, (y_true, y_pred)) in enumerate(matrices.items()):
        plt.subplot(2, 3, i+1)

        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

        plt.title(name)

    plt.tight_layout()
    plt.savefig("visuals/all_confusion_matrices.png", bbox_inches='tight')
    plt.show()