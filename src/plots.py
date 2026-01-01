import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels,
    out_path,
    title="Confusion Matrix",
    normalize=True
):
    """
    Dibuja y guarda una matriz de confusión (normalizada o no)
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_f1_per_class(
    y_true,
    y_pred,
    labels,
    out_path,
    title="F1-score per class"
):
    """
    Dibuja F1-score por clase
    """
    f1s = f1_score(y_true, y_pred, labels=labels, average=None)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=labels, y=f1s)
    plt.ylim(0, 1)
    plt.ylabel("F1-score")
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_model_comparison(
    metrics_dict,
    out_path,
    title="Model comparison"
):
    """
    metrics_dict = {
        "Classic": {"accuracy": 0.82, "macro_f1": 0.74},
        "Transformer": {"accuracy": 0.84, "macro_f1": 0.76}
    }
    """
    models = list(metrics_dict.keys())
    acc = [metrics_dict[m]["accuracy"] for m in models]
    f1 = [metrics_dict[m]["macro_f1"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(6, 5))
    plt.bar(x - width/2, acc, width, label="Accuracy")
    plt.bar(x + width/2, f1, width, label="Macro F1")

    plt.xticks(x, models)
    plt.ylim(0, 1)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
