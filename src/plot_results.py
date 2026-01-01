import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import REPORTS_DIR

def plot_confusion_from_csv(csv_path, out_path, title):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv(csv_path, index_col=0)

    short_labels = {
        "Expert Systems with Applications": "ESWA",
        "Journal of Visual Communication and Image Representation": "JVCIR",
        "Neural Networks": "NN",
        "Robotics and Autonomous Systems": "RAS"
    }

    df.index = [short_labels.get(i, i) for i in df.index]
    df.columns = [short_labels.get(c, c) for c in df.columns]

    # Normalización por fila
    df = df.div(df.sum(axis=1), axis=0)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar_kws={"shrink": 0.8}
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def extract_f1_from_report(report_path):
    """
    Extrae el F1-score por clase desde un classification_report en texto.
    """
    f1_scores = {}

    with open(report_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith((
                "Expert Systems",
                "Journal of Visual",
                "Neural Networks",
                "Robotics and Autonomous"
            )):
                parts = line.split()
                label = " ".join(parts[:-4])
                f1 = float(parts[-2])
                f1_scores[label] = f1

    return f1_scores


def plot_f1_dict(
    f1_dict,
    title,
    out_path
):
    """
    Dibuja F1-score por clase a partir de un diccionario.
    """
    labels = list(f1_dict.keys())
    values = list(f1_dict.values())

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("F1-score")
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_model_comparison(
    classic_metrics,
    transformer_metrics,
    out_path
):
    """
    Compara accuracy y macro-F1 entre modelos.
    """
    models = ["Classic", "Transformer"]
    accuracy = [
        classic_metrics["accuracy"],
        transformer_metrics["accuracy"]
    ]
    macro_f1 = [
        classic_metrics["macro_f1"],
        transformer_metrics["macro_f1"]
    ]

    x = range(len(models))
    width = 0.35

    plt.figure(figsize=(6, 5))
    plt.bar([i - width/2 for i in x], accuracy, width, label="Accuracy")
    plt.bar([i + width/2 for i in x], macro_f1, width, label="Macro F1")

    plt.xticks(x, models)
    plt.ylim(0, 1)
    plt.title("Classic vs Transformer – Overall Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # 1) Matrices de confusión
    # -------------------------------

    plot_confusion_from_csv(
        REPORTS_DIR / "classic_confusion_matrix.csv",
        REPORTS_DIR / "classic_confusion_matrix.png",
        "Confusion Matrix – Classic TF-IDF + SVM"
    )

    plot_confusion_from_csv(
        REPORTS_DIR / "transformer_confusion_matrix.csv",
        REPORTS_DIR / "transformer_confusion_matrix.png",
        "Confusion Matrix – Transformer (DistilBERT)"
    )

    # -------------------------------
    # 2) F1 por clase
    # -------------------------------

    classic_f1 = extract_f1_from_report(
        REPORTS_DIR / "classic_report.txt"
    )
    transformer_f1 = extract_f1_from_report(
        REPORTS_DIR / "transformer_report.txt"
    )

    plot_f1_dict(
        classic_f1,
        "F1-score per Journal – Classic Model",
        REPORTS_DIR / "classic_f1.png"
    )

    plot_f1_dict(
        transformer_f1,
        "F1-score per Journal – Transformer Model",
        REPORTS_DIR / "transformer_f1.png"
    )

    # -------------------------------
    # 3) Comparación global
    # -------------------------------

    classic_metrics = {
        "accuracy": 0.8213,
        "macro_f1": 0.7359
    }

    transformer_metrics = {
        "accuracy": 0.8393,
        "macro_f1": 0.7628
    }

    plot_model_comparison(
        classic_metrics,
        transformer_metrics,
        REPORTS_DIR / "model_comparison.png"
    )

    print(f"Gráficas generadas correctamente en: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
