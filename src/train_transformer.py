import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed
)

from config import DATASET_CSV, REPORTS_DIR, BERT_DIR

SEED = 42
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256

def compute_metrics(eval_pred):
    """
    Calcula métricas de evaluación a partir de las predicciones
    del modelo durante validación.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }

def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenamiento en device: {device}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    BERT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_CSV)
    df["text"] = df["text"].astype(str)
    df["journal"] = df["journal"].astype(str)

    # Codificación de etiquetas (revistas - enteros)
    labels_sorted = sorted(df["journal"].unique())
    label2id = {lab: i for i, lab in enumerate(labels_sorted)}
    id2label = {i: lab for lab, i in label2id.items()}
    df["label_id"] = df["journal"].map(label2id).astype(int)

    X = df["text"]
    y = df["label_id"]

    # División train / test estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )

    train_df = pd.DataFrame({"text": X_train.values, "labels": y_train.values})
    test_df = pd.DataFrame({"text": X_test.values, "labels": y_test.values})

    # Carga del tokenizer del modelo Transformer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_batch(batch):
        """
        Tokenización por lotes:
        - truncado de textos largos
        - padding hasta longitud fija
        """
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )

    # Conversión a HuggingFace Datasets
    from datasets import Dataset
    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)

    train_ds = train_ds.map(tokenize_batch, batched=True)
    test_ds = test_ds.map(tokenize_batch, batched=True)

    train_ds = train_ds.remove_columns(["text", "__index_level_0__"]) if "__index_level_0__" in train_ds.column_names else train_ds.remove_columns(["text"])
    test_ds = test_ds.remove_columns(["text", "__index_level_0__"]) if "__index_level_0__" in test_ds.column_names else test_ds.remove_columns(["text"])

    train_ds.set_format("torch")
    test_ds.set_format("torch")

    # 6) Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels_sorted),
        label2id=label2id,
        id2label=id2label
    )

    # Argumentos de entrenamiento
    args = TrainingArguments(
        output_dir=str(BERT_DIR / "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to=[],
        fp16=torch.cuda.is_available(),
        logging_steps=50
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    preds = trainer.predict(test_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    y_true = test_df["labels"].values

    report = classification_report(
        y_true, y_pred,
        target_names=labels_sorted,
        digits=4
    )

    print("\n--- INFORME DE CLASIFICACIÓN (TRANSFORMER) ---")
    print(report)

    (REPORTS_DIR / "transformer_report.txt").write_text(report, encoding="utf-8")

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)

    pd.DataFrame(
        cm,
        index=labels_sorted,
        columns=labels_sorted
    ).to_csv(
        REPORTS_DIR / "transformer_confusion_matrix.csv",
        encoding="utf-8"
    )

    model.save_pretrained(BERT_DIR / "final_model")
    tokenizer.save_pretrained(BERT_DIR / "final_model")

    print(f"\nModelo y tokenizer guardados en: {BERT_DIR / 'final_model'}")
    print(f"Reporte guardado en: {REPORTS_DIR / 'transformer_report.txt'}")


if __name__ == "__main__":
    main()
