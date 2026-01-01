import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from config import DATASET_CSV, MODELS_DIR


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Cargo el dataset
    df = pd.read_csv(DATASET_CSV)
    X = df["text"]
    y = df["journal"]

    print(f"Documentos: {len(df)}")
    print(f"Clases: {y.nunique()}")

    # Dividir train / test (estratificado)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Pesos por desbalanceo de clases
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=y_train.unique(),
        y=y_train
    )
    class_weight_dict = dict(zip(y_train.unique(), class_weights))

    # TF-IDF + Linear SVM
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            max_features=50000
        )),
        ("clf", LinearSVC(
            class_weight=class_weight_dict,
            random_state=42
        ))
    ])

    # Entrenamiento
    pipe.fit(X_train, y_train)

    # Predicciones
    y_pred = pipe.predict(X_test)

    # Evaluación
    print("\n--- INFORME DE CLASIFICACIÓN ---")
    print(classification_report(y_test, y_pred))

    print("Dimensión de la matriz de confusión:", confusion_matrix(y_test, y_pred).shape)

    # Guardo el modelo
    joblib.dump(pipe, MODELS_DIR / "tfidf_linear_svm.joblib")
    print(f"\nModelo guardado en {MODELS_DIR}")


if __name__ == "__main__":
    main()
