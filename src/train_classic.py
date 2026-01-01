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

from config import DATASET_CSV, MODELS_DIR, REPORTS_DIR


def main():
    # Crear carpetas si no existen
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Cargar dataset
    df = pd.read_csv(DATASET_CSV)
    X = df["text"]
    y = df["journal"]

    print(f"Documentos: {len(df)}")
    print(f"Clases: {y.nunique()}")

    # 2) División train / test estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 3) Pesos para compensar desbalance de clases
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=y_train.unique(),
        y=y_train
    )
    class_weight_dict = dict(zip(y_train.unique(), class_weights))

    # 4) Pipeline TF-IDF + Linear SVM
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

    # 5) Entrenamiento
    pipe.fit(X_train, y_train)

    # 6) Predicciones
    y_pred = pipe.predict(X_test)

    # 7) Evaluación
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- INFORME DE CLASIFICACIÓN (CLÁSICO) ---")
    print(report)
    print("Dimensión de la matriz de confusión:", cm.shape)

    # 8) Guardar reportes
    (REPORTS_DIR / "classic_report.txt").write_text(report, encoding="utf-8")
    pd.DataFrame(cm).to_csv(
        REPORTS_DIR / "classic_confusion_matrix.csv",
        index=False
    )

    # 9) Guardar modelo
    joblib.dump(pipe, MODELS_DIR / "tfidf_linear_svm.joblib")
    print(f"\nModelo guardado en {MODELS_DIR}")
    print(f"Reportes guardados en {REPORTS_DIR}")


if __name__ == "__main__":
    main()
