# Recomendador de revistas científicas

Este proyecto desarrolla un sistema de recomendación de revistas científicas a partir del contenido textual de artículos científicos. El objetivo es predecir en qué revista es más adecuado publicar un artículo utilizando técnicas de clasificación de texto.

El trabajo compara dos enfoques distintos:
- Un modelo clásico basado en **TF-IDF + SVM**
- Un modelo basado en **Transformers (DistilBERT)**

---

## Estructura del proyecto

```

RECOMENDADOR_REVISTAS/
│
├── data/
│   ├── raw/                      
│   │   ├── Expert Systems with Applications/
│   │   ├── Journal of Visual Communication and Image Representation/
│   │   ├── Neural Networks/
│   │   └── Robotics and Autonomous Systems/
│   │
│   └── processed/
│       └── dataset.csv           
│
├── models/
│   ├── bert/
│   │   ├── checkpoints/          # Checkpoints intermedios del entrenamiento
│   │   └── final_model/          # Modelo Transformer final entrenado
│   └── tfidf_linear_svm.joblib   # Modelo clásico TF-IDF + SVM
│
├── reports/
│   ├── classic_confusion_matrix.csv
│   ├── classic_confusion_matrix.png
│   ├── classic_f1.png
│   ├── classic_report.txt
│   ├── transformer_confusion_matrix.csv
│   ├── transformer_confusion_matrix.png
│   ├── transformer_f1.png
│   ├── transformer_report.txt
│   └── model_comparison.png
│
├── src/
│   ├── build_dataset_json.py     # Construcción del dataset a partir de JSON
│   ├── train_classic.py          # Entrenamiento del modelo clásico
│   ├── train_transformer.py      # Entrenamiento del modelo Transformer
│   └── plot_results.py           # Generación de gráficas y comparaciones
│
├── config.py                     
├── requirements.txt            
└── README.md

````

---

## Conjunto de datos

El conjunto de datos está formado por artículos científicos pertenecientes a cuatro revistas de la editorial Elsevier:

- **Expert Systems with Applications**
- **Journal of Visual Communication and Image Representation**
- **Neural Networks**
- **Robotics and Autonomous Systems**

Para cada artículo se utilizan los siguientes campos:
- Título
- Abstract
- Palabras clave

---

## Requisitos

Se recomienda usar un entorno virtual (por ejemplo con `conda` o `venv`).

Instalar las dependencias con:

```bash
pip install -r requirements.txt
````
---

## Ejecución del proyecto

### 1. Construir el dataset

Si se parte de los ficheros JSON originales:

```bash
python src/build_dataset_json.py
```

Esto generará el archivo `data/processed/dataset.csv`.

---

### 2. Entrenar el modelo clásico (TF-IDF + SVM)

```bash
python src/train_classic.py
```

Resultados generados:

* Modelo entrenado en `models/tfidf_linear_svm.joblib`
* Reportes y métricas en la carpeta `reports/`

---

### 3. Entrenar el modelo Transformer (DistilBERT)

```bash
python src/train_transformer.py
```

Resultados generados:

* Modelo final en `models/bert/final_model/`
* Métricas y matrices de confusión en `reports/`

> Nota: si hay GPU disponible, el entrenamiento utilizará aceleración automáticamente.

---

### 4. Generar gráficas y comparaciones

```bash
python src/plot_results.py
```

Se generan automáticamente:

* Matrices de confusión
* F1-score por revista
* Comparación global entre modelos

Todas las imágenes se guardan en la carpeta `reports/`.

---

## Resultados

* El modelo clásico obtiene un rendimiento sólido y sirve como buena línea base.
* El modelo Transformer mejora ligeramente las métricas globales y el equilibrio entre clases.
* El desbalanceo del conjunto de datos y el solapamiento temático entre revistas influyen notablemente en los errores de clasificación.
