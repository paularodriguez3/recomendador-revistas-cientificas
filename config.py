from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Datos
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dataset final
DATASET_CSV = PROCESSED_DATA_DIR / "dataset.csv"

# Modelos
MODELS_DIR = BASE_DIR / "models"
