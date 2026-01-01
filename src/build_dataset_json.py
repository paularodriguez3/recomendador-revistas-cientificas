import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import json
import pandas as pd
from tqdm import tqdm

from config import RAW_DATA_DIR, DATASET_CSV

def normalize_text(x):
    """
    Convierte el contenido a texto limpio:
    - None: ""
    - lista (keywords): string separado por ;
    - eliminar saltos de línea
    """
    if x is None:
        return ""
    if isinstance(x, list):
        return " ; ".join([str(i) for i in x])
    return str(x).replace("\n", " ").replace("\r", " ").strip()


def main():
    rows = []

    # Cada subcarpeta de RAW_DATA_DIR es una revista
    journal_dirs = [d for d in RAW_DATA_DIR.iterdir() if d.is_dir()]
    if not journal_dirs:
        raise RuntimeError("No se encontraron carpetas de revistas en data/raw")

    for journal_dir in tqdm(journal_dirs, desc="Procesando revistas"):
        journal_name = journal_dir.name.strip()

        json_files = list(journal_dir.glob("*.json"))
        if not json_files:
            continue

        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                articles = json.load(f)

            for art in articles:
                title = normalize_text(art.get("title"))
                abstract = normalize_text(art.get("abstract"))
                keywords = normalize_text(art.get("keywords"))
                year = art.get("year")

                # Texto que usará el modelo
                text = " ".join([title, abstract, keywords]).strip()

                rows.append({
                    "journal": journal_name,
                    "year": year,
                    "title": title,
                    "abstract": abstract,
                    "keywords": keywords,
                    "text": text
                })

    df = pd.DataFrame(rows)

    # Elimino  artículos sin texto
    df = df[df["text"].str.len() > 0]

    # Elimino duplicados 
    df.drop_duplicates(subset=["journal", "title"], inplace=True)

    # Se crea la carpeta processed si no existe
    DATASET_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Resumen para control y memoria
    print("\n--- RESUMEN DEL DATASET ---")
    print(f"Artículos totales: {len(df)}")
    print(f"Revistas distintas: {df['journal'].nunique()}")
    print("\nArtículos por revista:")
    print(df["journal"].value_counts())

    df.to_csv(DATASET_CSV, index=False, encoding="utf-8")
    print(f"\nDataset guardado en: {DATASET_CSV.resolve()}")


if __name__ == "__main__":
    main()
