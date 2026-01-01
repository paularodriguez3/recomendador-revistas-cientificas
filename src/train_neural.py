import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter

from config import DATASET_CSV

MAX_VOCAB_SIZE = 30000
MAX_SEQ_LEN = 300
BATCH_SIZE = 32
RANDOM_SEED = 42

def tokenize(text):
    return text.lower().split()

def build_vocab(texts, max_size):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for i, (word, _) in enumerate(counter.most_common(max_size - 2), start=2):
        vocab[word] = i

    return vocab

# Codificación de texto a IDs
def encode(text, vocab):
    tokens = tokenize(text)
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
    return ids[:MAX_SEQ_LEN]

# Dataset PyTorch
class JournalDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = encode(self.texts.iloc[idx], self.vocab)
        label = self.labels.iloc[idx]
        return torch.tensor(ids), torch.tensor(label)

# Padding dinámico
def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = [len(t) for t in texts]
    max_len = max(lengths)

    padded = torch.zeros(len(texts), max_len, dtype=torch.long)
    for i, t in enumerate(texts):
        padded[i, :len(t)] = t

    return padded, torch.tensor(labels)

def main():
    # Cargar dataset
    df = pd.read_csv(DATASET_CSV)

    # Mapeo de etiquetas a IDs
    label_to_id = {label: i for i, label in enumerate(sorted(df["journal"].unique()))}
    df["label_id"] = df["journal"].map(label_to_id)

    X = df["text"]
    y = df["label_id"]

    # Dividir en train y test estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y
    )

    # Construir vocabulario solo con train
    vocab = build_vocab(X_train, MAX_VOCAB_SIZE)

    print(f"Tamaño del vocabulario: {len(vocab)}")
    print(f"Número de clases: {len(label_to_id)}")

    # Datasets
    train_dataset = JournalDataset(X_train, y_train, vocab)
    test_dataset = JournalDataset(X_test, y_test, vocab)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    print("DataLoaders listos para usar.")


if __name__ == "__main__":
    main()
