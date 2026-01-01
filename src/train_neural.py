import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import torch
import torch.nn as nn
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

# Modelo BiLSTM
class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        padding_idx=0,
        dropout=0.3
    ):
        super().__init__()

        # Capa de embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Clasificador final
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        """
        x: [batch_size, seq_len]
        """
        embedded = self.embedding(x)
        # embedded: [batch_size, seq_len, embedding_dim]

        lstm_out, _ = self.lstm(embedded)
        # lstm_out: [batch_size, seq_len, hidden_dim*2]

        # Mean pooling sobre la dimensión temporal
        pooled = torch.mean(lstm_out, dim=1)
        # pooled: [batch_size, hidden_dim*2]

        pooled = self.dropout(pooled)

        logits = self.fc(pooled)
        # logits: [batch_size, output_dim]

        return logits

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

    # Probar modelo
    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=100,
        hidden_dim=128,
        output_dim=len(label_to_id)
    )

    sample_batch = next(iter(train_loader))[0]
    output = model(sample_batch)

    print("Salida del modelo:", output.shape)


if __name__ == "__main__":
    main()
