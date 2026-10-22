"""Training script for the IMDB sentiment analysis project.

The script ties together the utilities from ``src`` and implements all of the
requested improvements:

* **E** – uses ``nn.BCEWithLogitsLoss`` (no manual sigmoid).
* **F** – logs per‑epoch average loss and computes accuracy, precision, recall
  and F1 via ``src.utils.compute_metrics``.
* **G** – early stopping based on validation loss (``src.utils.EarlyStopping``).
* **H** – automatically selects the best torch device (CUDA > MPS > CPU).
* **I** – resolves the "non‑writable NumPy array" warning by copying the arrays
  before converting them to tensors.

Run with ``python -m src.train`` or ``python src/train.py`` from the project
root.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.model import RNN
from src.preprocess import load_dataset, preprocess_dataframe
from src.utils import EarlyStopping, compute_metrics, get_device


def prepare_data(
    csv_path: pathlib.Path,
) -> Tuple[
    torch.utils.data.TensorDataset,
    torch.utils.data.TensorDataset,
    TfidfVectorizer,
]:
    """Load, clean, vectorise and split the data into PyTorch ``TensorDataset`` objects.

    Returns ``(train_set, test_set, tf)``.
    """
    # 1️⃣ Load and clean
    df = load_dataset(str(csv_path))
    df = preprocess_dataframe(df)

    # 2️⃣ Encode labels (already ints from preprocessing step)
    y = df["sentiment"].values

    # 3️⃣ Train‑test split (raw text split to avoid leakage)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["review"], y, test_size=0.2, random_state=42
    )

    # 4️⃣ TF‑IDF (fit on train, transform test)
    tf = TfidfVectorizer(max_features=5000)
    X_train_sparse = tf.fit_transform(X_train_raw)
    X_test_sparse = tf.transform(X_test_raw)

    # 5️⃣ Convert to dense NumPy arrays and *copy* to avoid the non‑writable warning
    X_train = X_train_sparse.toarray().copy()
    X_test = X_test_sparse.toarray().copy()

    # 6️⃣ Torch tensors (float32).  Add a singleton dimension for the RNN seq_len=1.
    X_train_tensor = torch.from_numpy(X_train).float().unsqueeze(1)
    X_test_tensor = torch.from_numpy(X_test).float().unsqueeze(1)
    y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(1)
    y_test_tensor = torch.from_numpy(y_test).float().unsqueeze(1)

    train_set = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    test_set = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    return train_set, test_set, tf


def train(
    train_set: torch.utils.data.TensorDataset,
    val_set: torch.utils.data.TensorDataset,
    input_size: int,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 64,
    patience: int = 5,
) -> RNN:
    """Train the RNN model with early stopping.

    Returns the best model (stored on ``device``).
    """
    train_loader = torch.utils.data.DataLoader(
        train_set, shuffle=True, batch_size=batch_size
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, shuffle=False, batch_size=batch_size
    )

    model = RNN(input_size)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    early_stopper = EarlyStopping(patience=patience, mode="min")

    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb).squeeze(1)
            loss = criterion(outputs, yb.squeeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        avg_train_loss = running_loss / len(train_set)

        # Validation
        model.eval()
        val_losses = []
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb).squeeze(1)
                loss = criterion(outputs, yb.squeeze(1))
                val_losses.append(loss.item() * xb.size(0))
                preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
                all_preds.append(preds)
                all_labels.append(yb.squeeze(1).cpu().numpy())
        avg_val_loss = sum(val_losses) / len(val_set)
        metrics = compute_metrics(np.concatenate(all_labels), np.concatenate(all_preds))

        print(
            f"Epoch {epoch}/{epochs} | train loss: {avg_train_loss:.4f} | "
            f"val loss: {avg_val_loss:.4f} | "
            f"acc: {metrics['accuracy']*100:.2f}%"
        )

        # Early stopping check (monitor validation loss)
        if early_stopper(avg_val_loss):
            print("Early stopping triggered.")
            break
        # Keep best model (lowest val loss)
        if best_state is None or avg_val_loss < best_state[0]:
            best_state = (avg_val_loss, model.state_dict())

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state[1])
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IMDB sentiment RNN")
    parser.add_argument(
        "--data",
        type=pathlib.Path,
        default=pathlib.Path("IMDB_dataset.csv"),
        help="Path to the CSV dataset",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("models"),
        help="Directory to save the trained model and vectorizer",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    train_set, val_set, vectorizer = prepare_data(args.data)
    input_dim = train_set.tensors[0].size(2)  # (batch, seq_len=1, features)
    model = train(
        train_set,
        val_set,
        input_dim,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )

    # Save model and vectorizer for inference
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "rnn_model.pt"
    vectorizer_path = args.output_dir / "tfidf_vectorizer.pkl"
    torch.save(model.state_dict(), model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model and vectorizer saved: {model_path}, {vectorizer_path}")


if __name__ == "__main__":
    main()
