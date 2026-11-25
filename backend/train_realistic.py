
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from transformer_model import ActivityTransformer
from realistic_data import build_realistic_data


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "activity_transformer_realistic.pth"

SEED = 42
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
PATIENCE = 5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def evaluate(model, loader, loss_function):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, fb, yb in loader:
            xb = xb.to(DEVICE)
            fb = fb.to(DEVICE)
            yb = yb.to(DEVICE)

            outputs = model(xb, fb)
            loss = loss_function(outputs, yb)

            total_loss += loss.item() * yb.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == yb).sum().item()
            total += yb.size(0)

    return total_loss / total, correct / total


def main():
    print("Device:", DEVICE)

    print("\nGenerating training dataset...")
    x_train, f_train, y_train = build_realistic_data(
        n=5000,
        seed=42
    )

    print("Generating validation dataset...")
    x_val, f_val, y_val = build_realistic_data(
        n=1000,
        seed=2026
    )

    train_dataset = TensorDataset(
        x_train,
        f_train,
        y_train
    )

    val_dataset = TensorDataset(
        x_val,
        f_val,
        y_val
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("\nCreating Transformer model...")
    model = ActivityTransformer().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    loss_function = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    print("\nStarting Transformer training...\n")

    for epoch in range(EPOCHS):
        model.train()

        total_train_loss = 0.0
        correct = 0
        total = 0

        for xb, fb, yb in train_loader:
            xb = xb.to(DEVICE)
            fb = fb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(xb, fb)
            loss = loss_function(outputs, yb)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * yb.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == yb).sum().item()
            total += yb.size(0)

        train_loss = total_train_loss / total
        train_accuracy = correct / total

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            loss_function
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "num_activities": 8,
                    "num_features": 5,
                    "sequence_length": 12,
                    "d_model": 64,
                    "nhead": 4,
                    "num_layers": 2,
                    "dropout": 0.1,
                },
                MODEL_PATH
            )

            print("  Best model checkpoint saved.")

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    print("\n========== TRAINING COMPLETE ==========")
    print("Best epoch:", best_epoch)
    print("Best validation loss:", round(best_val_loss, 4))
    print("Checkpoint:", MODEL_PATH)


if __name__ == "__main__":
    main()
