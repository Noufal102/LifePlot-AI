
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from realistic_data import build_realistic_data
from transformer_model import ActivityTransformer


BASE_DIR = Path(__file__).resolve().parent

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TRAIN_SEEDS = [42, 123, 456]

EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PATIENCE = 5


def train_one_seed(seed, x_train, f_train, y_train,
                   x_val, f_val, y_val):

    print("\n" + "=" * 55)
    print(f"TRAINING BALANCED TRANSFORMER - SEED {seed}")
    print("=" * 55)

    # Reproducible model initialization and data shuffling
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    checkpoint_path = (
        BASE_DIR /
        f"activity_transformer_balanced_seed_{seed}.pth"
    )

    # Calculate class weights from training data only
    class_counts = torch.bincount(
        y_train,
        minlength=8
    ).float()

    class_weights = len(y_train) / (
        8 * class_counts.clamp(min=1)
    )

    print("Training class counts:", class_counts.tolist())

    train_dataset = TensorDataset(
        x_train.long(),
        f_train.float(),
        y_train.long()
    )

    val_dataset = TensorDataset(
        x_val.long(),
        f_val.float(),
        y_val.long()
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

    model = ActivityTransformer().to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):

        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for activities, features, targets in train_loader:

            activities = activities.to(DEVICE)
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(activities, features)

            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)

            predictions = outputs.argmax(dim=1)

            train_correct += (
                predictions == targets
            ).sum().item()

            train_total += targets.size(0)

        train_loss /= train_total
        train_accuracy = train_correct / train_total

        # Validation
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for activities, features, targets in val_loader:

                activities = activities.to(DEVICE)
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)

                outputs = model(activities, features)

                loss = criterion(outputs, targets)

                val_loss += loss.item() * targets.size(0)

                predictions = outputs.argmax(dim=1)

                val_correct += (
                    predictions == targets
                ).sum().item()

                val_total += targets.size(0)

        val_loss /= val_total
        val_accuracy = val_correct / val_total

        print(
            f"Seed {seed} | "
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "class_weights": class_weights,
                    "num_activities": 8,
                    "num_features": 5,
                    "sequence_length": 12,
                    "training_seed": seed,
                },
                checkpoint_path
            )

            print("  Best checkpoint saved.")

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"\nSeed {seed} training completed.")
    print("Best epoch:", best_epoch)
    print("Best validation loss:", round(best_val_loss, 4))
    print("Checkpoint:", checkpoint_path.name)

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "checkpoint": checkpoint_path.name,
    }


def main():

    print("========== BALANCED TRANSFORMER MULTI-SEED TRAINING ==========")

    print("\nGenerating shared training data...")

    x_train, f_train, y_train = build_realistic_data(
        n=5000,
        seed=42
    )

    print("Generating shared validation data...")

    x_val, f_val, y_val = build_realistic_data(
        n=1000,
        seed=2026
    )

    print("\nDevice:", DEVICE)

    results = []

    for seed in TRAIN_SEEDS:

        result = train_one_seed(
            seed,
            x_train,
            f_train,
            y_train,
            x_val,
            f_val,
            y_val
        )

        results.append(result)

    print("\n========== TRAINING SUMMARY ==========")

    for result in results:

        print(
            f"Seed {result['seed']} | "
            f"Best Epoch: {result['best_epoch']} | "
            f"Best Val Loss: {result['best_val_loss']:.4f} | "
            f"Checkpoint: {result['checkpoint']}"
        )

    print("\nAll training seeds completed.")


if __name__ == "__main__":
    main()
