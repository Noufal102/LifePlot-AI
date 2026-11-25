
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from realistic_data import build_realistic_data
from transformer_model import ActivityTransformer


BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "activity_transformer_balanced.pth"

# Reproducibility
torch.manual_seed(1234)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PATIENCE = 5


def main():
    print("Generating balanced training data...")

    x_train, f_train, y_train = build_realistic_data(
        n=5000,
        seed=42
    )

    x_val, f_val, y_val = build_realistic_data(
        n=1000,
        seed=2026
    )

    print("Training samples:", len(y_train))
    print("Validation samples:", len(y_val))

    # Calculate inverse-frequency class weights
    class_counts = torch.bincount(y_train, minlength=8).float()

    class_weights = len(y_train) / (
        8 * class_counts.clamp(min=1)
    )

    print("\nClass counts:")
    for i, count in enumerate(class_counts):
        print(f"Class {i}: {int(count)}")

    print("\nClass weights:")
    print(class_weights)

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

    print("\nDevice:", DEVICE)
    print("\nStarting class-balanced Transformer training...\n")

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
                },
                CHECKPOINT_PATH
            )

            print("  Best balanced checkpoint saved.")

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    print("\n========== TRAINING COMPLETE ==========")
    print("Best Epoch:", best_epoch)
    print("Best Validation Loss:", round(best_val_loss, 4))
    print("Checkpoint:", CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
