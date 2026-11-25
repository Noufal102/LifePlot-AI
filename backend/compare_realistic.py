
from pathlib import Path

import numpy as np
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from transformer_model import ActivityTransformer
from realistic_data import build_realistic_data


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "activity_transformer_realistic.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def to_rf_features(x, f, classes=8):
    one_hot = np.eye(classes)[x.numpy()]
    one_hot = one_hot.reshape(len(x), -1)

    numeric = f.numpy().reshape(len(f), -1)

    return np.concatenate([one_hot, numeric], axis=1)


def print_metrics(name, actual, predictions):
    print(f"\n{name}")

    print(f"Accuracy : {accuracy_score(actual, predictions):.4f}")
    print(
        f"Precision: {precision_score(actual, predictions, average='macro', zero_division=0):.4f}"
    )
    print(
        f"Recall   : {recall_score(actual, predictions, average='macro', zero_division=0):.4f}"
    )
    print(
        f"Macro F1 : {f1_score(actual, predictions, average='macro', zero_division=0):.4f}"
    )


def main():

    print("========== REALISTIC MODEL COMPARISON ==========")

    if not MODEL_PATH.exists():
        print("ERROR: Transformer checkpoint not found.")
        return

    # Same data seeds used during Transformer training/evaluation
    print("\nGenerating shared training dataset...")
    x_train, f_train, y_train = build_realistic_data(
        n=5000,
        seed=42
    )

    print("Generating shared test dataset...")
    x_test, f_test, y_test = build_realistic_data(
        n=1000,
        seed=98765
    )

    actual = y_test.numpy()

    # Load Transformer
    print("\nLoading Transformer...")
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    transformer = ActivityTransformer(
        num_activities=checkpoint["num_activities"],
        num_features=checkpoint["num_features"],
        d_model=checkpoint["d_model"],
        nhead=checkpoint["nhead"],
        num_layers=checkpoint["num_layers"],
        max_len=checkpoint["sequence_length"],
    )

    transformer.load_state_dict(checkpoint["model_state_dict"])
    transformer.to(DEVICE)
    transformer.eval()

    with torch.no_grad():
        transformer_predictions = transformer(
            x_test.to(DEVICE),
            f_test.to(DEVICE)
        ).argmax(dim=1).cpu().numpy()

    # Train Random Forest on exactly the same training data
    print("\nPreparing Random Forest features...")

    x_train_rf = to_rf_features(x_train, f_train)
    x_test_rf = to_rf_features(x_test, f_test)

    print("Training Random Forest...")

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(x_train_rf, y_train.numpy())

    rf_predictions = rf.predict(x_test_rf)

    # Majority-class baseline
    majority_class = np.bincount(
        y_train.numpy(),
        minlength=8
    ).argmax()

    majority_predictions = np.full(
        len(actual),
        majority_class
    )

    # Results
    print("\n========== COMPARISON RESULTS ==========")

    print_metrics(
        "Transformer",
        actual,
        transformer_predictions
    )

    print_metrics(
        "Random Forest",
        actual,
        rf_predictions
    )

    print_metrics(
        "Majority Class Baseline",
        actual,
        majority_predictions
    )

    agreement = np.mean(
        transformer_predictions == rf_predictions
    )

    print("\nPrediction agreement:", round(agreement, 4))

    print("\n========== COMPARISON COMPLETE ==========")


if __name__ == "__main__":
    main()
