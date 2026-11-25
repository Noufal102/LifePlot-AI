
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from transformer_model import ActivityTransformer
from realistic_data import build_realistic_data
from data_utils import ACTIVITIES


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "activity_transformer_realistic.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def main():

    print("========== REALISTIC MODEL EVALUATION ==========")

    if not MODEL_PATH.exists():
        print("ERROR: Realistic Transformer checkpoint not found.")
        return

    # Generate independent test dataset
    print("\nGenerating independent test dataset...")

    x_test, f_test, y_test = build_realistic_data(
        n=1000,
        seed=98765
    )

    print("Test samples:", len(y_test))

    # Load checkpoint
    print("\nLoading trained Transformer checkpoint...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model = ActivityTransformer(
        num_activities=checkpoint["num_activities"],
        num_features=checkpoint["num_features"],
        d_model=checkpoint["d_model"],
        nhead=checkpoint["nhead"],
        num_layers=checkpoint["num_layers"],
        max_len=checkpoint["sequence_length"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print("Checkpoint loaded successfully.")
    print("Best training epoch:", checkpoint["best_epoch"])
    print("Best validation loss:", checkpoint["best_val_loss"])

    # Predict
    print("\nRunning predictions...")

    with torch.no_grad():

        outputs = model(
            x_test.to(DEVICE),
            f_test.to(DEVICE)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        ).cpu().numpy()

    actual = y_test.numpy()

    # Calculate metrics
    accuracy = accuracy_score(actual, predictions)

    precision = precision_score(
        actual,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        actual,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        actual,
        predictions,
        average="macro",
        zero_division=0
    )

    print("\n========== TEST METRICS ==========")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")

    print("\n========== CLASSIFICATION REPORT ==========")

    print(
        classification_report(
            actual,
            predictions,
            labels=list(range(len(ACTIVITIES))),
            target_names=ACTIVITIES,
            zero_division=0
        )
    )

    print("\n========== CONFUSION MATRIX ==========")

    print(
        confusion_matrix(
            actual,
            predictions,
            labels=list(range(len(ACTIVITIES)))
        )
    )

    print("\n========== EVALUATION COMPLETE ==========")


if __name__ == "__main__":
    main()
