
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

from transformer_model import ActivityTransformer
from data_utils import ACTIVITIES
from train import build_demo_data


# Checkpoint location
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "activity_transformer.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    if not MODEL_PATH.exists():
        print("ERROR: Transformer checkpoint not found!")
        return

    print("Preparing synthetic holdout dataset...")

    # Reproduce the original 5,000 training samples
    # to advance the seeded generator past the training data.
    build_demo_data(n=5000)

    # Generate a separate 1,000-sample holdout set.
    x_test, f_test, y_test = build_demo_data(n=1000)

    print("Loading Transformer checkpoint...")

    model = ActivityTransformer()
    state = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    print("Running evaluation...")

    with torch.no_grad():
        logits = model(
            x_test.to(DEVICE),
            f_test.to(DEVICE)
        )
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

    actual = y_test.numpy()

    accuracy = accuracy_score(actual, predictions)

    precision, recall, macro_f1, _ = (
        precision_recall_fscore_support(
            actual,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    print("\n========== LIFEPLOT AI EVALUATION ==========")
    print(f"Evaluation samples : {len(actual)}")
    print(f"Accuracy           : {accuracy * 100:.2f}%")
    print(f"Macro Precision    : {precision:.4f}")
    print(f"Macro Recall       : {recall:.4f}")
    print(f"Macro F1-score     : {macro_f1:.4f}")

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            actual,
            predictions,
            labels=list(range(len(ACTIVITIES))),
            target_names=ACTIVITIES,
            zero_division=0
        )
    )

    print("CONFUSION MATRIX")
    print(confusion_matrix(
        actual,
        predictions,
        labels=list(range(len(ACTIVITIES)))
    ))

    print("\nEvaluation completed.")
    print("NOTE: These results measure performance on synthetic demo data.")


if __name__ == "__main__":
    main()
