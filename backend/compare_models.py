
from pathlib import Path
import random

import numpy as np
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from transformer_model import ActivityTransformer
from data_utils import ACTIVITIES
from train import build_demo_data


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "activity_transformer.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def convert_to_rf_features(activity_ids, numeric_features):
    ids = activity_ids.numpy()
    features = numeric_features.numpy()

    activity_one_hot = np.eye(len(ACTIVITIES))[ids]
    activity_one_hot = activity_one_hot.reshape(len(ids), -1)

    numeric_flat = features.reshape(len(features), -1)

    return np.concatenate(
        [activity_one_hot, numeric_flat],
        axis=1
    )


def calculate_metrics(actual, predicted):
    precision, recall, macro_f1, _ = (
        precision_recall_fscore_support(
            actual,
            predicted,
            average="macro",
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision,
        "recall": recall,
        "macro_f1": macro_f1
    }


def print_distribution(title, values):
    print(f"\n{title}")

    for i, name in enumerate(ACTIVITIES):
        count = int(np.sum(values == i))
        percentage = count / len(values) * 100

        print(
            f"{name:<18} {count:>4} "
            f"({percentage:>6.2f}%)"
        )


def main():

    if not MODEL_PATH.exists():
        print("ERROR: Transformer checkpoint not found.")
        return

    print("Generating shared synthetic dataset...")

    x_train, f_train, y_train = build_demo_data(n=5000)
    x_test, f_test, y_test = build_demo_data(n=1000)

    actual = y_test.numpy()

    # TRANSFORMER
    print("\nLoading Transformer checkpoint...")

    transformer = ActivityTransformer()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    transformer.load_state_dict(checkpoint)
    transformer.to(DEVICE)
    transformer.eval()

    print("Evaluating Transformer...")

    with torch.no_grad():
        logits = transformer(
            x_test.to(DEVICE),
            f_test.to(DEVICE)
        )

        transformer_predictions = (
            torch.argmax(logits, dim=1).cpu().numpy()
        )

    # RANDOM FOREST
    print("\nPreparing Random Forest features...")

    x_train_rf = convert_to_rf_features(x_train, f_train)
    x_test_rf = convert_to_rf_features(x_test, f_test)

    print("Training Random Forest...")

    random_forest = RandomForestClassifier(
        n_estimators=300,
        random_state=SEED,
        n_jobs=-1
    )

    random_forest.fit(x_train_rf, y_train.numpy())

    print("Evaluating Random Forest...")

    rf_predictions = random_forest.predict(x_test_rf)

    # METRICS
    transformer_metrics = calculate_metrics(
        actual, transformer_predictions
    )

    rf_metrics = calculate_metrics(
        actual, rf_predictions
    )

    print("\n========== MODEL COMPARISON ==========")

    print(
        f"{'Metric':<20}"
        f"{'Transformer':<18}"
        f"{'Random Forest':<18}"
    )

    for metric in ["accuracy", "precision", "recall", "macro_f1"]:
        print(
            f"{metric:<20}"
            f"{transformer_metrics[metric]:<18.4f}"
            f"{rf_metrics[metric]:<18.4f}"
        )

    # PREDICTION DISTRIBUTION
    print("\n========== PREDICTION DISTRIBUTION ==========")

    print_distribution("Actual labels:", actual)

    print_distribution(
        "Transformer predictions:",
        transformer_predictions
    )

    print_distribution(
        "Random Forest predictions:",
        rf_predictions
    )

    # AGREEMENT CHECK
    identical_count = int(
        np.sum(transformer_predictions == rf_predictions)
    )

    disagreement_count = len(actual) - identical_count

    agreement = identical_count / len(actual) * 100

    print("\n========== PREDICTION AGREEMENT ==========")

    print("Total samples:", len(actual))
    print("Identical predictions:", identical_count)
    print("Different predictions:", disagreement_count)
    print(f"Agreement percentage: {agreement:.2f}%")

    print(
        "Prediction arrays identical:",
        np.array_equal(
            transformer_predictions,
            rf_predictions
        )
    )

    print(
        "Transformer unique predictions:",
        np.unique(transformer_predictions)
    )

    print(
        "Random Forest unique predictions:",
        np.unique(rf_predictions)
    )

    # CLASSIFICATION REPORTS
    print("\nTRANSFORMER CLASSIFICATION REPORT")

    print(
        classification_report(
            actual,
            transformer_predictions,
            labels=list(range(len(ACTIVITIES))),
            target_names=ACTIVITIES,
            zero_division=0
        )
    )

    print("\nRANDOM FOREST CLASSIFICATION REPORT")

    print(
        classification_report(
            actual,
            rf_predictions,
            labels=list(range(len(ACTIVITIES))),
            target_names=ACTIVITIES,
            zero_division=0
        )
    )

    # CONFUSION MATRICES
    print("\nTRANSFORMER CONFUSION MATRIX")

    print(
        confusion_matrix(
            actual,
            transformer_predictions,
            labels=list(range(len(ACTIVITIES)))
        )
    )

    print("\nRANDOM FOREST CONFUSION MATRIX")

    print(
        confusion_matrix(
            actual,
            rf_predictions,
            labels=list(range(len(ACTIVITIES)))
        )
    )

    # FINAL DETAILS
    print("\n========== EVALUATION DETAILS ==========")

    print("Training samples:", len(y_train))
    print("Evaluation samples:", len(actual))
    print("Random Forest features:", x_train_rf.shape[1])
    print("Activity classes:", len(ACTIVITIES))

    print("\nNOTE:")
    print("Both models use synthetic demo data.")
    print("Results do not represent real-world accuracy.")

    print("\nModel comparison completed.")


if __name__ == "__main__":
    main()
