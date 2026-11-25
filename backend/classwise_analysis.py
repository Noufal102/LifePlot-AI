
from pathlib import Path

import numpy as np
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

from transformer_model import ActivityTransformer
from realistic_data import build_realistic_data
from data_utils import ACTIVITIES


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "activity_transformer_realistic.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TRAIN_SEED = 42

TEST_SEEDS = [
    98765,
    111,
    222,
    333,
    444,
]


def to_rf_features(x, f, classes=8):
    one_hot = np.eye(classes)[x.numpy()]
    one_hot = one_hot.reshape(len(x), -1)

    numeric = f.numpy().reshape(len(f), -1)

    return np.concatenate(
        [one_hot, numeric],
        axis=1
    )


def main():

    print("========== CLASS-WISE MODEL ANALYSIS ==========")

    if not MODEL_PATH.exists():
        print("ERROR: Transformer checkpoint not found.")
        return

    print("\nGenerating shared training data...")

    x_train, f_train, y_train = build_realistic_data(
        n=5000,
        seed=TRAIN_SEED
    )

    print("Training Random Forest...")

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(
        to_rf_features(x_train, f_train),
        y_train.numpy()
    )

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

    transformer.load_state_dict(
        checkpoint["model_state_dict"]
    )

    transformer.to(DEVICE)
    transformer.eval()

    all_actual = []
    all_transformer_predictions = []
    all_rf_predictions = []

    for seed in TEST_SEEDS:

        print(f"\nEvaluating test seed: {seed}")

        x_test, f_test, y_test = build_realistic_data(
            n=1000,
            seed=seed
        )

        actual = y_test.numpy()

        with torch.no_grad():

            outputs = transformer(
                x_test.to(DEVICE),
                f_test.to(DEVICE)
            )

            transformer_predictions = (
                outputs.argmax(dim=1).cpu().numpy()
            )

        rf_predictions = rf.predict(
            to_rf_features(x_test, f_test)
        )

        all_actual.extend(actual.tolist())

        all_transformer_predictions.extend(
            transformer_predictions.tolist()
        )

        all_rf_predictions.extend(
            rf_predictions.tolist()
        )

    all_actual = np.array(all_actual)

    all_transformer_predictions = np.array(
        all_transformer_predictions
    )

    all_rf_predictions = np.array(
        all_rf_predictions
    )

    labels = list(range(len(ACTIVITIES)))

    transformer_metrics = precision_recall_fscore_support(
        all_actual,
        all_transformer_predictions,
        labels=labels,
        zero_division=0
    )

    rf_metrics = precision_recall_fscore_support(
        all_actual,
        all_rf_predictions,
        labels=labels,
        zero_division=0
    )

    print("\n========== CLASS-WISE RESULTS ==========")

    print(
        f"{'Activity':<16}"
        f"{'TF Precision':>14}"
        f"{'RF Precision':>14}"
        f"{'TF Recall':>12}"
        f"{'RF Recall':>12}"
        f"{'TF F1':>10}"
        f"{'RF F1':>10}"
    )

    for i, activity in enumerate(ACTIVITIES):

        print(
            f"{activity:<16}"
            f"{transformer_metrics[0][i]:>14.3f}"
            f"{rf_metrics[0][i]:>14.3f}"
            f"{transformer_metrics[1][i]:>12.3f}"
            f"{rf_metrics[1][i]:>12.3f}"
            f"{transformer_metrics[2][i]:>10.3f}"
            f"{rf_metrics[2][i]:>10.3f}"
        )

    print("\n========== CLASS DISTRIBUTION ==========")

    counts = np.bincount(
        all_actual,
        minlength=len(ACTIVITIES)
    )

    for i, activity in enumerate(ACTIVITIES):
        print(f"{activity:<16}: {counts[i]}")

    print("\n========== ANALYSIS COMPLETE ==========")


if __name__ == "__main__":
    main()
