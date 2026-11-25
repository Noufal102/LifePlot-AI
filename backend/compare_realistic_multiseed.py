
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


def calculate_metrics(actual, predictions):
    return {
        "accuracy": accuracy_score(actual, predictions),
        "precision": precision_score(
            actual,
            predictions,
            average="macro",
            zero_division=0
        ),
        "recall": recall_score(
            actual,
            predictions,
            average="macro",
            zero_division=0
        ),
        "macro_f1": f1_score(
            actual,
            predictions,
            average="macro",
            zero_division=0
        ),
    }


def main():

    print("========== MULTI-SEED MODEL VERIFICATION ==========")

    if not MODEL_PATH.exists():
        print("ERROR: Transformer checkpoint not found.")
        return

    # Same training data for both models
    print("\nGenerating shared training data...")

    x_train, f_train, y_train = build_realistic_data(
        n=5000,
        seed=TRAIN_SEED
    )

    # Load Transformer
    print("\nLoading Transformer checkpoint...")

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

    # Train Random Forest on the same training data
    print("\nPreparing Random Forest training data...")

    x_train_rf = to_rf_features(
        x_train,
        f_train
    )

    print("Training Random Forest...")

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(
        x_train_rf,
        y_train.numpy()
    )

    results = {
        "Transformer": [],
        "Random Forest": [],
    }

    # Evaluate both models on 5 independent test datasets
    for seed in TEST_SEEDS:

        print(f"\n---------- TEST SEED: {seed} ----------")

        x_test, f_test, y_test = build_realistic_data(
            n=1000,
            seed=seed
        )

        actual = y_test.numpy()

        # Transformer prediction
        with torch.no_grad():
            outputs = transformer(
                x_test.to(DEVICE),
                f_test.to(DEVICE)
            )

            transformer_predictions = (
                outputs.argmax(dim=1).cpu().numpy()
            )

        # Random Forest prediction
        x_test_rf = to_rf_features(
            x_test,
            f_test
        )

        rf_predictions = rf.predict(x_test_rf)

        transformer_metrics = calculate_metrics(
            actual,
            transformer_predictions
        )

        rf_metrics = calculate_metrics(
            actual,
            rf_predictions
        )

        results["Transformer"].append(transformer_metrics)
        results["Random Forest"].append(rf_metrics)

        print("\nTransformer:")
        for metric, value in transformer_metrics.items():
            print(f"{metric}: {value:.4f}")

        print("\nRandom Forest:")
        for metric, value in rf_metrics.items():
            print(f"{metric}: {value:.4f}")

    # Calculate mean and standard deviation
    print("\n========== FINAL MULTI-SEED SUMMARY ==========")

    for model_name, model_results in results.items():

        print(f"\n{model_name}")

        for metric in [
            "accuracy",
            "precision",
            "recall",
            "macro_f1"
        ]:

            values = [
                result[metric]
                for result in model_results
            ]

            mean_value = np.mean(values)
            std_value = np.std(values)

            print(
                f"{metric}: "
                f"{mean_value:.4f} "
                f"+/- {std_value:.4f}"
            )

    # Compare average results
    transformer_f1 = np.mean([
        r["macro_f1"]
        for r in results["Transformer"]
    ])

    rf_f1 = np.mean([
        r["macro_f1"]
        for r in results["Random Forest"]
    ])

    print("\n========== MODEL DECISION ==========")

    if rf_f1 > transformer_f1:
        print("Random Forest has higher average Macro F1.")
    elif transformer_f1 > rf_f1:
        print("Transformer has higher average Macro F1.")
    else:
        print("Both models have equal average Macro F1.")

    print("\nMulti-seed verification completed.")


if __name__ == "__main__":
    main()
