
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

ORIGINAL_PATH = BASE_DIR / "activity_transformer_realistic.pth"
BALANCED_PATH = BASE_DIR / "activity_transformer_balanced.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TRAIN_SEED = 42

TEST_SEEDS = [98765, 111, 222, 333, 444]

ACTIVITIES = [
    "Sleep",
    "Work",
    "Eating",
    "Exercise",
    "Entertainment",
    "Study",
    "Travel",
    "Break",
]


def to_rf_features(x, f, classes=8):
    one_hot = np.eye(classes)[x.numpy()]
    one_hot = one_hot.reshape(len(x), -1)

    numeric = f.numpy().reshape(len(f), -1)

    return np.concatenate([one_hot, numeric], axis=1)


def calculate_metrics(actual, predictions):
    return {
        "accuracy": accuracy_score(actual, predictions),
        "precision": precision_score(
            actual, predictions,
            average="macro",
            zero_division=0
        ),
        "recall": recall_score(
            actual, predictions,
            average="macro",
            zero_division=0
        ),
        "macro_f1": f1_score(
            actual, predictions,
            average="macro",
            zero_division=0
        ),
    }


def load_transformer(path, label):
    print(f"\nLoading {label}...")

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=True
    )

    # Use architecture metadata when available.
    # Otherwise use the model class defaults.
    model = ActivityTransformer(
        num_activities=checkpoint.get("num_activities", 8),
        num_features=checkpoint.get("num_features", 5),
        d_model=checkpoint.get("d_model", 64),
        nhead=checkpoint.get("nhead", 4),
        num_layers=checkpoint.get("num_layers", 2),
        max_len=checkpoint.get("sequence_length", 12),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print("Checkpoint:", path.name)
    print("Best epoch:", checkpoint.get("best_epoch", "Not stored"))
    print("Best validation loss:", checkpoint.get("best_val_loss", "Not stored"))

    return model


def main():

    print("========== BALANCED MODEL COMPARISON ==========")

    if not ORIGINAL_PATH.exists():
        print("ERROR: Original Transformer checkpoint not found.")
        return

    if not BALANCED_PATH.exists():
        print("ERROR: Balanced Transformer checkpoint not found.")
        return

    print("\nGenerating shared training data...")

    x_train, f_train, y_train = build_realistic_data(
        n=5000,
        seed=TRAIN_SEED
    )

    original_transformer = load_transformer(
        ORIGINAL_PATH,
        "Original Transformer"
    )

    balanced_transformer = load_transformer(
        BALANCED_PATH,
        "Balanced Transformer"
    )

    x_train_rf = to_rf_features(x_train, f_train)
    y_train_np = y_train.numpy()

    print("\nTraining original Random Forest...")

    original_rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    original_rf.fit(x_train_rf, y_train_np)

    print("\nTraining balanced Random Forest...")

    balanced_rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    balanced_rf.fit(x_train_rf, y_train_np)

    models = {
        "Original Transformer": original_transformer,
        "Balanced Transformer": balanced_transformer,
    }

    rf_models = {
        "Original Random Forest": original_rf,
        "Balanced Random Forest": balanced_rf,
    }

    results = {
        name: [] for name in [
            "Original Transformer",
            "Balanced Transformer",
            "Original Random Forest",
            "Balanced Random Forest",
        ]
    }

    all_actual = []
    all_predictions = {
        name: [] for name in results
    }

    for seed in TEST_SEEDS:

        print(f"\n---------- TEST SEED: {seed} ----------")

        x_test, f_test, y_test = build_realistic_data(
            n=1000,
            seed=seed
        )

        actual = y_test.numpy()

        all_actual.extend(actual.tolist())

        # Evaluate both Transformers
        with torch.no_grad():

            for name, model in models.items():

                outputs = model(
                    x_test.to(DEVICE),
                    f_test.to(DEVICE)
                )

                predictions = outputs.argmax(dim=1).cpu().numpy()

                metrics = calculate_metrics(actual, predictions)

                results[name].append(metrics)
                all_predictions[name].extend(predictions.tolist())

                print(
                    f"{name}: "
                    f"Accuracy={metrics['accuracy']:.4f}, "
                    f"Macro F1={metrics['macro_f1']:.4f}"
                )

        # Evaluate both Random Forest models
        x_test_rf = to_rf_features(x_test, f_test)

        for name, model in rf_models.items():

            predictions = model.predict(x_test_rf)

            metrics = calculate_metrics(actual, predictions)

            results[name].append(metrics)
            all_predictions[name].extend(predictions.tolist())

            print(
                f"{name}: "
                f"Accuracy={metrics['accuracy']:.4f}, "
                f"Macro F1={metrics['macro_f1']:.4f}"
            )

    print("\n========== FINAL MULTI-SEED SUMMARY ==========")

    for name, model_results in results.items():

        print(f"\n{name}")

        for metric in [
            "accuracy",
            "precision",
            "recall",
            "macro_f1"
        ]:

            values = [r[metric] for r in model_results]

            print(
                f"{metric}: "
                f"{np.mean(values):.4f} "
                f"+/- {np.std(values):.4f}"
            )

    print("\n========== CLASS-WISE F1 COMPARISON ==========")

    actual_all = np.array(all_actual)

    for name, predictions in all_predictions.items():

        predictions_all = np.array(predictions)

        class_f1 = f1_score(
            actual_all,
            predictions_all,
            labels=list(range(8)),
            average=None,
            zero_division=0
        )

        print(f"\n{name}")

        for activity, score in zip(ACTIVITIES, class_f1):
            print(f"{activity:16s}: {score:.4f}")

    print("\n========== FINAL MODEL RANKING ==========")

    ranking = sorted(
        results.items(),
        key=lambda item: np.mean([
            r["macro_f1"] for r in item[1]
        ]),
        reverse=True
    )

    for position, (name, model_results) in enumerate(ranking, 1):

        mean_f1 = np.mean([
            r["macro_f1"] for r in model_results
        ])

        print(f"{position}. {name}: Macro F1 = {mean_f1:.4f}")

    print("\nBalanced comparison completed.")


if __name__ == "__main__":
    main()
