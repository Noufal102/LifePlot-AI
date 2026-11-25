
from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from transformer_model import ActivityTransformer
from realistic_data import build_realistic_data


BASE_DIR = Path(__file__).resolve().parent

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TRAIN_SEEDS = [42, 123, 456]

TEST_SEEDS = [98765, 111, 222, 333, 444]

ACTIVITIES = [
    "Sleep", "Work", "Eating", "Exercise",
    "Entertainment", "Study", "Travel", "Break"
]


def load_model(seed):

    path = BASE_DIR / (
        f"activity_transformer_balanced_seed_{seed}.pth"
    )

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint missing: {path}")

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=True
    )

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

    print(
        f"Loaded seed {seed} | "
        f"Best epoch: {checkpoint['best_epoch']} | "
        f"Validation loss: {checkpoint['best_val_loss']:.4f}"
    )

    return model


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


def main():

    print("========== BALANCED TRANSFORMER STABILITY TEST ==========")

    models = {}

    for seed in TRAIN_SEEDS:
        models[seed] = load_model(seed)

    all_actual = []
    all_predictions = {
        seed: [] for seed in TRAIN_SEEDS
    }

    results = {
        seed: [] for seed in TRAIN_SEEDS
    }

    for test_seed in TEST_SEEDS:

        print(f"\n---------- TEST SEED: {test_seed} ----------")

        x_test, f_test, y_test = build_realistic_data(
            n=1000,
            seed=test_seed
        )

        actual = y_test.numpy()
        all_actual.extend(actual.tolist())

        for train_seed, model in models.items():

            with torch.no_grad():

                outputs = model(
                    x_test.to(DEVICE),
                    f_test.to(DEVICE)
                )

                predictions = (
                    outputs.argmax(dim=1).cpu().numpy()
                )

            metrics = calculate_metrics(
                actual,
                predictions
            )

            results[train_seed].append(metrics)

            all_predictions[train_seed].extend(
                predictions.tolist()
            )

            print(
                f"Train Seed {train_seed} | "
                f"Accuracy: {metrics['accuracy']:.4f} | "
                f"Macro F1: {metrics['macro_f1']:.4f}"
            )

    print("\n========== TRAINING SEED SUMMARY ==========")

    for seed in TRAIN_SEEDS:

        print(f"\nTraining Seed: {seed}")

        for metric in [
            "accuracy",
            "precision",
            "recall",
            "macro_f1"
        ]:

            values = [
                item[metric] for item in results[seed]
            ]

            print(
                f"{metric}: "
                f"{np.mean(values):.4f} "
                f"+/- {np.std(values):.4f}"
            )

    print("\n========== CLASS-WISE F1 ==========")

    actual_all = np.array(all_actual)

    for seed in TRAIN_SEEDS:

        predictions = np.array(all_predictions[seed])

        class_scores = f1_score(
            actual_all,
            predictions,
            labels=list(range(8)),
            average=None,
            zero_division=0
        )

        print(f"\nTraining Seed {seed}")

        for activity, score in zip(ACTIVITIES, class_scores):
            print(f"{activity:16s}: {score:.4f}")

    print("\n========== CROSS-SEED STABILITY ==========")

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "macro_f1"
    ]:

        seed_means = [
            np.mean([
                item[metric] for item in results[seed]
            ])
            for seed in TRAIN_SEEDS
        ]

        print(
            f"{metric}: "
            f"Mean={np.mean(seed_means):.4f} | "
            f"Training-seed SD={np.std(seed_means):.4f}"
        )

    print("\nStability evaluation completed.")


if __name__ == "__main__":
    main()
