
from pathlib import Path

import numpy as np
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from realistic_data import build_realistic_data


TRAIN_SEED = 42
TRAIN_SAMPLES = 5000
TEST_SAMPLES = 1000

TRAINING_SEEDS = [42, 123, 456]
TEST_SEEDS = [98765, 111, 222, 333, 444]

N_ESTIMATORS = 300


def to_rf_features(activity_ids, numeric_features):
    activity_one_hot = torch.nn.functional.one_hot(
        activity_ids.long(),
        num_classes=8
    ).float()

    combined = torch.cat(
        [activity_one_hot, numeric_features.float()],
        dim=2
    )

    return combined.reshape(combined.shape[0], -1).numpy()


def get_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    }


def main():
    print("Generating shared training dataset...")

    train_x, train_f, train_y = build_realistic_data(
        n=TRAIN_SAMPLES,
        seed=TRAIN_SEED
    )

    X_train = to_rf_features(train_x, train_f)
    y_train = train_y.numpy()

    print("Training data shape:", X_train.shape)

    results = {
        "Original RF": [],
        "Balanced RF": [],
    }

    all_predictions = {
        "Original RF": [],
        "Balanced RF": [],
    }

    all_targets = []

    for rf_seed in TRAINING_SEEDS:
        print(f"\n{'=' * 55}")
        print(f"RANDOM FOREST TRAINING SEED: {rf_seed}")
        print(f"{'=' * 55}")

        models = {
            "Original RF": RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                random_state=rf_seed,
                n_jobs=-1
            ),
            "Balanced RF": RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                random_state=rf_seed,
                class_weight="balanced",
                n_jobs=-1
            ),
        }

        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            model.fit(X_train, y_train)

        seed_results = {
            "Original RF": [],
            "Balanced RF": [],
        }

        seed_predictions = {
            "Original RF": [],
            "Balanced RF": [],
        }

        for test_seed in TEST_SEEDS:
            print(f"\nEvaluating test seed: {test_seed}")

            test_x, test_f, test_y = build_realistic_data(
                n=TEST_SAMPLES,
                seed=test_seed
            )

            X_test = to_rf_features(test_x, test_f)
            y_test = test_y.numpy()

            for model_name, model in models.items():
                predictions = model.predict(X_test)

                metrics = get_metrics(y_test, predictions)
                seed_results[model_name].append(metrics)

                seed_predictions[model_name].extend(
                    predictions.tolist()
                )

                print(
                    f"{model_name:15} | "
                    f"Accuracy: {metrics['accuracy']:.4f} | "
                    f"Macro F1: {metrics['macro_f1']:.4f}"
                )

            if rf_seed == TRAINING_SEEDS[0]:
                all_targets.extend(y_test.tolist())

        for model_name in models:
            results[model_name].append(seed_results[model_name])

            all_predictions[model_name].append(seed_predictions[model_name])

    print("\n\n========== RF TRAINING-SEED SUMMARY ==========")

    for model_name in results:
        print(f"\n{model_name}")

        for metric in ["accuracy", "precision", "recall", "macro_f1"]:
            seed_means = []

            for training_seed_results in results[model_name]:
                values = [
                    item[metric] for item in training_seed_results
                ]
                seed_means.append(np.mean(values))

            print(
                f"{metric:12}: "
                f"{np.mean(seed_means):.4f} "
                f"+/- {np.std(seed_means):.4f}"
            )

    print("\n\n========== PER TRAINING SEED ==========")

    for model_name in results:
        print(f"\n{model_name}")

        for index, rf_seed in enumerate(TRAINING_SEEDS):
            seed_results = results[model_name][index]

            mean_f1 = np.mean([
                item["macro_f1"] for item in seed_results
            ])

            mean_acc = np.mean([
                item["accuracy"] for item in seed_results
            ])

            print(
                f"Seed {rf_seed}: "
                f"Accuracy={mean_acc:.4f}, "
                f"Macro F1={mean_f1:.4f}"
            )

    print("\n\n========== CLASS-WISE F1 ==========")

    combined_y_true = np.array(all_targets)

    for model_name in results:
        predictions = np.array(all_predictions[model_name])

        # Each RF training seed has predictions for the same 5 test sets.
        # Evaluate each seed separately, then average class-wise F1.
        class_f1_scores = []

        for seed_predictions in predictions:
            report = classification_report(
                combined_y_true,
                seed_predictions,
                labels=list(range(8)),
                output_dict=True,
                zero_division=0
            )

            class_f1_scores.append([
                report[str(class_id)]["f1-score"]
                for class_id in range(8)
            ])

        mean_class_f1 = np.mean(class_f1_scores, axis=0)

        print(f"\n{model_name}")
        for class_id, score in enumerate(mean_class_f1):
            print(f"Activity {class_id}: F1 = {score:.4f}")

    print("\nRF multi-seed stability experiment completed.")


if __name__ == "__main__":
    main()
