
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent

CHECKPOINTS = {
    "Original Transformer": BASE_DIR / "activity_transformer_realistic.pth",
    "Balanced Transformer": BASE_DIR / "activity_transformer_balanced.pth",
    "Balanced Transformer Seed 42": BASE_DIR / "activity_transformer_balanced_seed_42.pth",
    "Balanced Transformer Seed 123": BASE_DIR / "activity_transformer_balanced_seed_123.pth",
    "Balanced Transformer Seed 456": BASE_DIR / "activity_transformer_balanced_seed_456.pth",
}

RESULTS = {
    "Original Transformer": {
        "accuracy": 32.56,
        "macro_f1": 24.11,
        "note": "Single training seed; earlier experiment",
    },
    "Balanced Transformer": {
        "accuracy": 30.11,
        "macro_f1": 25.55,
        "note": "Three training seeds; stability evaluation",
    },
    "Original Random Forest": {
        "accuracy": 34.86,
        "macro_f1": 25.68,
        "note": "Three training seeds; five test datasets",
    },
    "Balanced Random Forest": {
        "accuracy": 33.61,
        "macro_f1": 26.44,
        "note": "Three training seeds; five test datasets",
    },
}


def load_checkpoint(path):
    try:
        try:
            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=True
            )
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")

        return checkpoint, None

    except Exception as error:
        return None, str(error)


def main():
    print("=" * 65)
    print("LIFEPLOT AI - FINAL MODEL SELECTION REPORT")
    print("=" * 65)

    print("\n1. TRANSFORMER CHECKPOINT VERIFICATION")

    for name, path in CHECKPOINTS.items():
        print(f"\n{name}")
        print("Path:", path)
        print("Exists:", path.exists())

        if not path.exists():
            continue

        checkpoint, error = load_checkpoint(path)

        if error:
            print("Loading error:", error)
            continue

        if not isinstance(checkpoint, dict):
            print("Checkpoint format: Unexpected")
            continue

        print("Checkpoint format: Loaded successfully")

        for key in [
            "best_epoch",
            "best_val_loss",
            "num_activities",
            "num_features",
            "sequence_length",
            "d_model",
            "num_heads",
            "num_layers",
            "dropout",
        ]:
            if key in checkpoint:
                print(f"{key}: {checkpoint[key]}")

    print("\n\n2. MODEL PERFORMANCE COMPARISON")
    print("-" * 65)

    print(
        f"{'Model':25} {'Accuracy':>12} {'Macro F1':>12}"
    )

    for name, metrics in RESULTS.items():
        print(
            f"{name:25} "
            f"{metrics['accuracy']:>11.2f}% "
            f"{metrics['macro_f1']:>11.2f}%"
        )

    winner = max(
        RESULTS.items(),
        key=lambda item: item[1]["macro_f1"]
    )

    print("\n3. CURRENT PERFORMANCE WINNER")
    print("Model:", winner[0])
    print("Macro F1:", f"{winner[1]['macro_f1']:.2f}%")
    print("Note:", winner[1]["note"])

    print("\n4. FINAL PROJECT DECISION")
    print("Primary architecture: Transformer-based")
    print("Benchmark model: Balanced Random Forest")
    print("Deployment decision: Pending API integration and testing")

    print("\n5. IMPORTANT NOTES")
    print("- Synthetic data was used for these experiments.")
    print("- Scores are experimental, not real-world guarantees.")
    print("- Transformer remains part of the project.")
    print("- Existing app and model files were not modified.")

    print("\n" + "=" * 65)
    print("MODEL SELECTION REPORT COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
