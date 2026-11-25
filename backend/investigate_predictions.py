
from pathlib import Path
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

from transformer_model import ActivityTransformer
from train import build_demo_data

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "activity_transformer.pth"

torch.manual_seed(42)
np.random.seed(42)

def to_rf_features(x, f, classes=8):
    one_hot = np.eye(classes)[x.numpy()]
    return np.concatenate(
        [one_hot.reshape(len(x), -1), f.numpy().reshape(len(x), -1)],
        axis=1
    )

def main():
    if not MODEL_PATH.exists():
        print("ERROR: Transformer checkpoint not found")
        return

    print("Generating independent datasets...")
    x_train, f_train, y_train = build_demo_data(5000)
    x_test, f_test, y_test = build_demo_data(1000)

    print("\nChecking test data...")
    print("Train X:", x_train.shape)
    print("Test X:", x_test.shape)
    print("Train labels:", y_train.shape)
    print("Test labels:", y_test.shape)

    print("\nLoading Transformer...")
    model = ActivityTransformer()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()

    with torch.no_grad():
        transformer_logits = model(x_test, f_test)
        transformer_probs = torch.softmax(transformer_logits, dim=1).numpy()
        transformer_preds = transformer_probs.argmax(axis=1)

    print("\nTraining independent Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(to_rf_features(x_train, f_train), y_train.numpy())

    rf_preds = rf.predict(to_rf_features(x_test, f_test))

    print("\n========== INVESTIGATION ==========")
    print("Transformer predictions:", len(transformer_preds))
    print("RF predictions:", len(rf_preds))

    print(
        "Transformer unique predictions:",
        np.unique(transformer_preds, return_counts=True)
    )
    print(
        "RF unique predictions:",
        np.unique(rf_preds, return_counts=True)
    )

    print(
        "\nTransformer vs RF identical:",
        np.array_equal(transformer_preds, rf_preds)
    )

    print(
        "Transformer vs actual identical:",
        np.array_equal(transformer_preds, y_test.numpy())
    )

    print(
        "RF vs actual identical:",
        np.array_equal(rf_preds, y_test.numpy())
    )

    print("\nFirst 30 predictions:")
    print("Actual:      ", y_test.numpy()[:30])
    print("Transformer: ", transformer_preds[:30])
    print("RandomForest:", rf_preds[:30])

    print("\nTransformer probability samples:")
    print(np.round(transformer_probs[:5], 4))

    print("\nRandom Forest probability samples:")
    print(np.round(rf.predict_proba(to_rf_features(x_test, f_test))[:5], 4))

    print("\nInvestigation completed.")

if __name__ == "__main__":
    main()
