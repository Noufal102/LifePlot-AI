
import os
import torch
import numpy as np

from transformer_model import ActivityTransformer
from data_utils import ACTIVITIES, encode_activity, make_features
from decision_engine import make_suggestion

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "activity_transformer_balanced_seed_42.pth"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_model = None


def load_model():
    global _model

    if _model is not None:
        return _model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Transformer checkpoint not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model = ActivityTransformer(
        num_activities=8,
        num_features=5,
        d_model=64,
        nhead=4,
        num_layers=2,
        max_len=12
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    _model = model
    return _model


def predict_next(history):

    if len(history) < 12:
        remaining = 12 - len(history)

        return {
            "predicted_activity": "Not Ready",
            "confidence": 0.0,
            "suggestion": (
                f"Add {remaining} more activities to enable "
                "a personalized Transformer prediction. "
                "A minimum of 12 activities is required."
            ),
            "top_predictions": []
        }

    model = load_model()
    recent = history[-12:]

    activity_ids = [
        encode_activity(item.activity)
        for item in recent
    ]

    features = [
        make_features(item)
        for item in recent
    ]

    activity_tensor = torch.tensor(
        [activity_ids],
        dtype=torch.long,
        device=DEVICE
    )

    feature_tensor = torch.tensor(
        [features],
        dtype=torch.float32,
        device=DEVICE
    )

    with torch.no_grad():
        logits = model(activity_tensor, feature_tensor)
        probabilities = torch.softmax(logits, dim=-1)[0]

        top_values, top_indices = torch.topk(
            probabilities,
            k=3
        )

        predicted_index = int(top_indices[0].item())
        confidence = float(top_values[0].item())

    top_predictions = []

    for probability, index in zip(
        top_values.tolist(),
        top_indices.tolist()
    ):
        top_predictions.append({
            "activity": ACTIVITIES[index],
            "confidence": round(float(probability), 4)
        })

    energy = int(round(
        np.mean([item.energy_level for item in recent])
    ))

    priority = int(round(
        np.mean([item.priority for item in recent])
    ))

    predicted_activity = ACTIVITIES[predicted_index]

    suggestion = make_suggestion(
        predicted_activity,
        confidence,
        energy,
        priority
    )

    return {
        "predicted_activity": predicted_activity,
        "confidence": round(confidence, 4),
        "suggestion": suggestion,
        "top_predictions": top_predictions
    }
