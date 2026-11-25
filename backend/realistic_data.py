
import random
from datetime import datetime, timedelta

import torch

from data_utils import ACTIVITIES, ACTIVITY_TO_ID


SEQUENCE_LENGTH = 12
NUM_ACTIVITIES = len(ACTIVITIES)


# Activity duration ranges in minutes
DURATION_RANGES = {
    "Sleep": (300, 480),
    "Work": (45, 180),
    "Eating": (15, 60),
    "Exercise": (20, 90),
    "Entertainment": (20, 120),
    "Study": (30, 150),
    "Travel": (10, 90),
    "Break": (5, 30),
}


def choose_activity(hour, previous_activity, rng):
    """
    Select the next activity using time-of-day patterns
    and activity transition preferences.
    """

    if 0 <= hour < 6:
        weights = {
            "Sleep": 90,
            "Break": 5,
            "Entertainment": 3,
            "Work": 2,
        }

    elif 6 <= hour < 9:
        weights = {
            "Sleep": 15,
            "Eating": 30,
            "Travel": 15,
            "Work": 15,
            "Exercise": 15,
            "Break": 10,
        }

    elif 9 <= hour < 12:
        weights = {
            "Work": 40,
            "Study": 30,
            "Break": 10,
            "Travel": 10,
            "Entertainment": 5,
            "Eating": 5,
        }

    elif 12 <= hour < 14:
        weights = {
            "Eating": 45,
            "Break": 20,
            "Work": 15,
            "Study": 10,
            "Entertainment": 10,
        }

    elif 14 <= hour < 18:
        weights = {
            "Work": 30,
            "Study": 25,
            "Entertainment": 10,
            "Exercise": 10,
            "Travel": 10,
            "Break": 15,
        }

    elif 18 <= hour < 22:
        weights = {
            "Entertainment": 25,
            "Eating": 20,
            "Exercise": 15,
            "Study": 15,
            "Travel": 10,
            "Break": 10,
            "Work": 5,
        }

    else:
        weights = {
            "Sleep": 55,
            "Entertainment": 15,
            "Eating": 10,
            "Break": 15,
            "Study": 5,
        }

    # Encourage plausible transitions.
    transitions = {
        "Sleep": {
            "Eating": 3,
            "Work": 2,
            "Exercise": 2,
        },
        "Work": {
            "Break": 3,
            "Eating": 2,
            "Travel": 2,
        },
        "Study": {
            "Break": 3,
            "Entertainment": 2,
        },
        "Exercise": {
            "Break": 3,
            "Eating": 2,
        },
        "Travel": {
            "Work": 2,
            "Eating": 2,
            "Entertainment": 2,
        },
        "Eating": {
            "Work": 2,
            "Study": 2,
            "Break": 2,
        },
        "Entertainment": {
            "Sleep": 2,
            "Break": 2,
        },
        "Break": {
            "Work": 2,
            "Study": 2,
            "Entertainment": 2,
        },
    }

    for activity, multiplier in transitions.get(
        previous_activity, {}
    ).items():
        if activity in weights:
            weights[activity] *= multiplier

    names = list(weights.keys())
    probabilities = list(weights.values())

    return rng.choices(names, weights=probabilities, k=1)[0]


def generate_one_sequence(rng):
    """
    Generate 13 consecutive activities.
    First 12 are inputs; the 13th is the prediction target.
    """

    start_time = datetime(
        year=2026,
        month=1,
        day=5 + rng.randrange(7),
        hour=rng.randrange(24),
        minute=rng.choice([0, 15, 30, 45]),
    )

    activities = []
    features = []

    previous_activity = "Sleep"

    for _ in range(SEQUENCE_LENGTH + 1):

        current_hour = start_time.hour

        activity = choose_activity(
            current_hour,
            previous_activity,
            rng,
        )

        min_duration, max_duration = DURATION_RANGES[activity]

        duration = rng.randint(min_duration, max_duration)

        if activity == "Sleep":
            energy = rng.randint(1, 3)
            priority = rng.randint(2, 3)

        elif activity in ["Work", "Study"]:
            energy = rng.randint(2, 5)
            priority = rng.randint(2, 3)

        elif activity == "Exercise":
            energy = rng.randint(3, 5)
            priority = rng.randint(1, 3)

        elif activity == "Break":
            energy = rng.randint(1, 4)
            priority = rng.randint(1, 2)

        else:
            energy = rng.randint(1, 5)
            priority = rng.randint(1, 3)

        activities.append(ACTIVITY_TO_ID[activity])

        features.append([
            min(duration / 600.0, 1.0),
            energy / 5.0,
            priority / 5.0,
            start_time.hour / 23.0,
            start_time.weekday() / 6.0,
        ])

        # Move time forward by activity duration and a short gap.
        gap = rng.randint(0, 20)
        start_time += timedelta(minutes=duration + gap)

        previous_activity = activity

    x = activities[:SEQUENCE_LENGTH]
    f = features[:SEQUENCE_LENGTH]
    y = activities[SEQUENCE_LENGTH]

    return x, f, y


def build_realistic_data(n=5000, seed=42):
    """
    Create a reproducible synthetic dataset.

    Returns:
        x: [n, 12] activity IDs
        f: [n, 12, 5] normalized numeric features
        y: [n] next activity IDs
    """

    rng = random.Random(seed)

    xs = []
    fs = []
    ys = []

    for _ in range(n):
        x, f, y = generate_one_sequence(rng)

        xs.append(x)
        fs.append(f)
        ys.append(y)

    return (
        torch.tensor(xs, dtype=torch.long),
        torch.tensor(fs, dtype=torch.float32),
        torch.tensor(ys, dtype=torch.long),
    )


if __name__ == "__main__":
    x, f, y = build_realistic_data(n=1000)

    print("Realistic synthetic dataset generated successfully!")
    print("Activities:", ACTIVITIES)
    print("Input activity shape:", x.shape)
    print("Input feature shape:", f.shape)
    print("Target shape:", y.shape)
    print("Sample activity IDs:", x[0].tolist())
    print("Sample target:", ACTIVITIES[y[0].item()])
