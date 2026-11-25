ACTIVITIES = ["Sleep", "Work", "Eating", "Exercise", "Entertainment", "Study", "Travel", "Break"]
ACTIVITY_TO_ID = {name: i for i, name in enumerate(ACTIVITIES)}
ID_TO_ACTIVITY = {i: name for name, i in ACTIVITY_TO_ID.items()}

def encode_activity(name):
    return ACTIVITY_TO_ID.get(name, ACTIVITY_TO_ID["Break"])

def make_features(item):
    # Consistent five numeric inputs: duration, energy, priority, hour, weekday.
    ts = item.timestamp
    return [
        min(float(item.duration) / 600.0, 1.0),
        float(item.energy_level) / 5.0,
        float(item.priority) / 5.0,
        ts.hour / 23.0,
        ts.weekday() / 6.0
    ]
