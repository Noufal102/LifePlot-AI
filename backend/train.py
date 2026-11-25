# Demo trainer: creates a synthetic sequence dataset so the project runs from a clean setup.
# Replace this with consented, real activity logs for meaningful personalization.
import random
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from transformer_model import ActivityTransformer

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

def build_demo_data(n=5000, seq_len=12, classes=8):
    xs, fs, ys = [], [], []
    # A simple synthetic daily-cycle pattern; this is NOT a real-world validated dataset.
    for _ in range(n):
        start = random.randrange(8)
        seq = [(start + j) % classes for j in range(seq_len)]
        # Add modest noise to avoid a perfectly deterministic sequence.
        seq = [v if random.random() > 0.08 else random.randrange(classes) for v in seq]
        features = []
        for j, activity_id in enumerate(seq):
            hour = (j * 2 + random.randrange(2)) % 24
            features.append([
                random.random() * 0.7 + 0.1,
                random.random(),
                random.random(),
                hour / 23.0,
                random.randrange(7) / 6.0
            ])
        target = (seq[-1] + 1) % classes
        if random.random() < 0.12:
            target = random.randrange(classes)
        xs.append(seq)
        fs.append(features)
        ys.append(target)
    return (torch.tensor(xs, dtype=torch.long),
            torch.tensor(fs, dtype=torch.float32),
            torch.tensor(ys, dtype=torch.long))

def main():
    x, f, y = build_demo_data()
    loader = DataLoader(TensorDataset(x, f, y), batch_size=64, shuffle=True)
    model = ActivityTransformer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(8):
        total = 0.0
        for xb, fb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb, fb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"Epoch {epoch + 1}/8 - loss: {total / len(loader):.4f}")
    path = "activity_transformer.pth"
    torch.save(model.state_dict(), path)
    print(f"Saved model: {path}")

if __name__ == "__main__":
    main()
