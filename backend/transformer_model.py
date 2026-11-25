import torch
from torch import nn

class ActivityTransformer(nn.Module):
    def __init__(self, num_activities=8, num_features=5, d_model=64,
                 nhead=4, num_layers=2, max_len=12):
        super().__init__()
        self.activity_embedding = nn.Embedding(num_activities, 16)
        self.feature_projection = nn.Linear(num_features, 48)
        self.position_embedding = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True,
            dropout=0.1, dim_feedforward=128
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_activities))
        self.max_len = max_len

    def forward(self, activity_ids, features):
        batch, seq_len = activity_ids.shape
        positions = torch.arange(seq_len, device=activity_ids.device).unsqueeze(0)
        x = torch.cat([
            self.activity_embedding(activity_ids),
            self.feature_projection(features)
        ], dim=-1)
        x = x + self.position_embedding(positions)
        x = self.encoder(x)
        return self.classifier(x[:, -1, :])
