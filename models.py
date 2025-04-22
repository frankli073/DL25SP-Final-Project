import torch
from torch import nn
from typing import List
import numpy as np


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class SimpleEncoder(nn.Module):
    def __init__(self, repr_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1),  # 65 → 33
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 33 → 17
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 17 → 9
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 9 * 9, repr_dim)  # fixed from 4096 to 5184

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class SimplePredictor(nn.Module):
    def __init__(self, repr_dim=256, action_dim=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(repr_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, repr_dim)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.mlp(x)


class JEPA(nn.Module):
    def __init__(self, repr_dim=256):
        super().__init__()
        self.encoder = SimpleEncoder(repr_dim)
        self.target_encoder = SimpleEncoder(repr_dim)
        self.predictor = SimplePredictor(repr_dim)
        self.repr_dim = repr_dim

    def forward(self, states, actions):
        B, T, C, H, W = states.shape
        device = states.device
        pred_states = []

        # Encode s0
        s = self.encoder(states[:, 0])  # [B, D]
        pred_states.append(s)

        for t in range(T - 1):
            s = self.predictor(s, actions[:, t])  # [B, D]
            pred_states.append(s)

        return torch.stack(pred_states, dim=1)  # [B, T, D]

    def compute_target_embeddings(self, states):
        B, T, C, H, W = states.shape
        targets = [self.target_encoder(states[:, t]) for t in range(T)]
        return torch.stack(targets, dim=1)  # [B, T, D]

