import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, action_space):  # Input image, must undergo downsampling
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(4, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2592, 256),
            nn.ReLU(),
            nn.Linear(256, action_space),
        )

    def forward(self, x):

        return self.layers(x / 255)
