import torch
import torch.nn as nn

class PencilNet(nn.Module):
    def __init__(self):
        super(PencilNet, self).__init__()

        self.input_size = (1, 160, 120)
        self.convolutions = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), # entering 1x160x120
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # entering 16x80x60
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1), # entering 32x40x30
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1), # entering 16x20x15
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1), # entering 16x10x7
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1), # entering 16x5x3 (and exiting 16x5x3)

        ) # After flatten = 240

        self.mlp = nn.Sequential(
            nn.Linear(240, 60)
            )
    
    def forward(self, x):
        x = self.convolutions(x)
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        x = torch.reshape(x, shape=(3, 4, 5))

        return x