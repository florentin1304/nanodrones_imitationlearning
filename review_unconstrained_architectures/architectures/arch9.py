import torch
import torch.nn as nn

class Arch9(nn.Module):
    def __init__(self):
        super(Arch9, self).__init__()

        self.input_size = (3, 320, 180)
        self.convolutions = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=2, padding=2), # 320x180x3 -> 160x90x12
            nn.ReLU(),

            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=5, stride=2, padding=2), # 160x90x12 -> 80x45x16
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=20, kernel_size=5, stride=2, padding=2), # 80x45x16 -> 40x23x20
            nn.ReLU(),
            
            nn.Conv2d(in_channels=20, out_channels=24, kernel_size=3, stride=2, padding=1), # 40x23x20 -> 20x12x24
            nn.ReLU(),
            
            nn.Conv2d(in_channels=24, out_channels=28, kernel_size=3, stride=2, padding=1), # 20x12x24 -> 10x6x28
            nn.ReLU(),
            
        ) # After flatten = 1680

        self.mlp = nn.Sequential(
            nn.Linear(in_features=1680, out_features=1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(in_features=1000, out_features=500),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(in_features=500, out_features=4)
        )
    
    def forward(self, x):
        x = self.convolutions(x)
        x = torch.flatten(x)
        x = self.mlp(x)

        return x