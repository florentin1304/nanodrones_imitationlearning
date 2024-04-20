import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 8 Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, padding=2), # 324 -> 324
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3), # 324 -> 108
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.2),

            nn.Conv2d(16, 32, kernel_size=5, padding=2), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3), # 108 -> 36
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3), # 36 -> 12
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 12 -> 6
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 6 -> 3
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.25),

            # nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2), # 21 -> 10
            # nn.Conv2d(256, 512, kernel_size=3, padding=1),
            # nn.ReLU(),            
            # nn.MaxPool2d(kernel_size=2, stride=2), # 10 -> 5
            # nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            # nn.ReLU()
        )

        self.feature_extractor_output_size = 256 * 3 * 3
        
        # 3 Fully connected layers for regression
        self.regressor = nn.Sequential(
            nn.Linear(self.feature_extractor_output_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),

            nn.Linear(256, 4)  # Output: 4 values for regression
        )

    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        
        # Flatten before passing to fully connected layers
        x = x.view(-1, self.feature_extractor_output_size)
        
        # Regression
        x = self.regressor(x)
        
        return x
