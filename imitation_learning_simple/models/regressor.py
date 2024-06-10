import numpy as np
import torch
import torch.nn as nn

class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_dims, output_size, dropout_prob=0.5):
        super(MLPRegressor, self).__init__()
        
        layers = []
        current_size = input_size
        
        for hidden_size in hidden_dims:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            current_size = hidden_size
            
        layers.append(nn.Linear(current_size, output_size))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

