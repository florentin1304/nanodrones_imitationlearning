import numpy as np
import torch
import torch.nn as nn
import torchvision

class MobileNetv2(nn.Module):
    def __init__(self):
        super(MobileNetv2, self).__init__()
        self.input_channels = 3
        self.width = 224
        self.height = 224

        self.mobilenet_v2 = torchvision.models.mobilenet_v2(
                weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
            )
        self.mobilenet_v2 = self.mobilenet_v2.features 

        self.input_shape = (self.input_channels, self.height, self.width)
        self.output_shape = self.__get_output_size()


    def forward(self, x):
        x = self.mobilenet_v2(x)
        x = x.mean([-2, -1])  # Global average pooling
        return x


    def __get_output_size(self):
        device = next(self.parameters()).device

        rand_vec = torch.rand(size=(self.input_channels, self.height, self.width)).to(device=device)
        rand_vec = rand_vec.unsqueeze(0)
        output_shape = self.forward(rand_vec).shape
        output_shape = output_shape[1:]

        return output_shape

if __name__ == "__main__":
    model = MobileNetv2()
    print(model.output_shape)


