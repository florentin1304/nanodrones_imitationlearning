import numpy as np
import torch
from PIL import Image

class AddGaussianNoise(object):
    def __init__(self, mean=0, std=1, prob=0.5):
        self.mean = mean
        self.std = std
        self.prob = prob
    
    def __call__(self, img):
        if torch.rand(1).item() < self.prob:
            img_array = np.array(img)
            noise = np.random.normal(self.mean, self.std, img_array.shape)
            img_array = img_array + noise
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)
        return img