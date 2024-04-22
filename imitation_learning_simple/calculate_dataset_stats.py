import torch
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import ToTensor
import numpy as np
from pathlib import *
import os
import tqdm
import matplotlib.pyplot as plt
import cv2

from utils.StackingDataset import StackingDataset


def calculate_mean_std(): 
    # Define dataset and transforms
    transform = None # Convert images to tensors

    curr_dir = os.path.dirname( os.path.abspath(__file__) )
    path = Path(curr_dir).parent.absolute()
    dataset_path = os.path.join(path, "nanodrones_sim")
    dataset_path = os.path.join(dataset_path, "data")
    dataset = StackingDataset(csv_dir=dataset_path, transform=transform)

    depth_channel_sum = torch.zeros(1)
    depth_channel_sum_sq = torch.zeros(1)
    pencil_channel_sum = torch.zeros(1)
    pencil_channel_sum_sq = torch.zeros(1)
    pixel_count = 0

    label_sum = torch.zeros(4)
    label_sum_sq = torch.zeros(4)
    label_count = 0
    
    for i in tqdm.tqdm(len(dataset)):
        img, label = dataset[i]
        pencil_image = img[0:1, :, :]
        depth_image = img[1:, :, :]

        # Compute cumulative sum of pixel values and pixel count
        pencil_channel_sum += torch.sum(pencil_image, dim=(1, 2))
        pencil_channel_sum_sq += torch.sum(torch.square(pencil_image), dim=(1, 2))
        depth_channel_sum += torch.sum(depth_image, dim=(1, 2))
        depth_channel_sum_sq += torch.sum(torch.square(depth_image), dim=(1, 2))
        pixel_count += img.shape[1] * img.shape[2]

        label_sum += label
        label_sum_sq += torch.square(label)
        label_count +=1
        if label_count > 57000:
            break


    # Calculate mean and standard deviation for each channel
    mean = torch.Tensor([depth_channel_sum, pencil_channel_sum]) / pixel_count
    std = torch.sqrt(torch.Tensor([depth_channel_sum_sq, pencil_channel_sum_sq]) / pixel_count - torch.square(mean))

    label_mean = label_sum / label_count
    label_std = torch.sqrt(label_sum_sq / label_count - torch.square(label_mean))

    return mean, std, label_mean, label_std

# Example usage:
if __name__ == "__main__":
    # Calculate mean and standard deviation
    mean, std, out_mean, out_std = calculate_mean_std()
    to_save = {}
    to_save['pencil'] = {'mean': mean[0], "std": std[0]}
    to_save['depth'] = {'mean': mean[1], "std": std[1]}

    # Print mean and standard deviation for each channel
    print("Mean of each channel:", mean)
    print("Standard deviation of each channel:", std)
    print("Mean of labels: ", out_mean)
    print("Standard deviation of labels: ", out_std)
