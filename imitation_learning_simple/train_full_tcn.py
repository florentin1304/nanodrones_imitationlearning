import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor, Normalize, Compose

from pathlib import *
import argparse

from utils.StackingDataset import StackingDataset  # Assuming you saved the custom dataset class in a file named custom_dataset.py
from models.resnet import ResNet18
from models.tcn import TCN
from Trainer import Trainer

def main(args):
    t = Trainer(args)
    t.train()

    # Define dataset and transforms
    # Mean of each channel: tensor([174.7335,   0.9116])
    # Standard deviation of each channel: tensor([108.9636,   0.1411])
    # Mean of labels:  tensor([ 5.5213e+01,  6.9066e-04, -5.0562e-03, -2.7119e-02])
    # Standard deviation of labels:  tensor([1.5544, 0.2182, 0.2224, 0.2800])
    # transform = Compose([
    #     Normalize(mean=[174.7335,   0.9116], std=[108.9636,   0.1411])
    # ])

    # output_transform = {
    #     "mean": torch.Tensor([ 5.5213e+01,  6.9066e-04, -5.0562e-03, -2.7119e-02]),
    #     "std": torch.Tensor([1.5544, 0.2182, 0.2224, 0.2800])
    # }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ### Training arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Size of the batch for training")
    parser.add_argument("--hist_size", type=int, default=32, help="Number of timesteps to stack when getting data from dataset (only applicable if `tcn_*` model is used )")
    parser.add_argument("--model", type=str, default="tcn_default", choices=["tcn_default", "resnet_default"], help="Type of model to use: resnet (no time information), tcn (time information)")

    parser.add_argument("--stats_file_name", type=str, default="stats.json", help="Name of the file containing the statistics (mean, std) of each column in dataset")
    parser.add_argument("--force_data_stats", action="store_true", help="Recompute the dataset stats even if config is found")
    parser.add_argument("--avoid_input_normalization", action="store_true", help="Stop input normalization")
    parser.add_argument("--avoid_output_normalization", action="store_true", help="Stop output normalization")

    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs to train the model")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Starting learning rate")
    parser.add_argument("--lr_scheduler_gamma", type=float, default=0.3, help="Muliply the learning rate by the gamma factor every \{args.lr_cheduler_step\} steps")
    parser.add_argument("--lr_scheduler_step", type=int, default=2, help="Every how many epochs apply the gamma to the learning rate")
    parser.add_argument("--patience_epochs", type=int, default=4, help="After how many epochs of not improving the validation score stop the training")

    parser.add_argument("--disable_cuda", action="store_true", help="Even if cuda is available, dont use it")

    # Optimizer arguments 

    # Wandb arguments
    

    args = parser.parse_args()
    main(args)