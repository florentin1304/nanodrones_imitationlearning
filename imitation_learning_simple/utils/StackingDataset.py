import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from utils.pencil_filter import PencilFilter

class StackingDataset(Dataset):
    def __init__(self, csv_dir, transform=None, pencil=True):
        """
        Args:
            csv_dir (string): Path to the directory containing CSV files with image paths and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_dir = csv_dir
        self.csv_files = [os.path.join(csv_dir, file) for file in os.listdir(csv_dir) if file.endswith('.csv')]
        self.transform = transform

        self.pencil_filter = PencilFilter() if pencil else None

        # Initialize an empty DataFrame with columns
        self.data_frame = pd.DataFrame()


        # Concatenate all CSV files into a single DataFrame
        for file in self.csv_files:
            df = pd.read_csv(file)
            self.data_frame = pd.concat([self.data_frame, df], ignore_index=True)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pencil_name = self.data_frame.loc[idx, 'pencil_img']
        pencil_path = os.path.join(self.csv_dir, "images", pencil_name+".png") 
        pencil_image = Image.open(pencil_path)
        pencil_image = np.array(pencil_image)
        if len(pencil_image) != 3:
            pencil_image = np.expand_dims(pencil_image, axis=0)

        # Read depth image
        depth_name = self.data_frame.loc[idx, 'depth_img']
        depth_path = os.path.join(self.csv_dir, "images", depth_name+".png") 
        depth_image = Image.open(depth_path)
        depth_array = np.array(depth_image)
        if len(depth_array) != 3:
            depth_array = np.expand_dims(depth_array, axis=0)
        
        # Stack depth image as a new channel
        final_array = np.concatenate([pencil_image, depth_array], axis=0)

        # Apply transformations if provided
        final_array = torch.Tensor(final_array) 
        if self.transform:
            final_array = self.transform(final_array)


        label = torch.Tensor([
            self.data_frame.loc[idx, 'alt_command'],
            self.data_frame.loc[idx, 'roll_command'],
            self.data_frame.loc[idx, 'pitch_command'],
            self.data_frame.loc[idx, 'yaw_command']
        ])
        
        return final_array, label