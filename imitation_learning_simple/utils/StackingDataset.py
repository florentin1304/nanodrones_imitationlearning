import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from utils.pencil_filter import PencilFilter
from torchvision.transforms import Resize

class StackingDataset(Dataset):
    def __init__(self, 
                 csv_dir, 
                 stats_dict=None,
                 norm_input=True,
                 norm_output=True,
                 transform=None, 
                 resize_size=84, 
                 max_hist=10):
        """
        Args:
            csv_dir (string): Path to the directory containing CSV files with image paths and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.stats_dict = stats_dict
        self.norm_input = norm_input
        self.norm_output = norm_output
        self.max_hist = max_hist


        self.csv_dir = csv_dir
        self.csv_files = [os.path.join(csv_dir, file) for file in os.listdir(csv_dir) if file.endswith('.csv')]
        self.transform = transform

        self.resize_size = resize_size
        self.resize = Resize((resize_size, resize_size))
        
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

        if self.max_hist <= 0 or self.max_hist is None:
            output = self.getFrameIdx(idx)
        else:
            output = self.getStackedHist(idx)

        label = torch.Tensor([
            self.data_frame.loc[idx, 'alt_command'],
            self.data_frame.loc[idx, 'roll_command'],
            self.data_frame.loc[idx, 'pitch_command'],
            self.data_frame.loc[idx, 'yaw_command']
        ])
        if self.norm_output:
            # TODO: HARDCODED, RENDERE GENERALE
            output_names = ["alt_command", "roll_command", "pitch_command", "yaw_command"]
            label_mean = torch.Tensor([self.stats_dict[x]['mean'][0] for x in output_names])
            label_std = torch.Tensor([self.stats_dict[x]['std'][0] for x in output_names])
            label = (label - label_mean) / label_std
            
        return output, label

    def getFrameIdx(self, idx):
            pencil_name = self.data_frame.loc[idx, 'pencil_img']
            pencil_path = os.path.join(self.csv_dir, "images", pencil_name+".png") 
            pencil_image = Image.open(pencil_path)
            pencil_image = np.array(pencil_image)
            if self.norm_input:
                pencil_image = (pencil_image - np.array(self.stats_dict['pencil_img']['mean'])) / np.array(self.stats_dict['pencil_img']['std']) 
            if len(pencil_image) != 3:
                pencil_image = np.expand_dims(pencil_image, axis=0)

            # Read depth image
            depth_name = self.data_frame.loc[idx, 'depth_img']
            depth_path = os.path.join(self.csv_dir, "images", depth_name+".png") 
            depth_image = Image.open(depth_path)
            depth_array = np.array(depth_image)
            if self.norm_input:
                depth_array = (depth_array - np.array(self.stats_dict['depth_img']['mean'])) / np.array(self.stats_dict['depth_img']['std']) 
            if len(depth_array) != 3:
                depth_array = np.expand_dims(depth_array, axis=0)
            
            # Stack depth image as a new channel 
            # shape = (2, H, W)
            final_array = np.concatenate([pencil_image, depth_array], axis=0)

            # Apply transformations if provided
            final_array = torch.as_tensor(final_array, dtype=torch.float) 
            if self.transform:
                final_array = self.transform(final_array)
            
            final_array = self.resize(final_array)
            return final_array
    
        
    def getStackedHist(self, idx):
        timeframes_tensor = []

        for i in range(1-self.max_hist, 1):
            # i is negative => going from 1-self.max_hist to 0, included (eg. -31 to 0, total 32)
            if self.isInvalidHistory(idx, idx+i):
                padding = torch.zeros(size=(2, self.resize_size, self.resize_size))
                timeframes_tensor.append(padding)
                continue

            tf_output = self.getFrameIdx(idx+i)
            tf_output.unsqueeze(dim=0)
            timeframes_tensor.append(tf_output)

        # shape = (max_hist, 2, H, W)
        timeframes_tensor = torch.stack(timeframes_tensor, dim=0)

        return timeframes_tensor

    def isInvalidHistory(self, idx, idx_hist):
        if idx_hist < 0: 
            return True
        if self.data_frame.loc[idx_hist, "index"] > self.data_frame.loc[idx, "index"] or \
            self.data_frame.loc[idx_hist, "run_name"] != self.data_frame.loc[idx, "run_name"]:
            return True
        return False
    
    def getDataFrame(self):
        return self.data_frame