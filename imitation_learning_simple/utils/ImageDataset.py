import os
import torch
import pandas as pd
import numpy as np
import json
import warnings
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from utils.StatsCalculator import StatsCalculator

class ImageDataset(Dataset):
    def __init__(self, 
                 csv_dir: str, 
                 force_stats=False,
                 transform=None,
                 input_type="RGB",
                 label_type="commands",
                 image_shape=(320, 320),
                 norm_input=True,
                 norm_label=True):
        """Image dataset

        Args:
            csv_dir (str): _description_
            transform (_type_, optional): _description_. Defaults to None.
            input_type (str, optional): _description_. Defaults to "RGB".
            label_type (str, optional): _description_. Defaults to "commands".
            image_shape (tuple, optional): _description_. Defaults to (320, 320).
            force_stats (bool, optional): _description_. Defaults to False.
            norm_input (bool, optional): _description_. Defaults to True.
            norm_label (bool, optional): _description_. Defaults to True.
        """
        self.csv_dir = csv_dir
        self.csv_files = [os.path.join(csv_dir, file) for file in os.listdir(csv_dir) if file.endswith('.csv')]

        
        # Initialize an empty DataFrame with columns
        self.data_frame = pd.DataFrame()

        # Concatenate all CSV files into a single DataFrame
        for file in self.csv_files:
            df = pd.read_csv(file)
            self.data_frame = pd.concat([self.data_frame, df], axis=0, ignore_index=True)
        self.data_frame.sort_values(by=["run_name", "index"])
        print("Dataset shape: ", self.data_frame.shape)

        ### Data stats
        stats_file_path = os.path.join(csv_dir, "stats.json")
        if not os.path.isfile(stats_file_path) or force_stats:
            print("stats.json not found")
            print("Creating stats file...")
            stats_calc = StatsCalculator(self.data_frame, ["run_name", "index"], csv_dir=csv_dir) 
            self.stats_dict = stats_calc.run()

            with open(stats_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.stats_dict, f, ensure_ascii=False, indent=4)
            print("done")
        else: 
            print("Loading 'stats.json'... ", end='')
            with open(stats_file_path, 'r') as f:
                self.stats_dict = json.load(f) 
            print("done")

        ### SET FUNCTIONS AND FUNCTIONALITIES BASED ON input_type AND output_type
        self.input_type = input_type
        self.label_type = label_type
        self.norm_input = norm_input
        self.norm_label = norm_label
        self.image_shape = image_shape
        self.transform = transform
        self.__configure()

    def __configure(self):
        ### Configure image get and transform
        transformation_list = []
        if self.transform is not None:
            transformation_list.append(self.transform)
        transformation_list.append(ToTensor())
        transformation_list.append(Resize(self.image_shape))

        if self.input_type == "RGB":
            self.get_image = lambda idx: self.__getRGBImage(idx)
            if self.norm_input:
                means = self.stats_dict["camera_img"]['mean']
                std = self.stats_dict["camera_img"]['std']
            
            
                transformation_list.append(Normalize(means, std))
            self.transform_image = Compose(transforms=transformation_list)
        
        elif self.input_type == "2CH":
            self.get_image = lambda idx: self.__get2CHImage(idx)
            if self.norm_input:
                means = [self.stats_dict["pencil_img"]['mean']/255, 0]
                std = [self.stats_dict["pencil_img"]['std']/255, 1]
                transformation_list.append(Normalize(means, std))
                
            self.transform_image = Compose(transforms=transformation_list)
        else:
            raise Exception(f"Unknown {self.input_type=}")
        
        ### Configure labels and transform
        if self.label_type =='commands':
            output_names = ["alt_command", "roll_command", "pitch_command", "yaw_command"]
        elif self.label_type =='d_commands':
            raise Exception(f"Not implemented {self.output_type=}")
            output_names = []
        elif self.label_type =='setpoints':
            raise Exception(f"Not implemented {self.output_type=}")
            output_names = []
        elif self.label_type =='d_setpoints':
            raise Exception(f"Not implemented {self.output_type=}")
            output_names = []
        else:
            raise Exception(f"Unknown {self.output_type=}")

        self.get_label = lambda idx: np.array(self.data_frame.iloc[idx][output_names].values, dtype=np.float32)

        if self.norm_label:
            label_mean = np.array([self.stats_dict[x]['mean'] for x in output_names])
            label_std = np.array([self.stats_dict[x]['std'] for x in output_names])
            self.transform_label = lambda label: (label - label_mean) / label_std
        else:
            self.transform_label = lambda label: label


    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Following functions change based on self.__configure()
        image = self.get_image(idx)
        image = self.transform_image(image)

        label = self.get_label(idx)
        label = self.transform_label(label)

        return torch.Tensor(image), torch.Tensor(label)
    
    def __getRGBImage(self, idx):
        rgb_name = self.data_frame.loc[idx, 'camera_img']
        rgb_path = os.path.join(self.csv_dir, "images", rgb_name+".png") 
        rgb_image = Image.open(rgb_path)
        rgb_image = np.array(rgb_image)

        return rgb_image
        

    def __get2CHImage(self, idx):
            # Get pencil
            pencil_name = self.data_frame.loc[idx, 'pencil_img']
            pencil_path = os.path.join(self.csv_dir, "images", pencil_name+".png") 
            pencil_image = Image.open(pencil_path)
            pencil_image = np.array(pencil_image)
    
            # Get depth
            depth_name = self.data_frame.loc[idx, 'depth_img']
            depth_path = os.path.join(self.csv_dir, "images", depth_name+".png") 
            depth_image = Image.open(depth_path)
            depth_array = np.array(depth_image)
            
            # Stack depth image as a new channel 
            # shape = (2, H, W)
            final_array = np.stack([pencil_image, depth_array], axis=-1)

            return final_array
    
    def get_row(self, idx):
        return self.data_frame.iloc[idx]

    def getDataFrame(self):
        return self.data_frame
    