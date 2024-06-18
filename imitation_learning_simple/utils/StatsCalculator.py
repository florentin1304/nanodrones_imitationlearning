import numpy as np
import pandas as pd
import os
import tqdm
import json
from PIL import Image
import cv2

class StatsCalculator:
    def __init__(self, df: pd.DataFrame, header_columns: list, csv_dir: str):
        self.df = df
        self.header_columns = header_columns
        self.csv_dir = csv_dir
        self.stats = {}

    def run(self, every_i_images=10):
        columns = [x for x in self.df.columns if x not in self.header_columns]
        columns_table = [x for x in columns if not x.endswith("img")]
        columns_images = [x for x in columns if x.endswith("img")]

        pencil_channel_sum = 0
        pencil_channel_sum_sq = 0
        pixel_count = 0
        for idx in tqdm.tqdm(range(0,  len(self.df[columns_images]['pencil_img']), every_i_images), desc="Computing pencil_img statistics"):
            pencil_name = self.df[columns_images]['pencil_img'].iloc[idx]
            pencil_path = os.path.join(self.csv_dir, "images", pencil_name+".png") 
            pencil_image = Image.open(pencil_path)
            pencil_image = np.array(pencil_image, dtype=float)
            pencil_image = np.expand_dims(pencil_image, axis=0)
            
            # Compute cumulative sum of pixel values and pixel count
            pencil_channel_sum += np.sum(pencil_image)
            pencil_channel_sum_sq += np.sum(np.square(pencil_image))

            pixel_count += pencil_image.shape[1] * pencil_image.shape[2]
            
        pencil_mean = pencil_channel_sum / pixel_count
        pencil_std = np.sqrt((pencil_channel_sum_sq / pixel_count) - np.square(pencil_mean))
        self.stats['pencil_img'] = {'mean':pencil_mean, 'std':pencil_std, 'min':0, 'max':255}

        depth_channel_sum = 0
        depth_channel_sum_sq = 0
        pixel_count = 0
        for idx in tqdm.tqdm(range(0,  len(self.df[columns_images]['depth_img']), every_i_images), desc="Computing depth_img statistics"):
            depth_name = self.df[columns_images]['depth_img'].iloc[idx]
            depth_path = os.path.join(self.csv_dir, "images", depth_name+".png") 
            depth_image = Image.open(depth_path)
            depth_image = np.array(depth_image, dtype=float)
            depth_image = np.expand_dims(depth_image, axis=0)
            
            # Compute cumulative sum of pixel values and pixel count
            depth_channel_sum += np.sum(depth_image)
            depth_channel_sum_sq += np.sum(np.square(depth_image))
            pixel_count += depth_image.shape[1] * depth_image.shape[2]

        img_grayscale_channel_sum = 0
        img_grayscale_channel_sum_sq = 0
        pixel_count = 0
        for idx in tqdm.tqdm(range(0,  len(self.df[columns_images]['depth_img']), every_i_images), desc="Computing grayscale_img statistics"):
            image_name = self.df[columns_images]['camera_img'].iloc[idx]
            image_path = os.path.join(self.csv_dir, "images", image_name+".png") 
            image_grayscale = Image.open(image_path).convert("RGB")
            image_grayscale = np.array(image_grayscale)
            image_grayscale = cv2.cvtColor(image_grayscale, cv2.COLOR_RGB2GRAY)
            image_grayscale = np.array(image_grayscale, dtype=float)
            image_grayscale = np.expand_dims(image_grayscale, axis=0)
            
            # Compute cumulative sum of pixel values and pixel count
            img_grayscale_channel_sum += np.sum(image_grayscale)
            img_grayscale_channel_sum_sq += np.sum(np.square(image_grayscale))
            pixel_count += image_grayscale.shape[1] * image_grayscale.shape[2]
            
        img_grayscale_mean = depth_channel_sum / pixel_count
        img_grayscale_std = np.sqrt((depth_channel_sum_sq / pixel_count) - np.square(img_grayscale_mean))
        self.stats['grayscale_img'] = {'mean':img_grayscale_mean, 'std':img_grayscale_std}
        
        # RGB columns
        self.stats['camera_img'] = {
            'mean': [0.485, 0.456, 0.406],
            'std':[0.229, 0.224, 0.225]
        }

        # Table columns
        table_statistics = self.df[columns_table].describe()
        for col in table_statistics.columns:
            col_statistics = table_statistics[col]
            self.stats[col] = {
                'mean':col_statistics['mean'], 
                'std':col_statistics['std'], 
                'min':col_statistics['min'], 
                'max':col_statistics['max']}
        

            
        return self.stats
            


        