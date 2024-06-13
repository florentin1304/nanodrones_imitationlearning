import os
import torch
import pandas as pd
import numpy as np
import json
from PIL import Image

from utils.ImageDataset import ImageDataset

class StackingImageDataset(ImageDataset):
    def __init__(self,  
                 max_hist=10, *args, **kwargs):
        
        super(StackingImageDataset, self).__init__(*args, **kwargs)

        self.max_hist = max_hist

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        run_id = super().get_row(idx)['run_name']

        image_list = [image]
        label_list = [label]
        for i in range(0, self.max_hist):
            hist_index = idx-(i+1)
            if hist_index < 0:
                break

            image, label = super(StackingImageDataset, self).__getitem__(hist_index)
            hist_run_id = super(StackingImageDataset, self).get_row(hist_index)['run_name']
            
            if run_id != hist_run_id:
                break
            
            image_list.append(image)
            label_list.append(label)
        
        # Before reverse we have [idx, idx-1, idx-2, ...]
        # After it will be in temporal order
        image_list.reverse()
        label_list.reverse()

        return torch.stack(image_list), torch.stack(label_list)

    @staticmethod
    def padding_collate_fn(batch):
        # data = (BATCH, TIMESTEP, C, H, W)
        data, labels = zip(*batch)

        # get shapes
        batch_size = len(batch)
        max_time_len = max([d.shape[0] for d in data])
        image_shape = data[0].shape[1:] # skip TIME 
        labels_shape = labels[0].shape[1:] # skip TIME

        # allign data with respect to max sequence len
        data_alligned = torch.zeros((batch_size, max_time_len, *image_shape))
        labels_alligned = torch.zeros((batch_size, max_time_len, *labels_shape))

        # 0 where we , FLO is happier this way
        mask = torch.zeros((len(batch), max_time_len))

        # fill with meaningfull data
        for i, d in enumerate(data):
            # B, T, C, H, W
            data_alligned[i, -d.shape[0]:, :, :, :] = d
            
            # B, T, L
            l = labels[i]
            labels_alligned[i, -l.shape[0]:, :] = l

            # B, T
            mask[i, -d.shape[0]:] = 1

        return data_alligned, labels_alligned, mask
    

# if __name__ == "__main__":
#     from torch.utils.data import DataLoader
#     ds = StackingImageDataset(max_hist=4, csv_dir="/home/florentin/Documents/repos/nanodrones_imitationlearning/nanodrones_sim/data")
#     dl = DataLoader(ds, collate_fn=StackingImageDataset.padding_collate_fn, batch_size=5)
#     for d, l, m in dl:
#         print(l)
#         print(m)
#         break
