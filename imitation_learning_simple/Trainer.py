import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from torchvision.transforms import ToTensor, Normalize, Compose

import numpy as np
import random

from pathlib import *
import os
import tqdm
import wandb
import argparse
import json
import copy

from models.full_model import FullModel
from utils.StackingImageDataset import StackingImageDataset
from utils.MaskedMSELoss import MaskedMSELoss
from utils.PerformanceCalculator import PerformanceCalculator


class Trainer:
    def __init__(self, args):
        self.args = args

        ### Init Wandb
        self.run_name = ""
        self.run_name += str(self.args.visual_extractor) + "_"
        self.run_name += str(self.args.input_type) + "_"
        self.run_name += str(self.args.label_type) + "_"
        self.run_name += "h" + str(self.args.hist_size) + "_"
        self.group_name = copy.deepcopy(self.run_name)

        self.run_name += f"seed{self.args.seed}" + "_" + self.get_random_string(5)

        wandb.init(
            mode=self.args.wandb_mode,
            project="nanodrones_imitation_learning",
            entity="udrea-florentin00",
            name=self.run_name,
            group=self.group_name,
            config= vars(self.args)
        )

        ### Get important paths
        self.curr_dir = os.path.dirname( os.path.abspath(__file__) )
        self.home_path = Path(self.curr_dir).parent.absolute()
        self.dataset_path = os.path.join(self.home_path,"nanodrones_sim" , "data")
        self.results_path = os.path.join(self.home_path, "imitation_learning_simple", "results")
        self.output_path = os.path.join(self.results_path, self.run_name)
        os.makedirs(self.output_path, exist_ok=True)

        ### Set seed
        self.set_seed(self.args.seed)

        ### Define Model
        self.model = FullModel(visual_fe=self.args.visual_extractor, 
                               history_len=self.args.hist_size, 
                               output_size=4)
        input_shape = self.model.vfe.get_input_shape()

        ### Get dataset and split
        transform = None
        self.dataset = StackingImageDataset(
                    max_hist=self.args.hist_size,
                    csv_dir=self.dataset_path, 
                    force_stats=self.args.force_data_stats,
                    transform=transform,
                    input_type=self.args.input_type,
                    label_type=self.args.label_type,
                    input_shape=input_shape,
                    norm_input=not(self.args.avoid_input_normalization),
                    norm_label=not(self.args.avoid_label_normalization)
                )

        ### TODO: Set different transformation, altrimenti non ha senso copiare i dataset
        self.train_dataset = copy.copy(self.dataset)
        self.val_dataset = copy.copy(self.dataset)
        self.test_dataset = copy.copy(self.dataset)

        train_ratio = 0.7
        val_ratio = 0.2

        len_dataset = len(self.dataset)
        max_index_train = int(train_ratio*len_dataset)
        max_index_val = int((train_ratio + val_ratio)*len_dataset)

        all_indexes = np.arange(len_dataset)
        train_indexes = all_indexes[:max_index_train]
        val_indexes = all_indexes[max_index_train : max_index_val]
        test_indexes = all_indexes[max_index_val:]

        train_sampler = SubsetRandomSampler(train_indexes)
        val_sampler = SubsetRandomSampler(val_indexes)
        test_sampler = SubsetRandomSampler(test_indexes)

        # Create DataLoader instances for train, validation, and test sets
        self.train_loader = DataLoader(self.train_dataset, 
                                       collate_fn=StackingImageDataset.padding_collate_fn, 
                                       batch_size=self.args.batch_size, 
                                       sampler=train_sampler,
                                       num_workers=self.args.num_workers)
        self.val_loader = DataLoader(self.val_dataset, 
                                     collate_fn=StackingImageDataset.padding_collate_fn, 
                                     sampler=val_sampler,
                                     batch_size=self.args.batch_size,
                                     num_workers=self.args.num_workers)
        self.test_loader = DataLoader(self.test_dataset, 
                                      collate_fn=StackingImageDataset.padding_collate_fn,
                                      sampler=test_sampler,
                                      batch_size=self.args.batch_size)

        # Load weights if necessary
        if self.args.load_model != "":
            if not(self.args.load_model.endswith(".pth") or self.args.load_model.endswith(".pt")):
                raise Exception("Weights file should end with .pt or .pth")
                
            self.model.load_state_dict(
                torch.load( os.join.path(self.output_path, self.args.load_model) )
            )

        ### Get device
        self.device = torch.device(
                    "cuda" if (torch.cuda.is_available() and not self.args.disable_cuda) else "cpu"
                )
        self.model.to(self.device)
        print(f"Working on {self.device}")

        # Define criterion
        self.criterion = MaskedMSELoss()  # Mean Squared Error loss for regression

    def train(self):
        print("=== Start training ===")
        print(f"Batch size: {self.args.batch_size}")
        print(f"Hist size: {self.args.hist_size}")
        # Define loss function and optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 
                                                    gamma=self.args.lr_scheduler_gamma, 
                                                    step_size=self.args.lr_scheduler_step)

        best_val_loss = float('inf')
        
        for epoch in range(self.args.num_epochs):
            ### Run epoch
            print( "="*25, f"EPOCH {epoch}", "="*25)
            epoch_loss = self.run_epoch()
            self.scheduler.step()
            print(f"Epoch [{epoch+1}/{self.args.num_epochs}], Train Loss: {epoch_loss:.4f}")

            ### Run validation
            validation_loss, mse, r2 = self.run_validation()            
            print(f"Validation Loss: {validation_loss:.4f}")


            ### Wandb logs
            wandb.log({"train_loss": epoch_loss,
                    "val_loss": validation_loss, 
                    "epoch": epoch,
                    "mse": sum(mse)/len(mse),
                    "r2": sum(r2)/len(r2)})
            
            ### Save model if best and early stopping
            torch.save(self.model.state_dict(), os.path.join(self.output_path, "last.pth"))
            if validation_loss < best_val_loss:
                print(f"Saving new model [New best loss {validation_loss:.4} vs Old best loss {best_val_loss:.4}]")
                best_val_loss = validation_loss
                waiting_epochs = 0
                torch.save(self.model.state_dict(), os.path.join(self.output_path, "best.pth"))
            else:
                waiting_epochs += 1
                if waiting_epochs >= self.args.patience_epochs:
                    print(f"Early stopping because ran more than {self.args.patience_epochs} without improvement")
                    break

    def run_epoch(self):
        self.model.train()
        
        pbar = tqdm.tqdm(self.train_loader)
        running_loss = 0
        num_samples = 0

        for images, labels, mask in pbar:
            pbar.set_description(f"Running loss: {running_loss/(num_samples+1e-5) :.4}")

            images, labels, mask = images.to(self.device), labels.to(self.device), mask.to(self.device)
            outputs = self.model(images)

            self.optimizer.zero_grad()
            loss = self.criterion(outputs, labels, mask)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            num_samples += self.args.batch_size

        return running_loss / (num_samples+1e-5)

    def run_validation(self):
        self.model.eval()
        performance = PerformanceCalculator()
        with torch.no_grad():
            pbar = tqdm.tqdm(self.val_loader)
            running_loss = 0
            num_samples = 0
            for images, labels, mask in pbar:
                pbar.set_description(f"Validation loss: {running_loss/(num_samples+1e-5) :.4}")

                images, labels, mask = images.to(self.device), labels.to(self.device), mask.to(self.device)

                outputs = self.model(images)

                self.optimizer.zero_grad()
                loss = self.criterion(outputs, labels, mask)
                running_loss += loss.item() * images.size(0)
                num_samples += self.args.batch_size

                ###
                for i in range(len(outputs)):
                    performance.extend(outputs[i], labels[i]) 
                ###

        print(performance)
        performance.plot(self.output_path, name="last_validation_outputs")
        return running_loss / (num_samples+1e-5), performance.mse(), performance.r2()
    
    def test(self):
        self.model.eval()
        performance = PerformanceCalculator()
        with torch.no_grad():
            pbar = tqdm.tqdm(self.test_loader)
            running_loss = 0
            num_samples = 0
            for images, labels, mask in pbar:
                pbar.set_description(f"Test loss: {running_loss/(num_samples+1e-5) :.4}")

                images, labels, mask = images.to(self.device), labels.to(self.device), mask.to(self.device)

                outputs = self.model(images)

                self.optimizer.zero_grad()
                loss = self.criterion(outputs, labels, mask)
                running_loss += loss.item() * images.size(0)
                num_samples += self.args.batch_size

                ###
                for i in range(len(outputs)):
                    performance.extend(outputs[i], labels[i]) 
                ###

        print(performance)
        performance.plot(self.output_path, name="test_ouputs")
        return running_loss / (num_samples+1e-5), performance.mse(), performance.r2()
    
    
    def get_random_string(self, n: int):
        random_string = ''
        for _ in range(n):
            if np.random.uniform() < 0.5:
                i = np.random.randint(65, 91)
            else:
                i = np.random.randint(97, 123)

            random_string += chr(i)
        return random_string

    def set_seed(self, random_seed):
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        return torch.Generator().manual_seed(random_seed)

    def compute_dataset_stats(self):
        # TODO: HARDCODED!!! -> Rifare in modo che lo faccia per tutte le colonne, in modo generale
        # fa popo cacare cosi oh
        print("Computing dataset statistics...")
        dataset = StackingImageDataset(csv_dir=self.dataset_path, norm_input=False, norm_output=False, transform=None, max_hist=-1)


        depth_channel_sum = torch.zeros(1)
        depth_channel_sum_sq = torch.zeros(1)
        pencil_channel_sum = torch.zeros(1)
        pencil_channel_sum_sq = torch.zeros(1)
        pixel_count = 0

        label_sum = torch.zeros(4)
        label_sum_sq = torch.zeros(4)
        label_count = 0
        
        for i in tqdm.tqdm(range(len(dataset))):
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


        # Calculate mean and standard deviation for each channel
        mean = torch.Tensor([depth_channel_sum, pencil_channel_sum]) / pixel_count
        std = torch.sqrt(torch.Tensor([depth_channel_sum_sq, pencil_channel_sum_sq]) / pixel_count - torch.square(mean))

        label_mean = label_sum / label_count
        label_std = torch.sqrt(label_sum_sq / label_count - torch.square(label_mean))

        statistics = {
            "depth_img": { 
                "mean": [mean[0].item()], 
                "std": [std[0].item()] 
                },
            "pencil_img": { 
                "mean": [mean[1].item()], 
                "std": [std[1].item()] 
                },
            "alt_command": { 
                "mean": [label_mean[0].item()], 
                "std": [label_std[0].item()] 
                },
            "roll_command": { 
                "mean": [label_mean[1].item()], 
                "std": [label_std[1].item()] 
                },
            "pitch_command": { 
                "mean": [label_mean[2].item()], 
                "std": [label_std[2].item()] 
                },
            "yaw_command": { 
                "mean": [label_mean[3].item()], 
                "std": [label_std[3].item()] 
                }
        }

        os.makedirs(self.datastats_path, exist_ok=True)
        with open(os.path.join(self.datastats_path, self.args.stats_file_name), 'w', encoding='utf-8') as f:
            json.dump(statistics, f, ensure_ascii=False, indent=4)




