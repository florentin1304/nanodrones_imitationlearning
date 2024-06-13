import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor, Normalize, Compose

import numpy as np
import random

from pathlib import *
import os
import tqdm
import wandb
import argparse
import json

from models.full_model import FullModel
from utils.StackingImageDataset import StackingImageDataset
from utils.MaskedMSELoss import MaskedMSELoss


class Trainer:
    def __init__(self, args):
        self.args = args

        ### Get important paths
        self.curr_dir = os.path.dirname( os.path.abspath(__file__) )
        self.home_path = Path(self.curr_dir).parent.absolute()
        self.dataset_path = os.path.join(self.home_path, "nanodrones_sim", "data")
        self.weights_path = os.path.join(self.home_path, "imitation_learning_simple", "weights")
        self.output_path = os.path.join(self.home_path, "imitation_learning_simple", "output")

        ### Set seed
        self.set_seed(self.args.seed)

        ### Get dataset and model type
        transform = None
        self.dataset = StackingImageDataset(
                    max_hist=self.args.hist_size,
                    csv_dir=self.dataset_path, 
                    force_stats=self.args.force_data_stats,
                    transform=transform,
                    input_type=self.args.input_type,
                    label_type=self.args.label_type,
                    image_shape=(320, 320),
                    norm_input=not(self.args.avoid_input_normalization),
                    norm_label=not(self.args.avoid_label_normalization)
                )

        self.model = FullModel(visual_fe=self.args.visual_extractor, 
                               input_type=self.args.input_type,
                               history_len=self.args.hist_size, 
                               output_size=4)
        
        # Load weights if necessary
        if self.args.load_model != "":
            if not(self.args.load_model.endswith(".pth") or self.args.load_model.endswith(".pt")):
                raise Exception("Weights file should end with .pt or .pth")
                
            self.model.load_state_dict(
                torch.load( os.join.path(self.weights_path, self.args.load_model) )
            )

        ### Get device
        self.device = torch.device(
                    "cuda" if (torch.cuda.is_available() and not self.args.disable_cuda) else "cpu"
                )
        self.model.to(self.device)
        print(f"Working on {self.device}")

        # Split dataset into train, validation, and test sets
        train_ratio = 0.7
        val_ratio = 0.2

        train_size = int(train_ratio * len(self.dataset))
        val_size = int(val_ratio * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [train_size, val_size, test_size])

        # Create DataLoader instances for train, validation, and test sets
        self.train_loader = DataLoader(self.train_dataset, 
                                       collate_fn=StackingImageDataset.padding_collate_fn, 
                                       batch_size=self.args.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, 
                                     collate_fn=StackingImageDataset.padding_collate_fn, 
                                     batch_size=self.args.batch_size)
        self.test_loader = DataLoader(self.test_dataset, 
                                      collate_fn=StackingImageDataset.padding_collate_fn, 
                                      batch_size=self.args.batch_size)

        # Define criterion
        self.criterion = MaskedMSELoss()  # Mean Squared Error loss for regression

        ### Init Wandb
        wandb.init(
            mode=self.args.wandb_mode,
            project="nanodrones_imitation_learning",
            entity="udrea-florentin00",
            name="megatest",
            config= vars(self.args)
        )

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
            validation_loss = self.run_validation()            
            print(f"Validation Loss: {validation_loss:.4f}")


            ### Wandb logs
            wandb.log({"train_loss": epoch_loss,
                    "val_loss": validation_loss, 
                    "epoch": epoch})
            
            ### Save model if best and early stopping
            if validation_loss < best_val_loss:
                print(f"Saving new model [New best loss {validation_loss:.4} vs Old best loss {best_val_loss:.4}]")
                best_val_loss = validation_loss
                waiting_epochs = 0
                torch.save(self.model.state_dict(), os.path.join(self.weights_path, "TCN_best.pth"))
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

        wandb.log({"train_loss_running": running_loss / (num_samples+1e-5)})
        return running_loss / (num_samples+1e-5)

    def run_validation(self):
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

        wandb.log({"val_loss": running_loss / (num_samples+1e-5)})
        return running_loss / (num_samples+1e-5)
    
    def run_test(self, save=True, plot_graphs=True):
        # Test the model
        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                num_samples += len(images)
            
        test_loss = total_loss / num_samples
        wandb.log({"test_loss": test_loss})
        print(f"Test Loss: {test_loss:.4f}") 

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




