import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor, Normalize, Compose

from pathlib import *
import os
import tqdm
import wandb
import argparse
import json

from utils.StackingDataset import StackingDataset
from models.resnet import ResNet18
from models.tcn import TCN


class Trainer:
    def __init__(self, args):
        self.args = args

        ### Get important paths
        self.curr_dir = os.path.dirname( os.path.abspath(__file__) )
        self.home_path = Path(self.curr_dir).parent.absolute()
        self.dataset_path = os.path.join(self.home_path, "nanodrones_sim", "data")
        self.weights_path = os.path.join(self.home_path, "imitation_learning_simple", "weights")
        self.datastats_path = os.path.join(self.home_path, "imitation_learning_simple", "dataset_stats")
        self.output_path = os.path.join(self.home_path, "imitation_learning_simple", "output")

        ### Load stats
        if not (self.args.avoid_input_normalization and self.args.avoid_output_normalization):
            stats_file_path = os.path.join(self.datastats_path, self.args.stats_file_name)
            if self.args.force_data_stats or not os.path.isfile(stats_file_path):
                self.compute_dataset_stats()
            
            stats_file = open(stats_file_path, "r") 
            stats = json.load(stats_file)


        ### Get dataset and model type
        transform = None

        if self.args.model == "tcn_default":
            self.dataset = StackingDataset(csv_dir=self.dataset_path, stats_dict=stats, transform=transform, max_hist=self.args.hist_size)
            self.model = TCN()
        elif self.args.model == "resnet_default":
            self.dataset = StackingDataset(csv_dir=self.dataset_path, stats_dict=stats, transform=transform, max_hist=None)
            self.model = ResNet18()
        else:
            raise Exception(f"Model argument {self.args.model} not recognised")
        
        ### Get device
        self.device = torch.device(
                    "cuda" if (torch.cuda.is_available() and not self.args.disable_cuda) else "cpu"
                )
        print(f"Working on {self.device}")

        # Split dataset into train, validation, and test sets
        train_ratio = 0.7
        val_ratio = 0.2

        train_size = int(train_ratio * len(self.dataset))
        val_size = int(val_ratio * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [train_size, val_size, test_size])

        # Create DataLoader instances for train, validation, and test sets
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.args.batch_size)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.args.batch_size)

        # Define criterion
        self.criterion = nn.MSELoss()  # Mean Squared Error loss for regression

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
            print( "="*60, f"EPOCH {epoch}", "="*60)
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
        i = 0

        for images, labels in pbar:
            pbar.set_description(f"Running loss: {running_loss/(i+1e-5) :.4}")

            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)

            self.optimizer.zero_grad()
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            i += self.args.batch_size
            # wandb.log({"train_loss_running": loss.item()})
        return running_loss / (i+1e-5)

    def run_validation(self):
        total_loss = 0.0
        num_samples = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                num_samples += len(images)

        validation_loss = total_loss / num_samples
        return validation_loss

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

    def compute_dataset_stats(self):
        # TODO: HARDCODED!!! -> Rifare in modo che lo faccia per tutte le colonne, in modo generale
        # fa popo cacare cosi oh
        print("Computing dataset statistics...")
        dataset = StackingDataset(csv_dir=self.dataset_path, norm_input=False, norm_output=False, transform=None, max_hist=-1)


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




