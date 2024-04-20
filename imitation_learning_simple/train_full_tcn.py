import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor, Normalize, Compose
from pathlib import *
import os
import tqdm
import wandb

from utils.StackingDataset import StackingDataset  # Assuming you saved the custom dataset class in a file named custom_dataset.py
# from models.simple_cnn import SimpleCNN  # Assuming you saved the custom CNN class in a file named custom_cnn.py
# from models.resnet import ResNet18
from models.tcn import TCN

MAX_HIST_SIZE = 32
BATCH_SIZE = 32

wandb.init(
    # set the wandb project where this run will be logged
    mode="disabled",
    name="TCN",
    project="imitation_learning",
    entity="udrea-florentin00",
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.01,
    "architecture": "TCN",
    "dataset": "100runs-norand",
    "epochs": 20,
    }
)


# Define dataset and transforms
# Mean of each channel: tensor([174.7335,   0.9116])
# Standard deviation of each channel: tensor([108.9636,   0.1411])
# Mean of labels:  tensor([ 5.5213e+01,  6.9066e-04, -5.0562e-03, -2.7119e-02])
# Standard deviation of labels:  tensor([1.5544, 0.2182, 0.2224, 0.2800])
transform = Compose([
    Normalize(mean=[174.7335,   0.9116], std=[108.9636,   0.1411])
])

output_transform = {
    "mean": torch.Tensor([ 5.5213e+01,  6.9066e-04, -5.0562e-03, -2.7119e-02]),
    "std": torch.Tensor([1.5544, 0.2182, 0.2224, 0.2800])
}


curr_dir = os.path.dirname( os.path.abspath(__file__) )
home_path = Path(curr_dir).parent.absolute()
dataset_path = os.path.join(home_path, "nanodrones_sim", "data")
weights_path = os.path.join(home_path, "imitation_learning_simple", "weights")
dataset = StackingDataset(csv_dir=dataset_path, transform=transform, max_hist=MAX_HIST_SIZE)


# Split dataset into train, validation, and test sets
train_ratio = 0.7
val_ratio = 0.2

train_size = int(train_ratio * len(dataset))
val_size = int(val_ratio * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoader instances for train, validation, and test sets
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Initialize the model
model = TCN()

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.005)
scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.3, step_size=3)

# Train the model
num_epochs = 20
waiting_epochs = 0
waiting_epochs_threshold = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device=}")
model.to(device)

best_val_loss = float('inf')
for epoch in range(num_epochs):
    print( "="*80, f"EPOCH {epoch}", "="*80)
    model.train()
    running_loss = 0.0

    pbar = tqdm.tqdm(train_loader)
    i = 0
    for images, labels in pbar:
        pbar.set_description(f"Running loss: {running_loss/(i+1e-5) :.4}")

        labels = (labels - output_transform['mean']) / output_transform['std']
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        i += BATCH_SIZE


    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}")
    scheduler.step()

    # Validate the model
    model.eval()
    total_loss = 0.0
    num_samples = 0
    with torch.no_grad():
        for images, labels in val_loader:
            labels = (labels - output_transform['mean']) / output_transform['std']
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            num_samples += len(images)
    validation_loss = total_loss / num_samples
    print(f"Validation Loss: {validation_loss:.4f}")
    wandb.log({"train_loss": epoch_loss,
               "val_loss": validation_loss, 
               "epoch": epoch})


    if validation_loss < best_val_loss:
        print(f"Saving new model: old loss {best_val_loss} > new loss {validation_loss}")
        best_val_loss = validation_loss
        waiting_epochs = 0
        torch.save(model.state_dict(), os.path.join(weights_path, "TCN_best.pth"))
    else:
        waiting_epochs += 1
        if waiting_epochs >= waiting_epochs_threshold:
            break

# Test the model
model.eval()
total_loss = 0.0
num_samples = 0
with torch.no_grad():
    for images, labels in test_loader:
        labels = (labels - output_transform['mean']) / output_transform['std']
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        num_samples += len(images)
    
    

test_loss = total_loss / num_samples
wandb.log({"test_loss": test_loss})
print(f"Test Loss: {test_loss:.4f}") 
