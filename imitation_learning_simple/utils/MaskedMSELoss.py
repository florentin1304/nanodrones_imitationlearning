import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, outputs, targets, mask):
        # Check that the dimensions match
        assert outputs.shape == targets.shape, "outputs and targets must have the same shape"
        assert mask.shape == (outputs.shape[0], outputs.shape[1]), "mask must have shape (batch_size, timesteps)"
        
        # Apply the mask to the outputs and targets
        mask = mask.unsqueeze(-1)  # Expand mask to match the dimensions of outputs and targets
        
        # Compute MSE loss
        loss = F.mse_loss(outputs, targets, reduction='none')  # Compute element-wise MSE
        
        # Apply the mask to the loss
        loss = loss * mask
        
        # Sum the losses for each valid element and divide by the number of valid elements
        loss_sum = loss.sum()
        mask_sum = mask.sum()
        
        # Compute the mean loss over all valid elements
        mean_loss = loss_sum / mask_sum
        
        return mean_loss