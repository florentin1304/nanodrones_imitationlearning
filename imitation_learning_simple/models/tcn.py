import torch
import torch.nn as nn
import pytorch_tcn
from models.resnet import ResNet, BasicBlock


class TCN(nn.Module):
    def __init__(self):
        super(TCN, self).__init__()
        
        self.image_feature_extractor = ResNet(BasicBlock, [2, 2, 2, 2])
        self.image_feature_extractor.fc = nn.Identity()
        self.tcn = pytorch_tcn.TCN(
                num_inputs = 512,
                num_channels = [256,256,256,256,256],
                kernel_size=2,
                dilation_reset = 16,
                input_shape='NLC' # batch, timesteps, features
        )
        self.regressor = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.25),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.25),

            nn.Linear(128,4)
        )

    def forward(self, x):
        ### x = (BATCH, TIME_STAMP, CHANNEL, H, W)
        # features_output = []
        # for batch in x:
        #     batch_output = self.image_feature_extractor(batch)
        #     features_output.append(batch_output)

        # # features_output_tensor = (BATCH, TIMESTAMP, FEATURES)
        # features_output_tensor = torch.stack(features_output,dim=0) 

        # METHOD2
        ### x = (BATCH, TIME_STAMP, CHANNEL, H, W)
        old_x_shape = x.shape
        new_x_shape = (-1, *x.shape[-3:])

        x_reshaped = x.reshape(new_x_shape)

        features_output_tensor = self.image_feature_extractor(x_reshaped)
        features_output_tensor = features_output_tensor.reshape((*old_x_shape[:2], 512))

        # temporal_features_tensor = (BATCH, TIMESTAMP, FEATURES)
        temporal_features_tensor = self.tcn(features_output_tensor)

        last_ts = temporal_features_tensor[:, -1 , :]

        output = self.regressor(last_ts)

        return output

    def print_devices(self):
        print(list(self.image_feature_extractor.parameters())[0].device)
        print(list(self.tcn.parameters())[0].device)
        print(list(self.regressor.parameters())[0].device)

