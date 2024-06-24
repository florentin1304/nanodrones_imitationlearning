import torch
import torch.nn as nn
import pytorch_tcn


class TCN(nn.Module):
    def __init__(self, input_size, hidden_dims=[256,256,256], output_size=None):
        super(TCN, self).__init__()
        
        self.tcn = pytorch_tcn.TCN(
                num_inputs = input_size,
                num_channels = hidden_dims,
                output_projection=output_size,
                kernel_size=2,
                dilation_reset = 8,
                
                input_shape='NLC' # batch, timesteps, features
        )

        self.output_shape = output_size if output_size is not None else hidden_dims[-1] 

    def forward(self, x):
        ### x = (BATCH, TIME_STAMP, CHANNEL, H, W)
        # features_output = []
        # for batch in x:
        #     batch_output = self.image_feature_extractor(batch)
        #     features_output.append(batch_output)

        # # features_output_tensor = (BATCH, TIMESTAMP, FEATURES)
        # features_output_tensor = torch.stack(features_output,dim=0) 

        # METHOD2

        # temporal_features_tensor = (BATCH, TIMESTAMP, FEATURES)
        output = self.tcn(x)

        # last_ts = temporal_features_tensor[:, -1 , :]

        # output = self.regressor(last_ts)

        return output
