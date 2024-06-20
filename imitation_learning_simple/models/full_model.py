import numpy as np
import torch
import torch.nn as nn

from models.tcn import TCN
from models.regressor import MLPRegressor
from models.visual_feature_extractors.FrontNet import Frontnet
from models.visual_feature_extractors.MobileNetv2 import MobileNetv2

visual_fe_dict = {"frontnet": Frontnet(),
                  "mobilenet": MobileNetv2()}

class FullModel(nn.Module):
    def __init__(self, visual_fe, history_len, output_size=4):
        super(FullModel, self).__init__()
        self.output_size = output_size
        self.history_len = history_len

        assert visual_fe_dict.get(visual_fe) is not None, f"Visual feature extractor '{visual_fe}' unrecognized"
        self.vfe = visual_fe_dict[visual_fe]
        self.vfe_output_shape = self.vfe.output_shape[0]
        self.input_to_classifier = self.vfe_output_shape

        # add TCN if we have 
        if history_len > 0:
            self.tcn = TCN(input_size=self.input_to_classifier,
                           hidden_dims=[128 for _ in range( 1 + int(np.log2(history_len)) )])
            self.input_to_classifier = self.tcn.output_shape 

            
        
        self.classifier = MLPRegressor(
            input_size=self.input_to_classifier,
            hidden_dims=[self.input_to_classifier//4],
            output_size=self.output_size
        )


    def forward(self, x, history_visual_features_array=None):
        ### x = (BATCH, TIME_STAMP, CHANNEL, H, W)
        # If doesnt have timestamp
        if len(x.shape) == 4:
            x = x.unsqueeze(1)

        assert len(x.shape) == 5, f"Unexpected tensor of {len(x.shape)=} num dimensions: {x.shape=} "

        # Extract visual features
        if history_visual_features_array is not None:
            assert history_visual_features_array.shape[0] == x.shape[0], "History array and Image input have different batch sizes"
            # Ensure you are looking at the last image, without dropping the dimension (-1:)
            x = x[:, -1:, :,:,:]
            old_x_shape = x.shape
            new_x_shape = (-1, *x.shape[-3:])
            x_reshaped = x.reshape(new_x_shape)
            features_output_tensor = self.vfe(x_reshaped)
            # batch, ts=1, features
            features_output_tensor = features_output_tensor.reshape((*old_x_shape[:2], self.vfe_output_shape))

            features_output_tensor = torch.cat([history_visual_features_array, features_output_tensor], dim=1) # on temporal dimension

        else:
            old_x_shape = x.shape
            new_x_shape = (-1, *x.shape[-3:])
            x_reshaped = x.reshape(new_x_shape)
            features_output_tensor = self.vfe(x_reshaped)
            features_output_tensor = features_output_tensor.reshape((*old_x_shape[:2], self.vfe_output_shape))
            
        ### features_output_tensor = (BATCH, TIME_STEP, FEATURES)
        if self.history_len > 0:
            features_output_tensor = self.tcn(features_output_tensor)

        ### features_output_tensor = (BATCH, TIME_STEP, FEATURES)
        old_x_shape = features_output_tensor.shape
        new_x_shape = (-1, *features_output_tensor.shape[-1:])
        features_output_tensor = features_output_tensor.reshape(new_x_shape)
        out = self.classifier(features_output_tensor)
        out = out.reshape((*old_x_shape[:2], self.output_size))
        
        ### features_output_tensor = (BATCH, TIME_STEP, OUTPUTS)
        return out