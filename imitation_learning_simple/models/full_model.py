import numpy as np
import torch
import torch.nn as nn

from tcn import TCN
from regressor import MLPRegressor
from visual_feature_extractors.FrontNet import Frontnet
from visual_feature_extractors.MobileNetv2 import MobileNetv2

visual_fe_dict = {"frontnet": Frontnet(),
                  "mobilenet": MobileNetv2()}

class FullModel(nn.Module):
    def __init__(self, visual_fe, history_len, output_size=4):
        super(FullModel, self).__init__()
        self.history_len = history_len

        assert visual_fe_dict.get(visual_fe) is not None, f"Visual feature extractor '{visual_fe}' unrecognized"
        self.vfe = visual_fe_dict[visual_fe]
        self.vfe_output_shape = self.vfe.output_shape

        self.input_to_classifier = self.vfe_output_shape 
        # add TCN if we have 
        if history_len > 0:
            self.tcn = TCN(input_size=self.vfe_output_shape,
                           hidden_dims=[256 for _ in range( 1 + int(np.log2(history_len)) )])
            self.input_to_classifier = self.tcn.output_shape 

            
        self.classifier = MLPRegressor(
            input_size=self.vfe_output_shape,
            hidden_dims=[self.vfe_output_shape//4],
            output_size=output_size
        )


    def forward(self, x):
        ### x = (BATCH, TIME_STAMP, CHANNEL, H, W)
        # If doesnt have timestamp
        if len(x.shape) == 4:
            x = x.unsqueeze(1)

        assert len(x.shape) == 5, f"Unexpected tensor of {len(x.shape)=} num dimensions: {x.shape=} "

        # Extract visual features
        old_x_shape = x.shape
        new_x_shape = (-1, *x.shape[-3:])

        x_reshaped = x.reshape(new_x_shape)

        features_output_tensor = self.image_feature_extractor(x_reshaped)
        features_output_tensor = features_output_tensor.reshape((*old_x_shape[:2], self.vfe_output_shape))

        ### features_output_tensor = (BATCH, TIME_STEP, FEATURES)
        if self.history_len > 0:
            features_output_tensor = self.tcn()

        ### features_output_tensor = (BATCH, TIME_STEP, FEATURES)
        out = self.classifier(features_output_tensor) # should go only for dim=-1
        
        ### features_output_tensor = (BATCH, TIME_STEP, OUTPUTS)
        return out