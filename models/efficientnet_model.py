# models/efficientnet_model.py

import torch.nn as nn
from torchvision import models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetModel(nn.Module):
    def __init__(self):
        super(EfficientNetModel, self).__init__()
       # self.base_model = models.efficientnet_b0(pretrained=True)
        #in_features = self.base_model.classifier[1].in_features
        weights = EfficientNet_B0_Weights.DEFAULT
        self.base_model = efficientnet_b0(weights=weights)
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(in_features, 2)
        
        self.base_model.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.base_model(x)
