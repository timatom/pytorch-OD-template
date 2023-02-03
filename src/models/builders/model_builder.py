import torch.nn as nn

from src.models.builders.base_model_builder import BaseNeuralNetBuilder

def forward(x, targets=None):
        # pass the inputs through the backbone to get the feature maps
        x = self._model.backbone(x)
        
        # pass the feature maps through the neck to get the feature maps
        x = self._model.neck(x)
        
        # pass the feature maps through the head to get the predictions
        x = self._model.head(x, targets)
             
        return x

class NeuralNetBuilder(BaseNeuralNetBuilder):
    def __init__(self):
        super().__init__()
        self._model = nn.Sequential()

    def build_backbone(self, backbone):
        self._model.add_module("backbone", backbone)

    def build_neck(self, neck):
        self._model.add_module("neck", neck)

    def build_head(self, head):
        self._model.add_module("head", head)
    
    def build_forward(self, x, targets=None):
        self._model.add_module("forward", forward)

    def get_model(self):
        return self._model
