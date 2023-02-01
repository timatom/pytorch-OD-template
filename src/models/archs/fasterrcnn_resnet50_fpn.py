import torch
import torch.nn as nn

from torchvision.models import resnet50 # backbone
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNNPredictor # neck and head

from src.models.archs.base_arch import Backbone, Neck, Head
    
class FasterRCNNResNet50FPNBackbone(Backbone):
    def build(self):
        return resnet50(pretrained=False)

class FasterRCNNResNet50FPNNeck(Neck):
    def build(self):
        return fasterrcnn_resnet50_fpn(pretrained=False).transform

class FasterRCNNResNet50FPNHead(Head):
    def build(self):
        return FastRCNNPredictor(pretrained=False)