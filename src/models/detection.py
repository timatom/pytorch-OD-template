'''
This module contains built model architectures for object detection.
'''

from src.models.builders.model_builder import NeuralNetBuilder

from src.models.archs.mobilenet import MobileNetBackbone, MobileNetNeck, MobileNetHead
from src.models.archs.resnet import ResNetBackbone, ResNetNeck, ResNetHead
# from src.models.detection.ssd import SSDBackbone, SSDNeck, SSDHead

def mobilenet():
    '''
    Builds a MobileNet model.
    '''
    builder = NeuralNetBuilder()
    builder.set_backbone(MobileNetBackbone())
    builder.set_neck(MobileNetNeck())
    builder.set_head(MobileNetHead())
    
    return builder.build_model()

def resnet():
    '''
    Builds a ResNet model.
    '''
    builder = NeuralNetBuilder()
    builder.set_backbone(ResNetBackbone())
    builder.set_neck(ResNetNeck())
    builder.set_head(ResNetHead())
    
    return builder.build_model()

# def ssd():
#     '''
#     Builds an SSD model.
#     '''
#     builder = NeuralNetBuilder()
#     builder.set_backbone(SSDBackbone())
#     builder.set_neck(SSDNeck())
#     builder.set_head(SSDHead())
    
#     return builder.build_model()
