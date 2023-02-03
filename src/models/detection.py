'''
This module contains built model architectures for object detection models.

Note: Function name typically takes the form of {backbone}_{neck}_{head}. The reason is remind
      others that data flows from the backbone to the neck to the head in that order within neural networks.
'''

from src.models.builders.model_builder import NeuralNetBuilder

# Import custom architectures
from src.models.archs.backbones.resnet import ResNet50Backbone
from src.models.archs.necks.fpn import FPNNeck
from src.models.archs.heads.retinanet import RetinaNetHead

# Import custom architectures blocks and bottlenecks
from src.models.archs.blocks.resnet import ResNetBlock, ResNetBottleneck
from src.models.archs.blocks.fpn import FPNBlock

def resnet50_fpn_retinanet():
    '''
    Builds a model with a custom head, backbone, and neck.
    '''
    # Configuring the model's input
    batch_size = 4
    backbone_channels = 3
    width = 300
    height = 300
    
    num_classes = 91
    
    # Configuring the model's backbone    
    input_shape = (batch_size, backbone_channels, width, height)
    
    backbone = ResNet50Backbone(input_shape=input_shape)
    
    backbone_output_shape = backbone.get_output_shape()
    
    # Configuring the model's neck
    input_shape = backbone_output_shape
    neck_channels = 256 # The number of channels in the FPN output
    
    neck = FPNNeck(input_shape, neck_channels)
    
    neck_output_shape = neck.get_output_shape()
    
    # Configuring the model's head
    head = RetinaNetHead(num_classes, neck_output_shape)
    
#     classification, regression = retinanet_head(neck_output_shape)

    # Building the model
    builder = NeuralNetBuilder()
    
    builder.build_backbone(backbone)
    builder.build_neck(neck)
    builder.build_head(head)

    return builder.get_model()