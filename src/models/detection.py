'''
This module contains built model architectures for object detection models.

Note: Function name typically takes the form of {backbone}_{neck}_{head}. The reason is remind
      others that data flows from the backbone to the neck to the head in that order within neural networks.
'''

from src.models.builders.model_builder import NeuralNetBuilder

# Import custom architectures
from src.models.archs.resnet import ResNetBackbone
from src.models.archs.fpn import FPNNeck
from src.models.archs.retinanet import RetinaNetHead

# Import custom architectures blocks and bottlenecks
from src.models.archs.blocks.resnet import ResNetBlock, ResNetBottleneck
from src.models.archs.blocks.fpn import FPNBlock

def resnet_fpn_retinanet():
    '''
    Builds a model with a custom head, backbone, and neck.
    '''
    # Configuring the model's backbone
    in_channels = 3
    out_channels = 64
    
    resnet_block = ResNetBlock
    resnet_block_layers = [3, 4, 6, 3]
    
    resnet = ResNetBackbone(resnet_block, resnet_block_layers, num_classes=91)
    
    # Configuring the model's neck
    fpn_block = FPNBlock
    
    in_channels = [256, 512, 1024, 2048]
    out_channels = 256
    
    num_outs = 5
    
    fpn = FPNNeck(fpn_block, in_channels, out_channels, num_outs)
    
    # Configuring the model's head
    in_channels = 256
    retinanet = RetinaNetHead(in_channels)

    # Building the model
    builder = NeuralNetBuilder()
    
    builder.build_backbone(resnet)
    builder.build_neck(fpn)
    builder.build_head(retinanet)

    return builder.get_model()