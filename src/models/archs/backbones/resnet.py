import torch
import torch.nn as nn
import torchvision.models as models

from src.models.archs.backbones.base_backbone import BaseBackbone

class ResNetBackbone(BaseBackbone):
    def __init__(self, input_shape, resnet_arch, *args, **kwargs):
        '''
        Initialize the ResNet backbone.
        Args:
            input_shape (tuple): The shape of the input tensor. It should be of the form (batch_size, channels, height, width)
            resnet_arch (nn.Module): The ResNet architecture.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
             
        Note: The ResNet model is used as a feature extractor. This means that the last two layers (avgpool and fc) are removed.
              This redults in a feature map that can be used by later components (e.g. Neck and Head). Feature maps are typically
              4D tensors of the form (batch_size, channels, height, width).
        '''
        self.resnet_arch = resnet_arch
        super().__init__(input_shape, *args, **kwargs)
        
    def _build(self, *args, **kwargs):
        '''
        Build the model architecture.
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        '''
        resnet_model = self.resnet_arch(pretrained=True)
        
        # Remove the last two layers (avgpool and fc)
        self.model = nn.Sequential(*list(resnet_model.children())[:-2])
        
    def forward(self, inputs):
        '''
        Forward pass.
        Args:
            inputs (torch.Tensor): Input tensor. It should be of the form (batch_size, channels, height, width)
        Returns:
            torch.Tensor: Output tensor. Its shape should be compatible with the Neck and Head components.
        '''
        # assert inputs.shape[1:] == self.input_shape[1:], 'The input tensor has an invalid shape.'
        # assert isinstance(inputs, torch.Tensor), 'The input tensor must be a PyTorch tensor.'
        
        # The input is a list of tensors. Each tensor is of the form (channels, height, width).
        # The number of tensors in the list is equal to the batch size.
        # The tensors in the list are stacked along the batch dimension to form a single tensor as follows:
        inputs = torch.stack(inputs, dim=0)
        
        assert inputs.shape[1:] == self.input_shape[1:], 'The input tensor has an invalid shape.'
        
        return self.model(inputs)
    
    def get_output_shape(self):
        '''
        Get the output shape of the backbone.
        Returns:
            tuple: The shape of the output tensor.
            
        Details: The output shape is determined by passing a dummy input tensor through the model.
        '''
        dummy_input = torch.zeros(self.input_shape)
        dummy_output = self.model(dummy_input)
        
        return dummy_output.shape
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.resnet_arch.__name__ + ')'
    
class ResNet18Backbone(ResNetBackbone):
    def __init__(self, input_shape, *args, **kwargs):
        '''
        Initialize the ResNet18 backbone.
        Args:
            input_shape (tuple): The shape of the input tensor. It should be of the form (batch_size, channels, height, width)
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        '''
        super().__init__(input_shape, models.resnet18, *args, **kwargs)
         
class ResNet34Backbone(ResNetBackbone):
    def __init__(self, input_shape, *args, **kwargs):
        '''
        Initialize the ResNet34 backbone.
        Args:
            input_shape (tuple): The shape of the input tensor. It should be of the form (batch_size, channels, height, width)
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        '''
        super().__init__(input_shape, models.resnet34, *args, **kwargs)
        
class ResNet50Backbone(ResNetBackbone):
    def __init__(self, input_shape, *args, **kwargs):
        '''
        Initialize the ResNet50 backbone.
        Args:
            input_shape (tuple): The shape of the input tensor. It should be of the form (batch_size, channels, height, width)
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        '''
        super().__init__(input_shape, models.resnet50, *args, **kwargs)
        
class ResNet101Backbone(ResNetBackbone):
    def __init__(self, input_shape, *args, **kwargs):
        '''
        Initialize the ResNet101 backbone.
        Args:
            input_shape (tuple): The shape of the input tensor. It should be of the form (batch_size, channels, height, width)
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        '''
        super().__init__(input_shape, models.resnet101, *args, **kwargs)
         
class ResNet152Backbone(ResNetBackbone):
    def __init__(self, input_shape, *args, **kwargs):
        '''
        Initialize the ResNet152 backbone.
        Args:
            input_shape (tuple): The shape of the input tensor. It should be of the form (batch_size, channels, height, width)
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        '''      
        super().__init__(input_shape, models.resnet152, *args, **kwargs)
         
class ResNeXt50_32x4dBackbone(ResNetBackbone):
    def __init__(self, input_shape, *args, **kwargs):
        '''
        Initialize the ResNeXt50_32x4d backbone.
        Args:
            input_shape (tuple): The shape of the input tensor. It should be of the form (batch_size, channels, height, width)
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        '''       
        super().__init__(input_shape, models.resnext50_32x4d, *args, **kwargs)
         
class ResNeXt101_32x8dBackbone(ResNetBackbone):
    def __init__(self, input_shape, *args, **kwargs):
        '''
        Initialize the ResNeXt101_32x8d backbone.
        Args:
            input_shape (tuple): The shape of the input tensor. It should be of the form (batch_size, channels, height, width)
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        '''
        super().__init__(input_shape, models.resnext101_32x8d, *args, **kwargs)
         
class WideResNet50_2Backbone(ResNetBackbone):
    def __init__(self, input_shape, *args, **kwargs):
        '''
        Initialize the WideResNet50_2 backbone.
        Args:
            input_shape (tuple): The shape of the input tensor. It should be of the form (batch_size, channels, height, width)
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        '''
        super().__init__(input_shape, models.wide_resnet50_2, *args, **kwargs)
         
class WideResNet101_2Backbone(ResNetBackbone):
    def __init__(self, input_shape, *args, **kwargs):
        '''
        Initialize the WideResNet101_2 backbone.
        Args:
            input_shape (tuple): The shape of the input tensor. It should be of the form (batch_size, channels, height, width)
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        '''
        super().__init__(input_shape, models.wide_resnet101_2, *args, **kwargs)
