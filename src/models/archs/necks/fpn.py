import torch
import torch.nn as nn

from src.models.archs.necks.base_neck import BaseNeck

class FPNNeck(BaseNeck):
    def __init__(self, input_shape, fpn_channels, *args, **kwargs):
        '''
        Initialize the FPN Neck component.
        Args:
            input_shape (tuple): The shape of the input tensor. It should be of the form (batch_size, channels, height, width)
            fpn_channels (int): The number of channels in the feature maps produced by the FPN.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        '''
        self.fpn_channels = fpn_channels
        super().__init__(input_shape, *args, **kwargs)
        
    def _build(self, *args, **kwargs):
        '''
        Build the model architecture.
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
             
        Details:
            The FPN Neck component consists of two parallel paths:
            1. Top-down path: This path consists of a sequence of Conv2d layers. The first layer is a 1x1 Conv2d layer that 
               reduces the number of channels in the input feature map to fpn_channels. The second layer is a 3x3 Conv2d layer 
               that upsamples the feature map to the desired size. The output of this path is added to the output of the lateral path.
            2. Lateral path: This path consists of a single 1x1 Conv2d layer that reduces the number of channels in the input 
               feature map to fpn_channels.
             
            The first layer in the top-down path is applied on the input feature map. The second layer in the top-down path is
            applied on the output of the first layer in the top-down path. And so on.
            
            The lateral path is applied on the input feature map. The output of the lateral path is added to the output of the
            top-down path. The output of the lateral path is added to the output of the top-down path applied on the input feature map.
            And so on.
        '''
        self.top_down_layers = nn.ModuleList([
            nn.Conv2d(self.input_shape[1], self.fpn_channels, 1),
            nn.Conv2d(self.fpn_channels, self.fpn_channels, 3, padding=1),
        ])
        self.lateral_layers = nn.ModuleList([
            nn.Conv2d(self.input_shape[1], self.fpn_channels, 1),
        ])
        
    def forward(self, inputs):
        '''
        Forward pass. Apply the FPN Neck on the input tensor.
        Args:
            inputs (torch.Tensor): Input tensor. It should be of the form (batch_size, channels, height, width)
        Returns:
            torch.Tensor: Output tensor. Its shape should be compatible with the Head components.
            
        Details:
            The forward pass consists of two steps:
            1. Apply the top-down path on the input feature map.
            2. Apply the lateral path on the input feature map.
            3. Add the output of the top-down path to the output of the lateral path.
            4. Repeat steps 1-3 for all the feature maps in the input tensor.
             
            The output of the forward pass is a list of feature maps. Each feature map is of the form (batch_size, fpn_channels, height, width).
        '''
        feature_maps = []
        x = inputs
        
        for top_down, lateral in zip(self.top_down_layers, self.lateral_layers):
            y = top_down(x)
            x = y + lateral(inputs)
            feature_maps.append(x)
            
        return feature_maps
    
    def get_output_shape(self):
        '''
        Return the output shape of the FPN Neck component.
        Returns:
            tuple: Output shape of the FPN Neck component. It should be of the form (batch_size, fpn_channels, height, width).
        '''
        # TODO: Verify that the output shape is correct!!!
        return (self.input_shape[0], self.fpn_channels, self.input_shape[2], self.input_shape[3])
    
    def __repr__(self):
        '''
        Return a string representation of the FPN Neck component.
        '''
        return f'FPNNeck(input_shape={self.input_shape}, fpn_channels={self.fpn_channels})'
