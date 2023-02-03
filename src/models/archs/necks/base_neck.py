from abc import ABCMeta, abstractmethod

import torch.nn as nn

class BaseNeck(nn.Module, metaclass=ABCMeta):
    def __init__(self, input_shape, *args, **kwargs):
        '''
        Initialize the BaseNeck component. 
        Args:
            input_shape (tuple): The shape of the input tensor. It should be of the form (batch_size, channels, height, width).
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        '''
        super().__init__(*args, **kwargs)
        self.input_shape = input_shape
        self._build(*args, **kwargs)
        
    @abstractmethod
    def _build(self, *args, **kwargs):
        '''
        Build the model architecture. To be implemented by subclasses.
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        '''
        pass
    
    @abstractmethod
    def forward(self, inputs):
        '''
        Forward pass. Apply the neck on the input tensor.
        To be implemented by subclasses.
        Args:
            inputs (torch.Tensor): Input tensor. It should be of the form (batch_size, channels, height, width).
                                   The input tensor is the output of the Backbone component.
        Returns:
            torch.Tensor: Output tensor. Its shape should be compatible with the Head components.
        '''
        pass

    @abstractmethod
    def get_output_shape(self):
        '''
        Return the output shape of the model.
        Returns:
            tuple: Output shape of the model. It should be of the form (batch_size, channels, height, width).
                   The output shape of the model is the shape of the tensor that is returned by the forward pass.
                   It is used to determine the input shape of the Head components.
        '''
        pass