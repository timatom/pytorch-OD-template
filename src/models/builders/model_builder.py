import torch.nn as nn

class NeuralNetBuilder:
    def __init__(self):
        self.backbone = None
        self.neck = None
        self.head = None
        
    def set_backbone(self, backbone):
        self.backbone = backbone
    
    def set_neck(self, neck):
        self.neck = neck
    
    def set_head(self, head):
        self.head = head
        
    def build_model(self):
        '''
        Builds the neural network.
        '''
        return nn.Sequential(
            self.backbone.build(),
            self.neck.build(),
            self.head.build()
        )
