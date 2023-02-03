import torch.nn as nn

from src.models.archs.heads.base_head import BaseHead

class RetinaNetHead(BaseHead):
    def __init__(self, num_classes, *args, **kwargs):
        self.num_classes = num_classes
        super().__init__(*args, **kwargs)

    def _build(self, *args, **kwargs):
        self.classification_layer = nn.Conv2d(256, self.num_classes, 1)
        self.regression_layer = nn.Conv2d(256, 4 * self.num_classes, 1)

    def forward(self, inputs):
        
        # TODO: This is a temporary fix for the issue of the model returning a list of tensors.
        if type(inputs) == list:
            inputs = inputs[0]
        
        classification = self.classification_layer(inputs)
        classification = classification.permute(0, 2, 3, 1)
        classification = classification.reshape(classification.shape[0], -1, self.num_classes)

        regression = self.regression_layer(inputs)
        regression = regression.permute(0, 2, 3, 1)
        regression = regression.reshape(regression.shape[0], -1, 4)

        return {"cls_logits": classification, "bbox_regression": regression }
    
    def get_output_shape(self):
        return self.input_shape[0], self.num_classes, self.input_shape[2], self.input_shape[3]
    
    def __repr__(self):
        return 'RetinaNetHead(num_classes={}, input_shape={})'.format(self.num_classes, self.input_shape)
