# TODO: Need to decide if this script belongs to the main `src` directory or if it belongs to the `src/models` package.

from pathlib import Path

# from model_factory import ModelFactory
from src.data.data_factory import DataFactory
from src.models.model_factory import ModelFactory

from src.models.builders.model_builder import NeuralNetBuilder

from src.models.detection import resnet50_fpn_retinanet

# Built-in models
from torchvision.models.detection import (fasterrcnn_resnet50_fpn, 
                                          fasterrcnn_mobilenet_v3_large_fpn, 
                                          fasterrcnn_mobilenet_v3_large_320_fpn, 
                                          ssd300_vgg16, 
                                          retinanet_resnet50_fpn, 
                                          maskrcnn_resnet50_fpn, 
                                          keypointrcnn_resnet50_fpn,
                                          ssdlite320_mobilenet_v3_large)

# import loss functions
import torch.nn.functional as F

# TODO: Need to implement a seperate loss class that will be used to calculate the loss of the model.
# Note: Make sure to do so in a way that is compatible with the `ModelFactory` class.
# Note: This seperation should still make make the codebase modular, maintainable, and easy to understand.
# Note: Some ways to achieve this are by using clear contracts, documenting the relationship between 
#       the loss function and the model, and using wrapper classes
def loss_fn(output, targets):
    '''
    This function is used to calculate the loss of the model.
    Args:
        output: The output of the model.
        targets: The ground truth labels and bounding boxes.
    Returns:
        losses (dict): A dictionary containing the loss values.
    '''
    losses = {}
    
    # TODO: Need to address error: "TypeError: list indices must be integers or slices, not str"
    loc_loss = F.smooth_l1_loss(output["bbox_regression"], targets["bbox_regression"])
    conf_loss = F.binary_cross_entropy_with_logits(output["cls_logits"], targets["cls_logits"])
    
    losses["loc_loss"] = loc_loss
    losses["conf_loss"] = conf_loss
    
    return losses

# Define the data path
data_path = Path("/workspaces/pytorch-od-template/datasets/v1/")

# Get model
model = resnet50_fpn_retinanet()

# model = fasterrcnn_resnet50_fpn(weights = "COCO_V1", pretrained=True)

# Get loss function
loss_fn = loss_fn

# Create instances of the classes
data_factory = DataFactory(data_path)
model_factory = ModelFactory(data_factory, model, loss_fn)

# Train the model
model = model_factory.train(epochs = 2)

# Test the model
# TODO: Implement the test_model() method
# model_factory.test_model()

# Save the model
model_factory.save_model("test_model.pth")

# Deploy the model
# TODO: Implement the deploy_model() method
# model_factory.deploy_model(model, "path/to/save/model.h5")
