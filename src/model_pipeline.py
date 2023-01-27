from pathlib import Path

from data_factory import DataFactory
from model_factory import ModelFactory

# Import the models
from torchvision.models.detection import (fasterrcnn_resnet50_fpn, 
                                          fasterrcnn_mobilenet_v3_large_fpn, 
                                          fasterrcnn_mobilenet_v3_large_320_fpn, 
                                          ssd300_vgg16, 
                                          retinanet_resnet50_fpn, 
                                          maskrcnn_resnet50_fpn, 
                                          keypointrcnn_resnet50_fpn,
                                          ssdlite320_mobilenet_v3_large)

# Define the data path
data_path = Path("/workspaces/pytorch-od-template/datasets/v1/")

# Create instances of the classes
data_factory = DataFactory(data_path)
model_factory = ModelFactory(data_factory, ssdlite320_mobilenet_v3_large)

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
