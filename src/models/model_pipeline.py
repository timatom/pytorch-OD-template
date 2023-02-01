# TODO: Need to decide if this script belongs to the main `src` directory or if it belongs to the `src/models` package.

from pathlib import Path

# from model_factory import ModelFactory
from src.data.data_factory import DataFactory
from src.models.model_factory import ModelFactory

from src.models.builders.model_builder import NeuralNetBuilder

from src.models.detection import mobilenet, resnet

# Built-in models
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

# Get model
model = mobilenet()

# Create instances of the classes
data_factory = DataFactory(data_path)
model_factory = ModelFactory(data_factory, model)

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
