from abc import ABC, abstractmethod

class BaseNeuralNetBuilder(ABC):
    def __init__(self):
        self._model = None
    
    def get_model(self):
        return self._model
    
    @abstractmethod
    def build_backbone(self):
        pass
    
    @abstractmethod
    def build_neck(self):
        pass
    
    @abstractmethod
    def build_head(self):
        pass
