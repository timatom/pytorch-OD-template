from abc import ABC, abstractmethod
    
class Backbone(ABC):
    @abstractmethod
    def build(self):
        pass

class Neck(ABC):
    @abstractmethod
    def build(self):
        pass

class Head(ABC):
    @abstractmethod
    def build(self):
        pass