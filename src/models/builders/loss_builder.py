import torch

class LossBuilder:
    def __init__(self):
        self._loss_fn = None

    def set_cross_entropy(self):
        self._loss_fn = torch.nn.CrossEntropyLoss()

    def set_mean_squared_error(self):
        self._loss_fn = torch.nn.MSELoss()

    def build(self):
        return self._loss_fn
