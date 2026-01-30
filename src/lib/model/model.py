import torch
import torch.nn as nn

from abc import abstractmethod

class Model(nn.Module):
    """
    Base NN model for function approximation RL
    """
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self) -> torch.Tensor:
        """
        Forward propagation of Neural Network model
        """
        raise NotImplementedError(f"Please implement the 'forward' method for {self.__class__.__name__}.")
    
    ## =============== Auxilary functions =============== ##

    def init_weights(self) -> None:
        """
        Orthogonal initialize the model weights

        The following layers will be initialized:
        - torch.nn.Linear
        """
        def _orthogonal_init(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)

        self.apply(_orthogonal_init)

    def init_biases(self, val: int = 0) -> None:
        """
        Constant initialize the model biases

        The following layers will be initialized:
        - torch.nn.Linear

        :param val: constant value to initialize biases
        :type val: int
        """
        def _constant_init(module):
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, val=val)

        self.apply(_constant_init)

    def save(self, path: str) -> None:
        """
        Save the model to the specific path

        :param path: Path to save the model to
        :type path: str
        """

        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """
        Load the model from the specific path

        The final storage device is determined by the constructor of the model

        :param path: Path to load the model from
        :type path: str
        """
        
        state_dict = torch.load(path, map_location=self.device, weights_only=False)  # prevent torch:FutureWarning
        self.load_state_dict(state_dict)