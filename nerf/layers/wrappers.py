import torch
import torch.nn.functional as F

__all__ = ["Linear", ]


class Linear(torch.nn.Linear):
    """
    A wrapper around :class:`torch.nn.Linear` to add the activation function.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:
        Args:
            activation (callable(Tensor) -> Tensor): a callable activation function
        It assumes that norm layer is used before activation.
        """
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.activation = activation

    def forward(self, x):
        x = F.linear(x, self.weight, self.bias)
        if self.activation is not None:
            x = self.activation(x)
        return x
