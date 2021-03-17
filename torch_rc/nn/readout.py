import torch
import torch.nn.functional


class Linear(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int):
        """
        A linear readout layer that can be used with the torch_rc optimizators.
        :param in_features: number of features in the input
        :param out_features: number of features in the output
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features)), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.empty((out_features,)), requires_grad=False)

    def forward(self, x):
        """
        :param x: tensor of shape (N, *, in_features) where `*` means any number of additional dimensions.
        :return: tensor of shape (N, *, out_features) where all but the last dimension are the same shape as the input.
        """
        return torch.nn.functional.linear(x, self.weight, self.bias)
