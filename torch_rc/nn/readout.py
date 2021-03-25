import torch
import torch.nn.functional


class Linear(torch.nn.Module):
    r"""A linear readout layer that can be used with the torch_rc optimizers.

    This layer is mostly identical to PyTorch's Linear layer, except that this one by default does not
    require gradients for its parameters.

    This layer applies the following linear transformation to the data: :math:`y = xA^T + b`.

    Since the parameters are set up to not require gradients, to tune them you should use one of the
    gradient-free optimizers from :py:mod:`torch_rc.optim`, such as the Ridge optimizers.

    Args:
        in_features: number of features in the input
        out_features: number of features in the output

    Shape:
        Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in_features}`
        Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out_features}`.

    Attributes:
        weight: the weights of the module of shape
            :math:`(\text{out_features}, \text{in_features})`. The values are
            initialized from an empty tensor.
        bias:   the bias of the module of shape :math:`(\text{out_features})`.
                The values are initialized from
                an empty tensor.

    Examples::

        >>> m = torch_rc.nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    def __init__(self, in_features: int, out_features: int):
        """"""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features)), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.empty((out_features,)), requires_grad=False)

    def forward(self, x):
        """
        :param x: tensor of shape `(N, *, in_features)` where `*` means any number of additional dimensions.
        :return: tensor of shape `(N, *, out_features)` where all but the last dimension are the same shape as the input.
        """
        return torch.nn.functional.linear(x, self.weight, self.bias)
