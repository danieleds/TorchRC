import torch
from torch_rc.nn.readout import Linear


@torch.jit.script
def _incremental_ridge_classifier_init(input_size: int, output_size: int):
    mat_a = torch.zeros(output_size, input_size + 1)
    mat_b = torch.zeros(input_size + 1, input_size + 1)
    ide = torch.eye(input_size + 1)
    return mat_a, mat_b, ide


@torch.jit.script
def _incremental_ridge_classifier_step(input, expected, l2_reg: float, mat_a, mat_b, ide):
    # Add bias
    s = torch.cat([input, torch.ones(input.shape[0], 1, device=input.device, dtype=input.dtype)], dim=1)

    # Convert y into "one-hot"-like tensor with values -1/+1
    y = torch.nn.functional.one_hot(expected, num_classes=mat_a.shape[0]).to(s.dtype).to(expected.device) * 2 - 1

    # s: (nb, nr+1)
    # y: (nb, ny)
    mat_a = mat_a + torch.einsum('br,by->yr', s, y)
    mat_b = mat_b + torch.einsum('br,bz->rz', s, s) + l2_reg * ide
    return mat_a, mat_b


@torch.jit.script
def _incremental_ridge_classifier_end(mat_a, mat_b):
    weights = mat_a @ torch.inverse(mat_b)  # (ny, nr+1)
    w, b = weights[:, :-1], weights[:, -1]
    return w, b


class RidgeIncrementalClassifier:

    def __init__(self, readout: Linear, l2_reg: float = 0):
        self.readout = readout
        self.l2_reg = l2_reg

        self._state_size = self.readout.weight.shape[1]
        self._output_size = self.readout.weight.shape[0]

        device = self.readout.weight.device

        mat_a, mat_b, ide = _incremental_ridge_classifier_init(self._state_size, self._output_size)
        self.mat_a = mat_a.to(device)
        self.mat_b = mat_b.to(device)
        self.ide = ide.to(device)

    def fit_step(self, states, expected):
        """

        Args:
            states: tensor of shape (batch, state_size) containing the states computed by the ESN
            expected: tensor of shape (batch,) containing the target classes

        Returns:

        """

        mat_a, mat_b = _incremental_ridge_classifier_step(states, expected, self.l2_reg, self.mat_a, self.mat_b, self.ide)
        self.mat_a = mat_a
        self.mat_b = mat_b

    def fit_end(self):
        w, b = _incremental_ridge_classifier_end(self.mat_a, self.mat_b)
        self._apply_weights(w, b)

    def _apply_weights(self, w, b):
        assert self.readout.weight.shape == w.shape
        assert self.readout.bias.shape == b.shape
        self.readout.weight.data = w
        self.readout.bias.data = b
