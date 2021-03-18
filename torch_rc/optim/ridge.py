import torch
import torch.nn.functional
from torch_rc.nn.readout import Linear


@torch.jit.script
def _incremental_ridge_init(input_size: int, output_size: int, device: torch.device):
    mat_a = torch.zeros(output_size, input_size + 1, device=device)
    mat_b = torch.zeros(input_size + 1, input_size + 1, device=device)
    ide = torch.eye(input_size + 1, device=device)
    return mat_a, mat_b, ide


@torch.jit.script
def _incremental_ridge_classifier_step(input, expected, mat_a, mat_b, classification: bool):
    # Add bias
    s = torch.cat([input, torch.ones(input.shape[0], 1, device=input.device, dtype=input.dtype)], dim=1)

    if classification:
        # Convert y into "one-hot"-like tensor with values -1/+1
        y = torch.nn.functional.one_hot(expected, num_classes=mat_a.shape[0]).to(s.dtype).to(expected.device) * 2 - 1
    else:
        y = expected

    # s: (nb, nr+1)
    # y: (nb, ny)
    mat_a = mat_a + torch.einsum('br,by->yr', s, y)
    mat_b = mat_b + torch.einsum('br,bz->rz', s, s)
    return mat_a, mat_b


@torch.jit.script
def _incremental_ridge_end(mat_a, mat_b, l2_reg: float, ide):
    # Compute A @ (B + l2_reg * I)^{-1}
    weights = torch.linalg.solve(mat_b + l2_reg * ide, mat_a.t()).t()  # (ny, nr+1)
    w, b = weights[:, :-1], weights[:, -1]
    return w, b


@torch.jit.script
def _direct_ridge(input_size: int, output_size: int, input, expected, l2_reg: float, classification: bool):
    """

    Args:
        input_size:
        output_size:
        input:
        expected:
        l2_reg:
        classification: True to perform classification, False to perform regression

    Returns:

    """
    assert input_size == input.shape[-1]

    ide = torch.eye(input_size + 1, device=input.device)

    # Add bias
    s = torch.cat([input, torch.ones(input.shape[0], 1, device=input.device, dtype=input.dtype)], dim=1)

    if classification:
        # Convert y into "one-hot"-like tensor with values -1/+1
        y = torch.nn.functional.one_hot(expected, num_classes=output_size).to(s.dtype).to(expected.device) * 2 - 1
    else:
        y = expected

    # s: (nb, nr+1)
    # y: (nb, ny)
    mat_a = torch.einsum('br,by->yr', s, y)
    mat_b = torch.einsum('br,bz->rz', s, s)

    # Compute A @ (B + l2_reg * I)^{-1}
    weights = torch.linalg.solve(mat_b + l2_reg * ide, mat_a.t()).t()  # (ny, nr+1)
    w, b = weights[:, :-1], weights[:, -1]
    return w, b


class _RidgeBase:

    def __init__(self, readout: Linear, l2_reg: float, classification: bool):
        self._readout = readout
        self._l2_reg = l2_reg
        self._classification = classification

        self._state_size = self._readout.weight.shape[1]
        self._output_size = self._readout.weight.shape[0]

    def fit(self, states, expected):
        w, b = _direct_ridge(self._state_size, self._output_size, states, expected, self._l2_reg, self._classification)
        self._apply_weights(w, b)

    def _apply_weights(self, w, b):
        assert self._readout.weight.shape == w.shape
        assert self._readout.bias.shape == b.shape
        self._readout.weight.data = w
        self._readout.bias.data = b


class _RidgeIncrementalBase:

    def __init__(self, readout: Linear, l2_reg: float, classification: bool):
        self._readout = readout
        self._l2_reg = l2_reg
        self._classification = classification

        self._state_size = self._readout.weight.shape[1]
        self._output_size = self._readout.weight.shape[0]

        device = self._readout.weight.device

        self.mat_a, self.mat_b, self.ide = _incremental_ridge_init(self._state_size, self._output_size, device)

    def fit_step(self, states, expected):
        """

        Args:
            states: tensor of shape (batch, state_size) containing the states computed by the ESN
            expected: tensor of shape (batch,) containing the target classes

        Returns:

        """

        mat_a, mat_b = _incremental_ridge_classifier_step(states, expected, self._mat_a, self._mat_b,
                                                          self._classification)
        self._mat_a = mat_a
        self._mat_b = mat_b

    def fit_apply(self):
        w, b = _incremental_ridge_end(self._mat_a, self._mat_b, self._l2_reg, self._ide)
        self._apply_weights(w, b)

    def _apply_weights(self, w, b):
        assert self._readout.weight.shape == w.shape
        assert self._readout.bias.shape == b.shape
        self._readout.weight.data = w
        self._readout.bias.data = b


class RidgeClassification(_RidgeBase):

    def __init__(self, readout: Linear, l2_reg: float = 0):
        super().__init__(readout, l2_reg, True)


class RidgeRegression(_RidgeBase):

    def __init__(self, readout: Linear, l2_reg: float = 0):
        super().__init__(readout, l2_reg, False)


class RidgeIncrementalClassification(_RidgeIncrementalBase):

    def __init__(self, readout: Linear, l2_reg: float = 0):
        super().__init__(readout, l2_reg, True)


class RidgeIncrementalRegression(_RidgeIncrementalBase):

    def __init__(self, readout: Linear, l2_reg: float = 0):
        super().__init__(readout, l2_reg, False)

