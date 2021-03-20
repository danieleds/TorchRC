import torch
import torch.nn.functional
from typing import Iterable


@torch.jit.script
def _incremental_ridge_init(input_size: int, output_size: int, device: torch.device):
    mat_a = torch.zeros(output_size, input_size + 1, device=device)
    mat_b = torch.zeros(input_size + 1, input_size + 1, device=device)
    ide = torch.eye(input_size + 1, device=device)
    return mat_a, mat_b, ide


@torch.jit.script
def _incremental_ridge_step(input, expected, mat_a, mat_b, classification: bool):
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

    return _incremental_ridge_end(mat_a, mat_b, l2_reg, ide)


def _detect_parameters(params: Iterable):
    if isinstance(params, torch.Tensor):
        raise TypeError("params argument given to the optimizer should be "
                        "an iterable of Tensors, but got " +
                        torch.typename(params))

    plist = list(params)
    if len(plist) == 0:
        raise ValueError("optimizer got an empty parameter list")

    if len(plist) != 2:
        raise ValueError(f"optimizer expected a list of 2 parameters, but got {len(plist)}")

    # Find which one is the weight and which one is the bias
    if len(plist[0].shape) > len(plist[1].shape):
        w, b = plist[0], plist[1]
    else:
        w, b = plist[1], plist[0]

    if w.shape[0] != b.shape[0]:
        raise ValueError(f"the first dimension of the tensors should match, but got {tuple(w.shape[0])} "
                         f"and {tuple(b.shape[0])}")

    return w, b


class _RidgeBase:

    def __init__(self, params: Iterable, l2_reg: float, classification: bool):
        self._l2_reg = l2_reg
        self._classification = classification

        w, b = _detect_parameters(params)
        self._params = {'w': w, 'b': b}

        self._input_size = w.shape[1]
        self._output_size = w.shape[0]

    def fit(self, input, expected):
        """

        Args:
            input: input tensor of shape (batch, n_features)
            expected: target tensor of shape:
                - (batch, n_targets) in case of regression
                - (batch,) in case of classification, with dtype=torch.Long

        """
        if self._classification:
            assert len(input.shape) == len(expected.shape) + 1, "Multi-target classification is not supported"
        else:
            assert len(input.shape) == len(expected.shape)

        w, b = _direct_ridge(self._input_size, self._output_size, input, expected, self._l2_reg, self._classification)
        self._apply_weights(w, b)

    def _apply_weights(self, w, b):
        assert self._params['w'].shape == w.shape
        assert self._params['b'].shape == b.shape
        self._params['w'].data = w
        self._params['b'].data = b


class _RidgeIncrementalBase:

    def __init__(self, params: Iterable, l2_reg: float, classification: bool):
        self._l2_reg = l2_reg
        self._classification = classification

        w, b = _detect_parameters(params)
        self._params = {'w': w, 'b': b}

        self._input_size = w.shape[1]
        self._output_size = w.shape[0]

        device = w.device

        self._mat_a, self._mat_b, self._ide = _incremental_ridge_init(self._input_size, self._output_size, device)

    def fit_step(self, input, expected):
        """

        Args:
            input: input tensor of shape (batch, n_features)
            expected: target tensor of shape:
                - (batch, n_targets) in case of regression
                - (batch,) in case of classification, with dtype=torch.Long

        """
        if self._classification:
            assert len(input.shape) == len(expected.shape) + 1, "Multi-target classification is not supported"
        else:
            assert len(input.shape) == len(expected.shape)

        mat_a, mat_b = _incremental_ridge_step(input, expected, self._mat_a, self._mat_b, self._classification)
        self._mat_a = mat_a
        self._mat_b = mat_b

    def fit_apply(self):
        w, b = _incremental_ridge_end(self._mat_a, self._mat_b, self._l2_reg, self._ide)
        self._apply_weights(w, b)

    def _apply_weights(self, w, b):
        assert self._params['w'].shape == w.shape
        assert self._params['b'].shape == b.shape
        self._params['w'].data = w
        self._params['b'].data = b


class RidgeClassification(_RidgeBase):

    def __init__(self, params: Iterable, l2_reg: float = 1):
        super().__init__(params, l2_reg, True)


class RidgeRegression(_RidgeBase):

    def __init__(self, params: Iterable, l2_reg: float = 1):
        super().__init__(params, l2_reg, False)


class RidgeIncrementalClassification(_RidgeIncrementalBase):

    def __init__(self, params: Iterable, l2_reg: float = 1):
        super().__init__(params, l2_reg, True)


class RidgeIncrementalRegression(_RidgeIncrementalBase):

    def __init__(self, params: Iterable, l2_reg: float = 1):
        super().__init__(params, l2_reg, False)
