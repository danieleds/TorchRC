import torch
from torch.nn import ModuleList
from typing import Optional, Final


@torch.jit.script
def to_sparse(tensor, density: float, sparse_repr: bool = False) -> torch.Tensor:
    if density < 1:
        if sparse_repr:
            cp = tensor.clone()
            cp[torch.rand_like(tensor) > density] = 0
            return cp.to_sparse()
        else:
            return tensor * (torch.rand_like(tensor) <= density).to(dtype=tensor.dtype).to(device=tensor.device)
    else:
        return tensor


class ESNCell(torch.nn.Module):
    use_bias: Final[bool]
    leaking_rate: Final[float]

    # TODO: density -> sparsity
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 bias: bool = True,
                 scale_rec: float = 0.9,
                 density_rec: float = 1.0,
                 scale_in: float = 1.0,
                 density_in: float = 1.0,
                 scale_bias: float = 1.0,
                 leaking_rate: float = 1.0,
                 rec_rescaling_method: str = 'specrad',  # Either "norm" or "specrad"
                 ):
        super(ESNCell, self).__init__()

        assert rec_rescaling_method in ['norm', 'specrad'], f"Invalid value for {rec_rescaling_method=}"
        assert 0 <= density_rec <= 1, f"Invalid range for {density_rec=}"
        assert 0 <= density_in <= 1, f"Invalid range for {density_in=}"
        assert 0 <= leaking_rate <= 1, f"Invalid range for {leaking_rate=}"

        self.use_bias = bias
        self.leaking_rate = leaking_rate

        # Reservoir
        W_in = torch.rand((output_size, input_size)) * 2 - 1
        W_hat = torch.rand((output_size, output_size)) * 2 - 1
        b = torch.rand(output_size) * 2 - 1 if self.use_bias else None

        # Sparsity
        W_in = to_sparse(W_in, density_in)
        W_hat = to_sparse(W_hat, density_rec)

        # Scale W_in
        W_in = scale_in * W_in

        # Scale W_hat
        W_hat = self.rescale_contractivity(W_hat, scale_rec, rec_rescaling_method)

        if self.use_bias:
            b = scale_bias * b

        # Assign as buffers
        self.register_buffer('W_in', W_in)
        self.register_buffer('W_hat', W_hat)
        if self.use_bias:
            self.register_buffer('b', b)

    def forward(self, input, hidden):
        """
        input: (batch, input_size)
        hidden: (batch, hidden_size)

        output: (batch, hidden_size)
        """

        if self.use_bias:
            h_tilde = torch.tanh(self.W_in @ input.t() + self.W_hat @ hidden.t() + self.b.view(-1, 1)).t()
        else:
            h_tilde = torch.tanh(self.W_in @ input.t() + self.W_hat @ hidden.t()).t()

        h = (1 - self.leaking_rate) * hidden + self.leaking_rate * h_tilde
        return h

    @staticmethod
    def rescale_contractivity(W, coeff, rescaling_method):
        if rescaling_method == 'norm':
            return W * coeff / W.norm()
        elif rescaling_method == 'specrad':
            return W * coeff / (W.eig()[0].abs().max())
        else:
            raise Exception("Invalid rescaling method used (must be either 'norm' or 'specrad')")


class ESNMultiringCell(torch.jit.ScriptModule):
    scale_rec: Final[float]
    use_bias: Final[bool]
    leaking_rate: Final[float]

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 bias: bool = True,
                 scale_rec: float = 0.9,
                 scale_in: float = 1.0,
                 density_in: float = 1.0,
                 scale_bias: float = 1.0,
                 leaking_rate: float = 1.0,
                 # rec_rescaling_method: str = 'specrad',  # Either "norm" or "specrad"
                 topology: str = 'multiring'):  # Either "ring" or "multiring"
        super(ESNMultiringCell, self).__init__()

        # assert rec_rescaling_method in ['norm', 'specrad'], f"Invalid value for {rec_rescaling_method=}"
        assert topology in ['ring', 'multiring'], f"Invalid value for {topology=}"
        assert 0 <= density_in <= 1, f"Invalid range for {density_in=}"
        assert 0 <= leaking_rate <= 1, f"Invalid range for {leaking_rate=}"

        self.use_bias = bias
        self.leaking_rate = leaking_rate

        # Reservoir
        W_in = torch.rand((output_size, input_size)) * 2 - 1
        b = torch.rand(output_size) * 2 - 1 if self.use_bias else None

        W_in = to_sparse(W_in, density_in)

        ring: torch.Tensor
        if topology == 'multiring':
            ring = torch.randperm(output_size)
        elif topology == 'ring':
            ring = torch.cat([torch.arange(1, output_size), torch.tensor([0])])
        else:
            assert False

        # Scale W_in
        W_in = scale_in * W_in

        # Scale W_hat
        # No need to compute the spectral radius for ring or multiring topologies since
        # the spectral radius is equal to the value of the nonzero elements
        self.scale_rec = scale_rec

        if self.use_bias:
            b = scale_bias * b

        self.register_buffer('W_in', W_in)
        self.register_buffer('ring', ring)
        if self.use_bias:
            self.register_buffer('b', b)

    @torch.jit.script_method
    def forward(self, input, hidden):
        """
        input: (batch, input_size)
        hidden: (batch, hidden_size)

        output: (batch, hidden_size)
        """
        # Add the bias column
        if self.use_bias:
            h_tilde = torch.tanh(self.W_in @ input.t() + self.scale_rec * hidden.t()[self.ring] + self.b.view(-1, 1)).t()
        else:
            h_tilde = torch.tanh(self.W_in @ input.t() + self.scale_rec * hidden.t()[self.ring]).t()

        h = (1-self.leaking_rate) * hidden + self.leaking_rate * h_tilde
        return h


class ESNBase(torch.nn.Module):
    bidirectional: Final[bool]

    def __init__(self, input_size, reservoir_size, cells: ModuleList, cells_bw: Optional[ModuleList], num_layers=1):
        """

        Args:
            input_size:
            reservoir_size:
            cells: Cells must be provided as a list. In case of a bidirectional architecture, they must be organized
            num_layers:
            dropout:
            bidirectional:
        """
        super(ESNBase, self).__init__()

        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.num_layers = num_layers
        self.bidirectional = cells_bw is not None
        self.num_directions = 2 if self.bidirectional else 1
        self._cells = cells
        self._cells_bw = cells_bw

    def forward(self, input, h_0: Optional[torch.Tensor] = None):
        """
        input: (seq_len, batch, input_size)
        h_0: (num_layers * num_directions, batch, hidden_size)

        output: (seq_len, batch, num_directions * hidden_size)  # Only the output of the last layer, for both directions
                                                                  concatenated
        h_n: (num_layers * num_directions, batch, hidden_size)  # Hidden state for the last step, in all layers
        """

        if self.bidirectional:
            return self._forward_bi(input, h_0)
        else:
            return self._forward_mono(input, h_0)

    def _forward_mono(self, input, h_0: Optional[torch.Tensor] = None):
        batch = input.shape[1]

        if h_0 is None:
            h_0 = input.new_zeros((self.num_layers * self.num_directions, batch, self.reservoir_size))

        # Separate the layers to avoid in-place operations
        h_l = list(h_0.unbind())

        next_layer_input = input
        for lyr, cell in enumerate(self._cells):
            layer_outputs = []  # list of (batch, hidden_size)
            step_h = h_l[lyr]
            for i, x_t in enumerate(next_layer_input):
                h = cell.forward(x_t, step_h)  # (batch, hidden_size)
                step_h = h
                layer_outputs.append(h)
            h_l[lyr] = step_h
            next_layer_input = torch.stack(layer_outputs)

        h_n = torch.stack(h_l)

        return next_layer_input, h_n

    def _forward_bi(self, input, h_0: Optional[torch.Tensor] = None):
        # Implementation of Deep-Bi type 2
        # https://github.com/pytorch/pytorch/issues/4930#issuecomment-361851298

        batch = input.shape[1]

        if h_0 is None:
            h_0 = input.new_zeros((self.num_layers * self.num_directions, batch, self.reservoir_size))

        # Separate the layers to avoid in-place operations
        # h_l = [l0d0, l0d1, l1d0, l1d1, ...]
        h_l = list(h_0.unbind())

        next_layer_input = input
        for lyr, (cell_fw, cell_bw) in enumerate(zip(self._cells, self._cells_bw)):
            layer_outputs_fw = []  # list of (batch, hidden_size)
            step_h = h_l[2 * lyr + 0]
            for x_t in next_layer_input:
                h = cell_fw.forward(x_t, step_h)  # (batch, hidden_size)
                step_h = h
                layer_outputs_fw.append(h)
            h_l[2 * lyr + 0] = step_h

            layer_outputs_bw = []  # list of (batch, hidden_size)
            step_h = h_l[2 * lyr + 1]
            for x_t in reversed(next_layer_input):
                h = cell_bw.forward(x_t, step_h)  # (batch, hidden_size)
                step_h = h
                layer_outputs_bw.append(h)
            h_l[2 * lyr + 1] = step_h

            # (seq_len, batch, self.num_directions * self.reservoir_size)
            next_layer_input = torch.cat((
                torch.stack(layer_outputs_fw),
                torch.stack(layer_outputs_bw)
            ), dim=2)

        h_n = torch.stack(h_l)

        return next_layer_input, h_n


class LeakyESN(ESNBase):

    def __init__(self, input_size: int, output_size: int, num_layers=1, bias=True, bidirectional=False,
                 scale_rec=0.9, density=1.0, scale_in=1.0, density_in=1.0,
                 scale_bias=1.0, leaking_rate=1.0, rescaling_method='specrad'):

        num_directions = 2 if bidirectional else 1

        cells = ModuleList([
            ESNCell(input_size if lyr == 0 else num_directions * output_size, output_size,
                    bias, scale_rec, density, scale_in, density_in,
                    scale_bias, leaking_rate, rescaling_method)
            for lyr in range(num_layers)
        ])

        cells_bw = None
        if bidirectional:
            cells_bw = ModuleList([
                ESNCell(input_size if lyr == 0 else num_directions * output_size, output_size,
                        bias, scale_rec, density, scale_in, density_in,
                        scale_bias, leaking_rate, rescaling_method)
                for lyr in range(num_layers)
            ])
        super().__init__(input_size, output_size, cells, cells_bw, num_layers)


class MultiringESN(ESNBase):

    def __init__(self, input_size: int, output_size: int, num_layers=1, bias=True, bidirectional=False,
                 scale_rec=1.0, scale_in=1.0, density_in=1.0,
                 scale_bias=1.0, leaking_rate=1.0, topology='multiring'):

        num_directions = 2 if bidirectional else 1

        cells = ModuleList([
            ESNMultiringCell(input_size if lyr == 0 else num_directions * output_size, output_size,
                             bias, scale_rec, scale_in, density_in,
                             scale_bias, leaking_rate, topology)
            for lyr in range(num_layers)
        ])

        cells_bw = None
        if bidirectional:
            cells_bw = ModuleList([
                ESNMultiringCell(input_size if lyr == 0 else num_directions * output_size, output_size,
                                 bias, scale_rec, scale_in, density_in,
                                 scale_bias, leaking_rate, topology)
                for lyr in range(num_layers)
            ])

        super().__init__(input_size, output_size, cells, cells_bw, num_layers)
