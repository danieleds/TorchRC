import torch
import torch.nn as nn
from torch.nn import ModuleList
import torch.nn.functional as F
from typing import Optional, Sequence, Final


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
        # https://github.com/pytorch/pytorch/issues/4930#issuecomment-361851298

        if self.bidirectional:
            assert False, "Not implemented"
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


#
# class ESNBaseOld(torch.nn.Module):
#
#     def __init__(self, input_size, reservoir_size, cells: ModuleList, num_layers=1, dropout=0, bidirectional=False):
#         super(ESNBaseOld, self).__init__()
#
#         self.input_size = input_size
#         self.reservoir_size = reservoir_size
#         self.num_layers = num_layers
#         self.dropout = dropout
#         self.bidirectional = bidirectional
#         self.num_directions = 2 if bidirectional else 1
#
#         # If True, the last state is automatically preserved for subsequent forward() calls
#         self.preserve_last_state = False
#         self.last_state = None
#
#         if self.bidirectional and self.preserve_last_state:
#             raise Exception("bidirectional=True and preserve_last_states=True are incompatible.")
#
#         num_directions = 2 if bidirectional else 1
#
#         for layer in range(num_layers):
#             for direction in range(num_directions):
#                 layer_input_size = input_size if layer == 0 else reservoir_size * num_directions
#                 suffix = '_reverse' if direction == 1 else ''
#
#                 # Instantiate a cell and save it as an instance on this RNN
#                 cell = cell_provider(layer_input_size, reservoir_size, layer, direction)
#                 setattr(self, 'cell_l{}{}'.format(layer, suffix), cell)
#
#     def forward_long_sequence_yld(self, input, h_0=None):
#         """
#         For when you need to feed very long sequences and need all states, but one at a time.
#         :param input: (seq_len, batch, input_size)
#         :param h_0: (num_layers * num_directions, batch, hidden_size)
#
#         Yields tensors of shape (num_directions, batch, hidden_size)
#         """
#         if self.num_layers > 1:
#             raise Exception("forward_long_sequence is not compatible with multilayer reservoirs yet.")
#
#         device = input.device
#
#         batch = input.size(1)
#
#         if h_0 is None:
#             h_0 = torch.zeros((self.num_directions, batch, self.reservoir_size), device=device)
#
#         h_l = [h_0[i] for i in range(self.num_directions)]
#
#         cell_fw = self.get_cell(0, 0)
#         cell_bw = self.get_cell(0, 1) if self.bidirectional else None
#         for i in range(len(input)):
#             step_fw = input[i]
#             h_l[0] = cell_fw.forward(step_fw, h_l[0])  # (batch, hidden_size)
#
#             if self.bidirectional:
#                 step_bw = input[-i-1]
#                 h_l[1] = cell_bw.forward(step_bw, h_l[1])  # (batch, hidden_size)
#
#             if self.bidirectional:
#                 yield torch.stack((h_l[0], h_l[1]))
#             else:
#                 yield h_l[0].unsqueeze(0)
#
#     def forward_long_sequence(self, input, h_0=None, seq_lengths=None):
#         """
#         For when you need to feed very long sequences and you only care about the last states.
#         :param input: (seq_len, batch, input_size)
#         :param h_0: (num_layers * num_directions, batch, hidden_size)
#         :return: h_n, which is a tensor of shape (num_directions, batch, hidden_size). It is compatible with the
#                  output tensor h_n from the forward method, which has shape (num_layers * num_directions, batch, hidden_size).
#                  It contains the hidden state for the last step of the sequence.
#         """
#         batch = input.size(1)
#
#         last_forward_states = torch.zeros(batch, self.reservoir_size, device=input.device)
#         last_backward_states = None  # (batch, self.reservoir_size)
#
#         for i, h in enumerate(self.forward_long_sequence_yld(input, h_0=h_0)):
#             if seq_lengths is None:
#                 last_forward_states = h[0]
#             else:
#                 # Find all batch indices which have been completely processed
#                 done_idx = [ j for j in range(batch) if seq_lengths[j] == i+1 ]
#                 last_forward_states[done_idx] = h[0, done_idx]
#             if self.bidirectional:
#                 last_backward_states = h[1]
#
#         if self.bidirectional:
#             return torch.stack((last_forward_states, last_backward_states))
#         else:
#             return last_forward_states.unsqueeze(0)
#
#     def forward(self, input, h_0=None):
#         """
#
#         :param input: tensor of shape (seq_len, batch, input_size)
#         :param h_0: tensor of shape (num_layers * num_directions, batch, hidden_size)
#         :param seq_lengths: length of each non-padded sequence in input.
#         :param return_all_states: whether you need the state associated to each step of the sequence, or just the
#                 last one (i.e. for GPU memory limitations).
#         :return: a tuple (output, h_n):
#                 - output: a tensor of shape (seq_len, batch, num_directions * hidden_size).
#                     It contains only the output of the last layer, for both directions concatenated.
#                     If return_all_states == False, it is of shape (1, batch, num_directions * hidden_size)
#                     and only contains the last step of the sequence
#                 - h_n: a tensor of shape (num_layers * num_directions, batch, hidden_size).
#                     It contains the hidden state for the last step, in all layers
#         """
#         if self.preserve_last_state and h_0 is None and self.last_state is not None:
#             h_0 = self.last_state
#
#         x, h = self._forward_type_2(input, h_0)
#
#         if self.preserve_last_state:
#             self.last_state = h
#
#         return x, h
#
#     def _forward_type_1(self, input, h_0=None):
#         """
#         input: (seq_len, batch, input_size)
#         h_0: (num_layers * num_directions, batch, hidden_size)
#
#         output: (seq_len, batch, num_directions * hidden_size)  # Only the output of the last layer, for both directions
#                                                                   concatenated
#         h_n: (num_layers * num_directions, batch, hidden_size)  # Hidden state for the last step, in all layers
#         """
#         # TODO: Deep Bidirectional implemented as two completely separate dynamics that are joined at the end
#         # https://github.com/pytorch/pytorch/issues/4930#issuecomment-361851298
#         raise Exception("Not implemented")
#
#     def _forward_type_2(self, input, h_0=None):
#         """
#         input: (seq_len, batch, input_size)
#         h_0: (num_layers * num_directions, batch, hidden_size)
#
#         output: (seq_len, batch, num_directions * hidden_size)  # Only the output of the last layer, for both directions
#                                                                   concatenated
#         h_n: (num_layers * num_directions, batch, hidden_size)  # Hidden state for the last step, in all layers
#         """
#         # https://github.com/pytorch/pytorch/issues/4930#issuecomment-361851298
#
#         device = input.device
#
#         seq_len = input.size(0)
#         batch = input.size(1)
#
#         if h_0 is None:
#             h_0 = torch.zeros((self.num_layers * self.num_directions, batch, self.reservoir_size), device=device)
#
#         h_l = [h_0[i] for i in range(self.num_layers * self.num_directions)]
#
#         # The reservoir activation for the whole sequence at all layers, for all time steps.
#         prev_output = torch.zeros((self.num_layers, seq_len, batch, self.num_directions * self.reservoir_size),
#                                   device=device)
#         for l in range(self.num_layers):
#             # shape: (seq_len, batch, num_directions * hidden_size)
#             layer_input = input if l == 0 else prev_output[l-1]
#
#             if l > 0 and self.dropout > 0:
#                 layer_input = F.dropout(layer_input, p=self.dropout)
#
#             forward_states = []  # list of seq_len tensors of size (batch, hidden_size)
#             backward_states = []  # list of seq_len tensors of size (batch, hidden_size)
#
#             # Forward pass
#             cell = getattr(self, 'cell_l{}'.format(l))
#             for step in layer_input:
#                 input_state = h_l[self.num_directions * l + 0]
#                 h = cell.forward(step, input_state)  # (batch, hidden_size)
#                 h_l[self.num_directions * l + 0] = h
#                 forward_states = forward_states + [h]
#
#             if self.bidirectional:
#                 # Backward pass
#                 cell = getattr(self, 'cell_l{}_reverse'.format(l))
#                 for step in reversed(layer_input):
#                     input_state = h_l[self.num_directions * l + 1]
#                     h = cell.forward(step, input_state)  # (batch, hidden_size)
#                     h_l[self.num_directions * l + 1] = h
#                     backward_states = [h] + backward_states
#
#             if self.bidirectional:
#                 prev_output_forward = torch.stack(forward_states)  # (seq_len, batch, hidden_size)
#                 prev_output_backward = torch.stack(backward_states)  # (seq_len, batch, hidden_size)
#                 prev_output[l] = torch.cat((prev_output_forward, prev_output_backward), dim=2)
#             else:
#                 prev_output[l] = torch.stack(forward_states)
#
#         return prev_output[-1], torch.stack(h_l)
#
#     def extract_last_states(self, states, seq_lengths=None):
#         """
#         Given the output of the GResNet for the whole sequence of a batch,
#         extracts the values associated to the last time step.
#         When seq_lengths is provided, this method takes particular care in ignoring the time steps
#         associated with padding inputs, also in the case of a bidirectional architecture. For example,
#         assume "P" indicates any state computed from a padding input, f1, ... fn are states computed in the
#         forward pass, and b1, ... bn are states computed in the backward pass. Then, this method transforms
#         the concatenated bidirectional output
#
#                 [ f1 f2 ... fn P P P | b1 b2 ... bn P P P ]
#
#         into
#
#                 [ fn b1 ]
#
#         while a naive approach would have selected
#
#                 [ P b1 ]
#
#         as the final state.
#
#         :param states: (seq_len, batch, num_directions * hidden_size)
#         :param seq_lengths: integer list, whose element i represents the length of the input sequence
#                             originally associated to the i-th state sequence in the 'states' minibatch.
#         :return:  (batch, num_directions * hidden_size)
#         """
#         if seq_lengths is None or len(seq_lengths) == 1:
#             return states[-1, :, :]
#
#         max_seq_len = states.shape[0]
#         batch_size = states.shape[1]
#
#         if self.bidirectional:
#             states = states.view(max_seq_len, batch_size, 2, -1)
#
#             final_states = []
#             for i in range(len(seq_lengths)):
#                 fw = states[seq_lengths[i] - 1, i, 0, :]
#                 bw = states[0, i, 1, :]
#                 final_states += [torch.cat([fw, bw], dim=0)]
#
#             return torch.stack(final_states)
#
#         else:
#             final_states = []
#             for i in range(len(seq_lengths)):
#                 fw = states[seq_lengths[i] - 1, i, :]
#                 final_states += [fw]
#
#             return torch.stack(final_states)
#
#     # def state_size(self):
#     #     return self.reservoir_size * self.num_directions
#
#     def get_cell(self, layer: int, direction: int = 0) -> torch.nn.Module:
#         """
#         Returns one of the cells in this model
#         :param layer: starts from zero
#         :param direction:
#         :return:
#         """
#         suffix = '_reverse' if direction == 1 else ''
#         return getattr(self, 'cell_l{}{}'.format(layer, suffix))
#
#     def get_cells(self) -> Sequence[torch.nn.Module]:
#         """
#         Returns all cells in this model
#         :return:
#         """
#         for layer in range(self.num_layers):
#             for direction in range(self.num_directions):
#                 yield self.get_cell(layer, direction=direction)


class LeakyESN(ESNBase):

    def __init__(self, input_size: int, output_size: int, num_layers=1, bias=True, bidirectional=False,
                 scale_rec=0.9, density=1.0, scale_in=1.0, density_in=1.0,
                 scale_bias=1.0, leaking_rate=1.0, rescaling_method='specrad'):

        assert not bidirectional, "Not implemented"

        cells = ModuleList([
            ESNCell(input_size if lyr == 0 else output_size, output_size,
                    bias, scale_rec, density, scale_in, density_in,
                    scale_bias, leaking_rate, rescaling_method)
            for lyr in range(num_layers)
        ])
        cells_bw = None
        super().__init__(input_size, output_size, cells, cells_bw, num_layers)


class MultiringESN(ESNBase):

    def __init__(self, input_size: int, output_size: int, num_layers=1, bias=True, bidirectional=False,
                 scale_rec=1.0, scale_in=1.0, density_in=1.0,
                 scale_bias=1.0, leaking_rate=1.0, topology='multiring'):

        assert not bidirectional, "Not implemented"

        cells = ModuleList([
            ESNMultiringCell(input_size if lyr == 0 else output_size, output_size,
                             bias, scale_rec, scale_in, density_in,
                             scale_bias, leaking_rate, topology)
            for lyr in range(num_layers)
        ])
        cells_bw = None
        super().__init__(input_size, output_size, cells, cells_bw, num_layers)
