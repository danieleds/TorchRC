import unittest
import torch

from torch_rc.nn.esn import ESNCell, LeakyESN, MultiringESN


class TestESN(unittest.TestCase):

    def test_esn_cell_is_scriptable(self):
        cell = ESNCell(2, 4)
        torch.jit.script(cell)

    def test_esn(self):
        esn = torch.jit.script(LeakyESN(1, 10, num_layers=3))
        esn.forward(torch.rand(15, 8, 1))

    def test_multiring_esn(self):
        esn = torch.jit.script(MultiringESN(1, 10, num_layers=3))
        esn.forward(torch.rand(15, 8, 1))


if __name__ == '__main__':
    unittest.main()
