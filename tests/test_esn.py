import unittest
import torch

from torch_rc.nn.esn import ESNCell, LeakyESN, MultiringESN


class TestESN(unittest.TestCase):

    def test_esn_cell_is_scriptable(self):
        cell = ESNCell(2, 4)
        torch.jit.script(cell)

    def test_esn(self):
        esn = torch.jit.script(LeakyESN(1, 10, num_layers=3))
        out, h = esn.forward(torch.rand(15, 8, 1))
        self.assertEqual(tuple(out.shape), (15, 8, 1 * 10))
        self.assertEqual(tuple(h.shape), (3 * 1, 8, 10))

    def test_bi_esn(self):
        esn = torch.jit.script(LeakyESN(1, 10, num_layers=3, bidirectional=True))
        out, h = esn.forward(torch.rand(15, 8, 1))
        self.assertEqual(tuple(out.shape), (15, 8, 2 * 10))
        self.assertEqual(tuple(h.shape), (3 * 2, 8, 10))

    def test_multiring_esn(self):
        esn = torch.jit.script(MultiringESN(1, 10, num_layers=3))
        out, h = esn.forward(torch.rand(15, 8, 1))
        self.assertEqual(tuple(out.shape), (15, 8, 1 * 10))
        self.assertEqual(tuple(h.shape), (3 * 1, 8, 10))

    def test_bi_multiring_esn(self):
        esn = torch.jit.script(MultiringESN(1, 10, num_layers=3, bidirectional=True))
        out, h = esn.forward(torch.rand(15, 8, 1))
        self.assertEqual(tuple(out.shape), (15, 8, 2 * 10))
        self.assertEqual(tuple(h.shape), (3 * 2, 8, 10))


if __name__ == '__main__':
    unittest.main()
