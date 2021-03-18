import unittest
import torch

from torch_rc.nn import Linear
from torch_rc.optim import RidgeClassification, RidgeIncrementalClassification


class TestGRU(unittest.TestCase):

    def test_ridge_classification(self):
        x = torch.as_tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 0],
            [1, 1, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
        ]).to(torch.float)
        y = torch.as_tensor([1, 1, 0, 0, 1, 1, 0, 0])  # not(x[:, 2])
        layer = Linear(4, 2)
        optim = RidgeClassification(layer, l2_reg=0)
        optim.fit(x, y)
        self.assertTrue(layer(torch.FloatTensor([[1, 1, 1, 0]])).argmax(-1) == 0)
        self.assertTrue(layer(torch.FloatTensor([[1, 1, 0, 0]])).argmax(-1) == 1)
        self.assertTrue(layer(torch.FloatTensor([[1, 1, 1, 1]])).argmax(-1) == 0)
        self.assertTrue(layer(torch.FloatTensor([[0, 0, 0, 0]])).argmax(-1) == 1)

    def test_ridge_incremental_classification(self):
        x = torch.as_tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 0],
            [1, 1, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
        ]).to(torch.float)
        y = torch.as_tensor([1, 1, 0, 0, 1, 1, 0, 0])  # not(x[:, 2])
        layer = Linear(4, 2)
        optim = RidgeIncrementalClassification(layer, l2_reg=0)
        for i in range(len(x)):
            optim.fit_step(x[i:i+1, :], y[i:i+1])
        optim.fit_apply()
        self.assertTrue(layer(torch.FloatTensor([[1, 1, 1, 0]])).argmax(-1) == 0)
        self.assertTrue(layer(torch.FloatTensor([[1, 1, 0, 0]])).argmax(-1) == 1)
        self.assertTrue(layer(torch.FloatTensor([[1, 1, 1, 1]])).argmax(-1) == 0)
        self.assertTrue(layer(torch.FloatTensor([[0, 0, 0, 0]])).argmax(-1) == 1)


if __name__ == '__main__':
    unittest.main()
