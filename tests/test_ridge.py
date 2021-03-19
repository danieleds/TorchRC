import unittest
import torch

from torch_rc.nn import Linear
from torch_rc.optim import RidgeClassification, RidgeIncrementalClassification, RidgeRegression,\
    RidgeIncrementalRegression


class TestRidge(unittest.TestCase):

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

    def test_ridge_regression(self):
        x = torch.as_tensor([
            [0.5, 0.6, 0.2, 0.6],
            [0.6, 0.2, 0.2, 0.3],
            [0.0, 0.9, 0.2, 0.6],
            [0.3, 0.3, 0.1, 0.3],
            [0.7, 0.2, 0.1, 0.1],
            [0.8, 0.3, 0.2, 0.3],
            [0.0, 0.0, 0.0, 0.5],
            [0.1, 0.3, 0.3, 0.7],
        ])
        y = (0.5 * x[:, 0] + -1 * x[:, 1] + 2 * x[:, 2] + -0.1 * x[:, 3]).view(-1, 1)
        layer = Linear(4, 1)
        optim = RidgeRegression(layer, l2_reg=0)
        optim.fit(x, y)
        self.assertTrue(torch.abs(layer(torch.as_tensor([[0.1, 0.3, 0.3, 0.7]])) - 0.28) < 1e-4)
        self.assertTrue(torch.abs(layer(torch.as_tensor([[0.4, 0.3, 0.2, 0.1]])) - 0.29) < 1e-4)
        self.assertTrue(torch.abs(layer(torch.as_tensor([[0.8, 0.9, 1.0, 1.1]])) - 1.39) < 1e-4)
        self.assertTrue(torch.abs(layer(torch.as_tensor([[0.5, -0.3, 0.0, 0.5]])) - 0.5) < 1e-4)

    def test_ridge_incremental_regression(self):
        x = torch.as_tensor([
            [0.5, 0.6, 0.2, 0.6],
            [0.6, 0.2, 0.2, 0.3],
            [0.0, 0.9, 0.2, 0.6],
            [0.3, 0.3, 0.1, 0.3],
            [0.7, 0.2, 0.1, 0.1],
            [0.8, 0.3, 0.2, 0.3],
            [0.0, 0.0, 0.0, 0.5],
            [0.1, 0.3, 0.3, 0.7],
        ])
        y = (0.5 * x[:, 0] + -1 * x[:, 1] + 2 * x[:, 2] + -0.1 * x[:, 3]).view(-1, 1)
        layer = Linear(4, 1)
        optim = RidgeIncrementalRegression(layer, l2_reg=0)
        for i in range(len(x)):
            optim.fit_step(x[i:i+1, :], y[i:i+1, :])
        optim.fit_apply()
        self.assertTrue(torch.abs(layer(torch.as_tensor([[0.1, 0.3, 0.3, 0.7]])) - 0.28) < 1e-4)
        self.assertTrue(torch.abs(layer(torch.as_tensor([[0.4, 0.3, 0.2, 0.1]])) - 0.29) < 1e-4)
        self.assertTrue(torch.abs(layer(torch.as_tensor([[0.8, 0.9, 1.0, 1.1]])) - 1.39) < 1e-4)
        self.assertTrue(torch.abs(layer(torch.as_tensor([[0.5, -0.3, 0.0, 0.5]])) - 0.5) < 1e-4)


if __name__ == '__main__':
    unittest.main()
