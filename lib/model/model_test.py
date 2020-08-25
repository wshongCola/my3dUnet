import unittest
from lib.model.network import UNet3D
import numpy as np
import torch
from torch.autograd import Variable
from torch import optim


class MyTestCase(unittest.TestCase):
    def test_something(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = Variable(torch.ones([8, 1, 32, 64, 160])).to(device)
        model = UNet3D(1, 1).to(device)
        out = model(data)
        print(out.detach().cpu().numpy().shape)


if __name__ == '__main__':
    unittest.main()
