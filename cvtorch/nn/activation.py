import torch.nn as nn

from cvtorch import CVTensor

class CPReLU(nn.Module):
    """Complex-Valed Parametric Rectified Linear Unit."""
    def __init__(self):
        super(CPReLU, self).__init__()
        self.cprelu_r = nn.PReLU()
        self.cprelu_i = nn.PReLU()
        
    def forward(self, x: CVTensor) -> CVTensor:
        return CVTensor(self.cprelu_r(x.real), self.cprelu_i(x.imag))

class CReLU(nn.Module):
    """Split Complex-Valued Rectified Linear Unit."""
    def __init__(self, inplace=True):
        super(CReLU, self).__init__()
        self.relu_r = nn.ReLU(inplace)
        self.relu_i = nn.ReLU(inplace)
        
    def forward(self, x: CVTensor) -> CVTensor:
        return CVTensor(self.relu_r(x.real), self.relu_i(x.imag))
    
class CTanh(nn.Module):
    """Split Complex-Valued Hyperbolic Tangent."""
    def __init__(self):
        super(CTanh, self).__init__()
        self.tanh_r = nn.Tanh()
        self.tanh_i = nn.Tanh()
        
    def forward(self, x: CVTensor) -> CVTensor:
        return CVTensor(self.tanh_r(x.real), self.relu_i(x.imag))

class CSigmoid(nn.Module):
    """Split Complex-Valued Sigmoid."""
    def __init__(self):
        super(CSigmoid, self).__init__()
        self.sigmoid_r = nn.Sigmoid()
        self.sigmoid_i = nn.Sigmoid()
        
    def forward(self, x: CVTensor) -> CVTensor:
        return CVTensor(self.sigmoid_r(x.real), self.sigmoid_i(x.imag))

