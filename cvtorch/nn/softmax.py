import torch.nn as nn

from cvtorch import CVTensor

class CVSoftMax(nn.Module):
    """Complex-Valued Softmax Layer."""
    def __init__(self, dim=None):
        super(CVSoftMax, self).__init__()
        
        self.softmax_r = nn.Softmax(dim)
        self.softmax_i = nn.Softmax(dim)
        
    def forward(self, x: CVTensor) -> CVTensor:
        return CVTensor(self.softmax_r(x.real), self.softmax_i(x.imag))
    
class MagSoftMax(nn.Module):
    """Magnitude Softmax Layer."""
    def __init__(self, dim=None):
        super(MagSoftMax, self).__init__()
        
        self.softmax = nn.Softmax(dim)
        
    def forward(self, x: CVTensor) -> CVTensor:
        return self.softmax(x.abs())
    
class MagMinMaxNorm(nn.Module):
    """Magnitude Min-Max Normalization Layer."""
    def __init__(self, dim=None):
        super(MagMinMaxNorm, self).__init__()
        
        self.dim = dim
        
    def forward(self, x: CVTensor) -> CVTensor:
        x_mag = x.abs()
        x_min = x_mag.min(self.dim, keepdim=True)[0]
        x_max = x_mag.max(self.dim, keepdim=True)[0]
        out = (x - x_min) / (x_max - x_min)
        return CVTensor(out.real, out.imag)
