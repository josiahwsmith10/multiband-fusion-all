import torch.nn as nn

from cvtorch import CVTensor

class CVLinear(nn.Module):
    """Complex-Valued Linear Layer."""
    def __init__(self, in_features, out_features, bias=False):
        super(CVLinear, self).__init__()
        
        self.linear_r = nn.Linear(in_features, out_features, bias)
        self.linear_i = nn.Linear(in_features, out_features, bias)
        
    def forward(self, x: CVTensor) -> CVTensor:
        return CVTensor(self.linear_r(x.real) - self.linear_i(x.imag), self.linear_r(x.imag) + self.linear_i(x.real))

