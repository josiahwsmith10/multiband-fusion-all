import torch.nn as nn

from cvtorch import CVTensor

class CVDropout(nn.Module):
    """Complex-Valued Dropout Layer."""
    def __init__(self, p=0.5, inplace=False):
        super(CVDropout, self).__init__()
        
        self.dropout_r = nn.Dropout(p, inplace)
        self.dropout_i = nn.Dropout(p, inplace)
        
    def forward(self, x: CVTensor) -> CVTensor:
        return CVTensor(self.dropout_r(x.real), self.dropout_i(x.imag))

