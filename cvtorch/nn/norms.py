import torch.nn as nn

from cvtorch import CVTensor

class CVLayerNorm(nn.Module):
    """Complex-Valued Layer Normalization.
        TODO: Implement complex-valued layer normalization."""
    def __init__(self, d_model, eps=1e-6):
        super(CVLayerNorm, self).__init__()
        self.LN_r = nn.LayerNorm(d_model, eps=eps)
        self.LN_i = nn.LayerNorm(d_model, eps=eps)
        
    def forward(self, x: CVTensor) -> CVTensor:
        return CVTensor(self.LN_r(x.real), self.LN_i(x.imag))

