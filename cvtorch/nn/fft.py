import torch
import torch.nn as nn

# TODO: add support for CPLX

class FFTBlock(nn.Module):
    """FFT Block."""
    def __init__(self, n=None, dim=-1, norm=None):
        super(FFTBlock, self).__init__()
        
        self.n = n
        self.dim = dim
        self.norm = norm
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft(x, n=self.n, dim=self.dim, norm=self.norm)
    
class IFFTBlock(nn.Module):
    """IFFT Block."""
    def __init__(self, n=None, dim=-1, norm=None):
        super(IFFTBlock, self).__init__()
        
        self.n = n
        self.dim = dim
        self.norm = norm
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifft(x, n=self.n, dim=self.dim, norm=self.norm)

