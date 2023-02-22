import torch
import torch.nn as nn

from cvtorch import CVTensor

def default_slow_cvconv1d(in_channels, out_channels, kernel_size, bias=False):
    """
    Default complex-valued 1D convolution.
        - Implemented using torch.nn.Conv1d and complex-valued tensors.
        - slower than using CVTensor
    """
    return SlowCVConv1d(
        in_channels=in_channels, out_channels=out_channels, 
        kernel_size=kernel_size, padding=kernel_size//2, bias=bias
    )
    
class SlowCVConv1d(nn.Module):
    """Complex-valued 1D convolution."""
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, 
        padding=0, dilation=1, groups=1, bias=False
    ):
        super(SlowCVConv1d, self).__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, dtype=torch.complex64)
        
    def forward(self, x: CVTensor) -> CVTensor:
        x = self.conv(x.complex)
        return CVTensor(x.real, x.imag)
    
def default_cvconv1d(in_channels, out_channels, kernel_size, bias=False):
    """Default naive complex-valued 1D convolution."""
    return CVConv1d(
        in_channels=in_channels, out_channels=out_channels, 
        kernel_size=kernel_size, padding=kernel_size//2, bias=bias
    )
    
class CVConv1d(nn.Module):
    """CVTensor-based complex-valued 1D convolution."""
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, 
        padding=0, dilation=1, groups=1, bias=False
    ):
        super(CVConv1d, self).__init__()
        
        self.conv_r = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
    def forward(self, x: CVTensor) -> CVTensor:
        return CVTensor(self.conv_r(x.real) - self.conv_i(x.imag), self.conv_r(x.imag) + self.conv_i(x.real))

