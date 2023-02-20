import torch
import torch.nn as nn

from model.common import CPLX

def complex_default_conv1d(in_channels, out_channels, kernel_size, bias=False):
    """Default complex-valued 1D convolution."""
    return CVConv1d(
        in_channels=in_channels, out_channels=out_channels, 
        kernel_size=kernel_size, padding=kernel_size//2, bias=bias
    )
    
class CVConv1d(nn.Module):
    """Complex-valued 1D convolution."""
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, 
        padding=0, dilation=1, groups=1, bias=False
    ):
        super(CVConv1d, self).__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, dtype=torch.complex64)
        
    def forward(self, x):
        x = self.conv(x.complex)
        return CPLX(x.real, x.imag)
    
def naive_complex_default_conv1d(in_channels, out_channels, kernel_size, bias=False):
    """Default naive complex-valued 1D convolution."""
    return NaiveCVConv1d(
        in_channels=in_channels, out_channels=out_channels, 
        kernel_size=kernel_size, padding=kernel_size//2, bias=bias
    )
    
class NaiveCVConv1d(nn.Module):
    """Naive complex-valued 1D convolution."""
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, 
        padding=0, dilation=1, groups=1, bias=False
    ):
        super(NaiveCVConv1d, self).__init__()
        
        self.conv_r = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
    def forward(self, x):
        return CPLX(self.conv_r(x.real) - self.conv_i(x.imag), self.conv_r(x.imag) + self.conv_i(x.real))
    
def fast_complex_default_conv1d(in_channels, out_channels, kernel_size, bias=False):
    """Default fast complex-valued 1D convolution."""
    return FastCVConv1d(
        in_channels=in_channels, out_channels=out_channels, 
        kernel_size=kernel_size, padding=kernel_size//2, bias=bias
    )

def fast_complex_conv(conv, input, weight, stride=1, padding=0, dilation=1):
    """Fast complex-valued convolution."""
    n_out = int(weight.shape[0])

    ww = torch.cat([weight.real, weight.imag], dim=0)
    wr = conv(input.real, ww, None, stride, padding, dilation, 1)
    wi = conv(input.imag, ww, None, stride, padding, dilation, 1)
    
    rwr, iwr = wr[:, :n_out], wr[:, n_out:]
    rwi, iwi = wi[:, :n_out], wi[:, n_out:]
    return CPLX(rwr - iwi, iwr + rwi)

class FastCVConv1d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, 
        padding=0, dilation=1, groups=1, bias=False
    ):
        super(FastCVConv1d, self).__init__()
        assert groups==1, "Groups are not supported."
        
        self.conv_r = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.Fconv = torch.functional.F.conv1d
        
    def forward(self, x):
        return fast_complex_conv(self.Fconv, x, CPLX(self.conv_r.weight, self.conv_i.weight), self.conv_r.stride, self.conv_r.padding, self.conv_r.dilation)

class CPReLU(nn.Module):
    """Complex-Valed Parametric Rectified Linear Unit."""
    def __init__(self):
        super(CPReLU, self).__init__()
        self.cprelu_r = nn.PReLU()
        self.cprelu_i = nn.PReLU()
        
    def forward(self, x):
        return CPLX(self.cprelu_r(x.real), self.cprelu_i(x.imag))
        #return self.cprelu_r(x.real) + 1j*self.cprelu_i(x.imag)

class CReLU(nn.Module):
    """Complex-Valued Rectified Linear Unit."""
    def __init__(self, inplace=True):
        super(CReLU, self).__init__()
        self.relu_r = nn.ReLU(inplace)
        self.relu_i = nn.ReLU(inplace)
        
    def forward(self, x):
        return self.relu_r(x.real) + 1j*self.relu_i(x.imag)
    
class CTanh(nn.Module):
    """Complex-Valued Hyperbolic Tangent."""
    def __init__(self):
        super(CTanh, self).__init__()
        self.tanh_r = nn.Tanh()
        self.tanh_i = nn.Tanh()
        
    def forward(self, x):
        return self.tanh_r(x.real) + 1j*self.relu_i(x.imag)

class FFTBlock(nn.Module):
    """FFT Block."""
    def __init__(self, n=None, dim=-1, norm=None):
        super(FFTBlock, self).__init__()
        
        self.n = n
        self.dim = dim
        self.norm = norm
        
    def forward(self, x):
        return torch.fft.fft(x, n=self.n, dim=self.dim, norm=self.norm)
    
class IFFTBlock(nn.Module):
    """IFFT Block."""
    def __init__(self, n=None, dim=-1, norm=None):
        super(IFFTBlock, self).__init__()
        
        self.n = n
        self.dim = dim
        self.norm = norm
        
    def forward(self, x):
        return torch.fft.ifft(x, n=self.n, dim=self.dim, norm=self.norm)

class ComplexValuedResBlock(nn.Module):
    """Complex-Valued Residual Block."""
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=False, act=CTanh(), res_scale=1):

        super(ComplexValuedResBlock, self).__init__()
        
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if i == 0:
                if type(act) == CPReLU:
                    m.append(CPReLU())
                else:
                    m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        # Complex residual block forward
        res = self.body(x).mul(self.res_scale)
        res.add(x)

        return res
    
class kRBlock(nn.Module):
    """kR-Block."""
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=False, act=CTanh(), res_scale=1, n_res_blocks=8):

        super(kRBlock, self).__init__()
        
        m = [
            ComplexValuedResBlock(
                conv, in_channels, kernel_size, bias=bias,
                act=act, res_scale=res_scale
            ) for _ in range(n_res_blocks)
        ]
        m.append(conv(in_channels, out_channels, kernel_size))
        
        self.body = nn.Sequential(*m)

    def forward(self, x):
        return self.body(x)

def kRBlockNEW(conv, in_channels, out_channels, kernel_size, bias=False, act=CTanh(), res_scale=1, n_res_blocks=8):
    """kR-Block."""
    m = [
        ComplexValuedResBlock(
            conv, in_channels, kernel_size, bias=bias,
            act=act, res_scale=res_scale
        ) for _ in range(n_res_blocks)
    ]
    m.append(conv(in_channels, out_channels, kernel_size))
    
    return nn.Sequential(*m)


