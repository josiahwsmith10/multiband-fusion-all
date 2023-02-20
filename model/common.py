import torch

class CPLX:
    def __init__(self, r, i):
        self.real = r
        self.imag = i
    
    def mul(self, other):
        if type(other) == CPLX:
            r = self.real*other.real - self.imag*other.imag
            i = self.real*other.imag + self.imag*other.real
            self.real, self.imag = r, i
        else:
            self.real *= other
            self.imag *= other
        return self
    
    def add(self, other):
        if type(other) == CPLX:
            self.real += other.real
            self.imag += other.imag
        else:
            x = self.complex + other
            self.real, self.imag = x.real, x.imag
    
    def view(self, *shape):
        self.real = self.real.view(*shape)
        self.imag = self.imag.view(*shape)
        return self
    
    def fft(self, n=None, dim=-1, norm="ortho"):
        out = torch.fft.fft(self.complex, n, dim, norm)
        self.real, self.imag = out.real, out.imag
        return self
        
    def ifft(self, n=None, dim=-1, norm="ortho"):
        out = torch.fft.ifft(self.complex, n, dim, norm)
        self.real, self.imag = out.real, out.imag
        return self
        
    def device(self):
        assert self.real.device == self.imag.device
        return self.real.device
    
    def to(self, device):
        self.real = self.real.to(device)
        self.imag = self.imag.to(device)
        return self
        
    def half(self):
        self.real = self.real.half()
        self.imag = self.imag.half()
        return self

    def abs(self):
        return self.complex.abs()
    
    @property
    def shape(self):
        assert self.real.shape == self.imag.shape
        return self.real.shape
    
    @property
    def complex(self):
        out = self.real + 1j*self.imag
        return out

