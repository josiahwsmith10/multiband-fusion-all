import torch
from scipy.constants import pi, c

class TIRadar():
    """Texas Instruments Radar class."""
    def __init__(self, f0=77e9, K=124.996e12, Nk=64, fS=2000e3):
        self.f0 = f0
        self.K = K
        self.Nk = Nk
        self.fS = fS
        self.Compute()
        
    def __str__(self):
        return f"TI Radar with f0={self.f0*1e-9} GHz, K={self.K*1e-12} MHz/us, Nk={self.Nk}, fS={self.fS*1e-3} ksps"
        
    def Compute(self):
        self.f = self.f0 + self.K/self.fS*(torch.arange(self.Nk))
        self.k = 2*pi/c*self.f
        self.range_max_m = self.fS*c/(2*self.K)
        self.f_c = self.f[self.Nk//2].numpy()
        self.lambda_c = c/self.f_c

