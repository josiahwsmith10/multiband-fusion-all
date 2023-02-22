import torch
import torch.nn as nn
import torch.functional as F

from cvtorch import CVTensor

class SSIM(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X: CVTensor, Y: CVTensor, data_range=None, full=False):
        assert isinstance(self.w, torch.Tensor)

        X = X.abs()
        Y = Y.abs()
        if data_range is None:
            data_range = torch.ones_like(Y) #* Y.max()
            p = (self.win_size - 1)//2
            data_range = data_range[:, :, p:-p, p:-p]
        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)  # typing: ignore
        uy = F.conv2d(Y, self.w)  #
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        if full:
            return S
        else:
            return S.mean()

class PerpLossSSIM(nn.Module):
    """From perp-loss paper 2022 Terpstra et al."""
    def __init__(self):
        super().__init__()
        
        self.ssim = SSIM()
        self.param = nn.Parameter(torch.ones(1)/2)
        
    def forward(self, X: CVTensor, Y: CVTensor):

        mag_input = torch.abs(X)
        mag_target = torch.abs(Y)
        cross = torch.abs(X.real * Y.imag - X.imag * Y.real)

        angle = torch.atan2(X.imag, X.real) - torch.atan2(Y.imag, Y.real)
        ploss = torch.abs(cross) / (mag_input + 1e-8)

        aligned_mask = (torch.cos(angle) < 0).bool()

        final_term = torch.zeros_like(ploss)
        final_term[aligned_mask] = mag_target[aligned_mask] + (mag_target[aligned_mask] - ploss[aligned_mask])
        final_term[~aligned_mask] = ploss[~aligned_mask]
        ssim_loss = (1 - self.ssim(X, Y))/ mag_input.shape[0]
        
        return (final_term.mean()*torch.clamp(self.param, 0, 1) + (1-torch.clamp(self.param, 0, 1))*ssim_loss)

class SplitL1(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1 = nn.L1Loss()
        
    def forward(self, x: CVTensor, y: CVTensor):
        return self.L1(x.real, y.real) + self.L1(x.imag, y.imag)
    
class SplitMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.MSE = nn.MSELoss()
        
    def forward(self, x: CVTensor, y: CVTensor):
        return self.MSE(x.real, y.real) + self.MSE(x.imag, y.imag)