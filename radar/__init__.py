import torch

from .multiradar import MultiRadar


def fft(x: torch.Tensor, n=None, dim=-1, norm="ortho") -> torch.Tensor:
    return torch.fft.fft(x, n=n, dim=dim, norm=norm)


def ifft(x: torch.Tensor, n=None, dim=-1, norm="ortho") -> torch.Tensor:
    return torch.fft.ifft(x, n=n, dim=dim, norm=norm)
