import numpy as np
import torch
import torch.nn as nn
import argparse

from cvtorch import CVTensor
import cvtorch.nn as cvnn

class SignalSelfCrossAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int, d_k: int, d_v: int, dropout: float = 0.1, SoftMaxClass=cvnn.MagMinMaxNorm,
                 conv=cvnn.default_cvconv1d, kernel_size=3):
        super(SignalSelfCrossAttention, self).__init__()
        
        self.AA = cvnn.CVMultiheadAttention(n_head, d_model, d_k, d_v, dropout, SoftMaxClass)
        self.AB = cvnn.CVMultiheadAttention(n_head, d_model, d_k, d_v, dropout, SoftMaxClass)
        self.BA = cvnn.CVMultiheadAttention(n_head, d_model, d_k, d_v, dropout, SoftMaxClass)
        self.BB = cvnn.CVMultiheadAttention(n_head, d_model, d_k, d_v, dropout, SoftMaxClass)
        
        self.conv = conv(in_channels=d_model*4, out_channels=d_model, kernel_size=kernel_size, bias=False)

    def forward(self, A: CVTensor, B: CVTensor):
        A, B = A.transpose(1, 2), B.transpose(1, 2)
        
        W = self.AA(A, A, A)
        X = self.BB(B, B, B)
        Y = self.AB(A, B, B)
        Z = self.BA(B, A, A)
        
        X = torch.cat([W.complex, X.complex, Y.complex, Z.complex], dim=2).transpose(1, 2)
        
        X = CVTensor(X.real, X.imag)
        
        return self.conv(X)

class SSCANet(nn.Module):
    """Signal self- and cross-attention model."""
    def __init__(self, args: argparse.Namespace, conv=cvnn.default_cvconv1d):
        super(SSCANet, self).__init__()