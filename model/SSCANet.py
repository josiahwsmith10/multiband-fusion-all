import torch
import torch.nn as nn
import argparse
from typing import Tuple

import cvtorch
from cvtorch import CVTensor
import cvtorch.nn as cvnn

from model.common import select_act
from model.kRNet import kRBlock

class SignalSelfCrossAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int, d_k: int, d_v: int, dropout: float = 0.1, SoftMaxClass=cvnn.MagMinMaxNorm,
                 conv=cvnn.default_cvconv1d, kernel_size=3, b=1, gamma=2):
        super(SignalSelfCrossAttention, self).__init__()
        
        self.AA = cvnn.CVMultiheadAttention(n_head, d_model, d_k, d_v, dropout, SoftMaxClass)
        self.AB = cvnn.CVMultiheadAttention(n_head, d_model, d_k, d_v, dropout, SoftMaxClass)
        self.BA = cvnn.CVMultiheadAttention(n_head, d_model, d_k, d_v, dropout, SoftMaxClass)
        self.BB = cvnn.CVMultiheadAttention(n_head, d_model, d_k, d_v, dropout, SoftMaxClass)
        
        self.chan_attn = cvnn.CVECA(channels=d_model*4, b=b, gamma=gamma)
        
        self.conv = conv(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size, bias=False)

    def forward(self, A: CVTensor, B: CVTensor):
        A, B = A.transpose(1, 2), B.transpose(1, 2)
        
        W = self.AA(A, A, A)
        X = self.BB(B, B, B)
        Y = self.AB(A, B, B)
        Z = self.BA(B, A, A)
        
        X = cvtorch.cat((W, X, Y, Z), dim=1).transpose(1, 2)
        
        X = self.chan_attn(X)
        
        return self.conv(X)

class SSCANet_Small(nn.Module):
    """Small signal self- and cross-attention model (x4 SR)."""
    def __init__(self, args: argparse.Namespace, mr, conv=cvnn.default_cvconv1d, SoftMaxClass=cvnn.MagMinMaxNorm):
        super(SSCANet_Small, self).__init__()
        
        assert mr.N_subbands == 2, "SSCANet only supports 2 subbands."

        self.args = args
        self.mr = mr
        
        self.act = select_act(args)
        
        d_model = args.n_feats
        B = args.n_res_blocks
        
        in_channels = args.in_channels
        out_channels = args.out_channels
        kernel_size = args.kernel_size
        act = self.act
        
        n_head = args.n_head
        d_k = args.d_k
        d_v = args.d_v
        dropout = args.dropout
        
        b = args.eca_b
        gamma = args.eca_gamma
        
        self.d_model = d_model
        
        self.A_head = conv(in_channels=in_channels, out_channels=d_model, kernel_size=kernel_size)
        self.B_head = conv(in_channels=in_channels, out_channels=d_model, kernel_size=kernel_size)
        
        self.R_SSCA = SignalSelfCrossAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, dropout=dropout, SoftMaxClass=SoftMaxClass,
                                               conv=conv, kernel_size=kernel_size, b=b, gamma=gamma)
        self.k_SSCA = SignalSelfCrossAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, dropout=dropout, SoftMaxClass=SoftMaxClass,
                                               conv=conv, kernel_size=kernel_size, b=b, gamma=gamma)
        
        self.R_CVECA = cvnn.CVECA(channels=d_model, b=b, gamma=gamma)
        self.k_CVECA = cvnn.CVECA(channels=d_model, b=b, gamma=gamma)
        
        self.refinement = kRBlock(conv=conv, in_channels=4*d_model, out_channels=out_channels, kernel_size=kernel_size, act=act, n_res_blocks=B)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get radar 1 and radar 2 data
        A = x[:, self.mr.idx[0]].unsqueeze(1)
        B = x[:, self.mr.idx[1]].unsqueeze(1)
        
        A = self.A_head(CVTensor(A.real, A.imag))
        B = self.B_head(CVTensor(B.real, B.imag))
        
        R = self.R_SSCA(A, B)
        k = self.k_SSCA(A, B)
        
        R = cvtorch.cat((R, k.fft()), dim=1) 
        k = cvtorch.cat((k, R[:, :self.d_model].ifft()), dim=1)
        
        R = self.R_CVECA(R)
        k = self.k_CVECA(k)
        
        X = cvtorch.cat((R.ifft(), k), dim=1)
        return self.refinement(X).complex.squeeze()
        
class RefinementBlock(nn.Module):
    """Refinement in the k- and R-domains using kR-Blocks."""
    def __init__(self, conv=cvnn.default_cvconv1d, in_channels=32, out_channels=32, 
                 kernel_size=3, act=cvnn.CPReLU, n_res_blocks=8):
        super(RefinementBlock, self).__init__()
        
        self.kr1 = kRBlock(conv=conv, in_channels=in_channels, out_channels=in_channels, 
                           kernel_size=kernel_size, act=act, n_res_blocks=n_res_blocks)
        
        self.kr2 = kRBlock(conv=conv, in_channels=in_channels, out_channels=in_channels, 
                           kernel_size=kernel_size, act=act, n_res_blocks=n_res_blocks)
        
        self.kr3 = kRBlock(conv=conv, in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=kernel_size, act=act, n_res_blocks=n_res_blocks)
        
    def forward(self, x: CVTensor) -> CVTensor:
        # x is in R-domain
        res = self.kr1(x).ifft()
        
        # res is in k-domain
        res = self.kr2(res)
        
        res += x.ifft()
        
        return self.kr3(res)

class SSCANet_Big(nn.Module):
    """Big signal self- and cross-attention model (x16)."""
    def __init__(self, args: argparse.Namespace, conv=cvnn.default_cvconv1d, SoftMaxClass=cvnn.MagMinMaxNorm):
        super(SSCANet_Big, self).__init__()

        self.args = args
        
        self.act = select_act(args)
        
        d_model = args.n_feats
        B = args.n_res_blocks
        
        in_channels = args.in_channels
        out_channels = args.out_channels
        kernel_size = args.kernel_size
        act = self.act
        
        n_head = args.n_head
        d_k = args.d_k
        d_v = args.d_v
        dropout = args.dropout
        
        b = args.eca_b
        gamma = args.eca_gamma
        
        self.head = conv(in_channels=in_channels, out_channels=d_model, kernel_size=kernel_size)
        
        self.R_SSCA1 = SignalSelfCrossAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, dropout=dropout, SoftMaxClass=SoftMaxClass, 
                                                conv=conv, kernel_size=kernel_size, b=b, gamma=gamma)
        self.k_SSCA1 = SignalSelfCrossAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, dropout=dropout, SoftMaxClass=SoftMaxClass,
                                                conv=conv, kernel_size=kernel_size, b=b, gamma=gamma)
        
        self.R_SSCA2 = SignalSelfCrossAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, dropout=dropout, SoftMaxClass=SoftMaxClass, 
                                                conv=conv, kernel_size=kernel_size, b=b, gamma=gamma)
        self.k_SSCA2 = SignalSelfCrossAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, dropout=dropout, SoftMaxClass=SoftMaxClass,
                                                conv=conv, kernel_size=kernel_size, b=b, gamma=gamma)
        
        self.R_CVECA1 = cvnn.CVECA(channels=d_model, b=b, gamma=gamma)

