import numpy as np
import torch
import torch.nn as nn
import argparse
from typing import Tuple

from model.layers import CPReLU, naive_complex_default_conv1d, CReLU, CTanh, kRBlock
from model.common import CPLX

class kRNet(nn.Module):
    """kR-Net model architecture following paper Fig. 4."""
    def __init__(self, args: argparse.Namespace, conv=naive_complex_default_conv1d):
        super(kRNet, self).__init__()
        
        self.args = args
        
        if args.act.lower() == 'crelu':
            self.act = CReLU(True)
        elif args.act.lower() == 'ctanh':
            self.act = CTanh()
        elif args.act.lower() == 'cprelu':
            self.act = CPReLU()
            
        # Following paper notation
        F = args.n_feats
        B = args.n_res_blocks
        
        in_channels = args.in_channels
        out_channels = args.out_channels
        kernel_size = args.kernel_size
        act = self.act
        
        self.head = conv(in_channels=in_channels, out_channels=F, kernel_size=kernel_size)
        
        self.kr1 = kRBlock(conv=conv, in_channels=F, out_channels=F, kernel_size=kernel_size, act=act, n_res_blocks=B)
        self.kr2 = kRBlock(conv=conv, in_channels=F, out_channels=F, kernel_size=kernel_size, act=act, n_res_blocks=B)
        self.kr3 = kRBlock(conv=conv, in_channels=F, out_channels=F, kernel_size=kernel_size, act=act, n_res_blocks=B)
        self.kr4 = kRBlock(conv=conv, in_channels=F, out_channels=F, kernel_size=kernel_size, act=act, n_res_blocks=B)
        self.kr5 = kRBlock(conv=conv, in_channels=F, out_channels=out_channels, kernel_size=kernel_size, act=act, n_res_blocks=B)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = CPLX(x.real, x.imag).view(x.shape[0], 1, -1)
        
        # x is in R-domain
        x = self.head(x)
        res = self.kr1(x).ifft()
        
        # res is in k-domain
        res = self.kr2(res).fft()
        
        # res is in R-domain
        res = self.kr3(res).ifft()
        
        # res is in k-domain
        res = self.kr4(res)
        res.add(x.ifft())
        
        return self.kr5(res).complex.squeeze(), None
        
    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])

class kNet(nn.Module):
    """k-Net model architecture."""
    def __init__(self, args: argparse.Namespace, conv=naive_complex_default_conv1d):
        super(kNet, self).__init__()
        
        self.args = args
        
        if args.act.lower() == 'crelu':
            self.act = CReLU(True)
        elif args.act.lower() == 'ctanh':
            self.act = CTanh()
        elif args.act.lower() == 'cprelu':
            self.act = CPReLU()
            
        # Following paper notation
        F = args.n_feats
        B = args.n_res_blocks
        
        in_channels = args.in_channels
        out_channels = args.out_channels
        kernel_size = args.kernel_size
        act = self.act
        
        self.head = conv(in_channels=in_channels, out_channels=F, kernel_size=kernel_size)
        
        self.kr1 = kRBlock(conv=conv, in_channels=F, out_channels=F, kernel_size=kernel_size, act=act, n_res_blocks=B)
        self.kr2 = kRBlock(conv=conv, in_channels=F, out_channels=F, kernel_size=kernel_size, act=act, n_res_blocks=B)
        self.kr3 = kRBlock(conv=conv, in_channels=F, out_channels=F, kernel_size=kernel_size, act=act, n_res_blocks=B)
        self.kr4 = kRBlock(conv=conv, in_channels=F, out_channels=F, kernel_size=kernel_size, act=act, n_res_blocks=B)
        self.kr5 = kRBlock(conv=conv, in_channels=F, out_channels=out_channels, kernel_size=kernel_size, act=act, n_res_blocks=B)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = CPLX(x.real, x.imag).view(x.shape[0], 1, -1).ifft()
        
        # x is in k-domain
        x = self.head(x)
        res = self.kr1(x)
        
        # res is in k-domain
        res = self.kr2(res)
        
        # res is in k-domain
        res = self.kr3(res)
        
        # res is in k-domain
        res = self.kr4(res)
        res.add(x)
        
        return self.kr5(res).complex.squeeze(), None
        
    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])

class RNet(nn.Module):
    """R-Net model architecture."""
    def __init__(self, args: argparse.Namespace, conv=naive_complex_default_conv1d):
        super(RNet, self).__init__()
        
        self.args = args
        
        if args.act.lower() == 'crelu':
            self.act = CReLU(True)
        elif args.act.lower() == 'ctanh':
            self.act = CTanh()
        elif args.act.lower() == 'cprelu':
            self.act = CPReLU()
            
        # Following paper notation
        F = args.n_feats
        B = args.n_res_blocks
        
        in_channels = args.in_channels
        out_channels = args.out_channels
        kernel_size = args.kernel_size
        act = self.act
        
        self.head = conv(in_channels=in_channels, out_channels=F, kernel_size=kernel_size)
        
        self.kr1 = kRBlock(conv=conv, in_channels=F, out_channels=F, kernel_size=kernel_size, act=act, n_res_blocks=B)
        self.kr2 = kRBlock(conv=conv, in_channels=F, out_channels=F, kernel_size=kernel_size, act=act, n_res_blocks=B)
        self.kr3 = kRBlock(conv=conv, in_channels=F, out_channels=F, kernel_size=kernel_size, act=act, n_res_blocks=B)
        self.kr4 = kRBlock(conv=conv, in_channels=F, out_channels=F, kernel_size=kernel_size, act=act, n_res_blocks=B)
        self.kr5 = kRBlock(conv=conv, in_channels=F, out_channels=out_channels, kernel_size=kernel_size, act=act, n_res_blocks=B)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = CPLX(x.real, x.imag).view(x.shape[0], 1, -1)
        
        # x is in R-domain
        x = self.head(x)
        res = self.kr1(x)
        
        # res is in R-domain
        res = self.kr2(res)
        
        # res is in R-domain
        res = self.kr3(res)
        
        # res is in R-domain
        res = self.kr4(res)
        res.add(x)
        
        return self.kr5(res).ifft().complex.squeeze(), None
        
    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])

class kRNet_v2(nn.Module):
    """kR-Net model v2 architecture."""
    def __init__(self, args: argparse.Namespace, conv=naive_complex_default_conv1d):
        super(kRNet, self).__init__()
        
        self.args = args
        
        if args.act.lower() == 'crelu':
            self.act = CReLU(True)
        elif args.act.lower() == 'ctanh':
            self.act = CTanh()
        elif args.act.lower() == 'cprelu':
            self.act = CPReLU()
            
        # Following paper notation
        F = args.n_feats
        B = args.n_res_blocks
        
        in_channels = args.in_channels
        out_channels = args.out_channels
        kernel_size = args.kernel_size
        act = self.act
        
        self.head = conv(in_channels=in_channels, out_channels=F, kernel_size=kernel_size)
        
        self.kr1 = kRBlock(conv=conv, in_channels=F, out_channels=F, kernel_size=kernel_size, act=act, n_res_blocks=B)
        self.kr2 = kRBlock(conv=conv, in_channels=F, out_channels=F, kernel_size=kernel_size, act=act, n_res_blocks=B)
        self.kr3 = kRBlock(conv=conv, in_channels=F, out_channels=F, kernel_size=kernel_size, act=act, n_res_blocks=B)
        self.kr4 = kRBlock(conv=conv, in_channels=F, out_channels=F, kernel_size=kernel_size, act=act, n_res_blocks=B)
        self.kr5 = kRBlock(conv=conv, in_channels=F, out_channels=out_channels, kernel_size=kernel_size, act=act, n_res_blocks=B)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = CPLX(x.real, x.imag).view(x.shape[0], 1, -1)
        
        # x is in R-domain
        x = self.head(x)
        res = self.kr1(x).ifft()    
        
        # res is in k-domain
        res = self.kr2(res).fft()
        
        # res is in R-domain
        res = self.kr3(res).ifft()
        
        # res is in k-domain
        res = self.kr4(res)
        res.add(x.ifft())
        
        return self.kr5(res).complex.squeeze(), None
        
    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])


