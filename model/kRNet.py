import numpy as np
import torch
import torch.nn as nn
import argparse
from typing import Tuple, Callable

import complextorch.nn as ctnn

from model.common import select_act_module, default_conv1d
import radar


class ComplexValuedResBlock(nn.Module):
    """Complex-Valued Residual Block."""

    def __init__(
        self,
        conv: Callable,
        n_feats: int,
        kernel_size: int,
        bias: bool = False,
        act_module: Callable = ctnn.CPReLU,
        res_scale: float = 1,
    ):
        super(ComplexValuedResBlock, self).__init__()

        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if i == 0:
                m.append(act_module())

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Complex residual block forward
        res = self.body(x) * self.res_scale
        res += x

        return res


class kRBlock(nn.Module):
    """kR-Block."""

    def __init__(
        self,
        conv: Callable,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool = False,
        act_module: Callable = ctnn.CPReLU,
        res_scale: float = 1,
        n_res_blocks: int = 8,
    ):
        super(kRBlock, self).__init__()

        m = [
            ComplexValuedResBlock(
                conv,
                in_channels,
                kernel_size,
                bias=bias,
                act_module=act_module,
                res_scale=res_scale,
            )
            for _ in range(n_res_blocks)
        ]
        m.append(conv(in_channels, out_channels, kernel_size))

        self.body = nn.Sequential(*m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class kRNet(nn.Module):
    """kR-Net model architecture following paper Fig. 4."""

    def __init__(self, args: argparse.Namespace, conv: Callable = default_conv1d):
        super(kRNet, self).__init__()

        self.args = args

        self.act_module = select_act_module(args)

        # Following paper notation
        F = args.n_feats
        B = args.n_res_blocks

        in_channels = args.in_channels
        out_channels = args.out_channels
        kernel_size = args.kernel_size
        act_module = self.act_module

        self.head = conv(
            in_channels=in_channels, out_channels=F, kernel_size=kernel_size
        )

        self.kr1 = kRBlock(
            conv=conv,
            in_channels=F,
            out_channels=F,
            kernel_size=kernel_size,
            act_module=act_module,
            n_res_blocks=B,
        )
        self.kr2 = kRBlock(
            conv=conv,
            in_channels=F,
            out_channels=F,
            kernel_size=kernel_size,
            act_module=act_module,
            n_res_blocks=B,
        )
        self.kr3 = kRBlock(
            conv=conv,
            in_channels=F,
            out_channels=F,
            kernel_size=kernel_size,
            act_module=act_module,
            n_res_blocks=B,
        )
        self.kr4 = kRBlock(
            conv=conv,
            in_channels=F,
            out_channels=F,
            kernel_size=kernel_size,
            act_module=act_module,
            n_res_blocks=B,
        )
        self.kr5 = kRBlock(
            conv=conv,
            in_channels=F,
            out_channels=out_channels,
            kernel_size=kernel_size,
            act_module=act_module,
            n_res_blocks=B,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.view(x.shape[0], 1, -1)

        # x is in R-domain
        x = self.head(x)
        res = radar.ifft(self.kr1(x))

        # res is in k-domain
        res = radar.fft(self.kr2(res))

        # res is in R-domain
        res = radar.ifft(self.kr3(res))

        # res is in k-domain
        res = self.kr4(res)
        res += radar.ifft(x)

        return self.kr5(res).squeeze(), None

    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])


class kNet(nn.Module):
    """k-Net model architecture."""

    def __init__(self, args: argparse.Namespace, conv: Callable = default_conv1d):
        super(kNet, self).__init__()

        self.args = args

        self.act = select_act_module(args)

        # Following paper notation
        F = args.n_feats
        B = args.n_res_blocks

        in_channels = args.in_channels
        out_channels = args.out_channels
        kernel_size = args.kernel_size
        act = self.act

        self.head = conv(
            in_channels=in_channels, out_channels=F, kernel_size=kernel_size
        )

        self.kr1 = kRBlock(
            conv=conv,
            in_channels=F,
            out_channels=F,
            kernel_size=kernel_size,
            act_module=act,
            n_res_blocks=B,
        )
        self.kr2 = kRBlock(
            conv=conv,
            in_channels=F,
            out_channels=F,
            kernel_size=kernel_size,
            act_module=act,
            n_res_blocks=B,
        )
        self.kr3 = kRBlock(
            conv=conv,
            in_channels=F,
            out_channels=F,
            kernel_size=kernel_size,
            act_module=act,
            n_res_blocks=B,
        )
        self.kr4 = kRBlock(
            conv=conv,
            in_channels=F,
            out_channels=F,
            kernel_size=kernel_size,
            act_module=act,
            n_res_blocks=B,
        )
        self.kr5 = kRBlock(
            conv=conv,
            in_channels=F,
            out_channels=out_channels,
            kernel_size=kernel_size,
            act_module=act,
            n_res_blocks=B,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.fft.ifft(x.view(x.shape[0], 1, -1), dim=-1, norm="ortho")

        # x is in k-domain
        x = self.head(x)
        res = self.kr1(x)

        # res is in k-domain
        res = self.kr2(res)

        # res is in k-domain
        res = self.kr3(res)

        # res is in k-domain
        res = self.kr4(res)
        res += x

        return self.kr5(res).squeeze(), None

    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])


class RNet(nn.Module):
    """R-Net model architecture."""

    def __init__(self, args: argparse.Namespace, conv=default_conv1d):
        super(RNet, self).__init__()

        self.args = args

        self.act = select_act_module(args)

        # Following paper notation
        F = args.n_feats
        B = args.n_res_blocks

        in_channels = args.in_channels
        out_channels = args.out_channels
        kernel_size = args.kernel_size
        act = self.act

        self.head = conv(
            in_channels=in_channels, out_channels=F, kernel_size=kernel_size
        )

        self.kr1 = kRBlock(
            conv=conv,
            in_channels=F,
            out_channels=F,
            kernel_size=kernel_size,
            act_module=act,
            n_res_blocks=B,
        )
        self.kr2 = kRBlock(
            conv=conv,
            in_channels=F,
            out_channels=F,
            kernel_size=kernel_size,
            act_module=act,
            n_res_blocks=B,
        )
        self.kr3 = kRBlock(
            conv=conv,
            in_channels=F,
            out_channels=F,
            kernel_size=kernel_size,
            act_module=act,
            n_res_blocks=B,
        )
        self.kr4 = kRBlock(
            conv=conv,
            in_channels=F,
            out_channels=F,
            kernel_size=kernel_size,
            act_module=act,
            n_res_blocks=B,
        )
        self.kr5 = kRBlock(
            conv=conv,
            in_channels=F,
            out_channels=out_channels,
            kernel_size=kernel_size,
            act_module=act,
            n_res_blocks=B,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.view(x.shape[0], 1, -1)

        # x is in R-domain
        x = self.head(x)
        res = self.kr1(x)

        # res is in R-domain
        res = self.kr2(res)

        # res is in R-domain
        res = self.kr3(res)

        # res is in R-domain
        res = self.kr4(res)
        res += x

        return radar.ifft(self.kr5(res)).squeeze(), None

    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])


class kRNet_v2(nn.Module):
    """kR-Net model v2 architecture."""

    def __init__(self, args: argparse.Namespace, conv=default_conv1d):
        super(kRNet_v2, self).__init__()

        self.args = args

        self.act = select_act_module(args)

        # Following paper notation
        F = args.n_feats
        B = args.n_res_blocks

        in_channels = args.in_channels
        out_channels = args.out_channels
        kernel_size = args.kernel_size
        act = self.act

        self.head = conv(
            in_channels=in_channels, out_channels=F, kernel_size=kernel_size
        )

        self.kr1 = kRBlock(
            conv=conv,
            in_channels=F,
            out_channels=F,
            kernel_size=kernel_size,
            act_module=act,
            n_res_blocks=B,
        )
        self.kr2 = kRBlock(
            conv=conv,
            in_channels=F,
            out_channels=F,
            kernel_size=kernel_size,
            act_module=act,
            n_res_blocks=B,
        )
        self.kr3 = kRBlock(
            conv=conv,
            in_channels=F,
            out_channels=F,
            kernel_size=kernel_size,
            act_module=act,
            n_res_blocks=B,
        )
        self.kr4 = kRBlock(
            conv=conv,
            in_channels=F,
            out_channels=F,
            kernel_size=kernel_size,
            act_module=act,
            n_res_blocks=B,
        )
        self.kr5 = kRBlock(
            conv=conv,
            in_channels=F,
            out_channels=out_channels,
            kernel_size=kernel_size,
            act_module=act,
            n_res_blocks=B,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.view(x.shape[0], 1, -1)

        # Intermediate representations in k-domain
        # TODO: ISSUE: how to store the intermediate representations since they have more channels than the input?
        intermediate = []

        # x is in R-domain
        x = self.head(x)
        intermediate.append(radar.ifft(x).clone().mean(dim=1))
        x = radar.ifft(self.kr1(x))

        # x is in k-domain
        intermediate.append(x.clone().mean(dim=1))
        x = radar.fft(self.kr2(x))

        # x is in R-domain
        intermediate.append(radar.ifft(x).clone().mean(dim=1))
        x = radar.ifft(self.kr3(x))

        # res is in k-domain
        intermediate.append(x.clone().mean(dim=1))
        x = radar.fft(self.kr4(x))

        # x is in R-domain
        intermediate.append(radar.ifft(x).clone().mean(dim=1))
        x = radar.ifft(self.kr5(x))

        return x.squeeze(), intermediate

    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])
