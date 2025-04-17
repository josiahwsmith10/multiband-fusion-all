import torch
import torch.nn as nn
import argparse
from typing import Tuple

import complextorch.nn as cvnn

from model.common import select_act, default_conv1d
from model.kRNet import kRBlock
from radar import MultiRadar
import radar


class SignalSelfCrossAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_model: int,
        d_k: int,
        d_v: int,
        dropout: float = 0.1,
        SoftMaxClass=cvnn.MagMinMaxNorm,
        conv=default_conv1d,
        kernel_size=3,
        b=1,
        gamma=2,
    ):
        super(SignalSelfCrossAttention, self).__init__()

        self.AA = cvnn.MultiheadAttention(
            n_heads, d_model, d_k, d_v, dropout, SoftMaxClass
        )
        self.AB = cvnn.MultiheadAttention(
            n_heads, d_model, d_k, d_v, dropout, SoftMaxClass
        )
        self.BA = cvnn.MultiheadAttention(
            n_heads, d_model, d_k, d_v, dropout, SoftMaxClass
        )
        self.BB = cvnn.MultiheadAttention(
            n_heads, d_model, d_k, d_v, dropout, SoftMaxClass
        )

        self.chan_attn = cvnn.EfficientChannelAttention1d(
            channels=d_model * 4, b=b, gamma=gamma
        )

        self.conv = conv(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            bias=False,
        )

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        A, B = A.transpose(1, 2), B.transpose(1, 2)

        W = self.AA(A, A, A)
        X = self.BB(B, B, B)
        Y = self.AB(A, B, B)
        Z = self.BA(B, A, A)

        X = torch.cat((W, X, Y, Z), dim=1).transpose(1, 2)

        X = self.chan_attn(X)

        return self.conv(X)


class DualRadarFusion(nn.Module):
    """Dual-Radar Fusion Block."""

    def __init__(
        self,
        conv,
        in_channels,
        kernel_size,
        n_heads,
        d_model,
        d_v,
        d_k,
        dropout,
        SoftMaxClass,
        b,
        gamma,
    ):
        super(DualRadarFusion, self).__init__()

        self.d_model = d_model

        self.head1 = conv(
            in_channels=in_channels, out_channels=d_model, kernel_size=kernel_size
        )
        self.head2 = conv(
            in_channels=in_channels, out_channels=d_model, kernel_size=kernel_size
        )

        self.R_SSCA = SignalSelfCrossAttention(
            n_heads=n_heads,
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            SoftMaxClass=SoftMaxClass,
            conv=conv,
            kernel_size=kernel_size,
            b=b,
            gamma=gamma,
        )
        self.k_SSCA = SignalSelfCrossAttention(
            n_heads=n_heads,
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            SoftMaxClass=SoftMaxClass,
            conv=conv,
            kernel_size=kernel_size,
            b=b,
            gamma=gamma,
        )

        self.R_CVECA = cvnn.EfficientChannelAttention1d(
            channels=d_model, b=b, gamma=gamma
        )
        self.k_CVECA = cvnn.EfficientChannelAttention1d(
            channels=d_model, b=b, gamma=gamma
        )

    def forward(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        # X1, X2 are in R-domain
        X1, X2 = self.head1(X1), self.head2(X2)

        R = self.R_SSCA(X1, X2)
        k = self.k_SSCA(radar.ifft(X1), radar.ifft(X2))

        R = torch.cat((R, radar.fft(k)), dim=1)
        k = torch.cat((k, radar.ifft(R[:, : self.d_model])), dim=1)

        R = self.R_CVECA(R)
        k = self.k_CVECA(k)
        return R, k


class DomainFusion(DualRadarFusion):
    def __init__(self, *args, **kwargs):
        super(DomainFusion, self).__init__(*args, **kwargs)

    def forward(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        # X1 is in R-domain, X2 is in k-domain
        X1, X2 = self.head1(X1), self.head2(X2)

        R = self.R_SSCA(X1, radar.fft(X2))
        k = self.k_SSCA(radar.ifft(X1), X2)

        R = torch.cat((R, radar.fft(k)), dim=1)
        k = torch.cat((k, radar.ifft(R[:, : self.d_model])), dim=1)

        R = self.R_CVECA(R)
        k = self.k_CVECA(k)
        return R, k


class RefinementBlock(nn.Module):
    """Refinement in the k- and R-domains using kR-Blocks."""

    def __init__(
        self,
        conv=default_conv1d,
        in_channels=32,
        out_channels=32,
        kernel_size=3,
        act=cvnn.CPReLU,
        n_res_blocks=8,
    ):
        super(RefinementBlock, self).__init__()

        self.kr1 = kRBlock(
            conv=conv,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            act=act,
            n_res_blocks=n_res_blocks,
        )

        self.kr2 = kRBlock(
            conv=conv,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            act=act,
            n_res_blocks=n_res_blocks,
        )

        self.kr3 = kRBlock(
            conv=conv,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            act=act,
            n_res_blocks=n_res_blocks,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is in R-domain
        res = radar.ifft(self.kr1(x))

        # res is in k-domain
        res = self.kr2(res)

        res += radar.ifft(x)

        return self.kr3(res)


class SSCANet_Small(nn.Module):
    """Small signal self- and cross-attention model (x4 SR)."""

    def __init__(
        self,
        args: argparse.Namespace,
        mr: MultiRadar,
        conv=default_conv1d,
        SoftMaxClass=cvnn.MagMinMaxNorm,
    ):
        super(SSCANet_Small, self).__init__()

        self.args = args
        self.mr = mr

        self.act = select_act(args)

        d_model = args.n_feats
        B = args.n_res_blocks

        in_channels = args.in_channels
        out_channels = args.out_channels
        kernel_size = args.kernel_size
        act = self.act

        n_heads = args.n_heads
        d_k = args.d_k
        d_v = args.d_v
        dropout = args.dropout

        b = args.eca_b
        gamma = args.eca_gamma

        self.dual_radar_fusion = DualRadarFusion(
            conv=conv,
            in_channels=in_channels,
            kernel_size=kernel_size,
            n_heads=n_heads,
            d_model=d_model,
            d_v=d_v,
            d_k=d_k,
            dropout=dropout,
            SoftMaxClass=SoftMaxClass,
            b=b,
            gamma=gamma,
        )

        self.refinement = kRBlock(
            conv=conv,
            in_channels=4 * d_model,
            out_channels=out_channels,
            kernel_size=kernel_size,
            act=act,
            n_res_blocks=B,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get radar 1 and radar 2 data in R-domain
        X1 = x[:, self.mr.idx[0]].unsqueeze(1)
        X2 = x[:, self.mr.idx[1]].unsqueeze(1)

        A_R, A_k = self.dual_radar_fusion(X1, X2)

        # Concatenate features in k-domain
        X = torch.cat((radar.ifft(A_R), A_k), dim=1)
        return self.refinement(X).squeeze(), None


class SSCANet_Big(nn.Module):
    """Big signal self- and cross-attention model (x16)."""

    def __init__(
        self,
        args: argparse.Namespace,
        mr: MultiRadar,
        conv=default_conv1d,
        SoftMaxClass=cvnn.MagMinMaxNorm,
    ):
        super(SSCANet_Big, self).__init__()

        self.args = args
        self.mr = mr

        self.act = select_act(args)

        d_model = args.n_feats
        B = args.n_res_blocks

        in_channels = args.in_channels
        out_channels = args.out_channels
        kernel_size = args.kernel_size
        act = self.act

        n_heads = args.n_heads
        d_k = args.d_k
        d_v = args.d_v
        dropout = args.dropout

        b = args.eca_b
        gamma = args.eca_gamma

        self.dual_radar_fusion = DualRadarFusion(
            conv=conv,
            in_channels=in_channels,
            kernel_size=kernel_size,
            n_heads=n_heads,
            d_model=d_model,
            d_v=d_v,
            d_k=d_k,
            dropout=dropout,
            SoftMaxClass=SoftMaxClass,
            b=b,
            gamma=gamma,
        )

        self.domain_fusion = DomainFusion(
            conv=conv,
            in_channels=2 * d_model,
            kernel_size=kernel_size,
            n_heads=n_heads,
            d_model=d_model,
            d_v=d_v,
            d_k=d_k,
            dropout=dropout,
            SoftMaxClass=SoftMaxClass,
            b=b,
            gamma=gamma,
        )

        self.refinement = RefinementBlock(
            conv=conv,
            in_channels=4 * d_model,
            out_channels=out_channels,
            kernel_size=kernel_size,
            act=act,
            n_res_blocks=B,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get radar 1 and radar 2 data in R-domain
        X1 = x[:, self.mr.idx[0]].unsqueeze(1)
        X2 = x[:, self.mr.idx[1]].unsqueeze(1)

        A_R, A_k = self.dual_radar_fusion(X1, X2)

        B_R, B_k = self.domain_fusion(A_R, A_k)

        # Concatenate features in k-domain
        X = torch.cat((radar.ifft(B_R), B_k), dim=1)
        return self.refinement(X).squeeze(), None
