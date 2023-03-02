import torch
import numpy as np
from scipy.constants import pi, c
import argparse

from .radar import TIRadar


class MultiRadar:
    """MultiRadar class."""

    def __init__(self, args: argparse.Namespace):
        # Check inputs
        assert len(args.f0) == len(args.Nk)

        self.f0 = args.f0
        self.K = args.K
        self.Nk = args.Nk
        self.Nk_fb = args.Nk_fb
        self.fS = args.fS

        self.N_subbands = len(args.f0)

        print(
            f"Total: \t\t{args.f0[0]*1e-9:.1f}-{(args.f0[0]+args.Nk_fb*args.K/args.fS)*1e-9:.1f} GHz"
        )
        for i, (f0_i, Nk_i) in enumerate(zip(args.f0, args.Nk)):
            print(
                f"Subband {i+1}: \t{f0_i*1e-9:.1f}-{(f0_i+Nk_i*args.K/args.fS)*1e-9:.1f} GHz"
            )

        self.Compute()

    def Compute(self):
        def idx_close(a, b, tol=1e-5):
            return np.where(np.abs(a - b) / ((a + b) / 2) < tol)[0]

        self.r = [
            TIRadar(f0=f0, K=self.K, Nk=Nk, fS=self.fS)
            for f0, Nk in zip(self.f0, self.Nk)
        ]
        self.r_fb = TIRadar(f0=self.f0[0], K=self.K, Nk=self.Nk_fb, fS=self.fS)

        self.idx = []
        self.subband_mask = []
        self.LR_mask = np.zeros(self.Nk_fb, dtype=bool)
        self.N_LR = 0
        for i, r in enumerate(self.r):
            beg_idx, end_idx = idx_close(r.f[0], self.r_fb.f), idx_close(
                r.f[-1], self.r_fb.f
            )

            self.idx.append(np.arange(beg_idx, end_idx + 1))
            self.N_LR += len(self.idx[i])

            x = np.zeros(self.Nk_fb, dtype=bool)
            x[self.idx[i]] = True
            self.subband_mask.append(x)

            self.LR_mask = self.LR_mask | x

        self.f_HR = self.r_fb.f
        self.k_HR = self.r_fb.k
        self.f_LR = torch.zeros(self.Nk_fb)
        self.f_LR[self.LR_mask] = self.f_HR[self.LR_mask]
        self.k_LR = 2 * pi / c * self.f_LR
