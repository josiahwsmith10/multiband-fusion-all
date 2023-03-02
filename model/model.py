import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from radar import MultiRadar

from model.kRNet import kRNet, kRNet_v2, kNet, RNet
from model.SSCANet import SSCANet_Big, SSCANet_Small


class ComplexModel(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(ComplexModel, self).__init__()

        self.args = args
        self.device = args.device

        self.train_loss = []
        self.val_loss = []

        if args.model.lower() == "kr-net":
            print("Making kR-Net model...")
            self.model = kRNet(args).to(self.device)
        elif args.model.lower() == "k-net":
            print("Making k-Net model...")
            self.model = kNet(args).to(self.device)
        elif args.model.lower() == "r-net":
            print("Making R-Net model...")
            self.model = RNet(args).to(self.device)
        elif args.model.lower() == "kr-net-v2":
            print("Making kR-Net v2 model...")
            self.model = kRNet_v2(args).to(self.device)
        elif args.model.lower() == "ssca-net-big":
            print("Making SSCA-Net model...")
            self.model = SSCANet_Big(args).to(self.device)
        elif args.model.lower() == "ssca-net-small":
            print("Making SSCA-Net model...")
            self.model = SSCANet_Small(args).to(self.device)
        else:
            assert False, "Must use kR-Net model"

        if args.precision.lower() == "half":
            self.model.half()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.model(x)
        else:
            return self.model.forward(x)

    def add_multiradar(self, mr: MultiRadar) -> None:
        if isinstance(self.model, (SSCANet_Big, SSCANet_Small)):
            self.model.mr = mr

    def run_sar_data(self, sar_data_LR, N_batch=None):
        def min_max_norm(x, dim):
            x_min = torch.min(torch.abs(x), dim=dim, keepdim=True)[0]
            x_max = torch.max(torch.abs(x), dim=dim, keepdim=True)[0]
            return (x - x_min) / (x_max - x_min)

        if N_batch is None:
            N_batch = self.args.batch_size

        Nx, Ny, Nk = sar_data_LR.shape

        sar_data_LR = torch.from_numpy(sar_data_LR).to(self.device).view(Nx * Ny, 1, Nk)
        sar_data_LR /= sar_data_LR[:, :, -1].view(Nx * Ny, 1, 1)
        if self.args.xyz_str.lower() == "xz":
            sar_data_LR = torch.fft.fft(sar_data_LR, dim=2, norm="ortho")
            sar_data_LR = min_max_norm(sar_data_LR, dim=2)

        sar_data_SR = torch.zeros_like(sar_data_LR).to("cpu")

        for i in tqdm(range(0, Nx * Ny, N_batch)):
            s_batch_SR = self.forward(sar_data_LR[i : i + N_batch])
            sar_data_SR[i : i + N_batch] = (
                (s_batch_SR.r + 1j * s_batch_SR.i).cpu().detach()
            )

        return sar_data_SR.numpy().reshape(Nx, Ny, Nk)
