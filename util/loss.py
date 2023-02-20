import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import os

from model.common import CPLX

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

    def forward(self, X: CPLX, Y: CPLX, data_range=None, full=False):
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
        
    def forward(self, X: CPLX, Y: CPLX):

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
        
    def forward(self, x, y):
        return self.L1(x.real, y.real) + self.L1(x.imag, y.imag)
    
class SplitMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.MSE = nn.MSELoss()
        
    def forward(self, x, y):
        return self.MSE(x.real, y.real) + self.MSE(x.imag, y.imag)

class Loss(nn.modules.loss._Loss):
    def __init__(self, args):
        super(Loss, self).__init__()
        print('Preparing complex loss function...')
        
        self.device = args.device
        self.batch_size = args.batch_size
        
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type[0] == 'i':
                intermediate_loss = True
                loss_type = loss_type[1:]
            else:
                intermediate_loss = False
                
            if loss_type == "SMSE":
                loss_function = SplitMSE()
            elif loss_type == "SL1":
                loss_function = SplitL1()
            elif loss_type == "PerpSSIM":
                loss_function = PerpLossSSIM()

            self.loss.append({
                'type': loss_type if not intermediate_loss else 'i' + loss_type,
                'weight': float(weight),
                'function': loss_function}
            )
                
        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})
            
        for l in self.loss:
            if l['function'] is not None:
                print('\t{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])
                
        self.log = torch.Tensor()
        
        self.loss_module.to(self.device)
        if args.precision == 'half':
            self.loss_module.half()
    
    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)
    
    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[0:epoch, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(apath, 'loss_{}.pdf'.format(l['type'])))
            
    def get_loss_module(self):
        return self.loss_module
        
    def forward(self, sr, hr, intermediate=None):
        # sr - super resolution (prediction)
        # hr - high resolution (label)
        losses = []
        for i, l in enumerate(self.loss):
            # Loss at output
            if l['function'] is not None and l['type'][0] != 'i':
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
            
            # Intermediate loss
            if l['function'] is not None and intermediate is not None and l['type'][0] == 'i':
                loss = sum([l['function'](inter, hr) for inter in intermediate])
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
                
        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()
            
        return loss_sum

