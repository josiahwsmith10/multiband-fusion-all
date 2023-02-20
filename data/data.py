import torch
from scipy.constants import pi
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from util.common import min_max_norm, apply_linear_transform

class MultiRadarDataset(torch.utils.data.Dataset):
    """MultiRadarDataset dataset."""
    
    def __init__(self, X, Y, args):
        """
        Args:
            X (tensor): Input data (complex-valued tensor)
            Y (tensor): Target data (complex-valued tensor)
        """
        assert X.shape == Y.shape, 'X and Y must have the same shape'
        
        self.X = X
        self.Y = Y
        
        self.N, self.d_model = self.X.shape
        
        self.args = args
        self.device = args.device
        
    def to(self, device):
        self.X = self.X.to(device)
        self.Y = self.Y.to(device)
        
        return self
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        lr = self.X[idx].to(self.device)
        hr = self.Y[idx].to(self.device)
        
        if self.args.precision == 'half':
            lr = lr.half()
            hr = hr.half()
            
        return lr, hr

class MultiRadarData:
    def __init__(self, multiradar, args, snr_range=[3, 50]):
        self.mr = multiradar
        self.args = args
        self.device = args.device
        self.snr_range = snr_range
        self.batch_size = args.batch_size
        
    def create_generic_dataset(self):        
        if self.args.xyz_str.lower() == 'xy':
            self.X = self.X.to(self.device)
            self.Y = self.Y.to(self.device)
            
            return MultiRadarDataset(self.X, self.Y, self.args)
            #return torch.utils.data.TensorDataset(self.X, self.Y)
        elif self.args.xyz_str.lower() == 'xz':
            self.X = self.X.to(self.device)
            self.Z = self.Z.to(self.device)
            
            return MultiRadarDataset(self.X, self.Z, self.args)
            #return torch.utils.data.TensorDataset(self.X, self.Z)
        
    def create_dataset_train(self, num_train, max_targets):
        self.max_targets = max_targets
        
        self.num_train = num_train
        
        self.create_XY(num_train)
        
        self.train_X = self.X
        self.train_Y = self.Y
        self.train_Z = self.Z
        self.train_SNR = self.SNR
        
        self.dataset_train = self.create_generic_dataset()
        
    def create_dataset_val(self, num_val, max_targets):
        self.max_targets = max_targets
        
        self.num_val = num_val
        
        self.create_XY(num_val)
        
        self.val_X = self.X
        self.val_Y = self.Y
        self.val_Z = self.Z
        self.val_SNR = self.SNR
        
        self.dataset_val = self.create_generic_dataset()
        
    def create_dataset_test(self, num_test, max_targets):
        self.max_targets = max_targets
        
        self.num_test = num_test
        
        self.create_XY(num_test)
        
        self.test_X = self.X
        self.test_Y = self.Y
        self.test_Z = self.Z
        self.test_SNR = self.SNR
        
        self.dataset_test = self.create_generic_dataset()
       
    def create_XY(self, num):
        def linear_power(x, N):
            return torch.sqrt(1/N * torch.sum(torch.abs(x)**2))
        
        X = torch.zeros((num, self.mr.Nk_fb), dtype=torch.complex64)
        Y = torch.zeros((num, self.mr.Nk_fb), dtype=torch.complex64)
        Z = torch.zeros((num, self.mr.Nk_fb), dtype=torch.complex64)
        SNR = np.zeros(num)
        
        k = self.mr.k_HR.reshape((1, self.mr.Nk_fb))
        
        for ind_num in tqdm(range(num)):
            # Handle noise
            SNR[ind_num] = self.snr_range[0] + (self.snr_range[1]-self.snr_range[0])*np.random.rand(1)
            n = torch.randn((self.mr.N_LR), dtype=torch.complex64)
            n_pow = linear_power(n, self.mr.N_LR)
            
            # Handle signal
            num_targets = np.random.randint(1, self.max_targets+1)
            R = 0.05*self.mr.r[0].range_max_m + 0.9*self.mr.r[0].range_max_m*torch.rand(num_targets, 1).reshape((num_targets, 1))
            
            amps = torch.randn(num_targets, 1).reshape((num_targets, 1))
            theta = 2*pi*torch.rand(num_targets, 1).reshape((num_targets, 1))
            
            amps = (0.5 + torch.abs(amps)) * torch.exp(1j*theta)
            
            x_HR = torch.sum(amps * torch.exp(1j*2*k*R), axis=0)
            
            x_LR = x_HR.clone()
            x_LR[~self.mr.LR_mask] = 0
            
            x_LR_pow = linear_power(x_LR, self.mr.N_LR)
            
            n = n/n_pow*x_LR_pow*10**(-SNR[ind_num]/10)
            
            x_LR[self.mr.LR_mask] += n
            
            norm = x_LR[-1]
            x_LR = x_LR/norm
            
            # Normalize Phase
            x_HR = x_HR/norm
            
            y_LR = torch.fft.fft(x_LR, self.mr.Nk_fb, dim=0, norm="ortho")
            y_HR = torch.fft.fft(x_HR, self.mr.Nk_fb, dim=0, norm="ortho")
            
            y_LR, a, b = min_max_norm(y_LR)
            y_HR = apply_linear_transform(y_HR, a, b)
            x_HR = apply_linear_transform(x_HR, a, b)
            
            X[ind_num] = y_LR # Frequency LR
            Y[ind_num] = y_HR # Frequency HR
            Z[ind_num] = x_HR # Time HR
            
        self.X = X
        self.Y = Y
        self.Z = Z
        self.SNR = SNR
     
    def test_net(self, args, model, data):
        """Tests the network on the test set and plots the results."""
        
        def test_one(ind_rand=None):
            if ind_rand is None:
                ind_rand = np.random.randint(0, data.num_test)
            
            print(f"Testing sample #{ind_rand} with SNR = {self.test_SNR[ind_rand]:.2f}")

            (x, y) = data.dataset_test[ind_rand]
            x = x.view(1, -1)

            y_pred = model.model(x).cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            x = x.cpu().detach().numpy().squeeze()
            
            if args.xyz_str.lower() == 'xz':
                y = np.fft.fft(y, norm="ortho")
                y_pred = np.fft.fft(y_pred, norm="ortho")
            else:
                y = y.numpy()

            fig = plt.figure(figsize=[12, 8])
            # Plot R-domain results
            y_max = np.max([np.abs(x).max(), np.abs(y).max(), np.abs(y_pred).max()]) + 0.1
            y_min = 0
            
            plt.subplot(241)
            plt.plot(np.abs(x))
            plt.title("Input " + str(ind_rand))
            plt.xlabel("R-domain")
            plt.ylim([y_min, y_max])

            plt.subplot(242)
            plt.plot(np.abs(y_pred))
            plt.title("Output")
            plt.xlabel("R-domain")
            plt.ylim([y_min, y_max])

            plt.subplot(243)
            plt.plot(np.abs(y))
            plt.title("Ground Truth")
            plt.xlabel("R-domain")
            plt.ylim([y_min, y_max])

            plt.subplot(244)
            err = np.abs(y - y_pred)
            plt.plot(err)
            plt.title("Error")
            plt.xlabel("R-domain")
            plt.ylim([y_min, y_max])

            # Plot k-domain results
            x = np.fft.ifft(x, norm="ortho")
            y = np.fft.ifft(y, norm="ortho")
            y_pred = np.fft.ifft(y_pred, norm="ortho")
            
            y_max = np.max([np.real(x).max(), np.real(y).max(), np.real(y_pred).max()]) + 0.1
            y_min = np.min([np.real(x).min(), np.real(y).min(), np.real(y_pred).min()]) - 0.1
                
            plt.subplot(245)
            plt.plot(np.real(x))
            plt.title("Input")
            plt.xlabel("k-domain")
            plt.ylim([y_min, y_max])

            plt.subplot(246)
            plt.plot(np.real(y_pred))
            plt.title("Output")
            plt.xlabel("k-domain")
            plt.ylim([y_min, y_max])

            plt.subplot(247)
            plt.plot(np.real(y))
            plt.title("Ground Truth")
            plt.xlabel("k-domain")
            plt.ylim([y_min, y_max])

            plt.subplot(248)
            err = y - y_pred
            plt.plot(np.real(err))
            plt.title("Error")
            plt.xlabel("k-domain")
            plt.ylim([y_min, y_max])

            plt.show()
        
        while True:
            print("Enter a test index or press enter to test a random index ('x'=exit):")
            user_input = input()
        
            if user_input == 'x' or user_input == 'X' or user_input == 'exit' or user_input == 'Exit':
                return
            elif user_input == '':
                test_one()
            elif user_input.isdigit():
                test_one(int(user_input))
        
    def Save(self, PATH):
        data_save = {
            'args': self.args,
            'dataset_train': self.dataset_train,
            'dataset_val': self.dataset_val
        }
        torch.save(data_save, PATH)
        print(f"Saved data to {PATH}")
        
    def Load(self, PATH):
        print(f"Loading data from {PATH}")
        data_save = torch.load(PATH)
        
        assert self.args.xyz_str == data_save['args']['xyz_str'], 'xyz_str must be the same as saved data!'
        self.dataset_train = data_save['dataset_train']
        self.dataset_val = data_save['dataset_val']
        
        
