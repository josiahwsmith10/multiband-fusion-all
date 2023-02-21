import torch
import time
import os
import argparse
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from model import ComplexModel
from data import MultiRadarData
from util import Loss
from util.saver import SaveModel
from util.common import make_optimizer

class Trainer():
    def __init__(self, args: argparse.Namespace, data: MultiRadarData, model: ComplexModel, loss: Loss, tb=True):
        print('Making the trainer...')
        self.args = args
        
        # Number of samples of GT high resolution signal
        self.N_HR = data.mr.Nk_fb
        
        self.loader_train = torch.utils.data.DataLoader(data.dataset_train, 
                                                    batch_size=args.batch_size, 
                                                    shuffle=True, 
                                                    pin_memory=False,
                                                    num_workers=0)
        
        self.loader_val = torch.utils.data.DataLoader(data.dataset_val, 
                                                    batch_size=args.batch_size, 
                                                    shuffle=True, 
                                                    pin_memory=False,
                                                    num_workers=0)
        
        self.batches_per_epoch = len(self.loader_train)
        self.model = model.to(args.device)
        self.loss = loss
        self.optimizer = make_optimizer(args, self.model)
        
        self.logs = {
            'log loss min': torch.inf, 
            'val_log loss min': torch.inf, 
            'log loss max': -torch.inf, 
            'val_log loss max': -torch.inf
            }
        
        # Create directory to save model checkpoints
        self.save_path = "./saved/models/" + args.model_name + "/"
        os.mkdir(self.save_path)
        
        if tb:
            print('Using tensorboard...')
            self.use_tensorboard = True
            
            # Initialize tensorboard writer
            self.writer = SummaryWriter(log_dir="./saved/models/" + args.model_name)
            
            # Save parameters
            for arg, value in vars(args).items():
                self.writer.add_text(arg, str(value))
        else:
            self.use_tensorboard = False
        
    def train(self):
        """Trains one epoch."""
        
        #torch.cuda.empty_cache()
        
        # Training phase
        self.loss.step()
        
        self.loss.start_log()
        self.model.train()
        
        self.tic = time.time()
        
        print('Training phase for epoch ', self.optimizer.get_last_epoch())
        for lr, hr in tqdm(self.loader_train):
            # lr - low resolution (feature) (batch_size x N_HR)
            # hr - high resolution (label) (batch_size x N_HR)
            
            # Zero gradients every batch
            self.optimizer.zero_grad()
            
            # Forward pass
            sr, intermediate = self.model(lr)

            # Compute loss
            loss = self.loss(sr, hr, intermediate)
            loss.backward()
            
            # Adjust learning weights
            self.optimizer.step()
            
        self.logs['log loss'], self.logs['train losses'] = self.loss.end_log(len(self.loader_train))
        self.optimizer.schedule()
        
        # Validation phase
        with torch.no_grad():
            self.loss.start_log()
            for lr, hr in self.loader_val:
                # lr - low resolution (feature) (batch_size x N_HR)
                # hr - high resolution (label) (batch_size x N_HR)

                # Forward pass
                sr, intermediate = self.model(lr)

                # Compute loss
                loss = self.loss(sr, hr, intermediate)
                
            self.logs['val_log loss'], self.logs['val losses'] = self.loss.end_log(len(self.loader_val))
        
        self.print_loss()
        self.save_checkpoint()
        
        self.model.train_loss.append(self.logs['log loss'])
        self.model.val_loss.append(self.logs['val_log loss'])
    
    def print_loss(self):
        """Prints the loss in readable format during training."""
        
        # Leave if print_every is 0 (never print loss)
        if self.args.print_every == 0:
            return
        
        # Always send to tensorboard if it is used
        if self.use_tensorboard:
            epoch = self.optimizer.get_last_epoch()
            add_scalar = self.writer.add_scalar
            
            add_scalar('Benchmark/train', self.logs['log loss'], epoch)
            add_scalar('Benchmark/val', self.logs['val_log loss'], epoch)
            
            for type, value in self.logs['train losses'].items():
                add_scalar(f'{type}/train', value, epoch)
            
            for type, value in self.logs['val losses'].items():
                add_scalar(f'{type}/val', value, epoch)
            self.writer.flush()
        
        # Leave if not the right epoch
        epoch = self.optimizer.get_last_epoch()
        if epoch % self.args.print_every != 0:
            return
            
        if self.logs['log loss'] < self.logs['log loss min']:
            self.logs['log loss min'] = self.logs['log loss']
        if self.logs['log loss'] > self.logs['log loss max']:
            self.logs['log loss max'] = self.logs['log loss']
        if self.logs['val_log loss'] < self.logs['val_log loss min']:
            self.logs['val_log loss min'] = self.logs['val_log loss']
        if self.logs['val_log loss'] > self.logs['val_log loss max']:
            self.logs['val_log loss max'] = self.logs['val_log loss']
        
        print(f"epoch {epoch} finished in {time.time() - self.tic:.3f} sec")
        print(f"training \t min:{self.logs['log loss min']:.3f}, max:{self.logs['log loss max']:.3f}, cur:{self.logs['log loss']:.3f}")
        print(f"validation \t min:{self.logs['val_log loss min']:.3f}, max:{self.logs['val_log loss max']:.3f}, cur:{self.logs['val_log loss']:.3f}\n")
        
    def save_checkpoint(self):
        """Saves checkpoint of the model at current epoch"""
        
        # Leave if save_every is 0 (never save model)
        if self.args.save_every == 0:
            return
        
        # Leave if not the right epoch
        epoch = self.optimizer.get_last_epoch()
        if epoch % self.args.save_every != 0:
            return
        
        # Save model
        SaveModel(args=self.args, model=self.model, loss=self.loss, trainer=self, PATH=self.save_path + f"epoch{epoch}_{self.args.model_name}.pt")
        
    def terminate(self):
        epoch = self.optimizer.get_last_epoch()
        return epoch >= self.args.epochs

