import argparse
import torch
from datetime import datetime
from uuid import uuid4

parser = argparse.ArgumentParser(description='kR-Net')

# Model-related arguments
parser.add_argument('--model', type=str, default='kR-Net',
                    help='Model type')

parser.add_argument('--xyz_str', type=str, default='xz',
                    help='xyz_str: xy = frequency(LR)-frequency(HR), xz = frequency(LR)-time/k(HR)')

parser.add_argument('--act', type=str, default='CPReLU',
                    help='Activation function: CTanh, CReLU, CPReLU')

parser.add_argument('--in_channels', type=int, default=1,
                    help='Number of input channels')

parser.add_argument('--out_channels', type=int, default=1,
                    help='Number of output channels')

parser.add_argument('--n_feats', type=int, default=32,
                    help='Number of feature maps')

parser.add_argument('--kernel_size', type=int, default=5,
                    help='Kernel size')

parser.add_argument('--res_scale', type=float, default=0.8,
                    help='Residual scaling factor')

parser.add_argument('--precision', type=str, default='single',
                    help='Precision: single or half')

# Used for kR-Net
parser.add_argument('--n_res_blocks', type=int, default=8,
                    help='Number of residual blocks')

# Training-related arguments
parser.add_argument('--loss', type=str, default='1*bSL1',
                    help='Loss function: loss functions separated by "+", each loss function has [weight]*[loss_type]')

parser.add_argument('--optimizer', type=str, default='ADAM',
                    help='Optimizer: ADAM or SGD')

parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate')

parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay')

parser.add_argument('--decay', type=str, default='200',
                    help='Decay milestones')

parser.add_argument('--gamma', type=float, default=0.5,
                    help='Decay factor at each milestone')

parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='Betas for ADAM optimizer')

parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='Epsilon for ADAM optimizer')

parser.add_argument('--print_every', type=int, default=1,
                    help='Print loss every n iterations (0 = never)')

parser.add_argument('--save_every', type=int, default=1,
                    help='Save model every n iterations (0 = never)')

parser.add_argument('--epochs', type=float, default=1e8,
                    help='Number of epochs')

parser.add_argument('--batch_size', type=int, default=1024,
                    help='Batch size')

# Checkpoint to load previously trained model
parser.add_argument('--checkpoint', type=str, default='',
                    help='Checkpoint name of model to load')

# Data-related arguments
parser.add_argument('--dataset', type=str, default='',
                    help='Dataset name to load (dataset_60GHz_77GHz_1048576_2048_Nt64.mrd)')

parser.add_argument('--num_train', type=int, default=1048576,
                    help='Number of training samples')

parser.add_argument('--num_val', type=int, default=2048,
                    help='Number of validation samples')

parser.add_argument('--num_test', type=int, default=2048,
                    help='Number of test samples')

parser.add_argument('--Nt', type=int, default=64,
                    help='Number of random target scatter components')

# Parse the arguments
args = parser.parse_args()

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
        
# Add cuda device
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args.model_name = datetime.now().strftime('%Y-%m-%d') + '_' + args.model + '_' + uuid4().hex[:8]

