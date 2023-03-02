import argparse
import torch
from datetime import datetime
from uuid import uuid4

parser = argparse.ArgumentParser(description="multiband-fusion-all")

# Model-related arguments
parser.add_argument("--model", type=str, default="kR-Net", help="Model type")

parser.add_argument(
    "--xyz_str",
    type=str,
    default="xz",
    help="xyz_str: xy = frequency(LR)-frequency(HR), xz = frequency(LR)-time/k(HR)",
)

parser.add_argument(
    "--act",
    type=str,
    default="CPReLU",
    help="Activation function: CTanh, CReLU, CPReLU",
)

parser.add_argument(
    "--in_channels", type=int, default=1, help="Number of input channels"
)

parser.add_argument(
    "--out_channels", type=int, default=1, help="Number of output channels"
)

parser.add_argument("--n_feats", type=int, default=32, help="Number of feature maps")

parser.add_argument("--kernel_size", type=int, default=5, help="Kernel size")

parser.add_argument(
    "--res_scale", type=float, default=0.8, help="Residual scaling factor"
)

parser.add_argument(
    "--precision", type=str, default="single", help="Precision: single or half"
)

# Used for kR-Blocks
parser.add_argument(
    "--n_res_blocks", type=int, default=8, help="Number of residual blocks"
)

# Used for SSCA-Net
parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")

parser.add_argument("--d_k", type=int, default=64, help="Dimension of key")

parser.add_argument("--d_v", type=int, default=64, help="Dimension of value")

parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

# Used for Efficient Channel Attention Module
parser.add_argument(
    "--eca_b",
    type=int,
    default=1,
    help="b hyper-parameter for adaptive kernel size formulation in ECA",
)

parser.add_argument(
    "--eca_gamma",
    type=float,
    default=2,
    help="gamma hyper-parameter for adaptive kernel size formulation in ECA",
)

# Training-related arguments
parser.add_argument(
    "--loss",
    type=str,
    default="1*bSL1",
    help='Loss function: loss functions separated by "+", each loss function has [weight]*[loss_type]',
)

parser.add_argument(
    "--optimizer", type=str, default="ADAM", help="Optimizer: ADAM or SGD"
)

parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay")

parser.add_argument("--decay", type=str, default="200", help="Decay milestones")

parser.add_argument(
    "--gamma", type=float, default=0.5, help="Decay factor at each milestone"
)

parser.add_argument(
    "--betas", type=tuple, default=(0.9, 0.999), help="Betas for ADAM optimizer"
)

parser.add_argument(
    "--epsilon", type=float, default=1e-8, help="Epsilon for ADAM optimizer"
)

parser.add_argument(
    "--print_every",
    type=int,
    default=1,
    help="Print loss every n iterations (0 = never)",
)

parser.add_argument(
    "--save_every",
    type=int,
    default=1,
    help="Save model every n iterations (0 = never)",
)

parser.add_argument("--epochs", type=float, default=1e8, help="Number of epochs")

parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")

# Checkpoint to load previously trained model
parser.add_argument(
    "--checkpoint", type=str, default="", help="Checkpoint name of model to load"
)

# Data-related arguments
parser.add_argument(
    "--dataset",
    type=str,
    default="",
    help="Dataset name to load (dataset_60GHz_77GHz_1048576_2048_Nt64.mrd)",
)

parser.add_argument(
    "--num_train", type=int, default=1048576, help="Number of training samples"
)

parser.add_argument(
    "--num_val", type=int, default=2048, help="Number of validation samples"
)

parser.add_argument("--num_test", type=int, default=2048, help="Number of test samples")

parser.add_argument(
    "--Nt", type=int, default=64, help="Number of random target scatter components"
)

# Radar-related arguments
parser.add_argument(
    "--f0",
    nargs="+",
    type=float,
    default=[60e9, 77e9],
    help="Starting frequency of each radar",
)

parser.add_argument("--K", type=float, default=124.996e12, help="Slope of FMCW chirp")

parser.add_argument(
    "--Nk", nargs="+", type=int, default=[64, 64], help="Number of samples per chirp"
)

parser.add_argument(
    "--Nk_fb", type=int, default=336, help="Number of samples of the full-band chirp"
)

parser.add_argument("--fS", type=float, default=2000e3, help="ADC sample rate")

# TensorBoard-related arguments
parser.add_argument(
    "--use_tensorboard",
    type=str,
    default="True",
    help="Boolean whether or not to use TensorBoard",
)

# Default options
parser.add_argument(
    "--defaults",
    nargs="+",
    type=str,
    default=[""],
    help="""Defaults to use (kR-Net, SSCA-Net, default datasets, etc.).
                    This overrides other arguments""",
)

# Parse the arguments
args = parser.parse_args()

for arg in vars(args):
    if vars(args)[arg] == "True":
        vars(args)[arg] = True
    elif vars(args)[arg] == "False":
        vars(args)[arg] = False

# args.defaults = ['SSCA-Net-Big', '64ghz_1024_small']
# args.use_tensorboard = False

# Parse the defaults
# First, datasets
if any([val.lower() == "21ghz_336_large" for val in args.defaults]):
    print("Using 21GHz_336_large dataset for SR")
    args.dataset = "dataset_60GHz_77GHz_1048576_2048_Nt64.mrd"
    args.num_train = 1048576
    args.num_val = 2048
    args.num_test = 2048
    args.Nk_fb = 336
if any([val.lower() == "21ghz_336_med" for val in args.defaults]):
    print("Using 21GHz_336_med dataset for SR")
    args.dataset = "dataset_60GHz_77GHz_16384_2048_Nt64.mrd"
    args.num_train = 16384
    args.num_val = 2048
    args.num_test = 2048
    args.Nk_fb = 336
if any([val.lower() == "21ghz_336_small" for val in args.defaults]):
    print("Using 21GHz_336_small dataset for SR")
    args.dataset = "dataset_60GHz_77GHz_2048_1024_Nt64.mrd"
    args.num_train = 2048
    args.num_val = 1024
    args.num_test = 1024
    args.Nk_fb = 336
if any([val.lower() == "64ghz_1024_large" for val in args.defaults]):
    print("Using 64GHz_1024_large dataset for 16x SR")
    args.dataset = "dataset_16x_1048576_2048_Nt64.mrd"
    args.num_train = 1048576
    args.num_val = 2048
    args.num_test = 2048
    args.Nk_fb = 1024
if any([val.lower() == "64ghz_1024_med" for val in args.defaults]):
    print("Using 64ghz_1024_med dataset for 16x SR")
    args.dataset = "dataset_16x_16384_2048_Nt64.mrd"
    args.num_train = 16384
    args.num_val = 2048
    args.num_test = 2048
    args.Nk_fb = 1024
if any([val.lower() == "64ghz_1024_small" for val in args.defaults]):
    print("Using 64ghz_1024_small dataset for 16x SR")
    args.dataset = "dataset_16x_2048_1024_Nt64.mrd"
    args.num_train = 2048
    args.num_val = 1024
    args.num_test = 1024
    args.Nk_fb = 1024

# Default models
if any([val.lower() == "ssca-net-big" for val in args.defaults]):
    print("Using SSCA-Net (big) defaults")
    args.model = "SSCA-Net-Big"
    args.d_v = 8
    args.d_k = 8
    args.n_heads = 8
    args.n_feats = 8
    args.batch_size = 512

# Set dataset type
if args.Nk_fb == 336:
    args.dataset_type = "60GHz_77GHz"
elif args.Nk_fb == 1024:
    args.dataset_type = "16x"

# Add cuda device
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create descriptive model name
args.model_name = (
    datetime.now().strftime("%Y-%m-%d")
    + "_"
    + args.model
    + "_"
    + args.dataset_type
    + "_"
    + uuid4().hex[:4]
)
