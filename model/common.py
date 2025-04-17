import argparse
import torch.nn as nn
import complextorch.nn as cvnn


def default_conv1d(*args, **kwargs) -> nn.Module:
    if "padding" not in kwargs:
        if "kernel_size" in kwargs:
            kwargs["padding"] = kwargs["kernel_size"] // 2
        else:
            kwargs["padding"] = args[2] // 2

    return cvnn.Conv1d(*args, **kwargs)


def select_act_module(args: argparse.Namespace):
    if args.act.lower() == "crelu":
        act_module = cvnn.CReLU
    elif args.act.lower() == "ctanh":
        act_module = cvnn.CTanh
    elif args.act.lower() == "cprelu":
        act_module = cvnn.CPReLU
    return act_module
