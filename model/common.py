import argparse

import cvtorch.nn as cvnn

def select_act(args: argparse.Namespace):
    if args.act.lower() == 'crelu':
        act = cvnn.CReLU(True)
    elif args.act.lower() == 'ctanh':
        act = cvnn.CTanh()
    elif args.act.lower() == 'cprelu':
        act = cvnn.CPReLU()
    return act

