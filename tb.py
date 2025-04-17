import torch

import complextorch.nn as cvnn

from radar import MultiRadar
from model.kRNet import kRNet
from util.option import args
from util.loss import Loss

from model.SSCANet import SignalSelfCrossAttention, SSCANet_Big, SSCANet_Small


def test_advanced_loss():
    args.loss = "1*SL1+1*iSL1"
    ll = Loss(args)

    ll.step()
    ll.start_log()

    sr = torch.randn(10, 10, dtype=torch.cfloat)
    hr = sr.clone() + 1

    ii = [sr] * 5

    print("loss=, ", ll(sr, hr, ii))


def scaled_dot_product_attention():
    X = torch.randn(64, 336, 128, dtype=torch.cfloat)

    m = cvnn.ScaledDotProductAttention(1, attn_dropout=0.0)

    Y = m(X, X, X)
    print(X.shape, Y.shape)


def multi_head_attention():
    X = torch.randn(64, 336, 128, dtype=torch.cfloat).to(args.device)

    m = cvnn.MultiheadAttention(n_heads=8, d_model=128, d_k=64, d_v=64, dropout=0.0).to(
        args.device
    )

    Y = m(X, X, X)
    print(X.shape, Y.shape)


def test_SSCA():
    A = torch.randn(16, 32, 64, dtype=torch.cfloat).to(args.device)
    B = torch.randn(16, 32, 64, dtype=torch.cfloat).to(args.device)

    m = SignalSelfCrossAttention(
        n_heads=8, d_model=32, d_k=64, d_v=64, dropout=0.0, kernel_size=3
    ).to(args.device)

    Y = m(A, B)
    print("SSCA test input/output", A.shape, B.shape, Y.shape)


def test_adaptive_pooling():
    X = torch.randn(64, 32, 336, dtype=torch.cfloat).to(args.device)

    m = cvnn.AdaptiveAvgPool1d(1).to(args.device)

    Y = m(X)
    print("Adaptive pooling test input/output", X.shape, Y.shape)


def test_CVECA():
    X = torch.randn(64, 32, 336, dtype=torch.cfloat).to(args.device)

    m = cvnn.EfficientChannelAttention1d(channels=32, b=1, gamma=2).to(args.device)

    Y = m(X)
    print("CVECA test input/output", X.shape, Y.shape)


def test_SSCA_Net_Small():
    mr = MultiRadar(args)
    x = torch.randn(40, 336, dtype=torch.cfloat).to(args.device)

    m = SSCANet_Small(args, mr).to(args.device)

    y = m(x)[0]
    print("SSCA Net Small test input/output", x.shape, y.shape, type(y))


def test_SSCA_Net_Big():
    mr = MultiRadar(args)
    x = torch.randn(40, 336, dtype=torch.cfloat).to(args.device)

    m = SSCANet_Big(args, mr).to(args.device)

    y = m(x)[0]
    print("SSCA Net Big test input/output", x.shape, y.shape, type(y))
    

def test_kRNet():
    mr = MultiRadar(args)
    x = torch.randn(40, mr.Nk_fb, dtype=torch.cfloat).to(args.device)

    m = kRNet(args).to(args.device)
    
    y = m(x)[0]
    print("kRNet test input/output", x.shape, y.shape, type(y))

if __name__ == "__main__":
    test_kRNet()

    # test_advanced_loss()

    # scaled_dot_product_attention()
    # multi_head_attention()

    # test_SSCA()
    # test_SSCA_Net_Small()
    # test_SSCA_Net_Big()

    # test_adaptive_pooling()

    # test_CVECA()

    pass
