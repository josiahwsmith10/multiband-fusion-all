import torch

from cvtorch import CVTensor
import cvtorch.nn as cvnn

from radar import MultiRadar
from model.kRNet import kRNet
from util.option import args
from util.loss import Loss

from model.SSCANet import SignalSelfCrossAttention, SSCANet_Big, SSCANet_Small
    
def test_complex_gradient():
    x = torch.ones(1, 1, 5, dtype=torch.complex64) + 1j*torch.ones(1, 1, 5, dtype=torch.complex64)
    z = torch.zeros(1, 1, 5, dtype=torch.complex64)
    
    x = CVTensor(x.real, x.imag)
    
    c = cvnn.SlowCVConv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
    nc = cvnn.CVConv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
    
    # Set conv weight
    c.conv.weight = torch.nn.Parameter(torch.ones_like(c.conv.weight) + 1j*torch.ones_like(c.conv.weight))
    nc.conv_r.weight = torch.nn.Parameter(torch.ones_like(nc.conv_r.weight))
    nc.conv_i.weight = torch.nn.Parameter(torch.ones_like(nc.conv_i.weight))
    
    # Set conv bias
    #c.conv.bias = torch.nn.Parameter(torch.ones_like(c.conv.bias) + 1j*torch.ones_like(c.conv.bias))
    #nc.conv_r.bias = torch.nn.Parameter(torch.ones_like(nc.conv_r.bias))
    #nc.conv_i.bias = torch.nn.Parameter(torch.ones_like(nc.conv_i.bias))
    
    L1 = torch.nn.L1Loss()
    
    opt1 = torch.optim.Adam(c.parameters(), lr=2e-1)
    opt2 = torch.optim.Adam(nc.parameters(), lr=2e-1)
    
    for i in range(5):
        print(f"\niteration: {i}")
        
        opt1.zero_grad()
        opt2.zero_grad()
        
        print("c  weight:", c.conv.weight.squeeze().detach().numpy())
        print("nc weight:", nc.conv_r.weight.squeeze().detach().numpy() + 1j*nc.conv_i.weight.squeeze().detach().numpy())
        #print("c  bias:", c.conv.bias.squeeze().detach().numpy())
        #print("nc bias:", nc.conv_r.bias.squeeze().detach().numpy() + 1j*nc.conv_i.bias.squeeze().detach().numpy())
        
        y1 = c(x).complex
        y2 = nc(x).complex
        
        print("y1:", y1.squeeze().detach().numpy())
        print("y2:", y2.squeeze().detach().numpy())
        
        l1 = L1(y1.real, z.real) + L1(y1.imag, z.imag)
        l2 = L1(y2.real, z.real) + L1(y2.imag, z.imag)
        
        l1.backward()
        l2.backward()
        
        opt1.step()
        opt2.step()
        
        print("c  weight:", c.conv.weight.squeeze().detach().numpy())
        print("nc weight:", nc.conv_r.weight.squeeze().detach().numpy() + 1j*nc.conv_i.weight.squeeze().detach().numpy())
        #print("c  bias:", c.conv.bias.squeeze().detach().numpy())
        #print("nc bias:", nc.conv_r.bias.squeeze().detach().numpy() + 1j*nc.conv_i.bias.squeeze().detach().numpy())

def test_advanced_loss():
    args.loss = '1*SL1+1*iSL1'
    ll = Loss(args)
    
    ll.step()
    ll.start_log()
    
    sr = torch.randn(10, 10, dtype=torch.complex64)
    hr = sr.clone() + 1
    
    ii = [sr] * 5
    
    print("loss=, ", ll(sr, hr, ii))

def scaled_dot_product_attention():
    x = torch.randn(64, 336, 128, dtype=torch.complex64)
    X = CVTensor(x.real, x.imag)
    
    m = cvnn.CVScaledDotProductAttention(1, attn_dropout=0.0)
    
    Y = m(X, X, X)
    print(X.shape, Y.shape)
    
def multi_head_attention():
    x = torch.randn(64, 336, 128, dtype=torch.complex64).to(args.device)
    X = CVTensor(x.real, x.imag)
    
    m = cvnn.CVMultiheadAttention(n_head=8, d_model=128, d_k=64, d_v=64, dropout=0.0).to(args.device)
    
    Y = m(X, X, X)
    print(X.shape, Y.shape)
    
def test_SSCA():
    a = torch.randn(16, 32, 64, dtype=torch.complex64).to(args.device)
    A = CVTensor(a.real, a.imag)
    
    b = torch.randn(16, 32, 64, dtype=torch.complex64).to(args.device)
    B = CVTensor(b.real, b.imag)
    
    m = SignalSelfCrossAttention(n_head=8, d_model=32, d_k=64, d_v=64, dropout=0.0, kernel_size=3).to(args.device)
    
    Y = m(A, B)
    print('SSCA test input/output', A.shape, B.shape, Y.shape)
    
def test_adaptive_pooling():
    x = torch.randn(64, 32, 336, dtype=torch.complex64).to(args.device)
    X = CVTensor(x.real, x.imag)
    
    m = cvnn.CVAdaptiveAvgPool1d(1).to(args.device)
    
    Y = m(X)
    print('Adaptive pooling test input/output', X.shape, Y.shape)
    
def test_CVECA():
    x = torch.randn(64, 32, 336, dtype=torch.complex64).to(args.device)
    X = CVTensor(x.real, x.imag)
    
    m = cvnn.CVECA(channels=32, b=1, gamma=2).to(args.device)
    
    Y = m(X)
    print('CVECA test input/output', X.shape, Y.shape)
    
def test_SSCA_Net():
    mr = MultiRadar(args)
    x = torch.randn(40, 336, dtype=torch.complex64).to(args.device)
    
    m = SSCANet_Small(args, mr).to(args.device)
    
    y = m(x)
    print('SSCA Net test input/output', x.shape, y.shape)
    

if __name__ == "__main__":
    #main_kRNet()
    #test_complex_gradient()
    
    #test_advanced_loss()
    
    #scaled_dot_product_attention()
    #multi_head_attention()
    
    #test_SSCA()
    test_SSCA_Net()
    
    #test_adaptive_pooling()
    
    #test_CVECA()

