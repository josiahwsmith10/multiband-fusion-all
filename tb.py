import torch

from model.kRNet import kRNet
from util.option import args

from model.layers import naive_complex_default_conv1d, complex_default_conv1d, fast_complex_default_conv1d
from model.common import CPLX

from util.loss import Loss

def main_kRNet():
    m = kRNet(args)
    
    x = torch.randn(64, 336, dtype=torch.complex64)
    y = m(x).to(args.device)
    print(x.shape, y.shape, m.num_params())
    
def test_complex_gradient():
    x = torch.ones(1, 1, 5, dtype=torch.complex64) + 1j*torch.ones(1, 1, 5, dtype=torch.complex64)
    z = torch.zeros(1, 1, 5, dtype=torch.complex64)
    
    x = CPLX(x.real, x.imag)
    
    c = complex_default_conv1d(in_channels=1, out_channels=1, kernel_size=3, bias=False)
    nc = naive_complex_default_conv1d(in_channels=1, out_channels=1, kernel_size=3, bias=False)
    fc = fast_complex_default_conv1d(in_channels=1, out_channels=1, kernel_size=3, bias=False)
    
    # Set conv weight
    c.conv.weight = torch.nn.Parameter(torch.ones_like(c.conv.weight) + 1j*torch.ones_like(c.conv.weight))
    nc.conv_r.weight = torch.nn.Parameter(torch.ones_like(nc.conv_r.weight))
    nc.conv_i.weight = torch.nn.Parameter(torch.ones_like(nc.conv_i.weight))
    fc.conv_r.weight = torch.nn.Parameter(torch.ones_like(fc.conv_r.weight))
    fc.conv_i.weight = torch.nn.Parameter(torch.ones_like(fc.conv_i.weight))
    
    # Set conv bias
    #c.conv.bias = torch.nn.Parameter(torch.ones_like(c.conv.bias) + 1j*torch.ones_like(c.conv.bias))
    #nc.conv_r.bias = torch.nn.Parameter(torch.ones_like(nc.conv_r.bias))
    #nc.conv_i.bias = torch.nn.Parameter(torch.ones_like(nc.conv_i.bias))
    
    L1 = torch.nn.L1Loss()
    
    opt1 = torch.optim.Adam(c.parameters(), lr=2e-1)
    opt2 = torch.optim.Adam(nc.parameters(), lr=2e-1)
    opt3 = torch.optim.Adam(fc.parameters(), lr=2e-1)
    
    for i in range(5):
        print(f"\niteration: {i}")
        
        opt1.zero_grad()
        opt2.zero_grad()
        
        print("c  weight:", c.conv.weight.squeeze().detach().numpy())
        print("nc weight:", nc.conv_r.weight.squeeze().detach().numpy() + 1j*nc.conv_i.weight.squeeze().detach().numpy())
        print("fc weight:", fc.conv_r.weight.squeeze().detach().numpy() + 1j*fc.conv_i.weight.squeeze().detach().numpy())
        #print("c  bias:", c.conv.bias.squeeze().detach().numpy())
        #print("nc bias:", nc.conv_r.bias.squeeze().detach().numpy() + 1j*nc.conv_i.bias.squeeze().detach().numpy())
        
        y1 = c(x).complex
        y2 = nc(x).complex
        y3 = fc(x).complex
        
        print("y1:", y1.squeeze().detach().numpy())
        print("y2:", y2.squeeze().detach().numpy())
        print("y3:", y3.squeeze().detach().numpy())
        
        l1 = L1(y1.real, z.real) + L1(y1.imag, z.imag)
        l2 = L1(y2.real, z.real) + L1(y2.imag, z.imag)
        l3 = L1(y3.real, z.real) + L1(y3.imag, z.imag)
        
        l1.backward()
        l2.backward()
        l3.backward()
        
        opt1.step()
        opt2.step()
        opt3.step()
        
        print("c  weight:", c.conv.weight.squeeze().detach().numpy())
        print("nc weight:", nc.conv_r.weight.squeeze().detach().numpy() + 1j*nc.conv_i.weight.squeeze().detach().numpy())
        print("fc weight:", fc.conv_r.weight.squeeze().detach().numpy() + 1j*fc.conv_i.weight.squeeze().detach().numpy())
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
    

if __name__ == "__main__":
    #main_kRNet()
    #test_complex_gradient()
    test_advanced_loss()
