import numpy as np
import torch
import torch.nn as nn

from cvtorch import CVTensor
import cvtorch.nn as cvnn

class CVScaledDotProductAttention(nn.Module):
    """Complex-Valued Scaled Dot-Product Attention."""
    def __init__(self, temperature, attn_dropout=0.1, SoftMaxClass=cvnn.MagMinMaxNorm):
        super(CVScaledDotProductAttention, self).__init__()
        
        self.temperature = temperature
        self.dropout = cvnn.CVDropout(attn_dropout)
        self.softmax = SoftMaxClass(dim=-1)
        
    def forward(self, q: CVTensor, k: CVTensor, v: CVTensor) -> CVTensor:
        attn = torch.matmul(q.complex / self.temperature, k.complex.transpose(-2, -1))
        
        attn = self.dropout(self.softmax(attn))
        output = torch.matmul(attn.complex, v.complex)
        return CVTensor(output.real, output.imag)

class CVMultiheadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int, d_k: int, d_v: int, dropout: float = 0.1, SoftMaxClass=cvnn.MagMinMaxNorm):
        super(CVMultiheadAttention, self).__init__()
        
        self.d_k = d_k
        self.d_v = d_v
        self.n_head = n_head
        
        self.w_q = cvnn.CVLinear(d_model, n_head * d_k, bias=False)
        self.w_k = cvnn.CVLinear(d_model, n_head * d_k, bias=False)
        self.w_v = cvnn.CVLinear(d_model, n_head * d_v, bias=False)
        self.fc = cvnn.CVLinear(n_head * d_v, d_model, bias=False)
        
        self.attention = CVScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout, SoftMaxClass=SoftMaxClass)
        
        self.dropout = cvnn.CVDropout(dropout)
        self.layer_norm = cvnn.CVLayerNorm(d_model, eps=1e-6)
        
    def forward(self, q: CVTensor, k: CVTensor, v: CVTensor):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        batch_size, len_q, len_k, len_v = q.shape[0], q.shape[1], k.shape[1], v.shape[1]
        
        res = q
        
        q = self.w_q(q).view(batch_size, len_q, n_head, d_k)
        k = self.w_k(k).view(batch_size, len_k, n_head, d_k)
        v = self.w_v(v).view(batch_size, len_v, n_head, d_v)
        
        q, k, v, = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        q = self.attention(q, k, v)
        
        q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        q = self.dropout(self.fc(q))
        q.add(res)
        
        return self.layer_norm(q)
    
class CVECA(nn.Module):
    """Complex-valued efficient channel attention."""
    def __init__(self, channels, b=1, gamma=2):
        super(CVECA, self).__init__()
        self.avg_pool = cvnn.CVAdaptiveAvgPool1d(1)
        self.channels = channels
        self.b = b
        self.gamma = gamma
        self.conv = cvnn.CVConv1d(1, 1, kernel_size=self.kernel_size(), padding=(self.kernel_size() - 1) // 2, bias=False) 
        self.sigmoid = cvnn.CSigmoid()

    def kernel_size(self):
        k = int(abs((np.log2(self.channels)/self.gamma)+ self.b/self.gamma))
        out = k if k % 2 else k+1
        return out

    def forward(self, x: CVTensor) -> CVTensor: 
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        
        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

