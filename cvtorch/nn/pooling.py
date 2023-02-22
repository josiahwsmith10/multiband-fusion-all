import torch.nn as nn

from cvtorch import CVTensor

class CVAdaptiveAvgPool1d(nn.AdaptiveAvgPool1d):
    def __init__(self, output_size) -> None:
        super().__init__(output_size)
    def forward(self, x: CVTensor) -> CVTensor:
        return CVTensor(super().forward(x.real), super().forward(x.imag))

