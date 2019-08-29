import torch
import torch.nn as nn

   
class LearnedTransform_unet(nn.Module):
    def __init__(self, network):
        super(LearnedTransform_unet, self).__init__()
        self.network = network
    def forward(self, x):
        out = self.network(x) + x
        return out, []
