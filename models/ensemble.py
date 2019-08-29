import torch
import torch.nn as nn
import torch.nn.functional as F


import torch.nn as nn
class MyEnsemble(nn.Module):
    def __init__(self, admm_model, denoise_model):
        super(MyEnsemble, self).__init__()
        self.admm_model = admm_model
        self.denoise_model = denoise_model
        
    def forward(self, x):
        
        admm_output = self.admm_model(x)
        final_output = self.denoise_model(admm_output) 
        
        return final_output
    
    def to(self, indevice):
        self = super().to(indevice)
        self.admm_model.to(indevice)
        self.admm_model.h_var.to(indevice)
        self.admm_model.h_zeros.to(indevice)
        self.admm_model.h_complex.to(indevice)
        self.admm_model.LtL.to(indevice)
        return self
    
   
class LearnedTransform_unet(nn.Module):
    def __init__(self, network):
        super(LearnedTransform_unet, self).__init__()
        self.network = network
    def forward(self, x):
        out = self.network(x) + x
        return out, []

