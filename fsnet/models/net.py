import torch
import torch.nn as nn

from models.ts2vec.encoder import TSEncoder
from models.ts2vec.fsnet import TSEncoder as FSNetTSEncoder



class TS2VecEncoderWrapper(nn.Module):
    def __init__(self, encoder, mask):
        super().__init__()
        self.encoder = encoder
        self.mask = mask

    def forward(self, input):
        return self.encoder(input.float(), mask=self.mask)[:, -1]

class Net(nn.Module):
    def __init__(self, 
                 enc_in:int, 
                 c_out:int,
                 output_dims:int,
                 hidden_dims:int,
                 depth:int,
                 regressor_dims:int,
                 pred_len:int,
                 device):
        
        super().__init__()
        self.device = device
        encoder = TSEncoder(input_dims=enc_in,
                             output_dims=output_dims,
                             hidden_dims=hidden_dims,
                             depth=depth) 
        self.encoder = TS2VecEncoderWrapper(encoder, mask='all_true').to(self.device)
        self.dim = c_out * pred_len
        
        #self.regressor = nn.Sequential(nn.Linear(320, 320), nn.ReLU(), nn.Linear(320, self.dim)).to(self.device)
        self.regressor = nn.Linear(regressor_dims, self.dim).to(self.device)
        
    def forward(self, x, x_mark=None):
        if x_mark is not None:
            x = torch.cat([x, x_mark], dim=-1).float().to(self.device)
        rep = self.encoder(x)
        y = self.regressor(rep)
        return y


class FSNet(nn.Module):
    def __init__(self, 
                 enc_in:int, 
                 c_out:int,
                 output_dims:int,
                 hidden_dims:int,
                 depth:int,
                 regressor_dims:int,
                 pred_len:int,
                 device):
        
        super().__init__()
        self.device = device
        encoder = FSNetTSEncoder(input_dims=enc_in,
                             output_dims=output_dims,
                             hidden_dims=hidden_dims,
                             depth=depth) 
        self.encoder = TS2VecEncoderWrapper(encoder, mask='all_true').to(self.device)
        self.dim = c_out * pred_len
        
        #self.regressor = nn.Sequential(nn.Linear(320, 320), nn.ReLU(), nn.Linear(320, self.dim)).to(self.device)
        self.regressor = nn.Linear(regressor_dims, self.dim).to(self.device)
        
    def forward(self, x, x_mark=None):
        if x_mark is not None:
            x = torch.cat([x, x_mark], dim=-1).float().to(self.device)
        rep = self.encoder(x)
        y = self.regressor(rep)
        return y
    
    def store_grad(self):
        for name, layer in self.encoder.named_modules():    
            if 'PadConv' in type(layer).__name__:
                #print('{} - {}'.format(name, type(layer).__name__))
                layer.store_grad()