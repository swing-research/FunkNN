"""
Autoencoder (to be used in generative network)
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
from funknn_model import squeeze


class autoencoder(nn.Module):

    def __init__(self, encoder=None, decoder=None):
        super(autoencoder, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder


class encoder(nn.Module):

    def __init__(self, latent_dim=256, in_res=64, c=3):
        super(encoder, self).__init__()
        
        self.in_res = in_res
        self.c = c
        prev_ch = c
        c_last = 256
        CNNs = []
        CNNs_add = []
        num_layers = [64,128,128,256]
        for i in range(len(num_layers)):
            CNNs.append(nn.Conv2d(prev_ch, num_layers[i] ,3,
                padding = 'same'))
            CNNs_add.append(nn.Conv2d(num_layers[i], num_layers[i] ,3,
                padding = 'same'))
            CNNs_add.append(nn.Conv2d(num_layers[i], num_layers[i] ,3,
                padding = 'same'))
            prev_ch = num_layers[i]
        
        if in_res == 64:
            num_layers = [c_last]
            for i in range(len(num_layers)):
                CNNs.append(nn.Conv2d(prev_ch, num_layers[i] ,3,
                                      padding = 'same'))
                CNNs_add.append(nn.Conv2d(num_layers[i], num_layers[i] ,3,
                    padding = 'same'))
                CNNs_add.append(nn.Conv2d(num_layers[i], num_layers[i] ,3,
                    padding = 'same'))
                prev_ch = num_layers[i]
        
        if in_res == 128:
            num_layers = [256,c_last]
            for i in range(len(num_layers)):
                CNNs.append(nn.Conv2d(prev_ch, num_layers[i] ,3,
                    padding = 'same'))
                CNNs_add.append(nn.Conv2d(num_layers[i], num_layers[i] ,3,
                    padding = 'same'))
                CNNs_add.append(nn.Conv2d(num_layers[i], num_layers[i] ,3,
                    padding = 'same'))
                prev_ch = num_layers[i]

            
            
        
        self.CNNs = nn.ModuleList(CNNs)
        self.CNNs_add = nn.ModuleList(CNNs_add)
        self.maxpool = nn.MaxPool2d(2, 2)
        
        feature_dim = 2 * 2 * c_last
        mlps = []
        mlps.append(nn.Linear(feature_dim, latent_dim))

        self.mlps = nn.ModuleList(mlps)
       

    def forward(self, x):

        x_skip = torch.mean(x , dim = 1, keepdim = True)
        for i in range(len(self.CNNs)):
            x = self.CNNs[i](x)
            x = F.relu(x)
            xm = self.maxpool(x)
            if i < 4:

                f = 2**(i+1)
                xm = xm + squeeze(x_skip , f).repeat_interleave(xm.shape[1]//(f**2) , dim = 1)

            x = self.CNNs_add[i*2](xm)
            x = F.relu(x)
            x = self.CNNs_add[i*2 + 1](x)
            x = F.relu(x)
            x = x + xm

        x = torch.flatten(x, 1)
        for i in range(len(self.mlps)-1):
            x = self.mlps[i](x)
            x = F.relu(x)
        
        x = self.mlps[-1](x)
        
        return x


class decoder(nn.Module):

    def __init__(self, latent_dim=256, in_res=64, c=3):
        super(decoder, self).__init__()
        
        self.in_res = in_res
        self.c = c
        prev_ch = 256
        t_CNNs = []
        CNNs = []

        if in_res == 128:
            num_layers = [256,256,128,128,64,self.c]
            for i in range(len(num_layers)):
                # t_CNNs.append(nn.ConvTranspose2d(prev_ch, num_layers[i] ,3,
                #     stride=2,padding = 1, output_padding=1))
                c_inter = 64 if num_layers[i] == self.c else num_layers[i]
                t_CNNs.append(nn.Conv2d(prev_ch, c_inter ,3,
                    padding = 'same'))
                CNNs.append(nn.Conv2d(c_inter, c_inter ,3,
                    padding = 'same'))
                CNNs.append(nn.Conv2d(c_inter, num_layers[i] ,3,
                    padding = 'same'))
                prev_ch = num_layers[i]

        elif in_res == 64:

            num_layers = [256,128,128,64,self.c]
            for i in range(len(num_layers)):

                c_inter = 64 if num_layers[i] == self.c else num_layers[i]
                t_CNNs.append(nn.Conv2d(prev_ch, c_inter ,3,
                    padding = 'same'))
                CNNs.append(nn.Conv2d(c_inter, c_inter ,3,
                    padding = 'same'))
                CNNs.append(nn.Conv2d(c_inter, num_layers[i] ,3,
                    padding = 'same'))
                prev_ch = num_layers[i]
        

            
        self.t_CNNs = nn.ModuleList(t_CNNs)
        self.CNNs = nn.ModuleList(CNNs)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.feature_dim = 2 * 2 * 256
        mlps = []
        mlps.append(nn.Linear(latent_dim , self.feature_dim))

        self.mlps = nn.ModuleList(mlps)
       
    def forward(self, x):

        for i in range(len(self.mlps)):
            x = self.mlps[i](x)
            x = F.relu(x)
        # x = squeeze(x , 16)
        b = x.shape[0]
        x = x.reshape([b, 256, 2, 2])
        
        for i in range(len(self.t_CNNs)-1):
            x = self.upsample(x)
            x = self.t_CNNs[i](x)
            xr = F.relu(x)
            x = self.CNNs[i*2](xr)
            x = F.relu(x)
            x = self.CNNs[i*2+1](x)
            x = F.relu(x)
            x = x + xr

        x = self.upsample(x)
        x = self.t_CNNs[-1](x)
        x = F.relu(x)
        x = self.CNNs[-2](x)
        x = F.relu(x)
        x = self.CNNs[-1](x)

        return x

