import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def pixel_shuffle(x, size=2):
    
    b, c, t = x.size()

    new_c, new_t = c // size, t * size
    
    return x.view(b, new_c, new_t) 



class Upsample1DBlock(nn.Module):
    
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding):
        
        super().__init__()

        self.conv = nn.Conv1d(dim_in, dim_out, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
        self.inst_norm = nn.InstanceNorm1d(dim_out)
        self.glu = nn.GLU(1)
        
    def forward(self, x):
        
        out = self.conv(x)
        out = pixel_shuffle(out)
        out = self.inst_norm(out)
        out = self.glu(out)
        
        return out 


class Downsample1DBlock(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size, stride, padding):
        
        super().__init__()

        self.conv1 = nn.Conv1d(dim_in, dim_out, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
        self.inst_norm = nn.InstanceNorm1d(dim_out)
        self.glu = nn.GLU(dim = 1)
    def forward(self, x):
        out = self.conv1(x)
        out = self.inst_norm(out)
        out = self.glu(out)

        return out

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv1d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.inst_norm = nn.InstanceNorm1d(dim_out)
        self.glu = nn.GLU(dim=1)

    def forward(self, x):
        x_ = self.conv(x)
        x_ = self.inst_norm(x_)
        x_ = self.glu(x_)

        return x_


class Generator(nn.Module):

    def __init__(self, input_dim = 36, conv_dim = 64, num_speakers = 4, repeat_num = 6):
        super().__init__()
        
        layers = []

        layers.append(nn.Conv1d(input_dim, 256, kernel_size = 15, stride = 1, padding = 7, bias = False))
        layers.append(nn.GLU(dim = 1))

        # Downsample

        layers.append(Downsample1DBlock(128, 256, 5, 2, 2))
        layers.append(Downsample1DBlock(128, 512, 5, 2, 2))

        # Bottleneck layers

        for _ in range(repeat_num):
            layers.append( ResidualBlock(256, 512))
        
        # Upsample
        
        layers.append(Upsample1DBlock(256, 512, 5, 1, 2))
        layers.append(Upsample1DBlock(128, 256, 5, 1, 2))
        
        layers.append(nn.Conv1d(64, input_dim, kernel_size = 7, stride = 1, padding = 3, bias = False  ))
        
        self.main = nn.Sequential(*layers)
    def forward(self, x):
        
        out = x
        out = self.main(out)
        return out



class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, input_size=(36, 256), conv_dim=64, repeat_num=5):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            #layers.append(nn.InstanceNorm2d(curr_dim*2))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size_0 = int(input_size[0] / np.power(2, repeat_num)) # 1
        kernel_size_1 = int(input_size[1] / np.power(2, repeat_num)) # 8
        self.main = nn.Sequential(*layers)
        self.conv_dis = nn.Conv2d(curr_dim, 1, kernel_size=(1, 4), stride=1, padding=0, bias=False) # padding should be 0
        #self.conv_clf_spks = nn.Conv2d(curr_dim, num_speakers, kernel_size=(kernel_size_0, kernel_size_1), stride=1, padding=0, bias=False)  # for num_speaker
        
    def forward(self, x):
        x = x.unsqueeze(1)
        h = self.main(x)
        out_src = self.conv_dis(h)
        #out_src = torch.sigmoid(out_src)
        #out_cls_spks = self.conv_clf_spks(h)
        return out_src #out_cls_spks.view(out_cls_spks.size(0), out_cls_spks.size(1))
        
        
