import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import math


class GLU(nn.Module):
    ''' GLU block, do not split channels dimension'''

    def __init__(self,):
        super().__init__()

    def forward(self, x):
        
        return x * torch.sigmoid(x)


class AdaptiveInstanceNormalisation(nn.Module):
    """AdaIN Block."""
    def __init__(self, dim_in, dim_c):
        super(AdaptiveInstanceNormalisation, self).__init__()

        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.dim_in = dim_in
        #self.style_num = style_num
        #self.gamma_s = nn.Linear(dim_c, dim_in)
        #self.beta_s = nn.Linear(dim_c, dim_in)
        
        self.gamma_t = nn.Linear(dim_c, dim_in)
        self.beta_t = nn.Linear(dim_c, dim_in)

    def forward(self, x, c_src, c_trg):
        u = torch.mean(x, dim=2, keepdim=True)
        var = torch.mean((x - u) * (x - u), dim=2, keepdim=True)
        std = torch.sqrt(var + 1e-8)

        # width = x.shape[2]
        #c = torch.cat([c_src, c_trg], dim = -1)
        
        gamma = self.gamma_t(c_trg.to(x.device)) #+ self.gamma_s(c_src.to(x.device))
        gamma = gamma.view(-1, self.dim_in, 1)
        beta = self.beta_t(c_trg.to(x.device)) #+ self.beta_s(c_src.to(x.device))
        beta = beta.view(-1, self.dim_in, 1)

        h = (x - u) / std
        h = h * gamma + beta

        return h
class ConditionalInstanceNormalisation(nn.Module):
    """CIN Block."""
    def __init__(self, dim_in, style_num):
        super(ConditionalInstanceNormalisation, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.dim_in = dim_in
        self.style_num = style_num
        self.gamma = nn.Linear(style_num, dim_in)
        self.beta = nn.Linear(style_num, dim_in)

    def forward(self, x, c):
        u = torch.mean(x, dim=2, keepdim=True)
        var = torch.mean((x - u) * (x - u), dim=2, keepdim=True)
        std = torch.sqrt(var + 1e-8)

        # width = x.shape[2]

        gamma = self.gamma(c.to(self.device))
        gamma = gamma.view(-1, self.dim_in, 1)
        beta = self.beta(c.to(self.device))
        beta = beta.view(-1, self.dim_in, 1)

        h = (x - u) / std
        h = h * gamma + beta

        return h


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.conv_1 = nn.Conv1d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.cin_1 = AdaptiveInstanceNormalisation(dim_out, 128)
        #self.relu_1 = nn.GLU(dim=1)

    def forward(self, x, c_src, c_trg):
        x_ = self.conv_1(x)
        x_ = self.cin_1(x_, c_src, c_trg)
        x_ = torch.sigmoid(x_) * x_
        #x_ = self.relu_1(x_)

        return x_

class SEBlock(nn.Module):
    '''Squeeze and Excitation Block'''
    
    def __init__(self, in_dim, hid_dim):
        
        super().__init__()
        
        self.conv = nn.Conv1d(in_dim, in_dim, kernel_size = 5, stride = 1, padding = 2, bias = False)

        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.relu1 = nn.ReLU()
        
        self.linear2 = nn.Linear(hid_dim, in_dim)
    
    def forward(self, x):
        '''
            x: input, shape: [b,c,t]
        '''
        conv_out = self.conv(x)

        mean = torch.mean(conv_out, dim = 2)
        
        z = self.linear1(mean)
        z = self.relu1(z)
        
        z = self.linear2(z)
        z = torch.sigmoid(z)

        # residual
        out = x +  conv_out * z.unsqueeze(2)

        return out

class SPEncoder(nn.Module):
    '''speaker encoder for adaptive instance normalization'''
    
    def __init__(self, num_speakers = 4, out_dim = 128, num_embeddings = None):
        
        super().__init__()
        
        if num_embeddings is None:
            num_embeddings = num_speakers
        
        self.num_embeddings = num_embeddings

        self.speaker_embeddings = nn.Embedding(num_embeddings, out_dim)
        self.speaker_embeddings.weight.data.uniform_(-1./ num_embeddings, 1./num_embeddings)

        self.down_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=5, stride = 1, padding=2, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.down_sample_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.down_sample_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.down_sample_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.down_sample_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.02),
        )
        
        self.linear1 = nn.Linear(256, 128)
        
        #self.unshared = nn.ModuleList()

        #for _ in range(num_speakers):
        #    self.unshared += [nn.Linear(256, 128)]

    def forward(self,x, trg_c = None):
        
        out = self.down_sample_1(x)
        
        out = self.down_sample_2(out)

        out = self.down_sample_3(out)
        
        out = self.down_sample_4(out)
        
        out = self.down_sample_5(out)
        
        b,c,h,w = out.size()
        out = out.view(b,c,h*w)
        out = torch.mean(out, dim = 2)

        s = self.linear1(out)
        
        scores = torch.matmul(s, self.speaker_embeddings.weight.t(),) \
            / math.sqrt(s.size(1))
        
        p_attn = F.softmax(scores, dim = -1)
        
        s = torch.matmul(p_attn, self.speaker_embeddings.weight)

        #res = []
        #for layer in self.unshared:
        #    res += [layer(out)]

        #res = torch.stack(res, dim = 1)
        
        #idx = torch.LongTensor(range(b)).to(x.device)
        #s = res[idx, trg_c.long()]

        return s


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, num_speakers=4):
        super(Generator, self).__init__()
        # Down-sampling layers
        '''
        self.down_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 9), padding=(1, 4), bias=False),
            nn.GLU(dim=1)
        )
        self.down_sample_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False),
            nn.InstanceNorm2d(num_features=256, affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )
        self.down_sample_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False),
            nn.InstanceNorm2d(num_features=512, affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )
        '''
        self.down_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 9), padding=(1, 4), bias=False),
            GLU()
        )
        self.down_sample_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False),
            nn.InstanceNorm2d(num_features=256, affine=True),
            GLU()
        )
        self.down_sample_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False),
            nn.InstanceNorm2d(num_features=256, affine=True),
            GLU()
        )
        # Down-conversion layers.
        self.down_conversion = nn.Sequential(
            nn.Conv1d(in_channels=2304,
                      out_channels=256,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.InstanceNorm1d(num_features=256, affine=True)
        )

        # Bottleneck layers.
        self.residual_1 = ResidualBlock(dim_in=256, dim_out=256)
        self.residual_2 = ResidualBlock(dim_in=256, dim_out=256)
        self.residual_3 = ResidualBlock(dim_in=256, dim_out=256)
        self.residual_4 = ResidualBlock(dim_in=256, dim_out=256)
        self.residual_5 = ResidualBlock(dim_in=256, dim_out=256)
        self.residual_6 = ResidualBlock(dim_in=256, dim_out=256)
        self.residual_7 = ResidualBlock(dim_in=256, dim_out=256)
        self.residual_8 = ResidualBlock(dim_in=256, dim_out=256)
        self.residual_9 = ResidualBlock(dim_in=256, dim_out=256)

        # Up-conversion layers.
        self.up_conversion = nn.Conv1d(in_channels=256,
                                       out_channels=2304,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bias=False)

        # Up-sampling layers.
        '''
        self.up_sample_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=256, affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )
        self.up_sample_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=128, affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )
        '''
        self.up_sample_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            GLU()
        )
        self.up_sample_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            GLU()
        )
        # Out.
        self.out = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, x, c_src, c_trg):
        width_size = x.size(3)

        x = self.down_sample_1(x)
        x = self.down_sample_2(x)
        x = self.down_sample_3(x)

        x = x.contiguous().view(-1, 2304, width_size // 4)
        x = self.down_conversion(x)

        x = self.residual_1(x, c_src, c_trg)
        x = self.residual_2(x, c_src, c_trg)
        x = self.residual_3(x, c_src, c_trg)
        x = self.residual_4(x, c_src, c_trg)
        x = self.residual_5(x, c_src, c_trg)
        x = self.residual_6(x, c_src, c_trg)
        x = self.residual_7(x, c_src, c_trg)
        x = self.residual_8(x, c_src, c_trg)
        x = self.residual_9(x, c_src, c_trg)

        x = self.up_conversion(x)
        x = x.view(-1, 256, 9, width_size // 4)

        x = self.up_sample_1(x)
        x = self.up_sample_2(x)
        x = self.out(x)

        return x


class Discriminator(nn.Module):
    """Discriminator network."""
    def __init__(self, num_speakers=10):
        super(Discriminator, self).__init__()

        self.num_speakers = num_speakers
        # Initial layers.
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            GLU()
        )
        #self.conv1 = nn.Conv2d(1, 128, kernel_size= (3,3), stride = 1, padding= 1)
        #self.gate1 = nn.Conv2d(1, 128, kernel_size = 3, stride = 1, padding = 1)

        # Down-sampling layers.
        self.down_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.InstanceNorm2d(256, affine = True),
            GLU()
        )
        #self.down_sample_1 = DisDown(128, 256, kernel_size = 3, stride = 2, padding = 1)

        self.down_sample_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.InstanceNorm2d(512, affine = True),
            GLU()
        )
        #self.down_sample_2 = DisDown(256, 512, kernel_size = 3, stride = 2, padding = 1)
        
        self.down_sample_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.InstanceNorm2d(1024, affine = True),
            GLU()
        )
        #self.down_sample_3 = DisDown(512, 1024, kernel_size = 3, stride = 2, padding = 1)
        self.down_sample_4 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), bias=False),
            GLU()
        )
        #self.down_sample_4 = DisDown(1024, 512, kernel_size = (1,5), stride = 1, padding = (0,2))
        # Fully connected layer.
        self.fully_connected = nn.Linear(in_features=512, out_features=num_speakers)
        #self.fully_connected = nn.Linear(in_features=512, out_features=512)

        # Projection.
        #self.projection = nn.Linear(self.num_speakers, 512)
        #self.projection_trg = nn.Linear(128, 512)
        #self.projection_src = nn.Linear(128, 512)
        #self.projection = nn.Linear(256, 512)

    def forward(self, x, c, c_):
        #c_onehot = torch.cat((c, c_), dim=1)
        #c_onehot = c_

        x = self.conv_layer_1(x)
        #x_conv = self.conv1(x)
        #x_gate = self.gate1(x)
        #out = x_conv * torch.sigmoid(x_gate)

        x = self.down_sample_1(x)
        x = self.down_sample_2(x)
        x = self.down_sample_3(x)
        x = self.down_sample_4(x)
        
        b, c, h, w = x.size()
        x = x.view(b,c, h*w)
        x = torch.mean(x, dim = 2)
        x = self.fully_connected(x)
        #x = torch.sigmoid(x)

        idx = torch.LongTensor(range(x.size(0))).to(x.device)

        x = x[idx, c_.long()]

        return x

class PatchDiscriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, num_speakers=10):
        super(PatchDiscriminator, self).__init__()

        self.num_speakers = num_speakers
        # Initial layers.
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            GLU()
        )
        #self.conv1 = nn.Conv2d(1, 128, kernel_size= (3,3), stride = 1, padding= 1)
        #self.gate1 = nn.Conv2d(1, 128, kernel_size = 3, stride = 1, padding = 1)

        # Down-sampling layers.
        self.down_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.InstanceNorm2d(256, affine = True),
            GLU()
        )
        #self.down_sample_1 = DisDown(128, 256, kernel_size = 3, stride = 2, padding = 1)

        self.down_sample_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.InstanceNorm2d(512, affine = True),
            GLU()
        )
        #self.down_sample_2 = DisDown(256, 512, kernel_size = 3, stride = 2, padding = 1)
        
        self.down_sample_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.InstanceNorm2d(1024, affine = True),
            GLU()
        )
        #self.down_sample_3 = DisDown(512, 1024, kernel_size = 3, stride = 2, padding = 1)
        self.down_sample_4 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), bias=False),
            GLU()
        )
        
        self.dis_conv = nn.Conv2d(512, num_speakers, kernel_size = 1, stride = 1, padding = 0, bias = False )

    def forward(self, x, c_src, c_):
        #c_onehot = torch.cat((c, c_), dim=1)
        #c_onehot = c_

        x = self.conv_layer_1(x)
        #x_conv = self.conv1(x)
        #x_gate = self.gate1(x)
        #out = x_conv * torch.sigmoid(x_gate)

        x = self.down_sample_1(x)
        x = self.down_sample_2(x)
        x = self.down_sample_3(x)
        x = self.down_sample_4(x)
        
        x = self.dis_conv(x)

        b, c, h, w = x.size()
        x = x.view(b,c, h*w)
        x = torch.mean(x, dim = 2)

        idx = torch.LongTensor(range(x.size(0))).to(x.device)

        x = x[idx, c_.long()]

        return x


