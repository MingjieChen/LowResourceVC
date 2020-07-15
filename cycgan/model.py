import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PixelShuffle(nn.Module):
    def __init__(self,):
        super().__init__()
    def forward(self, x):
        b, c, t = x.size()
        new_c, new_t = c // 2, t * 2
        
        return x.view(b, new_c, new_t)

def pixel_shuffle(x, size=2):
    
    b, c, t = x.size()

    new_c, new_t = c // size, t * size
    
    return x.view(b, new_c, new_t) 

class GLU(nn.Module):
    ''' GLU block, do not split channels dimension'''

    def __init__(self,):
        super().__init__()

    def forward(self, x):
        
        return x * torch.sigmoid(x)



class Upsample1DBlock(nn.Module):
    
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding):
        
        super().__init__()

        self.conv = nn.Conv1d(dim_in, dim_out, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
        self.inst_norm = nn.InstanceNorm1d(dim_out //2, affine = True)
        self.glu = GLU()

        '''
        self.conv = nn.Sequential(
                nn.Conv1d(dim_in, dim_out, kernel_size = kernel_size, stride = stride, padding = padding),
                PixelShuffle(),
                nn.InstanceNorm1d(dim_out //2, affine = True)
        )
        self.gates = nn.Sequential(
                nn.Conv1d(dim_in, dim_out, kernel_size = kernel_size, stride = stride, padding = padding),
                PixelShuffle(),
                nn.InstanceNorm1d(dim_out //2, affine = True)
        )
        '''
    def forward(self, x):
        
        out = self.conv(x)
        out = pixel_shuffle(out)
        out = self.inst_norm(out)
        out = self.glu(out)
        
        #out = self.conv(x) * torch.sigmoid(self.gates(x))


        return out  


class Downsample1DBlock(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size, stride, padding):
        
        super().__init__()

        self.conv1 = nn.Conv1d(dim_in, dim_out, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
        self.inst_norm = nn.InstanceNorm1d(dim_out, affine = True)
        self.glu = GLU()
        '''
        self.conv = nn.Sequential(
            nn.Conv1d(dim_in, dim_out, kernel_size = kernel_size, stride = stride, padding = padding),
            nn.InstanceNorm1d(dim_out, affine = True)
        )
        self.gates = nn.Sequential(
            nn.Conv1d(dim_in, dim_out, kernel_size = kernel_size, stride = stride, padding = padding),
            nn.InstanceNorm1d(dim_out, affine = True)
        )
        '''
    def forward(self, x):
        out = self.conv1(x)
        out = self.inst_norm(out)
        out = self.glu(out)

        #out = self.conv(x) * torch.sigmoid(self.gates(x))
        return out

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.inst_norm1 = nn.InstanceNorm1d(dim_out, affine = True)
        self.glu = GLU()

        self.conv2 = nn.Conv1d(dim_out, dim_in, kernel_size = 3, stride = 1, padding= 1, bias = False)
        self.inst_norm2 = nn.InstanceNorm1d(dim_in, affine = True)
        '''
        self.conv = nn.Sequential(
            nn.Conv1d(dim_in, dim_out, kernel_size = 3, stride = 1, padding = 1),
            nn.InstanceNorm1d(dim_out, affine = True)
        )
        self.gates = nn.Sequential(
            nn.Conv1d(dim_in, dim_out, kernel_size = 3, stride = 1, padding = 1),
            nn.InstanceNorm1d(dim_out, affine = True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv1d(dim_out, dim_in, kernel_size = 3, stride = 1, padding = 1),
            nn.InstanceNorm1d(dim_in, affine = True)
        )
        '''

    def forward(self, x):
        x_ = self.conv1(x)
        x_ = self.inst_norm1(x_)
        x_ = self.conv2(x_)
        x_ = self.inst_norm2(x_)
        x_ = self.glu(x_)
        '''
        h1_norm = self.conv(x)
        h1_gate = self.gates(x)

        h1_glu = h1_norm * torch.sigmoid(h1_gate)

        h2_norm = self.out_conv(h1_glu)

        return x + h2_norm
        '''    
        return x_ + x


class Generator(nn.Module):

    def __init__(self, input_dim = 36, conv_dim = 64, num_speakers = 4, repeat_num = 6):
        super().__init__()
        
        layers = []

        layers.append(nn.Conv1d(input_dim, 128, kernel_size = 15, stride = 1, padding = 7))
        layers.append(GLU())
        '''
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size = 15, stride = 1, padding = 7)
        self.gates1 = nn.Conv1d(input_dim, 128, kernel_size = 15, stride = 1, padding = 7)
        '''
        # Downsample

        layers.append(Downsample1DBlock(128, 256, 5, 2, 1))
        layers.append(Downsample1DBlock(256, 512, 5, 2, 2))

        # Bottleneck layers

        for _ in range(repeat_num):
            layers.append(ResidualBlock(512, 1024))
        
        # Upsample
        
        layers.append(Upsample1DBlock(512, 1024, 5, 1, 2))
        layers.append(Upsample1DBlock(512, 256, 5, 1, 2))
        
        layers.append(nn.Conv1d(128, input_dim, kernel_size = 15, stride = 1, padding = 7))
        
        self.main = nn.Sequential(*layers)
    def forward(self, x):
        
        #out = self.conv1(x) * torch.sigmoid(self.gates1(x))
        out = self.main(x)
        return out

class DisDownsample(nn.Module):
    
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding):
        
        super().__init__()

        self.conv = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size = kernel_size, stride = stride, padding = padding),
                nn.InstanceNorm2d(dim_out, affine = True)
        )
        self.gates = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size = kernel_size, stride = stride, padding = padding),
                nn.InstanceNorm2d(dim_out, affine = True)
        )

    def forward(self,x):
        
        return self.conv(x) * torch.sigmoid(self.gates(x))



class Discriminator(nn.Module):
    def __init__(self, input_size=(36, 256), conv_dim=128, repeat_num=4):
        super(Discriminator, self).__init__()
        layers = []
        
        layers.append(nn.Conv2d(1,128, kernel_size = [3,3], stride = [2,1], padding = 1))
        layers.append(GLU())
        #self.conv1 = nn.Conv2d(1, 128, kernel_size = [3,3], stride = [1,2], padding = [1,1])
        #self.gates1 = nn.Conv2d(1, 128, kernel_size = [3,3], stride = [1,2], padding = [1,1])

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim , curr_dim*2, kernel_size=3, stride=2, padding=1))
            layers.append(nn.InstanceNorm2d(curr_dim *2, affine = True))
            layers.append(GLU())
            curr_dim = curr_dim * 2
        #self.down1 = DisDownsample(128, 256, kernel_size = [3,3], stride = 2, padding = [1,1])
        #self.down2 = DisDownsample(256, 512, kernel_size = [3,3], stride = 2, padding = [1,1])
        #self.down3 = DisDownsample(512, 1024, kernel_size = [6,3], stride = [1,2], padding = [3,2])
        #self.conv_dis = nn.Conv2d(curr_dim, 1, kernel_size=[2,4],  stride=1, padding=0, bias=False)

        self.main = nn.Sequential(*layers)
        self.out_linear = nn.Linear(curr_dim, 1)
        
    def forward(self, x):
        x = x.unsqueeze(1) # b 1 36 256
        h = self.main(x) # b 1024 1 8
        
        #layer1 = self.conv1(x) * torch.sigmoid(self.gates1(x))
        
        #out = self.down1(layer1)
        #out = self.down2(out)
        #out = self.down3(out)
        
        #out_src = self.out_linear(out.permute(0,2,3,1))
        #out_src = self.conv_dis(h)
        out_src = self.out_linear(h.permute(0,2,3,1))
        #out_src = torch.sigmoid(out_src)
        #out_cls_spks = self.conv_clf_spks(h)
        return out_src #out_cls_spks.view(out_cls_spks.size(0), out_cls_spks.size(1))
        
        
