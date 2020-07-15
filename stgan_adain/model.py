import torch
import torch.nn as nn
import numpy as np
import argparse
from data_loader import get_loader, to_categorical
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
    
    def __init__(self, num_speakers = 4):
        
        super().__init__()
        
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
        
        #self.linear1 = nn.Linear(256, 128)
        
        self.unshared = nn.ModuleList()

        for _ in range(num_speakers):
            self.unshared += [nn.Linear(256, 128)]

    def forward(self,x, trg_c):
        
        out = self.down_sample_1(x)
        
        out = self.down_sample_2(out)

        out = self.down_sample_3(out)
        
        out = self.down_sample_4(out)
        
        out = self.down_sample_5(out)
        
        b,c,h,w = out.size()
        out = out.view(b,c,h*w)
        out = torch.mean(out, dim = 2)

        #out = self.linear1(out)
        res = []
        for layer in self.unshared:
            res += [layer(out)]

        res = torch.stack(res, dim = 1)
        
        idx = torch.LongTensor(range(b)).to(x.device)
        s = res[idx, trg_c.long()]

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

class PatchCondDiscriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, input_size=(36, 256), conv_dim=128, repeat_num=5, num_speakers=10):
        
        super(PatchCondDiscriminator, self).__init__()
        layers = []
        #layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        #layers.append(nn.LeakyReLU(0.01))
        self.first_conv = nn.Conv2d(1, conv_dim, kernel_size = 4, stride = 2, padding = 1)
        self.first_relu = nn.LeakyReLU(0.01)

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size_0 = int(input_size[0] / np.power(2, repeat_num)) # 1
        kernel_size_1 = int(input_size[1] / np.power(2, repeat_num)) # 8
        self.main = nn.Sequential(*layers)
        self.conv_dis = nn.Conv2d(curr_dim, 1, kernel_size=(kernel_size_0, kernel_size_1), stride=1, padding=0, bias=False) # padding should be 0
        #self.conv_clf_spks = nn.Conv2d(curr_dim, num_speakers, kernel_size=(kernel_size_0, kernel_size_1), stride=1, padding=0, bias=False)  # for num_speaker
        #self.projection = nn.Linear(2*128, 1024)
        #self.projection = nn.Linear(128, 64)
        
    def forward(self, x, c_src, c_trg):
        '''
            x: shape [b, 1, 36, 256]
            c: shape [b, 128]
        '''
        


        #c_onehot = torch.cat((c_src, c_trg), dim=1)
        c_cond = c_trg
        
        h = self.first_conv(x)
        h = self.first_relu(h)
        
        # concat
        
        h = h + c_cond.unsqueeze(2).unsqueeze(3)
        

        h = self.main(h) #b 1024 h w
        #p = self.projection(c_onehot) # b 1024
        #assert p.size(1) == h.size(1), f"p {p.size()} h {h.size()}"
        #h = h * p.unsqueeze(2).unsqueeze(3)
        out_src = self.conv_dis(h)
        return out_src
class PatchDiscriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, input_size=(36, 256), conv_dim=128, repeat_num=4, num_speakers=10):
        
        super(PatchDiscriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size_0 = int(input_size[0] / np.power(2, repeat_num)) # 1
        kernel_size_1 = int(input_size[1] / np.power(2, repeat_num)) # 8
        self.main = nn.Sequential(*layers)
        self.conv_dis = nn.Conv2d(curr_dim, 1, kernel_size=(kernel_size_0, kernel_size_1), stride=1, padding=0, bias=False) # padding should be 0
        #self.conv_clf_spks = nn.Conv2d(curr_dim, num_speakers, kernel_size=(kernel_size_0, kernel_size_1), stride=1, padding=0, bias=False)  # for num_speaker
        #self.projection = nn.Linear(2*128, 1024)
        #self.projection = nn.Linear(128, curr_dim)
        
    def forward(self, x, c_src, c_trg):
        
        #c_onehot = torch.cat((c_src, c_trg), dim=1)
        #c_onehot = c_trg
        h = self.main(x) #b 1024 h w
        #p = self.projection(c_onehot) # b 1024
        #assert p.size(1) == h.size(1), f"p {p.size()} h {h.size()}"
        #h = h + p.unsqueeze(2).unsqueeze(3)
        out_src = self.conv_dis(h)
        return out_src


# Just for testing shapes of architecture.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test G and D architecture')

    train_dir_default = '../data/VCTK-Data/mc/train'
    speaker_default = 'p229'

    # Data config.
    parser.add_argument('--train_dir', type=str, default=train_dir_default, help='Train dir path')
    parser.add_argument('--speakers', type=str, nargs='+', required=True, help='Speaker dir names')
    num_speakers = 4

    argv = parser.parse_args()
    train_dir = argv.train_dir
    speakers_using = argv.speakers

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    generator = Generator(num_speakers=num_speakers).to(device)
    discriminator = Discriminator(num_speakers=num_speakers).to(device)

    # Load data
    train_loader = get_loader(speakers_using, train_dir, 8, 'train', num_workers=1)
    data_iter = iter(train_loader)

    mc_real, spk_label_org, spk_c_org = next(data_iter)
    mc_real.unsqueeze_(1)  # (B, D, T) -> (B, 1, D, T) for conv2d

    spk_c = np.random.randint(0, num_speakers, size=mc_real.size(0))
    spk_c_cat = to_categorical(spk_c, num_speakers)
    spk_label_trg = torch.LongTensor(spk_c)
    spk_c_trg = torch.FloatTensor(spk_c_cat)

    mc_real = mc_real.to(device)              # Input mc.
    spk_label_org = spk_label_org.to(device)  # Original spk labels.
    spk_c_org = spk_c_org.to(device)          # Original spk acc conditioning.
    spk_label_trg = spk_label_trg.to(device)  # Target spk labels for classification loss for G.
    spk_c_trg = spk_c_trg.to(device)          # Target spk conditioning.

    print('------------------------')
    print('Testing Discriminator')
    print('-------------------------')
    print(f'Shape in: {mc_real.shape}')
    dis_real = discriminator(mc_real, spk_c_org, spk_c_trg)
    print(f'Shape out: {dis_real.shape}')
    print('------------------------')

    print('Testing Generator')
    print('-------------------------')
    print(f'Shape in: {mc_real.shape}')
    mc_fake = generator(mc_real, spk_c_trg)
    print(f'Shape out: {mc_fake.shape}')
    print('------------------------')
