import torch
import torch.nn as nn
import numpy as np
import argparse
from data_loader import get_loader, to_categorical


class ConditionalInstanceNormalisation(nn.Module):
    """CIN Block."""
    def __init__(self, dim_in, style_num):
        super(ConditionalInstanceNormalisation, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.dim_in = dim_in
        self.style_num = style_num
        self.gamma = nn.Linear(2*style_num, dim_in)
        self.beta = nn.Linear(2*style_num, dim_in)
        #self.gamma = nn.Linear(style_num, dim_in)
        #self.beta = nn.Linear(style_num, dim_in)

    def forward(self, x, c_src, c_trg):
        u = torch.mean(x, dim=2, keepdim=True)
        var = torch.mean((x - u) * (x - u), dim=2, keepdim=True)
        std = torch.sqrt(var + 1e-8)

        # width = x.shape[2]
        c = torch.cat([c_src, c_trg], dim = -1)
        #c = c_trg
        gamma = self.gamma(c.to(self.device))
        gamma = gamma.view(-1, self.dim_in, 1)
        beta = self.beta(c.to(self.device))
        beta = beta.view(-1, self.dim_in, 1)

        h = (x - u) / std
        h = h * gamma + beta

        return h
class GLU(nn.Module):
    ''' GLU block, do not split channels dimension'''

    def __init__(self,):
        super().__init__()

    def forward(self, x):
        
        return x * torch.sigmoid(x)

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out, style_num):
        super(ResidualBlock, self).__init__()
        self.conv_1 = nn.Conv1d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.cin_1 = ConditionalInstanceNormalisation(dim_out, style_num)
        #self.relu_1 = nn.GLU(dim=1)
        self.relu_1 = GLU()

    def forward(self, x, c_src, c_trg):
        x_ = self.conv_1(x)
        x_ = self.cin_1(x_, c_src, c_trg)
        x_ = self.relu_1(x_)

        return x_


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, num_speakers=4):
        super(Generator, self).__init__()
        # Down-sampling layers
        self.down_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 9), padding=(1, 4), bias=False),
            GLU()
        )
        self.down_sample_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False),
            nn.InstanceNorm2d(num_features=256, affine=True),
            GLU()
            #nn.GLU(dim=1)
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
        self.residual_1 = ResidualBlock(dim_in=256, dim_out=256, style_num=num_speakers)
        self.residual_2 = ResidualBlock(dim_in=256, dim_out=256, style_num=num_speakers)
        self.residual_3 = ResidualBlock(dim_in=256, dim_out=256, style_num=num_speakers)
        self.residual_4 = ResidualBlock(dim_in=256, dim_out=256, style_num=num_speakers)
        self.residual_5 = ResidualBlock(dim_in=256, dim_out=256, style_num=num_speakers)
        self.residual_6 = ResidualBlock(dim_in=256, dim_out=256, style_num=num_speakers)
        self.residual_7 = ResidualBlock(dim_in=256, dim_out=256, style_num=num_speakers)
        self.residual_8 = ResidualBlock(dim_in=256, dim_out=256, style_num=num_speakers)
        self.residual_9 = ResidualBlock(dim_in=256, dim_out=256, style_num=num_speakers)

        # Up-conversion layers.
        self.up_conversion = nn.Conv1d(in_channels=256,
                                       out_channels=2304,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bias=False)

        # Up-sampling layers.
        self.up_sample_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            GLU()
        )
        self.up_sample_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            GLU()
        )

        # Out.
        self.out = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=(5,15), stride=1, padding=(2,7), bias=False)

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

class PixelShuffle2d(nn.Module):
    '''pixel shuffle for up samplinga'''
    def __init__(self, factor = 2):
        super().__init__()

        self.factor = factor
    def forward(self, x):
        
        b, c, h, w = x.size()

        new_c, new_h, new_w = c // (2 *self.factor), h * self.factor, w * self.factor

        return x.view(b, new_c, new_h, new_w)

class Discriminator(nn.Module):
    """Discriminator network."""
    def __init__(self, num_speakers=10):
        super(Discriminator, self).__init__()

        self.num_speakers = num_speakers
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initial layers.
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            GLU()
        )

        # Down-sampling layers.
        self.down_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.InstanceNorm2d(num_features=256, affine=True),
            GLU()
        )
        self.down_sample_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.InstanceNorm2d(num_features=512, affine=True),
            GLU()
        )
        self.down_sample_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.InstanceNorm2d(num_features=1024, affine=True),
            GLU()
        )
        self.down_sample_4 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), bias=False),
            nn.InstanceNorm2d(num_features=512, affine=True),
            GLU()
        )

        # Fully connected layer.
        self.fully_connected = nn.Linear(in_features=512, out_features=1)
        #self.fully_connected = nn.Linear(in_features=512, out_features=512)

        # Projection.
        #self.projection = nn.Linear(self.num_speakers, 512)
        self.projection = nn.Linear(2*self.num_speakers, 512)

    def forward(self, x, c, c_):
        c_onehot = torch.cat((c, c_), dim=1).to(self.device)
        #c_onehot = c_

        x = self.conv_layer_1(x)

        x = self.down_sample_1(x)
        x = self.down_sample_2(x)
        x = self.down_sample_3(x)
        x_ = self.down_sample_4(x)

        h = torch.sum(x_, dim=(2, 3)) # b 512

        #x = self.fully_connected(h)
        x = self.fully_connected(x_.permute(0,2,3,1)).permute(0,3,1,2) # b 1 h w

        p = self.projection(c_onehot) #b 512

        in_prod = p * h
        
        x = x.view(x.size(0), -1)
        x = torch.cat([x ,in_prod], dim = -1)
        
        '''
        # old dis
        x = self.fully_connected(h)

        p = self.projection(c_onehot)

        x += torch.sum(p * h, dim=1, keepdim=True)
        '''
        x = torch.sigmoid(x)
        return x

class PatchDiscriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, input_size=(36, 256), conv_dim=64, repeat_num=5, num_speakers=10):
        
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
        self.projection = nn.Linear(2*num_speakers, 1024)
        #self.projection = nn.Linear(num_speakers, 1024)
        
    def forward(self, x, c_src, c_trg):
        
        c_onehot = torch.cat((c_src, c_trg), dim=1)
        #c_onehot = c_trg
        h = self.main(x) #b 1024 h w
        p = self.projection(c_onehot) # b 1024
        assert p.size(1) == h.size(1), f"p {p.size()} h {h.size()}"
        #h = h * p.unsqueeze(2).unsqueeze(3)
        h = h + p.unsqueeze(2).unsqueeze(3)
        out_src = self.conv_dis(h)
        out_src = torch.sigmoid(out_src)
        #out_cls_spks = self.conv_clf_spks(h)
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
