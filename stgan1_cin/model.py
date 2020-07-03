import torch
import torch.nn as nn
import numpy as np



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
    def __init__(self, dim_in, dim_out, style_num):
        super(ResidualBlock, self).__init__()
        self.conv_1 = nn.Conv1d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.cin_1 = ConditionalInstanceNormalisation(dim_out, style_num)
        self.relu_1 = nn.GLU(dim=1)

    def forward(self, x, c):
        x_ = self.conv_1(x)
        x_ = self.cin_1(x_, c)
        x_ = self.relu_1(x_)

        return x_


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, num_speakers=4):
        super(Generator, self).__init__()
        # Down-sampling layers
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
        self.residual_1 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_2 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_3 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_4 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_5 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_6 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_7 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_8 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_9 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)

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
            nn.InstanceNorm2d(num_features=256, affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )
        self.up_sample_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=128, affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )

        # Out.
        self.out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, x, c):
        width_size = x.size(3)

        x = self.down_sample_1(x)
        x = self.down_sample_2(x)
        x = self.down_sample_3(x)

        x = x.contiguous().view(-1, 2304, width_size // 4)
        x = self.down_conversion(x)

        x = self.residual_1(x, c)
        x = self.residual_2(x, c)
        x = self.residual_3(x, c)
        x = self.residual_4(x, c)
        x = self.residual_5(x, c)
        x = self.residual_6(x, c)
        x = self.residual_7(x, c)
        x = self.residual_8(x, c)
        x = self.residual_9(x, c)

        x = self.up_conversion(x)
        x = x.view(-1, 256, 9, width_size // 4)

        x = self.up_sample_1(x)
        x = self.up_sample_2(x)
        x = self.out(x)

        return x


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, input_size=(36, 256), conv_dim=64, repeat_num=5, num_speakers=10):
        super(Discriminator, self).__init__()
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
        self.conv_clf_spks = nn.Conv2d(curr_dim, num_speakers, kernel_size=(kernel_size_0, kernel_size_1), stride=1, padding=0, bias=False)  # for num_speaker
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv_dis(h)
        out_cls_spks = self.conv_clf_spks(h)
        return out_src, out_cls_spks.view(out_cls_spks.size(0), out_cls_spks.size(1))
