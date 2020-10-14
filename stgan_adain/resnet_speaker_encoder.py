#https://github.com/zhilangtaosha/SpeakerVerification_AMSoftmax_pytorch/blob/master/model/model.py


import torch
import torch.nn as nn
import math
from torch.nn import Parameter
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F

class ReLU(nn.Hardtanh):
    
    def __init__(self, inplace = False):
        
        super().__init__()

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + '(' + inplace_str + ')'

def conv3x3(in_planes, out_planes, stride = 1):
    
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride, padding = 1, bias = False)

class BasicBlock(nn.Module):
    expansion =1 
    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        
        super().__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = ReLU(inplace = True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        
        return out


class SEBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        super().__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.globalAvgPool = nn.AdaptiveAvgPool2d([1,1])
        self.fc1 = nn.Linear(in_features = planes, out_features = int(round(planes/16)))
        self.fc2 = nn.Linear(in_features = int(round(planes / 16)), out_features = planes)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        
        residual = x

        out = self.conv1d(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        original_out = out

        out = self.globalAvgPool(out)
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = out * original_out

        out += residual
        out = self.relu(out)
        return out



class MyResNet(nn.Module):
    
    def __init__(self, block, layers, num_classes, ratio = 1.0):
        
        super().__init__()

        self.relu = ReLU(inplace = True)
        
        self.inplanes = int(32 * ratio)
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size = 5, stride = 2, padding = 2, bias = False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.layer1 = self._make_layer(block, self.inplanes, layers[0])


        self.inplanes = int(64 * ratio)
        self.conv2 = nn.Conv2d(int(32*ratio), self.inplanes, kernel_size = 5, stride = 2, padding = 2, bias = False)
        self.bn2 = nn.BatchNorm2d(self.inplanes)
        self.layer2 = self._make_layer(block, self.inplanes, layers[1])

        self.inplanes = int(128 * ratio)
        self.conv3 = nn.Conv2d(int(64*ratio), self.inplanes, kernel_size = 5, stride = 2, padding = 2, bias = False)
        self.bn3 = nn.BatchNorm2d(self.inplanes)
        self.layer3 = self._make_layer(block, self.inplanes, layers[2])

        self.inplanes = int(256 * ratio)
        self.conv4 = nn.Conv2d(int(128*ratio), self.inplanes, kernel_size = 5, stride = 2, padding = 2, bias = False)
        self.bn4 = nn.BatchNorm2d(self.inplanes)
        self.layer4 = self._make_layer(block, self.inplanes, layers[3])


        self.avgpool = nn.AdaptiveAvgPool2d([4,1])
        self.fc = nn.Linear(self.inplanes * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride = 1):
        
        layers = []

        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        


class ResSPEncoder(nn.Module):

    def __init__(self, num_speakers, embedding_size = 128, ratio = 1.0, is_dropout = False, is_SENet = False, is_fn = False, spk_cls = False):
        
        super().__init__()

        self.is_fn = is_fn
        self.embedding_size = embedding_size
        self.spk_cls = spk_cls
        self.is_dropout = is_dropout
        if is_SENet:
            self.model = MyResNet(SEBasicBlock, [1,1,1,1], num_classes = num_speakers, ratio = ratio)
        else:
            self.model = MyResNet(BasicBlock, [1,1,1,1], num_classes = num_speakers, ratio = ratio)

        # feature norm
        if is_fn:
            self.model.fc = nn.Linear(int(256*ratio*4), embedding_size, bias = False)
            self.fn = nn.BatchNorm2d(embedding_size, affine = False)
        else:
            self.model.fc = nn.Linear(int(256*ratio*4), embedding_size)
        

        if is_dropout:
            self.dp = nn.Dropout(0.5)

        self.model.classifier = nn.Linear(self.embedding_size, num_speakers)

    def l2_norm(self, input):
        
        input_size = input.size()

        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)

        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1,1).expand_as(input))

        output = _output.view(input_size)

        return output
    def forward(self, x, c = None, cls_out = False):
        
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.layer1(x)


        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.relu(x)
        x = self.model.layer2(x)
        
        
        x = self.model.conv3(x)
        x = self.model.bn3(x)
        x = self.model.relu(x)
        x = self.model.layer3(x)

        
        x = self.model.conv4(x)
        x = self.model.bn4(x)
        x = self.model.relu(x)
        x = self.model.layer4(x)


        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.is_dropout:
            x = self.dp(x)

        x = self.model.fc(x)
        if self.is_fn:
            x = self.fn(x)

        features = self.l2_norm(x)
        
        x = self.model.classifier(features)
        if self.spk_cls and cls_out :
            return features, x
        else:
            return features
            
