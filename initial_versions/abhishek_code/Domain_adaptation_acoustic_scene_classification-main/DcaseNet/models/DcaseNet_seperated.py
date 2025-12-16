import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utilities import ConvBlock, init_gru, init_layer, interpolate

def get_DcaseNet_v3(**args):
    return DcaseNet_v3_FE(block = SEBasicBlock, **args)

class DcaseNet_v3_FE(nn.Module):
    def __init__(self, 
            block, 
            filts_ASC=256, 
            blocks_ASC=3, 
            strides_ASC=2, 
            pool_type='avg', 
            pool_size=(2,2)):
        super().__init__()
        
        self.pool_type = pool_type
        self.pool_size = pool_size

        # --- CNN Backbone ---
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4_1 = ConvBlock(in_channels=256, out_channels=256) # Common branch
        self.conv_block4_2 = ConvBlock(in_channels=256, out_channels=256) # Task-specific branch

        # --- Recurrent Layers ---
        self.gru_1 = nn.GRU(input_size=512, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.gru_2 = nn.GRU(input_size=512, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)

        # --- ASC Specific Feature Layers ---
        self.inplane = 256
        self.layer_ASC = self._make_layer(
            block=block,
            planes=384,
            blocks=blocks_ASC,
            stride=strides_ASC,
            reduction=16
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, block, planes, blocks, stride, reduction):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.inplane, planes, s, reduction))
            self.inplane = planes
        return nn.Sequential(*layers)

    def forward(self, x, mode=''):
        features = {}

        # --- CNN Backbone ---
        x = self.conv_block1(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block2(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block3(x, self.pool_type, pool_size=self.pool_size)
        x_1 = self.conv_block4_1(x, self.pool_type, pool_size=(2, 5))
        x_2 = self.conv_block4_2(x, self.pool_type, pool_size=(2, 5))

        # --- ASC Features ---
        if 'ASC' in mode:
            # These are the features before avg pooling
            out_ASC = self.layer_ASC(x_2)
            out_ASC = self.avgpool(out_ASC).view(x.size(0), -1)
            features['ASC'] = out_ASC

        return features

class Classifier(nn.Module):
    """Simplified Classifier"""
    def __init__(self, input_size=384, num_classes=10):
        super(Classifier, self).__init__()
        self.layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(input_size, 128),
            nn.Linear(128, num_classes),
        )

    def forward(self, h):
        return self.layer(h)

#####
# ASC
#####
def conv3x3(in_planes, out_planes, stride=1):
    #changed from 1d to 2d
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out += residual
        out = self.relu(out)

        return out





if __name__ == '__main__':
    pass