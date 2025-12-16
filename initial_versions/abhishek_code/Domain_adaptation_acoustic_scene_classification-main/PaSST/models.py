# models.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hear21passt.base import get_basic_model


class FeatureExtractor(nn.Module):
    """Feature Extractor using PaSST."""
    def __init__(self, device):
        super(FeatureExtractor, self).__init__()
        if get_basic_model is None:
            raise ImportError("hear21passt is required for FeatureExtractor.")
        self.model = get_basic_model(mode="embed_only")
        self.model.to(device)

    def forward(self, x):
        return self.model(x)

class Classifier(nn.Module):
    """Classifier head."""
    def __init__(self, input_size=768, num_classes=10):
        super(Classifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512 , 256 ),
            nn.ReLU(inplace=True),
            nn.Linear(256 , 128 ),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.layer(x)

class AdversarialNetwork(nn.Module):
    """Simplified Discriminator"""
    def __init__(self, input_size=768 , num_classes=1):
        super(AdversarialNetwork, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512 , 256 ),
            nn.ReLU(inplace=True),
            nn.Linear(256 , 128 ),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, h):
        return self.layer(h)

class Discriminator_bridge(nn.Module):
    def __init__(self, num_classes=1, flattened_size=768):
        super(Discriminator_bridge, self).__init__()
        hidden_size = 100
        self.fc1 = nn.Linear(flattened_size, hidden_size)
        self.dropout3 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.leakyRelu = nn.LeakyReLU(0.2, inplace=True)
        self.sig = nn.Sigmoid()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.leakyRelu(x)
        x = self.dropout3(x)
        y = self.fc3(x)
        x = self.fc2(x)
        # x = self.sig(x)
        return x,y

class Wrapper(nn.Module):
    def __init__(self , fe , g ):
        super(Wrapper , self).__init__()
        self.fe = fe
        self.g = g
    def forward(self, x):
        x = self.fe(x)
        x = self.g(x)
        return x

class Bridge(nn.Module):
    """ The GVB Bridge Network """
    def __init__(self, input_dim=768, output_dim=10):
        super(Bridge, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, output_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)
    
def GVBLoss(bridge_out, bridge_target=0.5):
    """ The GVB Loss Function """
    target = torch.tensor([bridge_target]).cuda().repeat(bridge_out.size(0), 1)
    return -(target * torch.log(bridge_out) + (1.0 - target) * torch.log(1.0 - bridge_out))
# --- A new, combined model for GVB integration ---
class passt_GVB(nn.Module):
    def __init__(self):
        super(passt_GVB, self).__init__()
        self.classifier = Classifier()
        self.bridge = Bridge()

    def forward(self, x):
        class_output = self.classifier(x)
        bridge_output = self.bridge(x)
        return  class_output, bridge_output
    
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float64(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)