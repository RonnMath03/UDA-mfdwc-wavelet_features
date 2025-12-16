# models.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Feature_extractor(nn.Module):
    def __init__(self):
        super(Feature_extractor, self).__init__()

        self.convolution_kernel_size = 7
        self.convolution_dropout = 0.3

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=self.convolution_kernel_size, padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(5, 5))
        self.dropout1 = nn.Dropout2d(p=self.convolution_dropout)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.convolution_kernel_size, padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 100))
        self.dropout2 = nn.Dropout2d(p=self.convolution_dropout)
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = x.view(x.size(0), -1)
        return x

class Classifier(nn.Module):
    def __init__(self, num_classes=10, feature_vector_length=128, input_sequence_length=500):
        super(Classifier, self).__init__()

        self.convolution_kernel_size = 7
        self.convolution_dropout = 0.3

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=self.convolution_kernel_size, padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(5, 5))
        self.dropout1 = nn.Dropout2d(p=self.convolution_dropout)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.convolution_kernel_size, padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 100))
        self.dropout2 = nn.Dropout2d(p=self.convolution_dropout)

        self.flattened_size = self._calculate_flattened_size(feature_vector_length, input_sequence_length)

        self.fc1 = nn.Linear(self.flattened_size, 100)
        self.dropout3 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(100, num_classes)
        self._initialize_weights()

    def _calculate_flattened_size(self, feature_length, sequence_length):
        x = torch.randn(1, 1, feature_length, sequence_length)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        return x.view(1, -1).size(1)

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
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

class Classifier_no_weights(nn.Module):
    def __init__(self, num_classes=10 ,flattened_size = 384 ):
        super(Classifier_no_weights, self).__init__()
        self.fc1 = nn.Linear(flattened_size, 100)
        self.dropout3 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(100, num_classes)
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
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

class Discriminator(nn.Module):
    def __init__(self, num_classes=1, feature_vector_length=128, input_sequence_length=500):
        super(Discriminator, self).__init__()

        self.convolution_kernel_size = 7
        self.convolution_dropout = 0.3

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=self.convolution_kernel_size, padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(5, 5))
        self.dropout1 = nn.Dropout2d(p=self.convolution_dropout)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.convolution_kernel_size, padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 100))
        self.dropout2 = nn.Dropout2d(p=self.convolution_dropout)

        self.flattened_size = self._calculate_flattened_size(feature_vector_length, input_sequence_length)

        self.fc1 = nn.Linear(self.flattened_size, 100)
        self.dropout3 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(100, num_classes)
        self.leakyRelu = nn.LeakyReLU(0.2, inplace=True)
        self.sig = nn.Sigmoid()
        self._initialize_weights()

    def _calculate_flattened_size(self, feature_length, sequence_length):
        x = torch.randn(1, 1, feature_length, sequence_length)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        return x.view(1, -1).size(1)

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
        x = self.fc2(x)
        x = self.sig(x)
        return x

class Discriminator_no_weights(nn.Module):
    def __init__(self, num_classes=1, flattened_size=384):
        super(Discriminator_no_weights, self).__init__()

        self.fc1 = nn.Linear(flattened_size, 100)
        self.dropout3 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(100, num_classes)
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
        x = self.fc2(x)
        x = self.sig(x)
        return x

class Discriminator_bridge(nn.Module):
    def __init__(self, num_classes=1, flattened_size=384):
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
    def __init__(self, input_dim=384, output_dim=10):
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
class cnn_GVB(nn.Module):
    def __init__(self):
        super(cnn_GVB, self).__init__()
        self.classifier = Classifier_no_weights()
        self.bridge = Bridge()

    def forward(self, x):
        class_output = self.classifier(x)
        bridge_output = self.bridge(x)
        return  class_output, bridge_output
    
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float64(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)