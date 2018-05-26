import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torchvision import models
import NN_models.ops as ops
class APX_SELU(Module):
    def __init__(self,file_name,inplace=False):
        super(APX_SELU, self).__init__()
        self.inplace = inplace
        self.file_name = file_name

    def forward(self, input):
        return ops.selu_apx(input,self.file_name)
class AlexNet(nn.Module):

    def __init__(self,file_name,num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            APX_SELU(file_name),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            APX_SELU(file_name),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            APX_SELU(file_name),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            APX_SELU(file_name),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            APX_SELU(file_name),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.BatchNorm1d(4096),
            APX_SELU(file_name),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            APX_SELU(file_name),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

models.vgg11_bn()