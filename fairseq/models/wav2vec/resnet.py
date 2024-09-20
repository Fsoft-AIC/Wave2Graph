#!/usr/bin/python                                                       
# Author: Siddhartha Gairola (t-sigai at microsoft dot com)                 
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, densenet121
#from torchsummary import summary

class ResNet(nn.Module):
    def __init__(self, num_out=256, drop_prob=0.2,pretrain=True):
        super(ResNet, self).__init__()

        # encoder
        self.model_ft = resnet18(pretrained=pretrain)
        num_ftrs = self.model_ft.fc.in_features#*4
        self.model_ft.fc = nn.Sequential(nn.Dropout(drop_prob), nn.Linear(num_ftrs, num_out), nn.ReLU(True), 
                nn.Dropout(drop_prob), nn.Linear(num_out, num_out), nn.ReLU(True)
                )

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1,3,1,1)
        x = self.model_ft(x)
        return x

    def fine_tune(self, block_layer=5):
        for idx, child in enumerate(self.model_ft.children()):
            if idx>block_layer:
                break
            for param in child.parameters():
                param.requires_grad = False
