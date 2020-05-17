import torch.nn as nn
import math
import utils 
import random

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            #3 224 128
            nn.Conv2d(3, 64, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            #64 112 64
            nn.Conv2d(64, 128, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            #128 56 32
            nn.Conv2d(128, 256, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            #256 28 16
            nn.Conv2d(256, 512, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, 3, padding=1),nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
            
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),nn.ReLU(inplace=True)
        )
            #512 14 8   
            # nn.Conv2d(512, 512, 3, padding=1),nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, 3, padding=1),nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, 3, padding=1),nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, 2)
        #512 7 4

        self.avg_pool = nn.AvgPool2d(14)
        #512 1 1
        # self. = nn.Linear(512, num_classes)
        self.classifier = nn.Linear(1024, num_classes)
        """
        self.fc1 = nn.Linear(512*2*2,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,10)
        """

    def forward(self, x, is_train=False, rand_index=None, lam=None):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        features = self.conv5(x)

        x = self.avg_pool(features)
        x = x.view(features.size(0), -1)
        x = self.classifier(x)
        return x