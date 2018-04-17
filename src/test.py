from torch.autograd import Variable
import torch.optim as optim
from extract_data import GetDataFromCSV, MyDataset
import torch
import numpy as np
from capsnet import CapsNet, MarginLoss
from torch.optim import lr_scheduler
import torch.nn as nn

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel(gray image), 16 output channels, 4x4 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 16, 4)
        self.pool1 = nn.MaxPool2d(3, 3)

        self.conv2 = nn.Conv2d(16, 16, 3)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(16, 16, 2)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 2 * 2, 120)
        self.fc2 = nn.Linear(120, 48)
        self.fc3 = nn.Linear(48, 7)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 16 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    model = model = CapsNet(3, 7)  #routing_iterations = 3
    print(model)