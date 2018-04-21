<<<<<<< HEAD
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from extract_data import GetDataFromCSV
import torch
import numpy as np

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


net = Net()
net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


datacsv = GetDataFromCSV()

image, labels = datacsv.get_training_data()
# Recreate array, this was necessary because the original format was:
# [28708,48,48], but Pytorch needs an indicator on depth so, added 1 as depth
f_l = np.empty([28708, 1, 48, 48])

for index, item in enumerate(f_l):  # Refill the list
    item[0] = image[index]

f_l = f_l.astype("float")
f_l = f_l / 255.0

f_l = torch.from_numpy(f_l)
labels = torch.from_numpy(labels)

batch_size = 4
trainset = torch.utils.data.TensorDataset(f_l, labels)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)


if __name__ == "__main__":

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            #inputs, labels = Variable(inputs), Variable(labels)

            labels = labels.resize(batch_size)
            #print(labels)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.float())
            loss = criterion(outputs, labels.type(torch.cuda.LongTensor))
            loss.backward()
            optimizer.step()

            #print statistics
            running_loss += loss.data[0]
            if i % 200 :    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    print('Finished Training')
=======
from torch.autograd import Variable
import torch
from src.cnn import Net


def Classifier(img):
    if img is None:
        return None
    else:
        model = Net()
        model.load_state_dict(torch.load('090_cnn_model_dict.pth'))
        model.eval()
        img = img.astype("float32")
        img = torch.from_numpy(img)
        img = img.view(-1, 1, 48, 48)
        img = Variable(img)
        probs = model(img)

        return probs
>>>>>>> fccba075dc509dcbf311fb90c05a749ca392ae15
