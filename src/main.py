from torch.autograd import Variable
import torch.optim as optim
from src.extract_data import GetDataFromCSV, MyDataset
import torch
import numpy as np
from src.capsnet import CapsNet, MarginLoss
from torch.optim import lr_scheduler

import argparse

# Training settings
parser = argparse.ArgumentParser(description='CapsNet')
parser.add_argument('--batchsize', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--routing_iterations', type=int, default=3)
parser.add_argument('--nclass', type=int, default=7)
parser.add_argument('--with_reconstruction', action='store_true', default=False)

args = parser.parse_args()



def train(epoch):
    datacsv = GetDataFromCSV()
    image, labels = datacsv.get_training_data()
    labels = torch.from_numpy(labels)
    f_l = np.empty([28708, 1, 48, 48])
    for index, item in enumerate(f_l):  # Refill the list
        item[0] = image[index]
    f_l = f_l.astype("float32")
    f_l = f_l / 255.0
    f_l = torch.from_numpy(f_l)
    train_dataset = MyDataset(f_l, labels)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=2)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
            target = target.type(torch.cuda.LongTensor)
        data, target = Variable(data), Variable(target, requires_grad=False)
        optimizer.zero_grad()
        output, probs = model(data)
        loss = loss_fn(probs, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))


if __name__ == "__main__":
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    model = CapsNet(args.routing_iterations, args.nclass)  #routing_iterations = 3
    if args.cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=15, min_lr=1e-6)
    loss_fn = MarginLoss(0.9, 0.1, 0.5)

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        torch.save(model.state_dict(),
                   '{:03d}_model_dict_{}routing_reconstruction{}.pth'.format(epoch, args.routing_iterations,
                                                                             args.with_reconstruction))