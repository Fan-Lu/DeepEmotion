from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.extract_data import GetDataFromCSV
import torch
import numpy as np
from src.cnn import Net

def Classifier(img):
    model = Net()
    model.load_state_dict(torch.load('090_cnn_model_dict.pth'))
    model.eval()

    img = torch.from_numpy(img)
    img = img.view(-1, 1, 48, 48)
    img = Variable(img)
    probs = model(img)

    return probs
