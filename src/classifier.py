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
