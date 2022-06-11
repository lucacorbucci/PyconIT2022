import torch
import torchvision
from torchvision import transforms, models
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
import time 
import streamlit as st

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 50)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 50)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        return F.log_softmax(self.fc3(x), dim=1)


def convert_image(image):
    transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    tensor = transform(image)
    return tensor

def create_net(device) -> Net:
    """
    This function creates the network we will use

    Args:
        device (torch.device): device to use

    Returns:
        Net: the network we will use
    """
    model = Net()
    net = model.to(device)
    return net

@st.cache
def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_net(device)
    model.load_state_dict(torch.load("../model/mnist_model.pt"))
    time.sleep(3)
    return model

def make_prediction(model, image):
    image_tensor = convert_image(image)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    single_loaded_img = image_tensor.to(device)
    single_loaded_img = single_loaded_img[None, None]
    single_loaded_img = single_loaded_img.type('torch.FloatTensor') # instead of DoubleTensor

    out_predict = model(single_loaded_img)
    pred = out_predict.max(1, keepdim=True)[1]
    return pred.item()