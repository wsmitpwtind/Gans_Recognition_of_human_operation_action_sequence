import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision import transforms
import numpy as np

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def name(self):
        return "LeNet"


model = LeNet().to(device)
model.load_state_dict(torch.load('./LeNET'))


with torch.no_grad():
    model.eval()
    input = cv2.imread('./predict/0.bmp', cv2.IMREAD_UNCHANGED)
    input = np.array(input)
    input = input[:,:,np.newaxis]
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,)),]
    )
    input = transform(input).to(device)
    input = input.unsqueeze(0)
    output = model(input)
    _, predicted = torch.max(output, 1)
    print(predicted)
