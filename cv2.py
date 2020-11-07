import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as dset
from torchvision import datasets, transforms
import torch.nn.functional as F
from tqdm import tqdm

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,)),]
)


trainSet = datasets.MNIST(root='./data', download=True, train=True, transform=transform)
testSet = datasets.MNIST(root='./data', download=True, train=False, transform=transform)
trainLoader = dset.DataLoader(trainSet, batch_size=64, shuffle=True)
testLoader = dset.DataLoader(testSet, batch_size=64, shuffle=False)

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def name(self):
        return "MLP"


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def name(self):
        return "LeNet"

model = LeNet().to(device)

epochs = 3
lr = 0.002
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)

for epoch in range(epochs):
    running_loss = 0.0

    for idx, data in enumerate(trainLoader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if idx % 100 == 99 or idx+1 == len(trainLoader):
            print('[%d/%d, %d/%d] loss: %.3f' % (epoch+1, epochs, idx+1, len(trainLoader), running_loss/2000))

print('Training Finished.')

correct = 0
total = 0

with torch.no_grad():
    for data in testLoader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # print(inputs)
        # print(labels)
        predictions = model(inputs)
        # print(predictions)
        predictions=torch.max(predictions,axis=1)
        # print(predictions[1])
        # print(labels)
        # print(predictions[1].size)
        # print(labels.size)
        ans = predictions[1]==labels
        # print(ans)
        correct += ans.sum().item()
        temp = ans.shape
        total += temp[0]


print('Accuracy of the network on the 10000 test images: %d %%' % (100*correct / total))

torch.save(model.state_dict(), model.name())
