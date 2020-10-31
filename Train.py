import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
import os
import numpy

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out


batch_size = 128
num_epoch = 100
z_dimension = 100

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize( mean=0.5, std=0.5)
])

mnist = datasets.MNIST('./data', transform=img_transform, download=True)
dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True,
                        num_workers=0)


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class generator(nn.Module):
    def __init__(self, input_size, num_feature):
        super(generator, self).__init__()
        self.fc = nn.Linear(input_size, num_feature)
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(1, 50, 3, stride=1, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(50, 25, 3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(True)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(25, 1, 2, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1, 56, 56)
        x = self.br(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        return x


D = discriminator().cuda()
G = generator(z_dimension, 3136).cuda()

criterion = nn.BCELoss()

d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)

for epoch in range(num_epoch):
    for i, (img, _) in enumerate(dataloader):
        num_img = img.size(0)
        real_img = Variable(img).cuda()
        real_label = Variable(torch.ones(num_img)).cuda()
        fake_label = Variable(torch.zeros(num_img)).cuda()
        real_out = D(real_img)
        d_loss_real = criterion(real_out, real_label)
        real_scores = real_out
        z = Variable(torch.randn(num_img, z_dimension)).cuda()
        fake_img = G(z)
        fake_out = D(fake_img)
        d_loss_fake = criterion(fake_out, fake_label)
        fake_scores = fake_out
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        z = Variable(torch.randn(num_img, z_dimension)).cuda()
        fake_img = G(z)
        output = D(fake_img)
        g_loss = criterion(output, real_label)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
                  'D real: {:.6f}, D fake: {:.6f}'
                  .format(epoch, num_epoch, d_loss.item(), g_loss.item(),
                          real_scores.data.mean(), fake_scores.data.mean()))
    if epoch == 0:
        real_images = to_img(real_img.cpu().data)
        save_image(real_images, './dc_img/real_images.png')

    fake_images = to_img(fake_img.cpu().data)
    save_image(fake_images, './dc_img/fake_images-{}.png'.format(epoch + 1))

torch.save(G.state_dict(), './generator.pth')
torch.save(D.state_dict(), './discriminator.pth')