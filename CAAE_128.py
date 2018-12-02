import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.nn.functional as F
import torchvision.utils as vutils
import pickle
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

n_channel = 3
n_disc = 16
n_gen = 64
n_encode = 64
n_l = 10
n_z = 50
img_size = 128
batchSize = 20
use_cuda = torch.cuda.is_available()
n_age = int(n_z/n_l)
n_gender = int(n_z/2)

des_dir = "./data/"

dataset = dset.ImageFolder(root=des_dir,
                           transform=transforms.Compose([
                               transforms.Scale(img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size= batchSize,
                                         shuffle=True)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            # input: 3*128*128
            nn.Conv2d(n_channel, n_encode, 5, 2, 2),
            nn.ReLU(),

            nn.Conv2d(n_encode, 2 * n_encode, 5, 2, 2),
            nn.ReLU(),

            nn.Conv2d(2 * n_encode, 4 * n_encode, 5, 2, 2),
            nn.ReLU(),

            nn.Conv2d(4 * n_encode, 8 * n_encode, 5, 2, 2),
            nn.ReLU(),

        )
        self.fc = nn.Linear(8 * n_encode * 8 * 8, 50)

    def forward(self, x):
        conv = self.conv(x).view(-1, 8 * n_encode * 8 * 8)
        out = self.fc(conv)
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(nn.Linear(n_z + n_l * n_age + n_gender,
                                          8 * 8 * n_gen * 16),
                                nn.ReLU())
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(16 * n_gen, 8 * n_gen, 4, 2, 1),
            nn.ReLU(),

            nn.ConvTranspose2d(8 * n_gen, 4 * n_gen, 4, 2, 1),
            nn.ReLU(),

            nn.ConvTranspose2d(4 * n_gen, 2 * n_gen, 4, 2, 1),
            nn.ReLU(),

            nn.ConvTranspose2d(2 * n_gen, n_gen, 4, 2, 1),
            nn.ReLU(),

            nn.ConvTranspose2d(n_gen, n_channel, 3, 1, 1),
            nn.Tanh(),

        )

    def forward(self, z, age, gender):
        l = age.repeat(1, n_age)
        k = gender.view(-1, 1).repeat(1, n_gender)

        x = torch.cat([z, l, k], dim=1)
        fc = self.fc(x).view(-1, 16 * n_gen, 8, 8)
        out = self.upconv(fc)
        return out


class Dimg(nn.Module):
    def __init__(self):
        super(Dimg, self).__init__()
        self.conv_img = nn.Sequential(
            nn.Conv2d(n_channel, n_disc, 4, 2, 1),
        )
        self.conv_l = nn.Sequential(
            nn.ConvTranspose2d(n_l * n_age + n_gender, n_l * n_age + n_gender, 64, 1, 0),
            nn.ReLU()
        )
        self.total_conv = nn.Sequential(
            nn.Conv2d(n_disc + n_l * n_age + n_gender, n_disc * 2, 4, 2, 1),
            nn.ReLU(),

            nn.Conv2d(n_disc * 2, n_disc * 4, 4, 2, 1),
            nn.ReLU(),

            nn.Conv2d(n_disc * 4, n_disc * 8, 4, 2, 1),
            nn.ReLU()
        )

        self.fc_common = nn.Sequential(
            nn.Linear(8 * 8 * img_size, 1024),
            nn.ReLU()
        )
        self.fc_head1 = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        self.fc_head2 = nn.Sequential(
            nn.Linear(1024, n_l),
            nn.Softmax()
        )

    def forward(self, img, age, gender):
        l = age.repeat(1, n_age, 1, 1, )
        k = gender.repeat(1, n_gender, 1, 1, )
        conv_img = self.conv_img(img)
        conv_l = self.conv_l(torch.cat([l, k], dim=1))
        catted = torch.cat((conv_img, conv_l), dim=1)
        total_conv = self.total_conv(catted).view(-1, 8 * 8 * img_size)
        body = self.fc_common(total_conv)

        head1 = self.fc_head1(body)
        head2 = self.fc_head2(body)

        return head1, head2


class Dz(nn.Module):
    def __init__(self):
        super(Dz, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_z, n_disc * 4),
            nn.ReLU(),

            nn.Linear(n_disc * 4, n_disc * 2),
            nn.ReLU(),

            nn.Linear(n_disc * 2, n_disc),
            nn.ReLU(),

            nn.Linear(n_disc, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)

def one_hot(labelTensor):
    oneHot = - torch.ones(batchSize*n_l).view(batchSize,n_l)
    for i,j in enumerate(labelTensor):
        oneHot[i,j] = 1
    if use_cuda:
        return Variable(oneHot).cuda()
    else:
        return Variable(oneHot)

def TV_LOSS(imgTensor):
    x = (imgTensor[:,:,1:,:]-imgTensor[:,:,:img_size-1,:])**2
    y = (imgTensor[:,:,:,1:]-imgTensor[:,:,:,:img_size-1])**2
    out = (x.mean(dim=1)+y.mean(dim=1)).mean()
    return out