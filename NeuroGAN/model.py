import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.nn as nn

#remember input is : 29*259
class ModelD(nn.Module):
    def __init__(self):
        super(ModelD, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (2,4), stride=(1,2), padding=(0,0))
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, (5,2), stride=(1,2), padding=(2,0))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, (5, 2), stride=(1, 2), padding=(2, 0))
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*28*32+1000, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.fc3 = nn.Linear(14, 1000)
        self.leaky=nn.LeakyReLU(0.2)
        # self.leaky=nn.ReLU()

    def forward(self, x, labels):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 29, 259)
        x = self.conv1(x)
        x = self.bn1(x)
        # x = F.relu(x)
        x = self.leaky(x)
        x = self.conv2(x)
        x = self.bn2(x)
        # x = F.relu(x)
        x =self.leaky(x)

        x = self.conv3(x)
        x = self.bn3(x)
        # x = F.relu(x)
        x = self.leaky(x)

        x = x.view(batch_size, 64*28*32)
        y_ = self.fc3(labels)
        # y_ = F.relu(y_)
        x = self.leaky(x)

        # print(x.shape, y_.shape)

        x = torch.cat([x, y_], 1)
        x = self.fc1(x)
        # x = F.relu(x)
        x = self.leaky(x)

        x = self.fc2(x)
        return F.sigmoid(x)

#z-dim=500
class ModelG(nn.Module):
    def __init__(self, z_dim):
        self.z_dim = z_dim
        super(ModelG, self).__init__()
        self.fc2 = nn.Linear(14, 1000)
        self.fc = nn.Linear(self.z_dim+1000, 64*28*32)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv1 = nn.ConvTranspose2d(64, 32, (5, 2), stride=(1, 2), padding=(2, 0))
        self.bn2 = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(
            32, 16,  (5, 2), stride=(1, 2), padding=(2, 0))
        self.bn3 = nn.BatchNorm2d(16)
        self.deconv3 = nn.ConvTranspose2d(16, 1,  (4, 5), stride=(1, 2), padding=(1, 0))
        self.tan=nn.Tanh()
        self.leaky=nn.LeakyReLU(0.2)

    def forward(self, x, labels):

        # print("in: ",x.shape)

        batch_size = x.size(0)
        # print("************",type(labels))
        y_ = self.fc2(labels)
        # y_ = F.relu(y_)
        y_=self.leaky(y_)
        x = torch.cat([x, y_], 1)
        # print("before fc: ",x.shape)
        x = self.fc(x)
        # print("after fc: ",x.shape)
        x = x.view(batch_size, 64, 28, 32)
        x = self.bn1(x)
        # x = F.relu(x)
        x=self.leaky(x)
        x = self.deconv1(x)
        # print("after dc1: ",x.shape)
        x = self.bn2(x)
        # x = F.relu(x)
        x = self.leaky(x)
        x = self.deconv2(x)
        # print("after dc2: ", x.shape)

        x = self.bn3(x)
        # x = F.relu(x)
        x=self.leaky(x)
        x = self.deconv3(x)
        # print("after dc3: ", x.shape)
        x = self.tan(x)
        return x
if __name__ == "__main__":
    a = ModelD()
    b=ModelG(500)
    print(a)
    print(b)