import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def stat(light):
    print(light.max(), light.min(), light.mean(), light.var())

# light = np.genfromtxt('20180827_Mouse1_reshape.csv', delimiter=',')


# stat(light)

# light=light.clip(50,1000)

# stat(light)

# scaler = MinMaxScaler(feature_range=(0, 1))

# print(scaler.fit(light))
# scaled_light=scaler.transform(light)
# stat(scaled_light)

# light = light.reshape(498, 29, 259)
scaler = MinMaxScaler(feature_range=(0, 1))


class CustomDatasetFromCSV(Dataset):
    def __init__(self, mode, transforms=None):

        # 498*7511
        self.data = np.genfromtxt('20180827_Mouse1_reshape.csv', delimiter=',')
        self.labels = np.genfromtxt('20180904_tag.csv', delimiter=',')
        self.labels-=1
        self.transforms = transforms
        #498*7511 (0,1)
        print(scaler.fit(self.data))
        self.data = self.data.clip(50, 1000)
        self.data = scaler.transform(self.data)

        if mode == "train":
            self.data = self.data[:400, :]
            self.labels = self.labels[:400]
        elif mode == "test":
            self.data = self.data[400:, :]
            self.labels = self.labels[400:]
        else:
            print("wrong mode!")

    def __getitem__(self, index):
        single_image_label = self.labels[index]
        img_as_img = (self.data[index]).reshape(29, 259)
        # 7511 (0,1) -> 29*259 (-1,1)
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return (self.data.shape[0])


