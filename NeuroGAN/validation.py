import numpy as np
import argparse
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch
import os
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.autograd import Variable
from torch import nn, optim
import matplotlib.pyplot as plt
from model import *
from loader import *
import pickle
from tqdm import trange


def generate_svm_data(batch_cnt):
    latent_dim = 500
    batch_size = 16
    # load data
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    light_train = CustomDatasetFromCSV("train", img_transform)

    train_loader = DataLoader(light_train,
                              batch_size=batch_size, shuffle=True)

    generator = ModelG(latent_dim)
    discriminator = ModelD()

    generator.cuda()
    discriminator.cuda()

    # generate z
    cur_dir = './model/model_g_epoch_{}.pth'.format(batch_cnt)
    generator.load_state_dict(torch.load(cur_dir)['state_dict'])
    z = Variable(torch.cuda.FloatTensor(
        np.random.normal(0, 1, (1, latent_dim))))

    stimuli = {}
    stimuli["data"] = []
    stimuli["label"] = []
    for cnt in trange(8):
        z = Variable(torch.cuda.FloatTensor(
            np.random.normal(0, 1, (1, latent_dim))))
        # labels = np.array([num for num in range(14)])

        for i in range(14):

            labels = (torch.tensor(np.array([i])))

            gen_label_onehot = torch.zeros(1, 14)
            # gen_label_onehot = gen_label_onehot
            gen_label_onehot.scatter_(1, labels.view(1, 1), 1)
            gen_label_onehot = gen_label_onehot.cuda()
            gen_imgs = generator(z, gen_label_onehot)
            # print(gen_imgs.shape)
            gen_imgs = gen_imgs.reshape(1, 29*259)
            dat = scaler.inverse_transform(gen_imgs.cpu().data)
            dat = dat.reshape(29, 259)
            plt.matshow(dat)
            info = "validate/"+"cnt_"+str(cnt)+"_label_"+str(i)
            plt.savefig(info)
            dat.reshape(29*259)
            stimuli["data"].append(dat)
            stimuli["label"].append(i)
    stimuli_svm = "stimuli_svm_"+str(batch_cnt)
    pickle.dump(stimuli, open(stimuli_svm, "wb"))
