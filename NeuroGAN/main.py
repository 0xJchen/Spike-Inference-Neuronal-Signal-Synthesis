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


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(
        0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))

    gen_label_onehot = torch.FloatTensor(n_row ** 2, 14)
    gen_label_onehot = gen_label_onehot.cuda()
    gen_label_onehot.scatter_(1, labels.view(n_row ** 2, 1), 1)
    gen_label_onehot = Variable(gen_label_onehot)

    gen_imgs = generator(z, gen_label_onehot)
    # gen_imgs=gen_img
    save_image(gen_imgs.data, "images/%d.png" %
               batches_done, nrow=n_row, normalize=True)


if __name__ == "__main__":
    os.makedirs("images", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=500,
                        help="dimensionality of the latent space")
    parser.add_argument("--n_classes", type=int, default=14,
                        help="number of classes for dataset")
    parser.add_argument("--sample_interval", type=int, default=25,
                        help="interval between image sampling")
    opt = parser.parse_args()
    print(opt)
    cuda = True if torch.cuda.is_available() else False

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    light_train = CustomDatasetFromCSV("train", img_transform)

    train_loader = DataLoader(light_train,
                              batch_size=opt.batch_size, shuffle=True)

    # Loss functions
    # adversarial_loss = torch.nn.MSELoss()
    adversarial_loss = torch.nn.BCEWithLogitsLoss()

    # Initialize generator and discriminator
    generator = ModelG(opt.latent_dim)
    discriminator = ModelD()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
    # Optimizers
    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(train_loader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(
            1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0),
                        requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(
            0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(torch.tensor(
            np.random.randint(0, opt.n_classes, batch_size)))
        gen_labels = gen_labels.cuda()
        # print(gen_labels.shape,batch_size,gen_labels)
        # Generate a batch of images

        gen_label_onehot = torch.zeros(opt.batch_size, 14)
        gen_label_onehot = gen_label_onehot.cuda()
        gen_label_onehot.resize_(batch_size, 14).zero_()
        gen_label_onehot.scatter_(1, gen_labels.view(batch_size, 1), 1)
        gen_label_onehot = Variable(gen_label_onehot)
        # print("before gen, z:{} gen_label:{}".format(z.shape,gen_label_onehot.shape))

        z = z.float()
        gen_label_onehot = gen_label_onehot.float()

        generator.forward(z, gen_label_onehot)
        gen_imgs = generator(z, gen_label_onehot)
        # print("####",gen_imgs.shape)
        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_label_onehot)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        if i%5==0:
            optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_onehot = torch.FloatTensor(batch_size, 14)
        real_onehot = real_onehot.cuda()
        real_onehot.resize_(batch_size, 14).zero_()
        real_onehot.scatter_(1, labels.view(batch_size, 1), 1)
        real_onehot = Variable(real_onehot)

        validity_real = discriminator(real_imgs, real_onehot)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_label_onehot)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        batches_done = epoch * len(train_loader) + i
        if batches_done % opt.sample_interval == 0:
            # sample_image(n_row=10, batches_done=batches_done)
            z = Variable(FloatTensor(
                np.random.normal(0, 1, (14, opt.latent_dim))))
            labels = np.array([num for num in range(14)])

            labels = Variable(LongTensor(labels))

            gen_label_onehot = torch.FloatTensor(14, 14)
            gen_label_onehot = gen_label_onehot.cuda()
            gen_label_onehot.scatter_(1, labels.view(14, 1), 1)
            gen_label_onehot = Variable(gen_label_onehot)

            gen_imgs = generator(z, gen_label_onehot)
            gen_imgs = gen_imgs.reshape(14, 29*259)
            dat = scaler.inverse_transform(gen_imgs.cpu().data)
            dat = dat.reshape(14, 29, 259)
            for nu in range(14):
                plt.matshow(dat[nu])
                info = "images/"+"batch_"+str(batches_done)+"_label_"+str(nu)
                plt.savefig(info)
        if epoch%5==0:
            torch.save({'state_dict': discriminator.state_dict()},
                       'model/model_d_epoch_{}_batch_{}.pth'.format(
                epoch,batches_done))
            torch.save({'state_dict': generator.state_dict()},
                       'model/model_g_epoch_{}_batch_{}.pth'.format(
                epoch, batches_done))
