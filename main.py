import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torchvision
from torchvision import models, transforms, datasets
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

batch_size=4
lr=0.0002 
beta1=0.5
beta2=0.999
NUM_EPOCHS = 50
ngpu = 1
L1_lambda = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_train_dir = "dataset/maps/train"


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 512)),
    transforms.CenterCrop((256, 512)),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

data_train = datasets.ImageFolder(root=data_train_dir, transform=transform)
dataloader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)

# def show_img(img):
#     img = img.numpy().transpose(1, 2, 0)
#     mean = np.array([0.5, 0.5, 0.5])
#     std = np.array([0.5, 0.5, 0.5])
    
#     img = img * std + mean
#     np.clip(img, 0, 1)
    
#     plt.figure(figsize=(5, 5))
#     plt.imshow(img)
#     plt.show()

# images,_ = next(iter(dataloader_train))

# sample_sat = images[0][:,:,:256]
# sample_map = images[0][:,:,256:]
# show_img(sample_sat)

def weights_init(m):
    name = m.__class__.__name__
    if (name.find("Conv") > -1):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif (name.find("BatchNorm") > -1):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super().__init__()
        self.ngpu = ngpu

        self.encoder1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        
        self.encoder2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        
        self.encoder3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
        )
        
        self.encoder4 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )
        
        self.encoder5 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )
        
        self.encoder6 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )
        
        self.encoder7 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)
        )

        self.decoder1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5)
        )

        self.decoder2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=512*2, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5)
        )
        
        self.decoder3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=512*2, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5)
        )
        
        self.decoder4 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=512*2, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            #nn.Dropout(0.5)
        )
        
        self.decoder5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=256*2, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            #nn.Dropout(0.5)
        )
        
        self.decoder6 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128*2, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            #nn.Dropout(0.5)
        )
        
        self.decoder7 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64*2, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)

        latent = self.encoder7(e6)

        d1 = torch.cat([self.decoder1(latent), e6], dim=1)
        d2 = torch.cat([self.decoder2(d1), e5], dim=1)
        d3 = torch.cat([self.decoder3(d2), e4], dim=1)
        d4 = torch.cat([self.decoder4(d3), e3], dim=1)
        d5 = torch.cat([self.decoder5(d4), e2], dim=1)
        d6 = torch.cat([self.decoder6(d5), e1], dim=1)
        
        out = self.decoder7(d6)
        
        return out
    
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        
        self.structure = nn.Sequential(
            nn.Conv2d(in_channels=3*2, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=64, out_channels= 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.structure(x)

if (__name__ == "__main__"):    
    gen = Generator(ngpu=1)
    dis = Discriminator(ngpu=1)
    gen.to(device)
    dis.to(device)

    criterion = nn.BCELoss()
    criterion.to(device)
    gen_optim = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta1, beta2))
    dis_optim = torch.optim.Adam(dis.parameters(), lr=lr, betas=(beta1, beta2))

    train_log = {"G_loss":[], "D_loss":[]}

    for epoch in range(NUM_EPOCHS):
        loop = 0
        for images, _ in iter(dataloader_train):
            loop += 1
            dis.zero_grad()
            inputs = images[:, :, :, :256].to(device)
            target = images[:, :, :, 256:].to(device)
            real_data = torch.cat([inputs, target], dim=1).to(device)
            # print(real_data.shape)
            fake_labels = gen(inputs)
            fake_data = torch.cat([inputs, fake_labels], dim=1)
            # print(fake_data.shape)

            D_real = dis(real_data)
            D_Loss_real = criterion(D_real, torch.ones_like(D_real).to(device))
            D_fake = dis(fake_data)
            D_loss_fake = criterion(D_fake, torch.zeros_like(D_fake).to(device))
            D_loss = (D_Loss_real + D_loss_fake)/2
            D_loss.backward()
            dis_optim.step()

            for i in range(2):
                gen.zero_grad()
                fake_labels = gen(inputs)
                gen_data = torch.cat([inputs, fake_labels], dim=1)
                gen_out = dis(gen_data)
                G_loss = criterion(gen_out, torch.ones_like(gen_out)) + L1_lambda*torch.abs(fake_labels - target).sum()
                G_loss.backward()
                gen_optim.step()
            print(
                    "[D loss: %f] [G loss: %f]"
                    % (D_loss.item(), G_loss.item())
                )
            train_log["G_loss"].append(G_loss.item())
            train_log["D_loss"].append(D_loss.item())

        torch.save({
                'epoch': epoch,
                'model_state_dict': gen.state_dict(),
                'optimizer_state_dict': gen_optim.state_dict(),
                'loss': G_loss,
                }, "save_model/pix2pix.pt")
        break

    train_log = pd.DataFrame(train_log)
    train_log.to_csv("training_log/pix2pix.csv")

print("done")