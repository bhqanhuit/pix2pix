from main import Generator
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision import models, transforms, datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np

model = Generator(ngpu=1)
optimizer = torch.optim.Adam(model.parameters())

checkpoint = torch.load("save_model/pix2pix.pt")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

data_test_dir = "dataset/maps/val"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 512)),
    transforms.CenterCrop((256, 512)),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

data_val = datasets.ImageFolder(root=data_test_dir, transform=transform)
dataloader_val = DataLoader(data_val, batch_size=3, shuffle=True)

fig = plt.figure(figsize=(10, 7)) 
rows = 3
cols = 3


def show_img(img, row_pos, pred_img):
    print(type(img))
    img = img.numpy().transpose(1, 2, 0)
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    
    img = img * std + mean
    np.clip(img, 0, 1)
    fig.add_subplot(rows, cols, row_pos*cols + 1) 
    plt.imshow(img[:, :256, :]) 
    plt.axis('off') 
    plt.title("Input")
    fig.add_subplot(rows, cols, row_pos*cols + 2) 
    plt.imshow(img[:, 256:512, :]) 
    plt.axis('off') 
    plt.title("Ground Truth")
    fig.add_subplot(rows, cols, row_pos*cols + 3) 
    plt.imshow(pred_img) 
    plt.axis('off') 
    plt.title("Generated image")

    

# show_img(data_val[1][0], 0)
# show_img(data_val[2][0], 1)

model = Generator(ngpu=10)
optimizer = torch.optim.Adam(model.parameters())

checkpoint = torch.load("save_model/pix2pixv2.pt")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

data = torch.stack([data_val[1][0], data_val[2][0], data_val[3][0]], dim=0)
# print(type(data))
# print(data.shape)

fake_img = model(data[:, :, :, :256])
# print(fake_img.shape)
# plt.imshow(fake_img[0].detach().T)
# plt.show()
show_img(data_val[1][0], 0, fake_img[0].detach().T)
show_img(data_val[2][0], 1, fake_img[1].detach().T)
show_img(data_val[3][0], 2, fake_img[2].detach().T)
plt.show()
