import os
from PIL import Image
import numpy as np
from IPython import display
from utils import Logger
import cv2

import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class mess(Dataset):

    def __init__(self, mess_patches, clean_patches, transform=None):
        self.clean_patches = clean_patches
        self.mess_patches = mess_patches
        self.transform = transform

    def __len__(self):
        return len(self.clean_patches)

    def __getitem__(self, idx):
        clean_image = self.clean_patches[idx]
        mess_image = self.mess_patches[idx]

        clean_image = np.asarray(clean_image)
        mess_image = np.asarray(mess_image)

        clean_image = Image.fromarray(clean_image.astype(np.uint8))
        mess_image = Image.fromarray(mess_image.astype(np.uint8))

        if self.transform:
            clean_image = self.transform(clean_image)
            mess_image = self.transform(mess_image)

        return mess_image, clean_image



'''Generator'''
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class GeneratorNet(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(GeneratorNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
    
def images_to_vectors(images):
    return images.view(images.size(0), 784)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 256, 256)



'''Discriminaotr'''
class DiscriminatorNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.disc = nn.Sequential(            
            nn.Conv2d(in_channels, 64, 3, 1, 1), # layer 1 64x256x256
            nn.LeakyReLU(0.01, inplace=True),                     
            nn.MaxPool2d(2),                     # layer 2 64x128x128
            nn.Conv2d(64, 128, 3, 1, 1),         # layer 3 128x128x128
            nn.LeakyReLU(0.01, inplace=True), 
            nn.MaxPool2d(2),                     # layer 4 128x64x64
            nn.Conv2d(128, 256, 3, 1, 1),        # layer 5 256x64x64
            nn.LeakyReLU(0.01, inplace=True),    # batch norm next
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),                     # layer 6 256x32x32
            nn.Conv2d(256, 256, 3, 1, 1),        # layer 7 256x32x32
            nn.LeakyReLU(0.01, inplace=True),    # batch norm next
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),                     # layer 8 256x16x16
            nn.Conv2d(256, 256, 3, 1, 1),        # layer 9 256x16x16
            nn.LeakyReLU(0.01, inplace=True),    # batch norm next
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 1, 3, 1, 1),          # layer 10 1x16x16
        )

    def forward(self, x,y):
        input = torch.cat([x,y], axis=1)
        return self.disc(input)
    
# Noise
def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available(): return n.cuda() 
    return n



'''Utils'''
def getPatches(mess_image,clean_image,mystride):
    mess_patches=[]
    clean_patches=[]

    #???????????? 256????????? ??? ?????????????????? padding??? ?????? ????????? ?????? ??????    
    h =  ((mess_image.shape [0] // 256) +1)*256 
    w =  ((mess_image.shape [1] // 256 ) +1)*256
    image_padding=np.ones((h,w))
    image_padding[:mess_image.shape[0],:mess_image.shape[1]]=mess_image

    #?????? ????????? ???????????? 256*256???????????? ????????? ????????? ??????
    for j in range (0,h-256,mystride): 
        for k in range (0,w-256,mystride):
            mess_patches.append(image_padding[j:j+256,k:k+256])
    
    h =  ((clean_image.shape [0] // 256) +1)*256 
    w =  ((clean_image.shape [1] // 256 ) +1)*256

    image_padding=np.ones((h,w))
    image_padding[:clean_image.shape[0],:clean_image.shape[1]]=clean_image

    for j in range (0,h-256,mystride):  
        for k in range (0,w-256,mystride):
            clean_patches.append(image_padding[j:j+256,k:k+256])  
            
    return np.array(mess_patches),np.array(clean_patches)



def create_patches():
    mess_patches = []
    clean_patches = []
    clean_image_path = './data/B/'
    mess_image_path = './data/A/'
  
    #????????? ?????? ????????? ?????? ??? ?????????
    for file in all_files:
        clean_img = cv2.imread(clean_image_path+file, cv2.IMREAD_GRAYSCALE)
        clean_t, t_otsu = cv2.threshold(clean_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        clean_thresh_np = np.zeros_like(clean_img)
        clean_thresh_np[clean_img > clean_t] = 255
        clean_image = np.asarray(clean_thresh_np)

        mess_img = cv2.imread(mess_image_path+file, cv2.IMREAD_GRAYSCALE)
        mess_t, t_otsu = cv2.threshold(mess_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        mess_thresh_np = np.zeros_like(mess_img)
        mess_thresh_np[mess_img > mess_t] = 255
        mess_image = np.asarray(mess_thresh_np)

        #getPatches ????????? ??? ??????????????? patch ?????? ??? ????????? ??????
        _mess_patches , _clean_patches = getPatches(mess_image, clean_image, mystride=128+64)
        mess_patches.append(_mess_patches)    
        clean_patches.append(_clean_patches)
    
    #concatenate?????? (????????? ??????, ???????????? patch ??????, 256, 256)??? (????????? ??????*???????????? patch, 256, 256)?????? ??????
    mess_patches = np.concatenate(mess_patches, axis=0)
    clean_patches = np.concatenate(clean_patches, axis=0)

    return mess_patches, clean_patches



DATA_FOLDER = './data/B'
all_files = os.listdir(DATA_FOLDER)
all_files.sort()

mess_patches, clean_patches = create_patches()

#Numpy ????????? ???????????? ????????? ????????? PIL???????????? ??????
dataset = mess(mess_patches, clean_patches, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]))

#???????????? DataLoader??? ?????????????????? ?????? ????????? ??????
bs = 16
data_loader = DataLoader(dataset, batch_size=bs, shuffle=True)
num_batches = len(data_loader)

discriminator = DiscriminatorNet(in_channels=2)
generator = GeneratorNet(in_channels=1, out_channels=1)

discriminator.cuda()
generator.cuda()

'''loss function'''
# Optimizers
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
loss1 = nn.MSELoss()
loss2 = nn.BCELoss()
loss3 = nn.BCEWithLogitsLoss()

# Number of steps to apply to the discriminator
d_steps = 1  # In Goodfellow et. al 2014 this variable is assigned to 1
# Number of epochs
num_epochs = 120

def test_noise():
  x,y = next(iter(data_loader))
  return Variable(x).float().cuda()

'''train'''
logger = Logger(model_name='DE-GAN', data_name='paper-cleaner')
num_test_samples = bs

for epoch in range(num_epochs):
    for n_batch, (mess_batch, clean_batch) in enumerate(data_loader):

        real_data = clean_batch.float().cuda()   #????????? ?????????
        noised_data = mess_batch.float().cuda()   #????????? ?????????
        
        fake_data = generator(noised_data)   #????????? ???????????? ????????? ????????? ????????? fake_data ??????

        d_optimizer.zero_grad()   #????????? ??? ????????? ????????? ?????? ???????????? ?????????
        
        prediction_real = discriminator(real_data, noised_data)   #????????? ???????????? ????????? ???????????? ?????? ?????????

        real_data_target = torch.ones_like(prediction_real)   #????????? ???????????? target????????? 1??? ?????????
        loss_real = loss1(prediction_real, real_data_target)   #??????????????? ????????? ??????

        prediction_fake = discriminator(fake_data, noised_data)  #???????????? ???????????? fake_data??? ????????? ???????????? ?????? ?????????

        fake_data_target = torch.zeros_like(prediction_real)   #????????? ???????????? target????????? 0?????? ?????????
        loss_fake = loss1(prediction_fake, fake_data_target)   #??????????????? ????????? ??????

        loss_d = (loss_real + loss_fake)/2   #????????? ????????? ?????????
        loss_d.backward(retain_graph=True)   #??????????????? ?????????
        
        d_optimizer.step()  #???????????? ????????????
  
        g_optimizer.zero_grad()   #????????? ??? ????????? ????????? ?????? ???????????? ?????????

        prediction = discriminator(fake_data, real_data)    #???????????? ?????? fake_data??? ????????? ???????????? ?????? ?????????
        
        real_data_target = torch.ones_like(prediction)   #???????????? ????????? ???????????? ???????????? ????????? 1??? ?????????

        loss_g1 = loss1(prediction, real_data_target)
        loss_g2 = loss1(fake_data, real_data)*500
        loss_g = loss_g1 + loss_g2

        loss_g.backward()

        g_optimizer.step()
                
        logger.log(loss_d, loss_g, epoch, n_batch, num_batches)

        # Display Progress
        if (n_batch) % 280 == 0:
            display.clear_output(True)
            # Display Images
            test_images = vectors_to_images(generator(test_noise())).data.cpu()
            logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches);
            # Display status Logs
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                loss_d, loss_g, prediction_real, prediction_fake
            )
        # Model Checkpoints
        if epoch % 10 == 0:
            logger.save_models(generator, discriminator, epoch)

path = './model/'
torch.save(generator.state_dict(), path + 'generator-final.pt')
torch.save(discriminator.state_dict(), path + 'discriminator-final.pt')