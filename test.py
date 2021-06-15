import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.datasets import MNIST

from models.gan import Generator, Discriminator
from mnist_loader import MNISTLoader
from utils import show

image_size = 28
batch_size = 256
nc = 1 # Number of channels in the training images. For color images this is
nz = 128 # Size of z latent vector (i.e. size of generator input)
ngf = 64 # Size of feature maps in generator
ndf = 64 # Size of feature maps in discriminator
alpha = 0.1 # loss ratio of Residual Loss, Discriminator Loss
save_fp = './results/exp06/'
netG_checkpoint = './results/exp06/checkpoints/G_epoch049.pth'
netD_checkpoint = './results/exp06/checkpoints/D_epoch049.pth'

# make dirs
os.makedirs(os.path.join(save_fp, 'test_img'), exist_ok=True)

# get model
netD = Discriminator(nc, ndf)
netG = Generator(nz, ngf, nc)

# load checkpoints
netD.load_state_dict(torch.load(netD_checkpoint))
netG.load_state_dict(torch.load(netG_checkpoint))

transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5))])

mnist = MNIST(root='./', train=False, transform=transform, download=False)

# Create the dataloader including only 9
dataloader = DataLoader(dataset=MNISTLoader(mnist, [9]),
                          batch_size=1,
                          shuffle=True,
                          drop_last=False)

# Decide which device we want to run on
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
netD.to(device)
netG.to(device)

# set loss function
criterion = torch.nn.MSELoss()

# get one real image
real, label = next(iter(dataloader))
real = real.to(device)

# get one random z vector
z_vector = torch.randn(1, nz, 1, 1, device=device, requires_grad=True)

# set optimizer to update z vector
optimizer = torch.optim.Adam([z_vector])

# iteration to update z vector
for i in tqdm(range(2001)):

    # generate fake from z
    fake = netG(z_vector)

    # get feature from discriminator
    f_real = netD(real)[1]
    f_fake = netD(fake)[1]

    # get loss
    lossR = criterion(real, fake)
    lossD = criterion(f_real, f_fake)
    loss = (1 - alpha) * lossR + alpha * lossD

    # update z vector
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # show, save real, fake image
    if i%200==0:
        hstack = torch.cat([real, fake], dim=0)
        hstack = vutils.make_grid(hstack.detach().cpu(), padding=2, normalize=True)
        show(hstack, title=f'Iteration: {i}', save_fp=os.path.join(save_fp, f'test_img/{i:04}.png'))

# visualize abnormal area (real - fake)
real_mask = real>0.2
real_mask = real_mask.type(torch.FloatTensor)
real_mask = vutils.make_grid(real_mask, normalize=True)

fake_mask = fake>0.2
fake_mask = fake_mask.type(torch.FloatTensor)
fake_mask = vutils.make_grid(fake_mask, normalize=True)

real_img = vutils.make_grid(real, normalize=True)
diff = real_mask - fake_mask
r, g, b = diff
real_img[0][r>0]=1.
real_img[1][g>0]=0.
real_img[2][b>0]=0.

# show image
print(f'Anomaly Score: {loss.item():.3f}')
show(real_img.detach().cpu(), title=f'Anomaly Score: {loss.item():.3f}', save_fp=os.path.join(save_fp, 'test_img/Anomaly.png'))