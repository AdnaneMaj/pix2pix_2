import torch
import torch.nn as nn
import torch.optim as optim
from pix2pix_2.utils import save_checkpoint,load_checkpoint,save_some_examples
from pix2pix_2 import config
from pix2pix_2.dataset import Costum
from pix2pix_2.generator import Generator
from pix2pix_2.discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm #for progress bar
from torchvision.utils import save_image

def train_fc(disc,gen,loader,opt_disc,opt_gen,L1_LOSS,BCE,g_scaler,d_scaler):
    loop = tqdm(loader,leave=True)

    for idx, (x,y) in enumerate(loop):
        x,y = x.to(config.DEVICE),y.to(config.DEVICE)

        #Train Discriminator
        with torch.cuda.amp.autocast():
            fake = gen(x)
            D_real = disc(x,y)
            D_fake = disc(x,fake.detach())
            D_real_loss = BCE(D_real,torch.ones_like(D_real))
            D_fake_loss = BCE(D_fake,torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss)/2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        #Train Generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x,fake)
            G_fake_loss = BCE(D_fake,torch.ones_like(D_fake))
            L1 = L1_LOSS(fake,y)
            G_loss = G_fake_loss + config.LAMBDA_L1*L1

        gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(D_real=torch.sigmoid(D_real).mean().item(),D_fake=torch.sigmoid(D_fake).mean().item())
    

def main():
    disc = Discriminator(in_channels_x=1,in_channels_y=3).to(config.DEVICE)
    gen = Generator(in_channels=1,out_channels=3).to(config.DEVICE)

    opt_disc = optim.Adam(disc.parameters(),lr=config.LEARNING_RATE,betas=(0.5,0.999))
    opt_gen = optim.Adam(gen.parameters(),lr=config.LEARNING_RATE,betas=(0.5,0.999))

    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN,gen,opt_gen,config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC,disc,opt_disc,config.LEARNING_RATE)

    train_dataset = Costum()
    train_loader = DataLoader(train_dataset,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=config.NUM_WORKERS)
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    val_dataset = Costum(val=True)
    val_loader = DataLoader(val_dataset,batch_size=1,shuffle=True)

    for epoch in range(config.NUM_EPOCHS):
        train_fc(
            disc,gen,train_loader,opt_disc,opt_gen,BCE,L1_LOSS,g_scaler,d_scaler
        )

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen,opt_gen,filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc,opt_disc,filename=config.CHECKPOINT_DISC)

        save_some_examples(gen,val_loader,epoch,folder="/kaggle/working/")
