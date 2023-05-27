import os 
import sys
sys.path.append(".")
import math
import time
import json
import random
import pickle

import cv2
from tqdm import tqdm
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import DataLoader
from torchsummary import summary

from pycommon.settings import *
# import pylitho.simple as litho
import pylitho.exact as litho

from lithobench.model import *
from lithobench.dataset import *


class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, leaky=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class deconv_block(nn.Module):
    """
    Deconvolution Block 
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, output_padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class Generator(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()

        n1 = 32
        filters = [n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()

        self.conv_head = conv_block(in_ch, n1, kernel_size=7, stride=1, padding=3)

        self.conv0 = conv_block(n1, filters[0], stride=2)
        self.conv1 = conv_block(filters[0], filters[1], stride=2)
        self.conv2 = conv_block(filters[1], filters[2], stride=2)
        self.conv3 = conv_block(filters[2], filters[3], stride=2)
        self.conv4 = conv_block(filters[3], filters[4], stride=2)

        self.res0 = conv_block(filters[4], filters[4], stride=1)
        self.res1 = conv_block(filters[4], filters[4], stride=1)
        self.res2 = conv_block(filters[4], filters[4], stride=1)
        self.res3 = conv_block(filters[4], filters[4], stride=1)
        self.res4 = conv_block(filters[4], filters[4], stride=1)
        self.res5 = conv_block(filters[4], filters[4], stride=1)
        self.res6 = conv_block(filters[4], filters[4], stride=1)
        self.res7 = conv_block(filters[4], filters[4], stride=1)
        self.res8 = conv_block(filters[4], filters[4], stride=1)

        self.deconv0 = deconv_block(filters[0], n1, stride=2)
        self.deconv1 = deconv_block(filters[1], filters[0], stride=2)
        self.deconv2 = deconv_block(filters[2], filters[1], stride=2)
        self.deconv3 = deconv_block(filters[3], filters[2], stride=2)
        self.deconv4 = deconv_block(filters[4], filters[3], stride=2)

        self.conv_tail_l = nn.Conv2d(n1, out_ch, kernel_size=7, stride=1, padding=3)
        self.conv_tail_r = nn.Conv2d(n1, out_ch, kernel_size=7, stride=1, padding=3)


    def forward(self, x):
        x_head = self.conv_head(x)

        x0_0 = self.conv0(x_head)
        x1_0 = self.conv1(x0_0)
        x2_0 = self.conv2(x1_0)
        x3_0 = self.conv3(x2_0)
        x4_0 = self.conv4(x3_0)

        xres = self.res0(x4_0)
        xres = self.res1(xres)
        xres = self.res2(xres)
        xres = self.res3(xres)
        xres = self.res4(xres)
        xres = self.res5(xres)
        xres = self.res6(xres)
        xres = self.res7(xres)
        xres = self.res8(xres)
        
        x4_1 = self.deconv4(xres)
        x3_1 = self.deconv3(x4_1)
        x2_1 = self.deconv2(x3_1)
        x1_1 = self.deconv1(x2_1)
        x0_1 = self.deconv0(x1_1)

        xl = self.conv_tail_l(x0_1)
        xr = self.conv_tail_r(x0_1)
        xl = self.sigmoid(xl)
        xr = self.sigmoid(xr)
        return xl, xr



class Discriminator(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.conv0_0 = conv_block(2, 64, kernel_size=4, stride=2, padding=1, leaky=True)
        self.conv1_0 = conv_block(64, 128, kernel_size=4, stride=1, padding='same', leaky=True)
        self.conv2_0 = conv_block(128, 1, kernel_size=4, stride=1, padding='same', leaky=True)
        self.flatten_0 = nn.Flatten()
        self.fc0_0 = nn.Linear(512**2, 1)
        self.sigmoid_0 = nn.Sigmoid()
        self.seq0 = nn.Sequential(self.conv0_0, self.conv1_0, self.conv2_0, self.flatten_0, self.fc0_0, self.sigmoid_0)
        
        self.conv0_1 = conv_block(2, 64, kernel_size=4, stride=2, padding=1, leaky=True)
        self.conv1_1 = conv_block(64, 128, kernel_size=4, stride=1, padding='same', leaky=True)
        self.conv2_1 = conv_block(128, 1, kernel_size=4, stride=1, padding='same', leaky=True)
        self.flatten_1 = nn.Flatten()
        self.fc0_1 = nn.Linear(256**2, 1)
        self.sigmoid_1 = nn.Sigmoid()
        self.seq1 = nn.Sequential(self.conv0_1, self.conv1_1, self.conv2_1, self.flatten_1, self.fc0_1, self.sigmoid_1)

    def forward(self, x): 
        x0 = self.seq0(x)
        x1 = self.seq1(F.interpolate(x, size=(512, 512)))
        return 0.5 * (x0 + x1)


class DAMOLitho(ModelLitho): 
    def __init__(self, size=(1024, 1024)): 
        super().__init__(size=size, name="DAMOLitho")
        self.simLitho = litho.LithoSim("./config/lithosimple.txt")
        self.netG = Generator()
        self.netD = Discriminator()
        if torch.cuda.is_available():
            self.netG = self.netG.cuda()
            self.netD = self.netD.cuda()
        self.netG = nn.DataParallel(self.netG)
        self.netD = nn.DataParallel(self.netD)
    
    @property
    def size(self): 
        return self._size
    @property
    def name(self): 
        return self._name

    def pretrain(self, train_loader, val_loader, epochs=1): 
        optimPre = optim.Adam(self.netG.parameters(), lr=1e-3)
        schedPre = lr_sched.StepLR(optimPre, 1, gamma=0.1)
        for epoch in range(epochs): 
            torch.cuda.empty_cache()
            print(f"[Epoch {epoch}] Training")
            self.netG.train()
            progress = tqdm(train_loader)
            for target, litho, label in progress: 
                if torch.cuda.is_available():
                    target = target.cuda()
                    litho = litho.cuda()
                    label = label.cuda()
                
                aerial, mask = self.netG(target)
                lossG = F.mse_loss(aerial, litho) + F.mse_loss(mask, label)
                
                optimPre.zero_grad()
                lossG.backward()
                optimPre.step()

                progress.set_postfix(lossG=lossG.item())

            print(f"[Epoch {epoch}] Testing")
            self.netG.eval()
            lossGs = []
            progress = tqdm(val_loader)
            for target, litho, label in progress: 
                with torch.no_grad():
                    if torch.cuda.is_available():
                        target = target.cuda()
                        litho = litho.cuda()
                        label = label.cuda()
                    
                    aerial, mask = self.netG(target)
                    lossG = F.mse_loss(aerial, litho) + F.mse_loss(mask, label)
                    lossGs.append(lossG.item())

                    progress.set_postfix(lossG=lossG.item())
            
            print(f"[Epoch {epoch}] lossG = {np.mean(lossGs)}")

            if epoch == epochs//2: 
                schedPre.step()

    def train(self, train_loader, val_loader, epochs=1): 
        def free_params(module: nn.Module):
            for p in module.parameters():
                p.requires_grad = True
        def frozen_params(module: nn.Module):
            for p in module.parameters():
                p.requires_grad = False
        optimG = optim.Adam(self.netG.parameters(), lr=1e-4)
        optimD = optim.Adam(self.netD.parameters(), lr=1e-4)
        schedG = lr_sched.StepLR(optimG, 1, gamma=0.1)
        schedD = lr_sched.StepLR(optimG, 1, gamma=0.1)

        print(f"[Initial] Testing")
        self.netG.eval()
        self.netD.eval()
        lossesLitho = []
        lossesResist = []
        mious = []
        mpas = []
        progress = tqdm(val_loader)
        for target, litho, label in progress: 
            with torch.no_grad():
                if torch.cuda.is_available():
                    target = target.cuda()
                    litho = litho.cuda()
                    label = label.cuda()
                aerial, mask = self.netG(target)
                aerial, mask = aerial.detach(), mask.detach()
                mask[mask > 0.5] = 1.0
                mask[mask <= 0.5] = 0.0
                ored = (mask > 0.5) | (label > 0.5)
                anded = (mask > 0.5) & (label > 0.5)

                lossLitho = F.mse_loss(aerial, litho)
                lossResist = F.mse_loss(mask, label)
                miou = anded.sum() / ored.sum()
                mpa = anded.sum() / label.sum()
                lossesLitho.append(lossLitho.item())
                lossesResist.append(lossResist.item())
                mious.append(miou.item())
                mpas.append(mpa.item())

                progress.set_postfix(lossLitho=lossLitho.item(), lossResist=lossResist.item(), IOU=miou.item(), PA=mpa.item())
            
        print(f"[Initialized] lossLitho = {np.mean(lossesLitho)} lossResist = {np.mean(lossesResist)} mIOU = {np.mean(mious)} mPA = {np.mean(mpas)}")

        for epoch in range(epochs): 
            print(f"[Epoch {epoch}] Training")
            self.netG.train()
            self.netD.train()
            progress = tqdm(train_loader)
            for target, litho, label in progress: 
                if torch.cuda.is_available():
                    target = target.cuda()
                    litho = litho.cuda()
                    label = label.cuda()
                # Train netD
                frozen_params(self.netG)
                free_params(self.netD)
                aerial, maskFake = self.netG(target)
                maskFake = torch.cat([maskFake, aerial], dim=1)
                zeros = torch.zeros([maskFake.shape[0]], dtype=maskFake.dtype, device=maskFake.device)
                maskTrue = torch.cat([label, litho], dim=1)
                ones = torch.ones([maskTrue.shape[0]], dtype=maskTrue.dtype, device=maskTrue.device)
                x = torch.cat([maskFake, maskTrue], dim=0)
                y = torch.cat([zeros, ones], dim=0).unsqueeze(1)
                pred = self.netD(x)
                lossD = F.binary_cross_entropy(pred, y)
                optimD.zero_grad()
                lossD.backward()
                optimD.step()
                # Train netG
                free_params(self.netG)
                frozen_params(self.netD)
                aerial, mask = self.netG(target)
                maskG = torch.cat([mask, aerial], dim=1)
                predD = self.netD(maskG)
                lossG1 = -torch.mean(torch.log(predD + 1e-6))
                lossG2 = F.mse_loss(aerial, litho) + F.mse_loss(mask, label)
                lossG = 0.01*lossG1 + lossG2
                optimG.zero_grad()
                lossG.backward()
                optimG.step()
                # Log
                progress.set_postfix(lossD=lossD.item(), lossG1=lossG1.item(), lossG2=lossG2.item())

            print(f"[Epoch {epoch}] Testing")
            self.netG.eval()
            self.netD.eval()
            lossesLitho = []
            lossesResist = []
            mious = []
            mpas = []
            progress = tqdm(val_loader)
            for target, litho, label in progress: 
                with torch.no_grad():
                    if torch.cuda.is_available():
                        target = target.cuda()
                        litho = litho.cuda()
                        label = label.cuda()
                    
                    aerial, mask = self.netG(target)
                    mask[mask > 0.5] = 1.0
                    mask[mask <= 0.5] = 0.0
                    ored = (mask > 0.5) | (label > 0.5)
                    anded = (mask > 0.5) & (label > 0.5)

                    lossLitho = F.mse_loss(aerial, litho)
                    lossResist = F.mse_loss(mask, label)
                    miou = anded.sum() / ored.sum()
                    mpa = anded.sum() / label.sum()
                    lossesLitho.append(lossLitho.item())
                    lossesResist.append(lossResist.item())
                    mious.append(miou.item())
                    mpas.append(mpa.item())

                    progress.set_postfix(lossLitho=lossLitho.item(), lossResist=lossResist.item(), IOU=miou.item(), PA=mpa.item())
            
            print(f"[Epoch {epoch}] lossLitho = {np.mean(lossesLitho)} lossResist = {np.mean(lossesResist)} mIOU = {np.mean(mious)} mPA = {np.mean(mpas)}")

            if epoch == epochs//2: 
                schedG.step()
                schedD.step()
    
    def save(self, filenames): 
        torch.save(self.netG.module.state_dict(), filenames[0])
        torch.save(self.netD.module.state_dict(), filenames[1])
    
    def load(self, filenames): 
        self.netG.module.load_state_dict(torch.load(filenames[0]))
        self.netD.module.load_state_dict(torch.load(filenames[1]))

    def run(self, target): 
        self.netG.eval()
        self.netD.eval()
        xl, xr = self.netG(target)
        xl, xr = xl.detach(), xr.detach()
        return xl, xr


if __name__ == "__main__": 
    Benchmark = "MetalSet"
    ImageSize = (1024, 1024)
    Epochs = 1
    BatchSize = 4
    NJobs = 8
    TrainOnly = False
    EvalOnly = False
    train_loader, val_loader = loadersLitho(Benchmark, ImageSize, BatchSize, NJobs)
    model = DAMOLitho(size=ImageSize)
    
    BatchSize = 200
    train_loader, val_loader = loadersLitho(Benchmark, ImageSize, BatchSize, NJobs)
    data = None
    for target, label1, label2 in train_loader: 
        data = target.cuda()
        break
    count = 0
    runtime = time.time()
    for idx in range(BatchSize): 
        if count >= BatchSize: 
            break
        print(f"\rEvaluating {count}/{BatchSize}", end="")
        model.run(data[idx][None, :, :, :])
        count += 1
    runtime = time.time() - runtime
    print(f"Average runtime: {runtime/count}s")
    exit(0)
    
    if not TrainOnly and not EvalOnly: 
        model.pretrain(train_loader, val_loader, epochs=Epochs)
        model.save(["trivial/damolitho/pretrainG.pth","trivial/damolitho/pretrainD.pth"])
    else: 
        model.load(["trivial/damolitho/pretrainG.pth","trivial/damolitho/pretrainD.pth"])
    if not EvalOnly: 
        model.train(train_loader, val_loader, epochs=Epochs)
        model.save(["trivial/damolitho/trainG.pth","trivial/damolitho/trainD.pth"])
    else: 
        model.load(["trivial/damolitho/trainG.pth","trivial/damolitho/trainD.pth"])
    model.evaluate(Benchmark, ImageSize, BatchSize, NJobs)


'''
[MetalSet]
[Evaluation] L2Aerial = 8.363129502507406e-06 L2Resist = 0.0007481179098143194 IOU = 0.968522677021305 PA = 0.9789443084213275
[ViaSet]
[Evaluation] L2Aerial = 2.9828278951728084e-06 L2Resist = 0.00014683662596047027 IOU = 0.9405426911085751 PA = 0.9569705623566429
[StdMetal]
[Evaluation] L2Aerial = 2.4977397629299958e-05 L2Resist = 0.0015019905917784747 IOU = 0.9545039648518843 PA = 0.9650201771189185
[StdContact]
[Evaluation] L2Aerial = 4.559633530984034e-05 L2Resist = 0.0016432759307679675 IOU = 0.8672596514225006 PA = 0.9333687084061759
'''
