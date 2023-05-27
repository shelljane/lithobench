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


def conv2d(chIn, chOut, kernel_size, stride, padding, bias=True, norm=True, relu=False): 
    layers = []
    layers.append(nn.Conv2d(chIn, chOut, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
    if norm: 
        layers.append(nn.BatchNorm2d(chOut, affine=bias))
    if relu: 
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def repeat2d(n, chIn, chOut, kernel_size, stride, padding, bias=True, norm=True, relu=False): 
    layers = []
    for idx in range(n): 
        layers.append(nn.Conv2d(chIn if idx == 0 else chOut, chOut, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        if norm: 
            layers.append(nn.BatchNorm2d(chOut, affine=bias))
        if relu: 
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def deconv2d(chIn, chOut, kernel_size, stride, padding, output_padding, bias=True, norm=True, relu=False): 
    layers = []
    layers.append(nn.ConvTranspose2d(chIn, chOut, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias))
    if norm: 
        layers.append(nn.BatchNorm2d(chOut, affine=bias))
    if relu: 
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def spsr(r, chIn, chOut, kernel_size, stride, padding, bias=True, norm=True, relu=False): 
    layers = []
    layers.append(nn.Conv2d(chIn, chOut*(r**2), kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
    layers.append(nn.PixelShuffle(r))
    if norm: 
        layers.append(nn.BatchNorm2d(chOut, affine=bias))
    if relu: 
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def linear(chIn, chOut, bias=True, norm=True, relu=False): 
    layers = []
    layers.append(nn.Linear(chIn, chOut, bias=bias))
    if norm: 
        layers.append(nn.BatchNorm1d(chOut, affine=bias))
    if relu: 
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class Generator(nn.Module): 
    def __init__(self, Cin, Cout): 
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = conv2d(Cin, 64,  kernel_size=5, stride=1, padding='same', relu=True)
        self.conv2 = conv2d(64,  128, kernel_size=5, stride=1, padding='same', relu=True)
        self.conv3 = conv2d(128, 256, kernel_size=5, stride=1, padding='same', relu=True)
        self.conv4 = conv2d(256, 512, kernel_size=5, stride=1, padding='same', relu=True)
        self.conv5 = conv2d(512, 512, kernel_size=5, stride=1, padding='same', relu=True)
        self.conv6 = conv2d(512, 512, kernel_size=5, stride=1, padding='same', relu=True)
        self.conv7 = conv2d(512, 512, kernel_size=5, stride=1, padding='same', relu=True)
        self.conv8 = conv2d(512, 512, kernel_size=5, stride=1, padding='same', relu=True)
        self.deconv8 = conv2d(512, 512,  kernel_size=5, stride=1, padding='same', relu=True)
        self.deconv7 = conv2d(512, 512,  kernel_size=5, stride=1, padding='same', relu=True)
        self.deconv6 = conv2d(512, 512,  kernel_size=5, stride=1, padding='same', relu=True)
        self.deconv5 = conv2d(512, 512,  kernel_size=5, stride=1, padding='same', relu=True)
        self.deconv4 = conv2d(512, 256,  kernel_size=5, stride=1, padding='same', relu=True)
        self.deconv3 = conv2d(256, 128,  kernel_size=5, stride=1, padding='same', relu=True)
        self.deconv2 = conv2d(128, 64,   kernel_size=5, stride=1, padding='same', relu=True)
        self.deconv1l = conv2d(64,  Cout, kernel_size=5, stride=1, padding='same', norm=False, relu=False)
        self.deconv1r = conv2d(64,  Cout, kernel_size=5, stride=1, padding='same', norm=False, relu=False)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x= self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = self.conv6(x)
        x = self.pool(x)
        x = self.conv7(x)
        x = self.pool(x)
        x = self.conv8(x)
        x = self.pool(x)
        x = self.upscale(x)
        x = self.deconv8(x)
        x = self.upscale(x)
        x = self.deconv7(x)
        x = self.upscale(x)
        x = self.deconv6(x)
        x = self.upscale(x)
        x = self.deconv5(x)
        x = self.upscale(x)
        x = self.deconv4(x)
        x = self.upscale(x)
        x = self.deconv3(x)
        x = self.upscale(x)
        x = self.deconv2(x)
        x = self.upscale(x)
        xl = self.sigmoid(self.deconv1l(x))
        xr = self.sigmoid(self.deconv1r(x))

        return xl, xr

class Discriminator(nn.Module): 
    def __init__(self, CinA, CinB): 
        super().__init__()
        pool = nn.MaxPool2d(kernel_size=2, stride=2)
        sigmoid = nn.Sigmoid()
        flatten = nn.Flatten()
        conv1l = conv2d(CinA, 64,  kernel_size=5, stride=1, padding='same', relu=True)  
        conv2l = conv2d(64,  128, kernel_size=5, stride=1, padding='same', relu=True) 
        conv3l = conv2d(128, 256, kernel_size=5, stride=1, padding='same', relu=True) 
        conv4l = conv2d(256, 512, kernel_size=5, stride=1, padding='same', relu=True) 
        conv5l = conv2d(512, 1,   kernel_size=5, stride=1, padding='same', relu=True) 
        fc1l = linear(32*32*1, 1, norm=False, relu=False)
        conv1r = conv2d(CinB, 64,  kernel_size=5, stride=1, padding='same', relu=True)  
        conv2r = conv2d(64,  128, kernel_size=5, stride=1, padding='same', relu=True) 
        conv3r = conv2d(128, 256, kernel_size=5, stride=1, padding='same', relu=True) 
        conv4r = conv2d(256, 512, kernel_size=5, stride=1, padding='same', relu=True) 
        conv5r = conv2d(512, 1,   kernel_size=5, stride=1, padding='same', relu=True) 
        fc1r = linear(32*32*1, 1, norm=False, relu=False)
        self._seql = nn.Sequential(conv1l, pool, conv2l, pool, conv3l, pool, 
                                   conv4l, conv5l, flatten, fc1l, sigmoid)
        self._seqr = nn.Sequential(conv1r, pool, conv2r, pool, conv3r, pool, 
                                   conv4r, conv5r, flatten, fc1r, sigmoid)

    def forward(self, xl, xr):
        return self._seql(xl), self._seqr(xr) 


class LithoGAN(ModelLitho): 
    def __init__(self, size=(256, 256)): 
        super().__init__(size=size, name="LithoGAN")
        self.simLitho = litho.LithoSim("./config/lithosimple.txt")
        self.netG = Generator(1, 1)
        self.netD = Discriminator(1, 1)
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
                aerial, resist = self.netG(target)
                xl = torch.cat([aerial, litho], dim=0)
                xr = torch.cat([resist, label], dim=0)
                predl, predr = self.netD(xl, xr)
                zeros = torch.zeros([aerial.shape[0]], dtype=aerial.dtype, device=aerial.device)
                ones = torch.ones([litho.shape[0]], dtype=litho.dtype, device=litho.device)
                y = torch.cat([zeros, ones], dim=0).unsqueeze(1)
                lossD = 0.5 * (F.binary_cross_entropy(predl, y) + F.binary_cross_entropy(predr, y))
                optimD.zero_grad()
                lossD.backward()
                optimD.step()
                # Train netG
                free_params(self.netG)
                frozen_params(self.netD)
                aerial, resist = self.netG(target)
                predl, predr = self.netD(aerial, resist)
                lossG1 = 0.5 * (-torch.mean(torch.log(predl+1e-6)) + -torch.mean(torch.log(predr+1e-6)))
                lossG2 = F.mse_loss(aerial, litho) + F.mse_loss(resist, label)
                lossG = 0.001*lossG1 + lossG2
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
                    
                    aerial, resist = self.netG(target)
                    resist[resist > 0.5] = 1.0
                    resist[resist <= 0.5] = 0.0
                    ored = (resist > 0.5) | (label > 0.5)
                    anded = (resist > 0.5) & (label > 0.5)

                    lossLitho = F.mse_loss(aerial, litho)
                    lossResist = F.mse_loss(resist, label)
                    miou = anded.sum() / ored.sum()
                    mpa = anded.sum() / label.sum()
                    lossesLitho.append(lossLitho.item())
                    lossesResist.append(lossResist.item())
                    mious.append(miou.item())
                    mpas.append(mpa.item())

                    progress.set_postfix(l2Litho=lossLitho.item(), l2Resist=lossResist.item(), IOU=miou.item(), PA=mpa.item())
            
            print(f"[Epoch {epoch}] l2Litho = {np.mean(lossesLitho)} l2Resist = {np.mean(lossesResist)} mIOU = {np.mean(mious)} mPA = {np.mean(mpas)}")

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
    ImageSize = (256, 256)
    Epochs = 1
    BatchSize = 32
    NJobs = 8
    TrainOnly = False
    EvalOnly = False
    train_loader, val_loader = loadersLitho(Benchmark, ImageSize, BatchSize, NJobs)
    model = LithoGAN(size=ImageSize)
    
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
    
    if not EvalOnly: 
        model.train(train_loader, val_loader, epochs=Epochs)
        model.save(["trivial/lithogan/trainG.pth","trivial/lithogan/trainD.pth"])
    else: 
        model.load(["trivial/lithogan/trainG.pth","trivial/lithogan/trainD.pth"])
    model.evaluate(Benchmark, ImageSize, BatchSize, NJobs)



'''
[MetalSet]
[Evaluation] L2Aerial = 0.0009863558012651852 L2Resist = 0.016525732030948766 IOU = 0.3787635908677028 PA = 0.4293446924823981

[ViaSet]
[Evaluation] L2Aerial = 0.0002597144537323582 L2Resist = 0.001408949532740555 IOU = 0.4722621119529991 PA = 0.526229334033125

[StdMetal]
[Evaluation] L2Aerial = 0.0013601601951652104 L2Resist = 0.025917949361933604 IOU = 0.3038685123125712 PA = 0.34473183419969344

[StdContact]
[Evaluation] L2Aerial = 0.00273817692262431 L2Resist = 0.011855735909193754 IOU = 0.009275606940112388 PA = 0.009513129043625668
'''
