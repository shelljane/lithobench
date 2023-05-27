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
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
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


def deconv2d(chIn, chOut, kernel_size, stride, padding, output_padding, bias=True, norm=True, relu=False): 
    layers = []
    layers.append(nn.ConvTranspose2d(chIn, chOut, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias))
    if norm: 
        layers.append(nn.BatchNorm2d(chOut, affine=bias))
    if relu: 
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def sepconv2d(chIn, chOut, kernel_size, stride, padding, bias=True, norm=True, relu=False): 
    layers = []
    layers.append(nn.Conv2d(chIn, chOut, groups=chIn, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
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

def split(x, size=16): 
    return rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=size, s2=size)

class CFNO(nn.Module): 
    def __init__(self, c=1, d=16, k=16, s=1, size=(128, 128)): 
        super().__init__()
        self.c = c
        self.d = d
        self.k = k
        self.s = s
        self.size = size
        self.fc = nn.Linear(self.c*(self.k**2), self.d, dtype=COMPLEXTYPE)
        self.conv = sepconv2d(self.d, self.d, kernel_size=2*self.s+1, stride=1, padding="same", relu=False)

    def forward(self, x): 
        batchsize = x.shape[0]
        c = x.shape[1]
        h = x.shape[2]//self.k
        w = x.shape[3]//self.k
        patches = split(x, self.k)
        patches = patches.view(-1, self.c*(self.k**2))
        fft = torch.fft.fft(patches, dim=-1)
        fc = self.fc(fft)
        ifft = torch.fft.ifft(fc).real
        ifft = rearrange(ifft, '(b h w) d -> b d h w', h=h, w=w)
        conved = self.conv(ifft)
        return F.interpolate(conved, size=self.size)


class CFNONet(nn.Module): 
    def __init__(self): 
        super().__init__()
        
        self.cfno0 = CFNO(c=1, d=16, k=16, s=1)
        self.cfno1 = CFNO(c=1, d=32, k=32, s=1)
        self.cfno2 = CFNO(c=1, d=64, k=64, s=1)

        self.conv0a = conv2d(1, 32, kernel_size=3, stride=2, padding=1, relu=True)
        self.conv0b = repeat2d(2, 32, 32, kernel_size=3, stride=1, padding=1, relu=True)
        self.conv1a = conv2d(32, 64, kernel_size=3, stride=2, padding=1, relu=True)
        self.conv1b = repeat2d(2, 64, 64, kernel_size=3, stride=1, padding=1, relu=True)
        self.conv2a = conv2d(64, 128, kernel_size=3, stride=2, padding=1, relu=True)
        self.conv2b = repeat2d(2, 128, 128, kernel_size=3, stride=1, padding=1, relu=True)
        self.branch = nn.Sequential(self.conv0a, self.conv0b, self.conv1a, self.conv1b, self.conv2a, self.conv2b)

        self.deconv0a = deconv2d(16+32+64+128, 128, kernel_size=3, stride=2, padding=1, output_padding=1, relu=True)
        self.deconv0b = repeat2d(2, 128, 128, kernel_size=3, stride=1, padding=1, relu=True)
        self.deconv1a = deconv2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, relu=True)
        self.deconv1b = repeat2d(2, 64, 64, kernel_size=3, stride=1, padding=1, relu=True)
        self.deconv2a = deconv2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, relu=True)
        self.deconv2b = repeat2d(2, 32, 32, kernel_size=3, stride=1, padding=1, relu=True)

        self.conv3 = conv2d(32, 32, kernel_size=3, stride=1, padding=1, relu=True)
        self.conv4 = conv2d(32, 32, kernel_size=3, stride=1, padding=1, relu=True)
        self.conv5 = conv2d(32, 32, kernel_size=3, stride=1, padding=1, relu=True)
        self.conv6l = conv2d(32, 1,  kernel_size=3, stride=1, padding=1, norm=False, relu=False)
        self.conv6r = conv2d(32, 1,  kernel_size=3, stride=1, padding=1, norm=False, relu=False)
        
        self.sigmoid = nn.Sigmoid()

        self.tail = nn.Sequential(self.deconv0a, self.deconv0b, self.deconv1a, self.deconv1b, self.deconv2a, self.deconv2b, 
                                  self.conv3, self.conv4, self.conv5)

    def forward(self, x): 
        
        br0 = self.cfno0(x)
        br1 = self.cfno1(x)
        br2 = self.cfno2(x)
        br3 = self.branch(x)

        feat = torch.cat([br0, br1, br2, br3], dim=1)
        x = self.tail(feat)
        xl = self.conv6l(x)
        xr = self.conv6r(x)
        xl = self.sigmoid(xl)
        xr = self.sigmoid(xr)

        return xl, xr


class CFNOLitho(ModelLitho): 
    def __init__(self, size=(1024, 1024)): 
        super().__init__(size=size, name="CFNOLitho")
        self.simLitho = litho.LithoSim("./config/lithosimple.txt")
        self.net = CFNONet()
        if torch.cuda.is_available():
            self.net = self.net.cuda()
    
    @property
    def size(self): 
        return self._size
    @property
    def name(self): 
        return self._name

    def pretrain(self, train_loader, val_loader, epochs=1): 
        pass

    def train(self, train_loader, val_loader, epochs=1): 
        opt = optim.Adam(self.net.parameters(), lr=1e-3)
        sched = lr_sched.StepLR(opt, 1, gamma=0.1)
        for epoch in range(epochs): 
            print(f"[Epoch {epoch}] Training")
            self.net.train()
            progress = tqdm(train_loader)
            for target, litho, label in progress: 
                if torch.cuda.is_available():
                    target = target.cuda()
                    litho = litho.cuda()
                    label = label.cuda()
                    
                aerial, mask = self.net(target)
                loss = F.mse_loss(aerial, litho) + F.mse_loss(mask, label)
                
                opt.zero_grad()
                loss.backward()
                opt.step()

                progress.set_postfix(loss=loss.item())

            print(f"[Epoch {epoch}] Testing")
            self.net.eval()
            lossesLitho = []
            lossesResist = []
            mious = []
            mpas = []
            progress = tqdm(val_loader)
            with torch.no_grad(): 
                for target, litho, label in progress: 
                    if torch.cuda.is_available():
                        target = target.cuda()
                        litho = litho.cuda()
                        label = label.cuda()

                    aerial, resist = self.net(target)
                    aerial, resist = aerial.detach(), resist.detach()
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

                    progress.set_postfix(l2Litho=lossLitho.item(), lossResist=lossResist.item(), IOU=miou.item(), PA=mpa.item())
            
            print(f"[Epoch {epoch}] l2Litho = {np.mean(lossesLitho)} l2Resist = {np.mean(lossesResist)} mIOU = {np.mean(mious)} mPA = {np.mean(mpas)}")

            if epoch == epochs//2: 
                sched.step()
    
    def save(self, filenames): 
        filename = filenames[0] if isinstance(filenames, list) else filenames
        torch.save(self.net.state_dict(), filename)
    
    def load(self, filenames): 
        filename = filenames[0] if isinstance(filenames, list) else filenames
        self.net.load_state_dict(torch.load(filename))

    def run(self, target): 
        self.net.eval()
        xl, xr = self.net(target)
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
    model = CFNOLitho(size=ImageSize)
    
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
        model.save(["trivial/cfnolitho/trainG.pth","trivial/cfnolitho/trainD.pth"])
    else: 
        model.load(["trivial/cfnolitho/trainG.pth","trivial/cfnolitho/trainD.pth"])
    model.evaluate(Benchmark, ImageSize, BatchSize, NJobs)


'''
[MetalSet]
[Evaluation] L2Aerial = 1.8668543756854496e-05 L2Resist = 0.0014741282973556333 IOU = 0.9391338304591585 PA = 0.9660255279564219
[ViaSet]
[Evaluation] L2Aerial = 3.772498979087505e-06 L2Resist = 0.00021254133825980277 IOU = 0.9168913381405198 PA = 0.958326921721825
[StdMetal]
[Evaluation] L2Aerial = 2.6138177511658048e-05 L2Resist = 0.002287564621142605 IOU = 0.931980715955005 PA = 0.9583481532685897
[StdContact]
[Evaluation] L2Aerial = 2.136819395153517e-05 L2Resist = 0.002199736379441761 IOU = 0.8250303736754826 PA = 0.9006112998440152
'''
