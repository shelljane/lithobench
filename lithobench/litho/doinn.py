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


class RFNO(nn.Module):
    def __init__(self, out_channels, modes1, modes2):
        super(RFNO, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / out_channels)
        self.weights0 = nn.Parameter(self.scale * torch.rand(1, out_channels, 1, 1, dtype=COMPLEXTYPE))
        self.weights1 = nn.Parameter(self.scale * torch.rand(1, out_channels, self.modes1, self.modes2, dtype=COMPLEXTYPE))
        self.weights2 = nn.Parameter(self.scale * torch.rand(1, out_channels, self.modes1, self.modes2, dtype=COMPLEXTYPE))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        x_ft = x_ft * self.weights0

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=COMPLEXTYPE, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class RFNONet(nn.Module): 
    def __init__(self): 
        super().__init__()
        
        self.rfno = RFNO(64, modes1=32, modes2=32)

        self.conv0 = conv2d(1,  16, kernel_size=3, stride=2, padding=1, relu=True)
        self.conv1 = conv2d(16, 32, kernel_size=3, stride=2, padding=1, relu=True)
        self.conv2 = conv2d(32, 64, kernel_size=3, stride=2, padding=1, relu=True)

        self.deconv0 = deconv2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1, relu=True)
        self.deconv1 = deconv2d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1, relu=True)
        self.deconv2 = deconv2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, relu=True)

        self.conv3 = conv2d(16, 16, kernel_size=3, stride=1, padding=1, relu=True)
        self.conv4 = conv2d(16, 16, kernel_size=3, stride=1, padding=1, relu=True)
        self.conv5 = conv2d(16, 8, kernel_size=3, stride=1, padding=1, relu=True)
        self.conv6l = conv2d(8, 1,  kernel_size=3, stride=1, padding=1, norm=False, relu=False)
        self.conv6r = conv2d(8, 1,  kernel_size=3, stride=1, padding=1, norm=False, relu=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): 
        
        br0 = self.rfno(F.avg_pool2d(x, kernel_size=8, stride=8))
        
        br1_0 = self.conv0(x)
        br1_1 = self.conv1(br1_0)
        br1_2 = self.conv2(br1_1)

        joined = self.deconv0(torch.cat([br0, br1_2], dim=1))
        joined = self.deconv1(torch.cat([joined, br1_1], dim=1))
        joined = self.deconv2(torch.cat([joined, br1_0], dim=1))

        joined = self.conv3(joined)
        joined = self.conv4(joined)
        joined = self.conv5(joined)
        xl = self.conv6l(joined)
        xr = self.conv6r(joined)
        xl = self.sigmoid(xl)
        xr = self.sigmoid(xr)

        return xl, xr

class DOINN(ModelLitho): 
    def __init__(self, size=(1024, 1024)): 
        super().__init__(size=size, name="DOINN")
        self.simLitho = litho.LithoSim("./config/lithosimple.txt")
        self.net = RFNONet()
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
    BatchSize = 16
    NJobs = 8
    TrainOnly = False
    EvalOnly = False
    train_loader, val_loader = loadersLitho(Benchmark, ImageSize, BatchSize, NJobs)
    model = DOINN(size=ImageSize)
    
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
        model.save(["trivial/doinn/trainG.pth","trivial/doinn/trainD.pth"])
    else: 
        model.load(["trivial/doinn/trainG.pth","trivial/doinn/trainD.pth"])
    model.evaluate(Benchmark, ImageSize, BatchSize, NJobs)


'''
[MetalSet]
[Evaluation] L2Aerial = 8.542811870426607e-06 L2Resist = 0.000664127220340905 IOU = 0.9719989467593073 PA = 0.9836789925121566
[ViaSet]
[Evaluation] L2Aerial = 1.9461552496921456e-06 L2Resist = 0.0001019230555887059 IOU = 0.959123646083114 PA = 0.9772722411614198
[StdMetal]
[Evaluation] L2Aerial = 1.756155854630325e-05 L2Resist = 0.001241809072192101 IOU = 0.9619989956126493 PA = 0.975569462074953
[StdContact]
[Evaluation] L2Aerial = 2.3732940793376077e-05 L2Resist = 0.0012705808357250962 IOU = 0.8958707451820374 PA = 0.9400871775367043
'''
