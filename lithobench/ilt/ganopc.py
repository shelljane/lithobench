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
    def __init__(self): 
        super().__init__()
        conv1 = conv2d(1,   16,   kernel_size=5, stride=2, padding=2, relu=True) # 128x128
        conv2 = conv2d(16,  64,   kernel_size=5, stride=2, padding=2, relu=True) # 64x64
        conv3 = conv2d(64,  128,  kernel_size=5, stride=2, padding=2, relu=True) # 32x32
        conv4 = conv2d(128, 512,  kernel_size=5, stride=2, padding=2, relu=True) # 16x16
        conv5 = conv2d(512, 1024, kernel_size=5, stride=2, padding=2, relu=True) # 8x8
        spsr5 = spsr(2, 1024, 512, kernel_size=3, stride=1, padding=1, relu=True) # 16x16
        spsr4 = spsr(2, 512,  128, kernel_size=3, stride=1, padding=1, relu=True) # 32x32
        spsr3 = spsr(2, 128,  64,  kernel_size=3, stride=1, padding=1, relu=True) # 64x64
        spsr2 = spsr(2, 64,   16,  kernel_size=3, stride=1, padding=1, relu=True) # 128x128
        spsr1 = spsr(2, 16,   1,   kernel_size=3, stride=1, padding=1, norm=False, relu=False) # 256x256
        self._seq = nn.Sequential(conv1, conv2, conv3, conv4, conv5, 
                                  spsr5, spsr4, spsr3, spsr2, spsr1)

    def forward(self, x): 
        return self._seq(x) 

class Discriminator(nn.Module): 
    def __init__(self): 
        super().__init__()
        repeat2a = repeat2d(2, 1, 64, kernel_size=3, stride=1, padding=1, relu=True)
        conv1 = conv2d(64, 64, kernel_size=3, stride=2, padding=1, relu=True) # 128x128
        repeat2b = repeat2d(2, 64, 128, kernel_size=3, stride=1, padding=1, relu=True)
        conv2 = conv2d(128, 128, kernel_size=3, stride=2, padding=1, relu=True) # 64x64
        repeat3a = repeat2d(3, 128, 256, kernel_size=3, stride=1, padding=1, relu=True)
        conv3 = conv2d(256, 256, kernel_size=3, stride=2, padding=1, relu=True) # 32x32
        repeat3b = repeat2d(3, 256, 512, kernel_size=3, stride=1, padding=1, relu=True)
        conv4 = conv2d(512, 512, kernel_size=3, stride=2, padding=1, relu=True) # 16x16
        repeat3c = repeat2d(3, 512, 512, kernel_size=3, stride=1, padding=1, relu=True)
        conv5 = conv2d(512, 512, kernel_size=3, stride=2, padding=1, relu=True) # 8x8
        flatten = nn.Flatten()
        fc1 = linear(8*8*512, 2048, relu=True)
        fc2 = linear(2048, 512, relu=True)
        fc3 = linear(512, 1, relu=True)
        sigmoid = nn.Sigmoid()
        self._seq = nn.Sequential(repeat2a, conv1, repeat2b, conv2, 
                                  repeat3a, conv3, repeat3b, conv4, repeat3c, conv5, 
                                  flatten, fc1, fc2, fc3, sigmoid)

    def forward(self, x): 
        return self._seq(x) 


class GANOPC(ModelILT): 
    def __init__(self, size=(256, 256)): 
        super().__init__(size=size, name="GANOPC")
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
            print(f"[Pre-Epoch {epoch}] Training")
            self.netG.train()
            progress = tqdm(train_loader)
            for target, label in progress: 
                if torch.cuda.is_available():
                    target = target.cuda()
                    label = label.cuda()
                
                params = self.netG(target)
                mask = torch.sigmoid(params.squeeze(1))
                printedNom, printedMax, printedMin = self.simLitho(mask)
                l2loss = F.mse_loss(printedNom.unsqueeze(1), target)
                lossG2 = F.mse_loss(mask.unsqueeze(1), label)
                lossG = l2loss + lossG2
                
                optimPre.zero_grad()
                lossG.backward()
                optimPre.step()

                progress.set_postfix(l2loss=l2loss.item(), lossG2=lossG2.item())

            print(f"[Pre-Epoch {epoch}] Testing")
            self.netG.eval()
            l2losses = []
            lossG2s = []
            progress = tqdm(val_loader)
            for target, label in progress: 
                with torch.no_grad():
                    if torch.cuda.is_available():
                        target = target.cuda()
                        label = label.cuda()
                    
                    params = self.netG(target)
                    mask = torch.sigmoid(params.squeeze(1))
                    printedNom, printedMax, printedMin = self.simLitho(mask)
                    l2loss = F.mse_loss(printedNom.unsqueeze(1), target)
                    lossG2 = F.mse_loss(mask.unsqueeze(1), label)
                    lossG = l2loss + lossG2
                    l2losses.append(l2loss.item())
                    lossG2s.append(lossG.item())

                    progress.set_postfix(l2loss=l2loss.item(), lossG2=lossG2.item())
            
            print(f"[Pre-Epoch {epoch}] L2 loss = {np.mean(l2losses)}, lossG2 = {np.mean(lossG2s)}")

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
        l2losses = []
        lossG2s = []
        progress = tqdm(val_loader)
        for target, label in progress: 
            with torch.no_grad():
                if torch.cuda.is_available():
                    target = target.cuda()
                    label = label.cuda()
                
                params = self.netG(target)
                mask = torch.sigmoid(params.squeeze(1))
                printedNom, printedMax, printedMin = self.simLitho(mask)
                l2loss = F.mse_loss(printedNom.unsqueeze(1), target)
                lossG2 = F.mse_loss(mask.unsqueeze(1), label)
                lossG = l2loss + lossG2
                l2losses.append(l2loss.item())
                lossG2s.append(lossG2.item())
                
                printedNom, printedMax, printedMin = self.simLitho(label.squeeze(1))
                l2ref = F.mse_loss(printedNom.unsqueeze(1), target)

                progress.set_postfix(l2loss=l2loss.item(), l2ref=l2ref.item(), lossG2=lossG2.item())
        
        print(f"[Initial] L2 loss = {np.mean(l2losses)}, lossG2 = {np.mean(lossG2s)}")

        for epoch in range(epochs): 
            print(f"[Epoch {epoch}] Training")
            self.netG.train()
            self.netD.train()
            progress = tqdm(train_loader)
            for target, label in progress: 
                if torch.cuda.is_available():
                    target = target.cuda()
                    label = label.cuda()
                # Train netD
                frozen_params(self.netG)
                free_params(self.netD)
                params = self.netG(target)
                maskFake = torch.sigmoid(params.squeeze(1)).unsqueeze(1)
                zeros = torch.zeros([maskFake.shape[0]], dtype=maskFake.dtype, device=maskFake.device)
                maskTrue = label
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
                params = self.netG(target)
                maskG = torch.sigmoid(params.squeeze(1)).unsqueeze(1)
                predD = self.netD(maskG)
                lossG1 = -torch.mean(torch.log(predD))
                lossG2 = F.mse_loss(maskG, label)
                lossG = lossG1 + lossG2
                optimG.zero_grad()
                lossG.backward()
                optimG.step()
                # Log
                progress.set_postfix(lossD=lossD.item(), lossG1=lossG1.item(), lossG2=lossG2.item())

            print(f"[Epoch {epoch}] Testing")
            self.netG.eval()
            self.netD.eval()
            l2losses = []
            lossG2s = []
            l2refs = []
            progress = tqdm(val_loader)
            for target, label in progress: 
                with torch.no_grad():
                    if torch.cuda.is_available():
                        target = target.cuda()
                        label = label.cuda()
                    
                    params = self.netG(target)
                    mask = torch.sigmoid(params.squeeze(1))
                    printedNom, printedMax, printedMin = self.simLitho(mask)
                    l2loss = F.mse_loss(printedNom.unsqueeze(1), target)
                    lossG2 = F.mse_loss(mask.unsqueeze(1), label)
                    lossG = l2loss + lossG2
                    l2losses.append(l2loss.item())
                    lossG2s.append(lossG2.item())
                    
                    printedNom, printedMax, printedMin = self.simLitho(label.squeeze(1))
                    l2ref = F.mse_loss(printedNom.unsqueeze(1), target)
                    l2refs.append(l2ref.item())

                    progress.set_postfix(l2loss=l2loss.item(), l2ref=l2ref.item(), lossG2=lossG2.item())
            
            print(f"[Epoch {epoch}] L2 loss = {np.mean(l2losses)}, l2ref = {np.mean(l2refs)}, lossG2 = {np.mean(lossG2s)}")

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
        return torch.sigmoid(self.netG(target)[0, 0]).detach()


if __name__ == "__main__": 
    Benchmark = "MetalSet"
    ImageSize = (256, 256)
    Epochs = 1
    BatchSize = 64
    NJobs = 8
    TrainOnly = True
    EvalOnly = True
    # train_loader, val_loader = loadersILT(Benchmark, ImageSize, BatchSize, NJobs)
    targets = evaluate.getTargets(samples=None, dataset=Benchmark)
    ilt = GANOPC(size=ImageSize)
    
    BatchSize = 200
    train_loader, val_loader = loadersILT(Benchmark, ImageSize, BatchSize, NJobs)
    data = None
    for target, label in train_loader: 
        data = target
        break
    count = 0
    runtime = time.time()
    for idx in range(BatchSize): 
        if count >= BatchSize: 
            break
        print(f"\rEvaluating {count}/{BatchSize}", end="")
        ilt.run(data[idx][None, :, :, :])
        count += 1
    runtime = time.time() - runtime
    print(f"Average runtime: {runtime/count}s")
    exit(0)
    
    if not TrainOnly and not EvalOnly: 
        ilt.pretrain(train_loader, val_loader, epochs=Epochs)
        ilt.save(["trivial/ganopc/pretrainG.pth","trivial/ganopc/pretrainD.pth"])
    else: 
        ilt.load(["trivial/ganopc/pretrainG.pth","trivial/ganopc/pretrainD.pth"])
    if not EvalOnly: 
        ilt.train(train_loader, val_loader, epochs=Epochs)
        ilt.save(["trivial/ganopc/trainG.pth","trivial/ganopc/trainD.pth"])
    else: 
        ilt.load(["trivial/ganopc/trainG.pth","trivial/ganopc/trainD.pth"])
    ilt.evaluate(targets, finetune=True, folder="trivial/ganopc")


'''
[MetalSet]
[Evaluation]: Getting masks from the model
[Testcase 1]: L2 60679; PVBand 47173; EPE 17; Shots: 653
[Testcase 2]: L2 43374; PVBand 39006; EPE 4; Shots: 531
[Testcase 3]: L2 91493; PVBand 70775; EPE 47; Shots: 724
[Testcase 4]: L2 18245; PVBand 22118; EPE 2; Shots: 572
[Testcase 5]: L2 42248; PVBand 50576; EPE 1; Shots: 562
[Testcase 6]: L2 46808; PVBand 47026; EPE 3; Shots: 653
[Testcase 7]: L2 32133; PVBand 40955; EPE 0; Shots: 630
[Testcase 8]: L2 22528; PVBand 20645; EPE 2; Shots: 446
[Testcase 9]: L2 56176; PVBand 57564; EPE 6; Shots: 571
[Testcase 10]: L2 20461; PVBand 17058; EPE 5; Shots: 403
[Initialized]: L2 43414; PVBand 41290; EPE 8.7; Runtime: 0.19s; Shots: 574
[Evaluation]: Finetuning the masks with pixel-based ILT method
[Testcase 1]: L2 38998; PVBand 48635; EPE 3; Shots: 592
[Testcase 2]: L2 30467; PVBand 39165; EPE 0; Shots: 563
[Testcase 3]: L2 63429; PVBand 76224; EPE 17; Shots: 681
[Testcase 4]: L2 8849; PVBand 23660; EPE 0; Shots: 487
[Testcase 5]: L2 29142; PVBand 54071; EPE 0; Shots: 562
[Testcase 6]: L2 30140; PVBand 48343; EPE 0; Shots: 583
[Testcase 7]: L2 16210; PVBand 41724; EPE 0; Shots: 505
[Testcase 8]: L2 11651; PVBand 21050; EPE 0; Shots: 532
[Testcase 9]: L2 34338; PVBand 62001; EPE 0; Shots: 615
[Testcase 10]: L2 7682; PVBand 16805; EPE 0; Shots: 403
[Finetuned]: L2 27091; PVBand 43168; EPE 2.0; Shots: 552

[ViaSet]
[Evaluation]: Getting masks from the model
[Testcase 1]: L2 9005; PVBand 4792; EPE 6; Shots: 89
[Testcase 2]: L2 8696; PVBand 2510; EPE 6; Shots: 66
[Testcase 3]: L2 13024; PVBand 4485; EPE 8; Shots: 148
[Testcase 4]: L2 5229; PVBand 6429; EPE 0; Shots: 149
[Testcase 5]: L2 20038; PVBand 8438; EPE 10; Shots: 256
[Testcase 6]: L2 22567; PVBand 2718; EPE 16; Shots: 105
[Testcase 7]: L2 13382; PVBand 3454; EPE 10; Shots: 148
[Testcase 8]: L2 27106; PVBand 20833; EPE 10; Shots: 435
[Testcase 9]: L2 18544; PVBand 13196; EPE 9; Shots: 242
[Testcase 10]: L2 10082; PVBand 0; EPE 8; Shots: 27
[Initialized]: L2 14767; PVBand 6686; EPE 8.3; Runtime: 2.09s; Shots: 166
[Evaluation]: Finetuning the masks with pixel-based ILT method
[Testcase 1]: L2 2629; PVBand 4568; EPE 0; Shots: 133
[Testcase 2]: L2 2630; PVBand 4529; EPE 0; Shots: 156
[Testcase 3]: L2 4518; PVBand 8612; EPE 0; Shots: 315
[Testcase 4]: L2 3396; PVBand 6158; EPE 0; Shots: 191
[Testcase 5]: L2 7304; PVBand 12612; EPE 0; Shots: 378
[Testcase 6]: L2 6192; PVBand 10961; EPE 0; Shots: 377
[Testcase 7]: L2 3768; PVBand 6752; EPE 0; Shots: 207
[Testcase 8]: L2 11123; PVBand 21055; EPE 0; Shots: 561
[Testcase 9]: L2 8109; PVBand 14198; EPE 1; Shots: 413
[Testcase 10]: L2 3917; PVBand 5021; EPE 1; Shots: 135
[Finetuned]: L2 5359; PVBand 9447; EPE 0.2; Shots: 287

[StdMetal]
[Evaluation]: Getting masks from the model
[Testcase 1]: L2 19236; PVBand 19632; EPE 6; Shots: 369
[Testcase 2]: L2 10947; PVBand 7083; EPE 2; Shots: 227
[Testcase 3]: L2 15817; PVBand 13066; EPE 7; Shots: 494
[Testcase 4]: L2 27058; PVBand 14210; EPE 16; Shots: 409
[Testcase 5]: L2 8067; PVBand 5138; EPE 1; Shots: 262
[Testcase 6]: L2 4467; PVBand 6406; EPE 0; Shots: 165
[Testcase 7]: L2 24981; PVBand 11350; EPE 6; Shots: 391
[Testcase 8]: L2 6425; PVBand 6669; EPE 0; Shots: 323
[Testcase 9]: L2 23882; PVBand 17569; EPE 7; Shots: 453
[Testcase 10]: L2 20977; PVBand 24987; EPE 0; Shots: 544
[Testcase 11]: L2 25952; PVBand 30284; EPE 2; Shots: 683
[Testcase 12]: L2 7208; PVBand 5998; EPE 0; Shots: 295
[Testcase 13]: L2 43968; PVBand 42636; EPE 8; Shots: 511
[Testcase 14]: L2 24584; PVBand 22359; EPE 6; Shots: 496
[Testcase 15]: L2 22338; PVBand 27437; EPE 2; Shots: 502
[Testcase 16]: L2 17496; PVBand 14981; EPE 1; Shots: 553
[Testcase 17]: L2 18725; PVBand 18137; EPE 1; Shots: 416
[Testcase 18]: L2 26540; PVBand 29237; EPE 0; Shots: 559
[Testcase 19]: L2 10717; PVBand 7171; EPE 2; Shots: 354
[Testcase 20]: L2 22541; PVBand 13329; EPE 8; Shots: 373
[Testcase 21]: L2 38951; PVBand 46078; EPE 5; Shots: 469
[Testcase 22]: L2 20174; PVBand 20662; EPE 2; Shots: 398
[Testcase 23]: L2 52261; PVBand 37342; EPE 12; Shots: 861
[Testcase 24]: L2 23777; PVBand 19075; EPE 7; Shots: 376
[Testcase 25]: L2 17496; PVBand 13597; EPE 3; Shots: 589
[Testcase 26]: L2 18649; PVBand 20058; EPE 0; Shots: 340
[Testcase 27]: L2 25890; PVBand 20413; EPE 5; Shots: 511
[Testcase 28]: L2 5162; PVBand 6556; EPE 0; Shots: 288
[Testcase 29]: L2 51831; PVBand 49276; EPE 5; Shots: 468
[Testcase 30]: L2 48504; PVBand 49466; EPE 6; Shots: 639
[Testcase 31]: L2 58522; PVBand 36073; EPE 22; Shots: 886
[Testcase 32]: L2 9772; PVBand 8682; EPE 0; Shots: 369
[Testcase 33]: L2 9343; PVBand 10966; EPE 0; Shots: 309
[Testcase 34]: L2 17190; PVBand 7434; EPE 10; Shots: 248
[Testcase 35]: L2 20280; PVBand 19750; EPE 1; Shots: 557
[Testcase 36]: L2 51048; PVBand 57834; EPE 10; Shots: 591
[Testcase 37]: L2 25507; PVBand 16974; EPE 8; Shots: 487
[Testcase 38]: L2 99190; PVBand 79976; EPE 24; Shots: 752
[Testcase 39]: L2 14693; PVBand 7336; EPE 3; Shots: 208
[Testcase 40]: L2 16765; PVBand 13281; EPE 0; Shots: 518
[Testcase 41]: L2 94176; PVBand 84679; EPE 33; Shots: 622
[Testcase 42]: L2 13259; PVBand 10081; EPE 2; Shots: 328
[Testcase 43]: L2 39497; PVBand 37536; EPE 5; Shots: 569
[Testcase 44]: L2 76985; PVBand 70035; EPE 12; Shots: 705
[Testcase 45]: L2 25238; PVBand 23148; EPE 3; Shots: 445
[Testcase 46]: L2 21589; PVBand 22402; EPE 2; Shots: 506
[Testcase 47]: L2 21173; PVBand 21958; EPE 2; Shots: 443
[Testcase 48]: L2 13461; PVBand 14327; EPE 2; Shots: 400
[Testcase 49]: L2 38181; PVBand 43983; EPE 0; Shots: 592
[Testcase 50]: L2 20749; PVBand 19091; EPE 5; Shots: 462
[Testcase 51]: L2 7471; PVBand 6077; EPE 0; Shots: 343
[Testcase 52]: L2 6827; PVBand 6358; EPE 0; Shots: 320
[Testcase 53]: L2 46426; PVBand 32007; EPE 10; Shots: 862
[Testcase 54]: L2 23860; PVBand 26539; EPE 2; Shots: 559
[Testcase 55]: L2 19236; PVBand 19632; EPE 6; Shots: 366
[Testcase 56]: L2 9204; PVBand 5872; EPE 0; Shots: 278
[Testcase 57]: L2 53384; PVBand 43121; EPE 9; Shots: 506
[Testcase 58]: L2 7105; PVBand 6752; EPE 0; Shots: 320
[Testcase 59]: L2 27448; PVBand 24442; EPE 1; Shots: 567
[Testcase 60]: L2 35393; PVBand 30391; EPE 7; Shots: 523
[Testcase 61]: L2 7162; PVBand 5382; EPE 0; Shots: 289
[Testcase 62]: L2 43525; PVBand 48042; EPE 1; Shots: 675
[Testcase 63]: L2 48784; PVBand 48672; EPE 16; Shots: 506
[Testcase 64]: L2 21895; PVBand 18447; EPE 2; Shots: 465
[Testcase 65]: L2 18985; PVBand 18573; EPE 0; Shots: 338
[Testcase 66]: L2 58998; PVBand 48324; EPE 12; Shots: 674
[Testcase 67]: L2 24193; PVBand 17106; EPE 3; Shots: 408
[Testcase 68]: L2 8755; PVBand 6104; EPE 1; Shots: 237
[Testcase 69]: L2 25962; PVBand 30189; EPE 2; Shots: 558
[Testcase 70]: L2 17214; PVBand 22446; EPE 2; Shots: 440
[Testcase 71]: L2 20174; PVBand 20662; EPE 2; Shots: 389
[Testcase 72]: L2 18813; PVBand 15626; EPE 3; Shots: 351
[Testcase 73]: L2 39983; PVBand 54109; EPE 4; Shots: 623
[Testcase 74]: L2 24271; PVBand 16447; EPE 10; Shots: 493
[Testcase 75]: L2 16835; PVBand 15546; EPE 4; Shots: 366
[Testcase 76]: L2 8067; PVBand 5138; EPE 1; Shots: 266
[Testcase 77]: L2 7208; PVBand 5998; EPE 0; Shots: 292
[Testcase 78]: L2 8703; PVBand 5192; EPE 0; Shots: 307
[Testcase 79]: L2 42598; PVBand 37595; EPE 9; Shots: 537
[Testcase 80]: L2 17979; PVBand 19983; EPE 1; Shots: 388
[Testcase 81]: L2 19133; PVBand 21113; EPE 0; Shots: 460
[Testcase 82]: L2 23828; PVBand 19294; EPE 5; Shots: 364
[Testcase 83]: L2 12922; PVBand 13523; EPE 1; Shots: 379
[Testcase 84]: L2 27716; PVBand 29365; EPE 8; Shots: 504
[Testcase 85]: L2 8278; PVBand 7310; EPE 0; Shots: 367
[Testcase 86]: L2 13461; PVBand 14327; EPE 2; Shots: 376
[Testcase 87]: L2 19046; PVBand 19053; EPE 2; Shots: 487
[Testcase 88]: L2 28868; PVBand 23485; EPE 4; Shots: 472
[Testcase 89]: L2 49916; PVBand 55774; EPE 4; Shots: 536
[Testcase 90]: L2 5621; PVBand 6232; EPE 0; Shots: 349
[Testcase 91]: L2 67102; PVBand 61845; EPE 11; Shots: 745
[Testcase 92]: L2 16519; PVBand 21177; EPE 1; Shots: 535
[Testcase 93]: L2 8681; PVBand 5332; EPE 1; Shots: 269
[Testcase 94]: L2 15071; PVBand 14138; EPE 3; Shots: 525
[Testcase 95]: L2 69235; PVBand 51468; EPE 27; Shots: 762
[Testcase 96]: L2 18592; PVBand 10769; EPE 7; Shots: 300
[Testcase 97]: L2 37040; PVBand 32803; EPE 3; Shots: 567
[Testcase 98]: L2 18985; PVBand 18573; EPE 0; Shots: 341
[Testcase 99]: L2 5504; PVBand 6264; EPE 0; Shots: 345
[Testcase 100]: L2 27058; PVBand 14210; EPE 16; Shots: 410
[Testcase 101]: L2 24584; PVBand 22359; EPE 6; Shots: 512
[Testcase 102]: L2 10510; PVBand 7977; EPE 0; Shots: 282
[Testcase 103]: L2 43472; PVBand 47269; EPE 3; Shots: 712
[Testcase 104]: L2 34532; PVBand 36346; EPE 5; Shots: 468
[Testcase 105]: L2 26169; PVBand 18294; EPE 7; Shots: 522
[Testcase 106]: L2 4467; PVBand 6406; EPE 0; Shots: 165
[Testcase 107]: L2 27681; PVBand 18366; EPE 8; Shots: 428
[Testcase 108]: L2 20024; PVBand 18227; EPE 2; Shots: 370
[Testcase 109]: L2 5830; PVBand 6315; EPE 0; Shots: 313
[Testcase 110]: L2 19860; PVBand 18175; EPE 0; Shots: 432
[Testcase 111]: L2 17945; PVBand 20049; EPE 0; Shots: 366
[Testcase 112]: L2 32613; PVBand 48253; EPE 1; Shots: 594
[Testcase 113]: L2 17945; PVBand 20049; EPE 0; Shots: 355
[Testcase 114]: L2 6787; PVBand 6077; EPE 0; Shots: 294
[Testcase 115]: L2 34836; PVBand 27021; EPE 8; Shots: 525
[Testcase 116]: L2 66167; PVBand 31712; EPE 37; Shots: 660
[Testcase 117]: L2 34402; PVBand 42973; EPE 4; Shots: 627
[Testcase 118]: L2 6408; PVBand 6324; EPE 0; Shots: 316
[Testcase 119]: L2 20277; PVBand 18732; EPE 3; Shots: 413
[Testcase 120]: L2 11713; PVBand 14064; EPE 0; Shots: 445
[Testcase 121]: L2 25750; PVBand 23744; EPE 3; Shots: 457
[Testcase 122]: L2 28603; PVBand 27601; EPE 4; Shots: 559
[Testcase 123]: L2 5504; PVBand 6264; EPE 0; Shots: 350
[Testcase 124]: L2 9581; PVBand 6452; EPE 0; Shots: 381
[Testcase 125]: L2 18251; PVBand 16033; EPE 4; Shots: 346
[Testcase 126]: L2 32419; PVBand 25464; EPE 7; Shots: 509
[Testcase 127]: L2 36462; PVBand 44187; EPE 2; Shots: 674
[Testcase 128]: L2 51857; PVBand 38738; EPE 23; Shots: 582
[Testcase 129]: L2 18059; PVBand 17881; EPE 1; Shots: 327
[Testcase 130]: L2 13718; PVBand 14497; EPE 0; Shots: 444
[Testcase 131]: L2 9961; PVBand 7298; EPE 2; Shots: 323
[Testcase 132]: L2 25046; PVBand 16810; EPE 6; Shots: 491
[Testcase 133]: L2 72648; PVBand 37181; EPE 38; Shots: 615
[Testcase 134]: L2 19511; PVBand 16419; EPE 1; Shots: 509
[Testcase 135]: L2 70479; PVBand 35303; EPE 41; Shots: 705
[Testcase 136]: L2 20862; PVBand 25428; EPE 0; Shots: 499
[Testcase 137]: L2 6408; PVBand 6324; EPE 0; Shots: 316
[Testcase 138]: L2 23719; PVBand 12886; EPE 6; Shots: 435
[Testcase 139]: L2 43347; PVBand 30026; EPE 12; Shots: 908
[Testcase 140]: L2 7856; PVBand 11117; EPE 0; Shots: 333
[Testcase 141]: L2 37384; PVBand 37572; EPE 4; Shots: 461
[Testcase 142]: L2 7687; PVBand 5945; EPE 1; Shots: 299
[Testcase 143]: L2 21519; PVBand 19070; EPE 5; Shots: 662
[Testcase 144]: L2 7471; PVBand 6077; EPE 0; Shots: 356
[Testcase 145]: L2 20356; PVBand 14927; EPE 5; Shots: 366
[Testcase 146]: L2 8703; PVBand 5192; EPE 0; Shots: 313
[Testcase 147]: L2 18649; PVBand 20058; EPE 0; Shots: 346
[Testcase 148]: L2 5929; PVBand 6233; EPE 0; Shots: 341
[Testcase 149]: L2 15667; PVBand 8167; EPE 4; Shots: 271
[Testcase 150]: L2 15071; PVBand 14138; EPE 3; Shots: 558
[Testcase 151]: L2 5621; PVBand 6232; EPE 0; Shots: 331
[Testcase 152]: L2 5162; PVBand 6556; EPE 0; Shots: 291
[Testcase 153]: L2 33626; PVBand 42399; EPE 4; Shots: 582
[Testcase 154]: L2 29895; PVBand 42892; EPE 2; Shots: 521
[Testcase 155]: L2 40856; PVBand 35992; EPE 6; Shots: 414
[Testcase 156]: L2 15920; PVBand 17181; EPE 0; Shots: 393
[Testcase 157]: L2 11392; PVBand 15521; EPE 0; Shots: 358
[Testcase 158]: L2 60491; PVBand 57951; EPE 19; Shots: 566
[Testcase 159]: L2 21631; PVBand 20499; EPE 2; Shots: 479
[Testcase 160]: L2 28803; PVBand 32928; EPE 2; Shots: 588
[Testcase 161]: L2 22668; PVBand 24452; EPE 4; Shots: 468
[Testcase 162]: L2 5929; PVBand 6233; EPE 0; Shots: 343
[Testcase 163]: L2 18251; PVBand 16033; EPE 4; Shots: 353
[Testcase 164]: L2 8382; PVBand 6203; EPE 0; Shots: 300
[Testcase 165]: L2 39442; PVBand 45908; EPE 2; Shots: 642
[Testcase 166]: L2 38129; PVBand 35341; EPE 4; Shots: 496
[Testcase 167]: L2 54759; PVBand 41347; EPE 13; Shots: 820
[Testcase 168]: L2 15057; PVBand 19413; EPE 1; Shots: 457
[Testcase 169]: L2 40765; PVBand 40696; EPE 5; Shots: 425
[Testcase 170]: L2 32644; PVBand 34726; EPE 1; Shots: 536
[Testcase 171]: L2 7710; PVBand 12607; EPE 0; Shots: 333
[Testcase 172]: L2 16043; PVBand 16538; EPE 3; Shots: 399
[Testcase 173]: L2 54075; PVBand 57177; EPE 13; Shots: 575
[Testcase 174]: L2 49075; PVBand 52119; EPE 6; Shots: 558
[Testcase 175]: L2 64598; PVBand 50682; EPE 13; Shots: 652
[Testcase 176]: L2 32472; PVBand 17810; EPE 9; Shots: 585
[Testcase 177]: L2 22602; PVBand 20210; EPE 0; Shots: 358
[Testcase 178]: L2 24981; PVBand 11350; EPE 6; Shots: 392
[Testcase 179]: L2 16765; PVBand 13281; EPE 0; Shots: 519
[Testcase 180]: L2 20637; PVBand 24627; EPE 1; Shots: 573
[Testcase 181]: L2 26944; PVBand 23702; EPE 2; Shots: 447
[Testcase 182]: L2 4467; PVBand 6406; EPE 0; Shots: 163
[Testcase 183]: L2 15751; PVBand 17089; EPE 0; Shots: 471
[Testcase 184]: L2 38005; PVBand 37901; EPE 10; Shots: 557
[Testcase 185]: L2 18173; PVBand 19478; EPE 0; Shots: 545
[Testcase 186]: L2 60428; PVBand 66814; EPE 12; Shots: 644
[Testcase 187]: L2 22541; PVBand 13329; EPE 8; Shots: 374
[Testcase 188]: L2 15436; PVBand 11488; EPE 2; Shots: 408
[Testcase 189]: L2 26482; PVBand 35741; EPE 2; Shots: 487
[Testcase 190]: L2 11252; PVBand 7500; EPE 5; Shots: 249
[Testcase 191]: L2 21519; PVBand 19070; EPE 5; Shots: 663
[Testcase 192]: L2 36452; PVBand 42349; EPE 0; Shots: 517
[Testcase 193]: L2 42838; PVBand 45282; EPE 8; Shots: 667
[Testcase 194]: L2 37040; PVBand 32803; EPE 3; Shots: 563
[Testcase 195]: L2 22274; PVBand 25118; EPE 1; Shots: 449
[Testcase 196]: L2 30835; PVBand 22652; EPE 5; Shots: 558
[Testcase 197]: L2 10510; PVBand 7977; EPE 0; Shots: 276
[Testcase 198]: L2 23828; PVBand 19294; EPE 5; Shots: 362
[Testcase 199]: L2 25120; PVBand 30755; EPE 2; Shots: 505
[Testcase 200]: L2 19814; PVBand 26657; EPE 1; Shots: 445
[Testcase 201]: L2 5830; PVBand 6315; EPE 0; Shots: 306
[Testcase 202]: L2 10717; PVBand 7171; EPE 2; Shots: 340
[Testcase 203]: L2 37728; PVBand 37001; EPE 3; Shots: 551
[Testcase 204]: L2 7382; PVBand 6205; EPE 0; Shots: 337
[Testcase 205]: L2 74831; PVBand 75345; EPE 18; Shots: 606
[Testcase 206]: L2 20013; PVBand 20976; EPE 4; Shots: 440
[Testcase 207]: L2 39214; PVBand 55447; EPE 2; Shots: 437
[Testcase 208]: L2 24981; PVBand 11350; EPE 6; Shots: 379
[Testcase 209]: L2 17312; PVBand 16482; EPE 4; Shots: 420
[Testcase 210]: L2 59373; PVBand 50334; EPE 11; Shots: 571
[Testcase 211]: L2 8229; PVBand 9362; EPE 0; Shots: 377
[Testcase 212]: L2 7925; PVBand 5404; EPE 0; Shots: 277
[Testcase 213]: L2 23777; PVBand 19075; EPE 7; Shots: 373
[Testcase 214]: L2 22245; PVBand 20642; EPE 2; Shots: 500
[Testcase 215]: L2 50490; PVBand 54129; EPE 4; Shots: 602
[Testcase 216]: L2 22967; PVBand 23283; EPE 1; Shots: 579
[Testcase 217]: L2 38017; PVBand 43350; EPE 7; Shots: 588
[Testcase 218]: L2 20024; PVBand 18227; EPE 2; Shots: 359
[Testcase 219]: L2 17912; PVBand 18323; EPE 3; Shots: 345
[Testcase 220]: L2 51011; PVBand 45530; EPE 6; Shots: 634
[Testcase 221]: L2 25238; PVBand 23148; EPE 3; Shots: 439
[Testcase 222]: L2 14696; PVBand 8015; EPE 0; Shots: 369
[Testcase 223]: L2 6910; PVBand 6177; EPE 0; Shots: 200
[Testcase 224]: L2 17979; PVBand 19983; EPE 1; Shots: 376
[Testcase 225]: L2 25636; PVBand 16084; EPE 7; Shots: 499
[Testcase 226]: L2 17420; PVBand 26402; EPE 0; Shots: 462
[Testcase 227]: L2 24435; PVBand 28009; EPE 3; Shots: 505
[Testcase 228]: L2 10078; PVBand 23249; EPE 0; Shots: 374
[Testcase 229]: L2 46943; PVBand 46139; EPE 8; Shots: 535
[Testcase 230]: L2 65562; PVBand 55198; EPE 21; Shots: 717
[Testcase 231]: L2 22394; PVBand 25678; EPE 2; Shots: 651
[Testcase 232]: L2 31182; PVBand 36700; EPE 3; Shots: 443
[Testcase 233]: L2 27783; PVBand 20794; EPE 8; Shots: 463
[Testcase 234]: L2 13461; PVBand 14327; EPE 2; Shots: 391
[Testcase 235]: L2 41695; PVBand 36875; EPE 10; Shots: 688
[Testcase 236]: L2 16835; PVBand 15546; EPE 4; Shots: 374
[Testcase 237]: L2 19751; PVBand 20188; EPE 1; Shots: 355
[Testcase 238]: L2 21427; PVBand 23684; EPE 4; Shots: 432
[Testcase 239]: L2 41568; PVBand 44263; EPE 6; Shots: 668
[Testcase 240]: L2 24981; PVBand 11350; EPE 6; Shots: 386
[Testcase 241]: L2 21966; PVBand 24741; EPE 1; Shots: 579
[Testcase 242]: L2 6827; PVBand 6358; EPE 0; Shots: 317
[Testcase 243]: L2 7997; PVBand 7244; EPE 0; Shots: 321
[Testcase 244]: L2 19751; PVBand 20188; EPE 1; Shots: 343
[Testcase 245]: L2 21488; PVBand 31405; EPE 0; Shots: 591
[Testcase 246]: L2 37525; PVBand 34957; EPE 5; Shots: 571
[Testcase 247]: L2 27716; PVBand 29365; EPE 8; Shots: 505
[Testcase 248]: L2 76559; PVBand 63679; EPE 11; Shots: 725
[Testcase 249]: L2 7925; PVBand 5404; EPE 0; Shots: 286
[Testcase 250]: L2 22602; PVBand 20210; EPE 0; Shots: 359
[Testcase 251]: L2 13461; PVBand 14327; EPE 2; Shots: 394
[Testcase 252]: L2 13667; PVBand 10591; EPE 1; Shots: 427
[Testcase 253]: L2 7105; PVBand 6752; EPE 0; Shots: 325
[Testcase 254]: L2 18059; PVBand 17881; EPE 1; Shots: 332
[Testcase 255]: L2 37237; PVBand 26031; EPE 7; Shots: 431
[Testcase 256]: L2 7382; PVBand 6205; EPE 0; Shots: 338
[Testcase 257]: L2 82301; PVBand 72231; EPE 19; Shots: 604
[Testcase 258]: L2 33915; PVBand 29788; EPE 3; Shots: 531
[Testcase 259]: L2 44770; PVBand 45613; EPE 7; Shots: 589
[Testcase 260]: L2 9772; PVBand 8682; EPE 0; Shots: 368
[Testcase 261]: L2 103750; PVBand 55055; EPE 40; Shots: 641
[Testcase 262]: L2 69528; PVBand 29082; EPE 36; Shots: 655
[Testcase 263]: L2 11310; PVBand 17177; EPE 0; Shots: 481
[Testcase 264]: L2 20327; PVBand 21640; EPE 3; Shots: 498
[Testcase 265]: L2 13737; PVBand 8922; EPE 4; Shots: 240
[Testcase 266]: L2 38340; PVBand 37318; EPE 5; Shots: 550
[Testcase 267]: L2 20277; PVBand 18732; EPE 3; Shots: 415
[Testcase 268]: L2 14693; PVBand 7336; EPE 3; Shots: 219
[Testcase 269]: L2 14140; PVBand 11115; EPE 4; Shots: 354
[Testcase 270]: L2 17847; PVBand 23896; EPE 0; Shots: 484
[Testcase 271]: L2 15667; PVBand 8167; EPE 4; Shots: 268
[Initialized]: L2 25929; PVBand 23715; EPE 4.6; Runtime: 0.04s; Shots: 457
[Evaluation]: Finetuning the masks with pixel-based ILT method
[Testcase 1]: L2 8788; PVBand 20191; EPE 0; Shots: 404
[Testcase 2]: L2 5146; PVBand 7289; EPE 0; Shots: 295
[Testcase 3]: L2 7230; PVBand 12243; EPE 0; Shots: 470
[Testcase 4]: L2 8861; PVBand 16787; EPE 0; Shots: 397
[Testcase 5]: L2 4346; PVBand 6049; EPE 0; Shots: 327
[Testcase 6]: L2 3343; PVBand 6156; EPE 0; Shots: 208
[Testcase 7]: L2 9341; PVBand 13373; EPE 0; Shots: 476
[Testcase 8]: L2 3735; PVBand 7366; EPE 0; Shots: 339
[Testcase 9]: L2 10713; PVBand 18927; EPE 0; Shots: 450
[Testcase 10]: L2 10684; PVBand 30336; EPE 0; Shots: 445
[Testcase 11]: L2 14462; PVBand 32718; EPE 0; Shots: 560
[Testcase 12]: L2 4711; PVBand 6463; EPE 0; Shots: 321
[Testcase 13]: L2 24087; PVBand 43489; EPE 0; Shots: 417
[Testcase 14]: L2 10462; PVBand 23856; EPE 0; Shots: 436
[Testcase 15]: L2 13567; PVBand 25579; EPE 0; Shots: 546
[Testcase 16]: L2 10373; PVBand 15618; EPE 0; Shots: 563
[Testcase 17]: L2 9754; PVBand 18505; EPE 0; Shots: 412
[Testcase 18]: L2 13755; PVBand 32607; EPE 0; Shots: 478
[Testcase 19]: L2 4922; PVBand 7325; EPE 0; Shots: 415
[Testcase 20]: L2 7934; PVBand 13493; EPE 0; Shots: 446
[Testcase 21]: L2 20227; PVBand 46754; EPE 0; Shots: 400
[Testcase 22]: L2 9843; PVBand 21621; EPE 0; Shots: 441
[Testcase 23]: L2 19966; PVBand 40613; EPE 0; Shots: 549
[Testcase 24]: L2 9420; PVBand 19354; EPE 0; Shots: 419
[Testcase 25]: L2 9025; PVBand 15433; EPE 0; Shots: 439
[Testcase 26]: L2 9200; PVBand 19885; EPE 0; Shots: 382
[Testcase 27]: L2 13740; PVBand 21795; EPE 0; Shots: 493
[Testcase 28]: L2 2945; PVBand 6686; EPE 0; Shots: 339
[Testcase 29]: L2 28871; PVBand 54340; EPE 0; Shots: 443
[Testcase 30]: L2 27470; PVBand 50044; EPE 0; Shots: 601
[Testcase 31]: L2 21057; PVBand 40139; EPE 0; Shots: 638
[Testcase 32]: L2 6282; PVBand 8890; EPE 0; Shots: 408
[Testcase 33]: L2 5979; PVBand 10507; EPE 0; Shots: 321
[Testcase 34]: L2 5660; PVBand 7875; EPE 0; Shots: 369
[Testcase 35]: L2 9055; PVBand 19564; EPE 0; Shots: 564
[Testcase 36]: L2 28126; PVBand 62670; EPE 1; Shots: 516
[Testcase 37]: L2 11498; PVBand 18857; EPE 0; Shots: 466
[Testcase 38]: L2 53347; PVBand 81866; EPE 0; Shots: 525
[Testcase 39]: L2 5398; PVBand 7503; EPE 0; Shots: 335
[Testcase 40]: L2 8773; PVBand 14331; EPE 0; Shots: 433
[Testcase 41]: L2 53775; PVBand 88385; EPE 3; Shots: 496
[Testcase 42]: L2 6940; PVBand 10334; EPE 0; Shots: 366
[Testcase 43]: L2 21348; PVBand 39848; EPE 0; Shots: 546
[Testcase 44]: L2 40330; PVBand 75930; EPE 0; Shots: 516
[Testcase 45]: L2 11255; PVBand 25097; EPE 0; Shots: 432
[Testcase 46]: L2 10908; PVBand 23458; EPE 0; Shots: 493
[Testcase 47]: L2 11607; PVBand 23448; EPE 0; Shots: 443
[Testcase 48]: L2 7800; PVBand 14640; EPE 0; Shots: 373
[Testcase 49]: L2 18374; PVBand 47563; EPE 0; Shots: 539
[Testcase 50]: L2 10969; PVBand 19472; EPE 0; Shots: 476
[Testcase 51]: L2 4761; PVBand 6982; EPE 0; Shots: 363
[Testcase 52]: L2 3873; PVBand 6888; EPE 0; Shots: 334
[Testcase 53]: L2 17621; PVBand 35090; EPE 0; Shots: 573
[Testcase 54]: L2 13025; PVBand 25889; EPE 0; Shots: 583
[Testcase 55]: L2 8788; PVBand 20191; EPE 0; Shots: 389
[Testcase 56]: L2 5185; PVBand 6736; EPE 0; Shots: 300
[Testcase 57]: L2 24703; PVBand 46458; EPE 0; Shots: 463
[Testcase 58]: L2 4674; PVBand 7284; EPE 0; Shots: 349
[Testcase 59]: L2 13857; PVBand 24304; EPE 0; Shots: 552
[Testcase 60]: L2 18325; PVBand 33239; EPE 0; Shots: 576
[Testcase 61]: L2 4098; PVBand 6032; EPE 0; Shots: 349
[Testcase 62]: L2 28379; PVBand 52365; EPE 0; Shots: 552
[Testcase 63]: L2 24109; PVBand 49195; EPE 0; Shots: 448
[Testcase 64]: L2 10799; PVBand 19303; EPE 0; Shots: 415
[Testcase 65]: L2 9775; PVBand 18159; EPE 0; Shots: 356
[Testcase 66]: L2 31335; PVBand 53372; EPE 1; Shots: 558
[Testcase 67]: L2 12385; PVBand 18981; EPE 0; Shots: 433
[Testcase 68]: L2 4657; PVBand 6367; EPE 0; Shots: 280
[Testcase 69]: L2 14677; PVBand 30307; EPE 0; Shots: 472
[Testcase 70]: L2 7628; PVBand 21846; EPE 0; Shots: 400
[Testcase 71]: L2 9843; PVBand 21621; EPE 0; Shots: 431
[Testcase 72]: L2 9667; PVBand 15926; EPE 0; Shots: 447
[Testcase 73]: L2 24206; PVBand 54929; EPE 0; Shots: 508
[Testcase 74]: L2 8513; PVBand 16610; EPE 0; Shots: 488
[Testcase 75]: L2 9358; PVBand 15631; EPE 0; Shots: 393
[Testcase 76]: L2 4346; PVBand 6049; EPE 0; Shots: 320
[Testcase 77]: L2 4711; PVBand 6463; EPE 0; Shots: 332
[Testcase 78]: L2 4574; PVBand 6260; EPE 0; Shots: 396
[Testcase 79]: L2 20465; PVBand 40453; EPE 0; Shots: 374
[Testcase 80]: L2 8993; PVBand 19988; EPE 0; Shots: 416
[Testcase 81]: L2 8470; PVBand 23134; EPE 0; Shots: 401
[Testcase 82]: L2 9727; PVBand 19204; EPE 0; Shots: 413
[Testcase 83]: L2 7634; PVBand 13285; EPE 0; Shots: 398
[Testcase 84]: L2 14564; PVBand 29877; EPE 0; Shots: 491
[Testcase 85]: L2 5067; PVBand 7513; EPE 0; Shots: 425
[Testcase 86]: L2 7800; PVBand 14640; EPE 0; Shots: 374
[Testcase 87]: L2 10058; PVBand 19354; EPE 0; Shots: 501
[Testcase 88]: L2 9875; PVBand 25103; EPE 0; Shots: 493
[Testcase 89]: L2 24977; PVBand 58955; EPE 0; Shots: 460
[Testcase 90]: L2 3614; PVBand 6231; EPE 0; Shots: 370
[Testcase 91]: L2 36226; PVBand 70312; EPE 0; Shots: 498
[Testcase 92]: L2 7653; PVBand 21135; EPE 0; Shots: 495
[Testcase 93]: L2 4915; PVBand 6066; EPE 0; Shots: 340
[Testcase 94]: L2 8226; PVBand 14504; EPE 0; Shots: 474
[Testcase 95]: L2 27608; PVBand 52624; EPE 1; Shots: 551
[Testcase 96]: L2 7110; PVBand 10502; EPE 0; Shots: 394
[Testcase 97]: L2 18511; PVBand 35253; EPE 0; Shots: 553
[Testcase 98]: L2 9775; PVBand 18159; EPE 0; Shots: 375
[Testcase 99]: L2 3611; PVBand 6343; EPE 0; Shots: 381
[Testcase 100]: L2 8861; PVBand 16787; EPE 0; Shots: 392
[Testcase 101]: L2 10462; PVBand 23856; EPE 0; Shots: 417
[Testcase 102]: L2 4880; PVBand 7968; EPE 0; Shots: 312
[Testcase 103]: L2 26459; PVBand 50608; EPE 0; Shots: 583
[Testcase 104]: L2 20117; PVBand 38188; EPE 0; Shots: 446
[Testcase 105]: L2 11999; PVBand 19618; EPE 0; Shots: 488
[Testcase 106]: L2 3343; PVBand 6156; EPE 0; Shots: 214
[Testcase 107]: L2 10694; PVBand 20183; EPE 0; Shots: 484
[Testcase 108]: L2 8382; PVBand 18661; EPE 0; Shots: 373
[Testcase 109]: L2 3676; PVBand 7072; EPE 0; Shots: 342
[Testcase 110]: L2 8887; PVBand 19239; EPE 0; Shots: 410
[Testcase 111]: L2 9246; PVBand 19791; EPE 0; Shots: 362
[Testcase 112]: L2 21331; PVBand 47448; EPE 0; Shots: 508
[Testcase 113]: L2 9246; PVBand 19791; EPE 0; Shots: 363
[Testcase 114]: L2 4019; PVBand 6333; EPE 0; Shots: 342
[Testcase 115]: L2 16037; PVBand 27226; EPE 0; Shots: 554
[Testcase 116]: L2 17700; PVBand 33489; EPE 0; Shots: 571
[Testcase 117]: L2 22938; PVBand 46167; EPE 0; Shots: 459
[Testcase 118]: L2 4214; PVBand 6981; EPE 0; Shots: 313
[Testcase 119]: L2 11294; PVBand 17899; EPE 0; Shots: 440
[Testcase 120]: L2 8149; PVBand 13905; EPE 0; Shots: 406
[Testcase 121]: L2 11495; PVBand 24595; EPE 0; Shots: 435
[Testcase 122]: L2 18184; PVBand 31087; EPE 0; Shots: 529
[Testcase 123]: L2 3611; PVBand 6343; EPE 0; Shots: 376
[Testcase 124]: L2 5067; PVBand 7139; EPE 0; Shots: 449
[Testcase 125]: L2 9345; PVBand 16573; EPE 0; Shots: 411
[Testcase 126]: L2 13076; PVBand 25334; EPE 0; Shots: 489
[Testcase 127]: L2 20070; PVBand 46072; EPE 0; Shots: 568
[Testcase 128]: L2 16784; PVBand 38953; EPE 0; Shots: 506
[Testcase 129]: L2 8971; PVBand 18394; EPE 0; Shots: 384
[Testcase 130]: L2 8242; PVBand 14026; EPE 0; Shots: 408
[Testcase 131]: L2 5275; PVBand 8062; EPE 0; Shots: 363
[Testcase 132]: L2 10669; PVBand 18265; EPE 0; Shots: 480
[Testcase 133]: L2 22820; PVBand 37595; EPE 0; Shots: 546
[Testcase 134]: L2 9540; PVBand 15928; EPE 0; Shots: 519
[Testcase 135]: L2 19998; PVBand 37451; EPE 0; Shots: 599
[Testcase 136]: L2 12284; PVBand 25574; EPE 0; Shots: 399
[Testcase 137]: L2 4214; PVBand 6981; EPE 0; Shots: 315
[Testcase 138]: L2 9630; PVBand 14095; EPE 0; Shots: 444
[Testcase 139]: L2 16775; PVBand 32881; EPE 0; Shots: 594
[Testcase 140]: L2 4711; PVBand 13097; EPE 0; Shots: 298
[Testcase 141]: L2 19995; PVBand 40090; EPE 0; Shots: 394
[Testcase 142]: L2 4368; PVBand 6313; EPE 0; Shots: 397
[Testcase 143]: L2 11810; PVBand 20302; EPE 0; Shots: 528
[Testcase 144]: L2 4761; PVBand 6982; EPE 0; Shots: 362
[Testcase 145]: L2 9703; PVBand 15510; EPE 0; Shots: 418
[Testcase 146]: L2 4574; PVBand 6260; EPE 0; Shots: 419
[Testcase 147]: L2 9200; PVBand 19885; EPE 0; Shots: 368
[Testcase 148]: L2 4217; PVBand 6441; EPE 0; Shots: 415
[Testcase 149]: L2 5520; PVBand 8492; EPE 0; Shots: 351
[Testcase 150]: L2 8226; PVBand 14504; EPE 0; Shots: 458
[Testcase 151]: L2 3614; PVBand 6231; EPE 0; Shots: 366
[Testcase 152]: L2 2945; PVBand 6686; EPE 0; Shots: 338
[Testcase 153]: L2 16298; PVBand 46506; EPE 0; Shots: 490
[Testcase 154]: L2 18173; PVBand 41969; EPE 0; Shots: 450
[Testcase 155]: L2 19807; PVBand 39513; EPE 0; Shots: 409
[Testcase 156]: L2 9153; PVBand 17435; EPE 0; Shots: 424
[Testcase 157]: L2 5553; PVBand 15361; EPE 0; Shots: 383
[Testcase 158]: L2 36272; PVBand 64821; EPE 0; Shots: 505
[Testcase 159]: L2 9131; PVBand 19739; EPE 0; Shots: 479
[Testcase 160]: L2 16864; PVBand 33583; EPE 0; Shots: 561
[Testcase 161]: L2 11959; PVBand 25072; EPE 0; Shots: 422
[Testcase 162]: L2 4217; PVBand 6441; EPE 0; Shots: 418
[Testcase 163]: L2 9345; PVBand 16573; EPE 0; Shots: 410
[Testcase 164]: L2 4279; PVBand 6374; EPE 0; Shots: 361
[Testcase 165]: L2 22743; PVBand 48405; EPE 0; Shots: 542
[Testcase 166]: L2 19824; PVBand 38447; EPE 0; Shots: 434
[Testcase 167]: L2 22753; PVBand 44000; EPE 0; Shots: 589
[Testcase 168]: L2 8690; PVBand 20079; EPE 0; Shots: 477
[Testcase 169]: L2 18292; PVBand 40349; EPE 0; Shots: 489
[Testcase 170]: L2 15918; PVBand 34402; EPE 0; Shots: 487
[Testcase 171]: L2 4691; PVBand 12399; EPE 0; Shots: 326
[Testcase 172]: L2 4846; PVBand 14547; EPE 0; Shots: 378
[Testcase 173]: L2 34690; PVBand 56053; EPE 0; Shots: 595
[Testcase 174]: L2 26405; PVBand 53197; EPE 0; Shots: 523
[Testcase 175]: L2 32453; PVBand 55401; EPE 0; Shots: 548
[Testcase 176]: L2 14401; PVBand 21089; EPE 0; Shots: 529
[Testcase 177]: L2 9611; PVBand 19729; EPE 0; Shots: 377
[Testcase 178]: L2 9341; PVBand 13373; EPE 0; Shots: 469
[Testcase 179]: L2 8773; PVBand 14331; EPE 0; Shots: 455
[Testcase 180]: L2 12165; PVBand 25794; EPE 0; Shots: 535
[Testcase 181]: L2 10920; PVBand 24989; EPE 0; Shots: 467
[Testcase 182]: L2 3343; PVBand 6156; EPE 0; Shots: 207
[Testcase 183]: L2 6871; PVBand 17466; EPE 0; Shots: 440
[Testcase 184]: L2 18863; PVBand 38900; EPE 0; Shots: 528
[Testcase 185]: L2 7886; PVBand 20012; EPE 0; Shots: 499
[Testcase 186]: L2 35269; PVBand 72367; EPE 0; Shots: 502
[Testcase 187]: L2 7934; PVBand 13493; EPE 0; Shots: 442
[Testcase 188]: L2 8252; PVBand 11927; EPE 0; Shots: 482
[Testcase 189]: L2 13256; PVBand 36499; EPE 0; Shots: 437
[Testcase 190]: L2 5019; PVBand 7420; EPE 0; Shots: 330
[Testcase 191]: L2 11810; PVBand 20302; EPE 0; Shots: 525
[Testcase 192]: L2 14921; PVBand 45726; EPE 0; Shots: 519
[Testcase 193]: L2 19534; PVBand 43785; EPE 0; Shots: 560
[Testcase 194]: L2 18511; PVBand 35253; EPE 0; Shots: 551
[Testcase 195]: L2 11098; PVBand 25239; EPE 0; Shots: 401
[Testcase 196]: L2 13551; PVBand 23321; EPE 0; Shots: 486
[Testcase 197]: L2 4880; PVBand 7968; EPE 0; Shots: 296
[Testcase 198]: L2 9727; PVBand 19204; EPE 0; Shots: 389
[Testcase 199]: L2 12122; PVBand 30490; EPE 0; Shots: 511
[Testcase 200]: L2 12219; PVBand 25270; EPE 0; Shots: 453
[Testcase 201]: L2 3676; PVBand 7072; EPE 0; Shots: 340
[Testcase 202]: L2 4922; PVBand 7325; EPE 0; Shots: 406
[Testcase 203]: L2 22706; PVBand 38268; EPE 0; Shots: 516
[Testcase 204]: L2 4677; PVBand 6530; EPE 0; Shots: 415
[Testcase 205]: L2 42502; PVBand 80329; EPE 1; Shots: 469
[Testcase 206]: L2 7417; PVBand 21035; EPE 0; Shots: 434
[Testcase 207]: L2 23000; PVBand 56667; EPE 0; Shots: 400
[Testcase 208]: L2 9341; PVBand 13373; EPE 0; Shots: 473
[Testcase 209]: L2 9632; PVBand 17301; EPE 0; Shots: 432
[Testcase 210]: L2 36051; PVBand 56841; EPE 0; Shots: 572
[Testcase 211]: L2 4750; PVBand 9711; EPE 0; Shots: 311
[Testcase 212]: L2 4927; PVBand 5996; EPE 0; Shots: 335
[Testcase 213]: L2 9420; PVBand 19354; EPE 0; Shots: 397
[Testcase 214]: L2 13509; PVBand 21047; EPE 0; Shots: 471
[Testcase 215]: L2 32313; PVBand 55802; EPE 0; Shots: 579
[Testcase 216]: L2 11939; PVBand 25379; EPE 0; Shots: 497
[Testcase 217]: L2 26815; PVBand 46121; EPE 3; Shots: 536
[Testcase 218]: L2 8382; PVBand 18661; EPE 0; Shots: 369
[Testcase 219]: L2 8347; PVBand 17927; EPE 0; Shots: 383
[Testcase 220]: L2 19825; PVBand 49503; EPE 0; Shots: 506
[Testcase 221]: L2 11255; PVBand 25097; EPE 0; Shots: 399
[Testcase 222]: L2 7651; PVBand 9343; EPE 0; Shots: 480
[Testcase 223]: L2 4349; PVBand 6267; EPE 0; Shots: 258
[Testcase 224]: L2 8993; PVBand 19988; EPE 0; Shots: 423
[Testcase 225]: L2 12703; PVBand 17333; EPE 0; Shots: 519
[Testcase 226]: L2 8118; PVBand 28720; EPE 0; Shots: 417
[Testcase 227]: L2 14437; PVBand 28498; EPE 0; Shots: 533
[Testcase 228]: L2 6279; PVBand 22861; EPE 0; Shots: 329
[Testcase 229]: L2 22839; PVBand 50064; EPE 0; Shots: 496
[Testcase 230]: L2 28147; PVBand 54111; EPE 0; Shots: 503
[Testcase 231]: L2 12783; PVBand 27308; EPE 0; Shots: 558
[Testcase 232]: L2 15040; PVBand 39122; EPE 0; Shots: 436
[Testcase 233]: L2 12337; PVBand 20933; EPE 0; Shots: 441
[Testcase 234]: L2 7800; PVBand 14640; EPE 0; Shots: 397
[Testcase 235]: L2 18743; PVBand 38893; EPE 0; Shots: 512
[Testcase 236]: L2 9358; PVBand 15631; EPE 0; Shots: 398
[Testcase 237]: L2 9178; PVBand 20017; EPE 0; Shots: 381
[Testcase 238]: L2 12290; PVBand 25173; EPE 0; Shots: 432
[Testcase 239]: L2 21288; PVBand 48309; EPE 0; Shots: 543
[Testcase 240]: L2 9341; PVBand 13373; EPE 0; Shots: 471
[Testcase 241]: L2 9057; PVBand 25881; EPE 0; Shots: 540
[Testcase 242]: L2 3873; PVBand 6888; EPE 0; Shots: 328
[Testcase 243]: L2 4727; PVBand 6935; EPE 0; Shots: 384
[Testcase 244]: L2 9178; PVBand 20017; EPE 0; Shots: 359
[Testcase 245]: L2 15662; PVBand 34605; EPE 0; Shots: 496
[Testcase 246]: L2 20800; PVBand 37537; EPE 0; Shots: 564
[Testcase 247]: L2 14564; PVBand 29877; EPE 0; Shots: 490
[Testcase 248]: L2 40231; PVBand 71691; EPE 0; Shots: 551
[Testcase 249]: L2 4927; PVBand 5996; EPE 0; Shots: 345
[Testcase 250]: L2 9611; PVBand 19729; EPE 0; Shots: 370
[Testcase 251]: L2 7800; PVBand 14640; EPE 0; Shots: 394
[Testcase 252]: L2 7008; PVBand 11057; EPE 0; Shots: 443
[Testcase 253]: L2 4674; PVBand 7284; EPE 0; Shots: 337
[Testcase 254]: L2 8971; PVBand 18394; EPE 0; Shots: 380
[Testcase 255]: L2 16684; PVBand 28771; EPE 0; Shots: 557
[Testcase 256]: L2 4677; PVBand 6530; EPE 0; Shots: 413
[Testcase 257]: L2 46273; PVBand 80826; EPE 2; Shots: 432
[Testcase 258]: L2 14868; PVBand 31229; EPE 0; Shots: 538
[Testcase 259]: L2 17223; PVBand 46043; EPE 0; Shots: 439
[Testcase 260]: L2 6282; PVBand 8890; EPE 0; Shots: 407
[Testcase 261]: L2 41943; PVBand 61066; EPE 0; Shots: 543
[Testcase 262]: L2 18141; PVBand 31155; EPE 0; Shots: 593
[Testcase 263]: L2 6568; PVBand 16758; EPE 0; Shots: 470
[Testcase 264]: L2 10356; PVBand 22374; EPE 0; Shots: 476
[Testcase 265]: L2 5374; PVBand 8765; EPE 0; Shots: 285
[Testcase 266]: L2 21441; PVBand 39833; EPE 0; Shots: 540
[Testcase 267]: L2 11294; PVBand 17899; EPE 0; Shots: 447
[Testcase 268]: L2 5398; PVBand 7503; EPE 0; Shots: 327
[Testcase 269]: L2 6084; PVBand 11715; EPE 0; Shots: 396
[Testcase 270]: L2 11125; PVBand 23935; EPE 0; Shots: 452
[Testcase 271]: L2 5520; PVBand 8492; EPE 0; Shots: 365
[Finetuned]: L2 12841; PVBand 24859; EPE 0.0; Shots: 441

[StdContact]
[Evaluation]: Getting masks from the model
[Testcase 1]: L2 89796; PVBand 10709; EPE 79
[Testcase 2]: L2 91476; PVBand 987; EPE 84
[Testcase 3]: L2 66941; PVBand 4006; EPE 60
[Testcase 4]: L2 78408; PVBand 0; EPE 72
[Testcase 5]: L2 78374; PVBand 4688; EPE 70
[Testcase 6]: L2 91476; PVBand 0; EPE 84
[Testcase 7]: L2 65834; PVBand 2617; EPE 61
[Testcase 8]: L2 91388; PVBand 4241; EPE 83
[Testcase 9]: L2 113256; PVBand 0; EPE 104
[Testcase 10]: L2 63515; PVBand 11080; EPE 53
[Testcase 11]: L2 82121; PVBand 1424; EPE 75
[Testcase 12]: L2 81878; PVBand 4107; EPE 73
[Testcase 13]: L2 113622; PVBand 2597; EPE 105
[Testcase 14]: L2 82764; PVBand 436; EPE 76
[Testcase 15]: L2 70266; PVBand 11677; EPE 58
[Testcase 16]: L2 81753; PVBand 8609; EPE 73
[Testcase 17]: L2 95929; PVBand 6804; EPE 86
[Testcase 18]: L2 96000; PVBand 2083; EPE 89
[Testcase 19]: L2 68922; PVBand 2949; EPE 63
[Testcase 20]: L2 64106; PVBand 2281; EPE 59
[Testcase 21]: L2 81954; PVBand 2825; EPE 76
[Testcase 22]: L2 75340; PVBand 11167; EPE 67
[Testcase 23]: L2 76614; PVBand 2513; EPE 70
[Testcase 24]: L2 81606; PVBand 3725; EPE 74
[Testcase 25]: L2 86364; PVBand 14693; EPE 72
[Testcase 26]: L2 69284; PVBand 9065; EPE 62
[Testcase 27]: L2 69696; PVBand 366; EPE 64
[Testcase 28]: L2 79653; PVBand 4767; EPE 72
[Testcase 29]: L2 73303; PVBand 24381; EPE 61
[Testcase 30]: L2 62424; PVBand 8154; EPE 54
[Testcase 31]: L2 69573; PVBand 14687; EPE 59
[Testcase 32]: L2 93024; PVBand 6156; EPE 84
[Testcase 33]: L2 74052; PVBand 0; EPE 68
[Testcase 34]: L2 50295; PVBand 2541; EPE 46
[Testcase 35]: L2 129327; PVBand 4197; EPE 118
[Testcase 36]: L2 126324; PVBand 469; EPE 116
[Testcase 37]: L2 91476; PVBand 0; EPE 84
[Testcase 38]: L2 80463; PVBand 11126; EPE 71
[Testcase 39]: L2 76602; PVBand 3117; EPE 69
[Testcase 40]: L2 69386; PVBand 1104; EPE 64
[Testcase 41]: L2 67705; PVBand 2951; EPE 61
[Testcase 42]: L2 81320; PVBand 3508; EPE 74
[Testcase 43]: L2 65569; PVBand 9788; EPE 59
[Testcase 44]: L2 108900; PVBand 0; EPE 100
[Testcase 45]: L2 74052; PVBand 0; EPE 68
[Testcase 46]: L2 76538; PVBand 4317; EPE 70
[Testcase 47]: L2 64685; PVBand 3291; EPE 60
[Testcase 48]: L2 74110; PVBand 3238; EPE 68
[Testcase 49]: L2 69733; PVBand 10455; EPE 59
[Testcase 50]: L2 81731; PVBand 1945; EPE 74
[Testcase 51]: L2 85585; PVBand 9581; EPE 74
[Testcase 52]: L2 56119; PVBand 2043; EPE 52
[Testcase 53]: L2 60583; PVBand 1802; EPE 55
[Testcase 54]: L2 70445; PVBand 11827; EPE 60
[Testcase 55]: L2 82764; PVBand 148; EPE 76
[Testcase 56]: L2 100188; PVBand 380; EPE 92
[Testcase 57]: L2 64832; PVBand 11067; EPE 55
[Testcase 58]: L2 72159; PVBand 4816; EPE 65
[Testcase 59]: L2 113256; PVBand 0; EPE 104
[Testcase 60]: L2 73639; PVBand 1339; EPE 68
[Testcase 61]: L2 110122; PVBand 5267; EPE 99
[Testcase 62]: L2 108064; PVBand 4315; EPE 99
[Testcase 63]: L2 55968; PVBand 2626; EPE 50
[Testcase 64]: L2 82764; PVBand 0; EPE 76
[Testcase 65]: L2 53285; PVBand 6028; EPE 47
[Testcase 66]: L2 81891; PVBand 2125; EPE 75
[Testcase 67]: L2 94484; PVBand 2477; EPE 86
[Testcase 68]: L2 85980; PVBand 2761; EPE 77
[Testcase 69]: L2 80823; PVBand 4450; EPE 72
[Testcase 70]: L2 56104; PVBand 4391; EPE 49
[Testcase 71]: L2 72970; PVBand 10795; EPE 63
[Testcase 72]: L2 68461; PVBand 2068; EPE 62
[Testcase 73]: L2 108270; PVBand 3977; EPE 98
[Testcase 74]: L2 84609; PVBand 13990; EPE 73
[Testcase 75]: L2 84516; PVBand 5513; EPE 76
[Testcase 76]: L2 82764; PVBand 0; EPE 76
[Testcase 77]: L2 60144; PVBand 7973; EPE 51
[Testcase 78]: L2 100041; PVBand 1659; EPE 92
[Testcase 79]: L2 86272; PVBand 7761; EPE 77
[Testcase 80]: L2 94931; PVBand 2453; EPE 86
[Testcase 81]: L2 98926; PVBand 3788; EPE 89
[Testcase 82]: L2 91476; PVBand 0; EPE 84
[Testcase 83]: L2 91476; PVBand 82; EPE 84
[Testcase 84]: L2 71545; PVBand 4288; EPE 65
[Testcase 85]: L2 78773; PVBand 17737; EPE 66
[Testcase 86]: L2 81021; PVBand 3367; EPE 73
[Testcase 87]: L2 104544; PVBand 0; EPE 96
[Testcase 88]: L2 49948; PVBand 3362; EPE 45
[Testcase 89]: L2 82764; PVBand 0; EPE 76
[Testcase 90]: L2 99623; PVBand 1436; EPE 91
[Testcase 91]: L2 87120; PVBand 0; EPE 80
[Testcase 92]: L2 78415; PVBand 8202; EPE 70
[Testcase 93]: L2 69363; PVBand 1886; EPE 63
[Testcase 94]: L2 100188; PVBand 0; EPE 92
[Testcase 95]: L2 73609; PVBand 2854; EPE 67
[Testcase 96]: L2 85313; PVBand 3870; EPE 77
[Testcase 97]: L2 78444; PVBand 3684; EPE 72
[Testcase 98]: L2 81708; PVBand 9251; EPE 71
[Testcase 99]: L2 79067; PVBand 12250; EPE 71
[Testcase 100]: L2 76756; PVBand 2379; EPE 69
[Testcase 101]: L2 60984; PVBand 0; EPE 56
[Testcase 102]: L2 92535; PVBand 6635; EPE 82
[Testcase 103]: L2 117612; PVBand 64; EPE 108
[Testcase 104]: L2 100060; PVBand 900; EPE 92
[Testcase 105]: L2 113256; PVBand 296; EPE 104
[Testcase 106]: L2 68539; PVBand 7251; EPE 59
[Testcase 107]: L2 65810; PVBand 14502; EPE 55
[Testcase 108]: L2 82509; PVBand 22193; EPE 69
[Testcase 109]: L2 111256; PVBand 2784; EPE 100
[Testcase 110]: L2 53850; PVBand 4637; EPE 46
[Testcase 111]: L2 120029; PVBand 5698; EPE 109
[Testcase 112]: L2 104442; PVBand 866; EPE 95
[Testcase 113]: L2 72166; PVBand 19964; EPE 60
[Testcase 114]: L2 58786; PVBand 4422; EPE 53
[Testcase 115]: L2 59388; PVBand 8579; EPE 50
[Testcase 116]: L2 94809; PVBand 8221; EPE 85
[Testcase 117]: L2 69696; PVBand 0; EPE 64
[Testcase 118]: L2 81608; PVBand 3180; EPE 74
[Testcase 119]: L2 104544; PVBand 960; EPE 96
[Testcase 120]: L2 60984; PVBand 0; EPE 56
[Testcase 121]: L2 130663; PVBand 840; EPE 120
[Testcase 122]: L2 64244; PVBand 2850; EPE 58
[Testcase 123]: L2 63262; PVBand 4784; EPE 56
[Testcase 124]: L2 68938; PVBand 3101; EPE 63
[Testcase 125]: L2 87190; PVBand 4769; EPE 79
[Testcase 126]: L2 64324; PVBand 1766; EPE 58
[Testcase 127]: L2 108900; PVBand 0; EPE 100
[Testcase 128]: L2 52085; PVBand 1033; EPE 48
[Testcase 129]: L2 103235; PVBand 10531; EPE 93
[Testcase 130]: L2 76775; PVBand 17017; EPE 62
[Testcase 131]: L2 92316; PVBand 12781; EPE 84
[Testcase 132]: L2 76840; PVBand 2441; EPE 70
[Testcase 133]: L2 74668; PVBand 5025; EPE 66
[Testcase 134]: L2 60669; PVBand 12839; EPE 53
[Testcase 135]: L2 81950; PVBand 3597; EPE 75
[Testcase 136]: L2 77334; PVBand 1994; EPE 70
[Testcase 137]: L2 74114; PVBand 12439; EPE 65
[Testcase 138]: L2 59820; PVBand 2114; EPE 55
[Testcase 139]: L2 65527; PVBand 7216; EPE 58
[Testcase 140]: L2 70285; PVBand 7640; EPE 63
[Testcase 141]: L2 121937; PVBand 3098; EPE 110
[Testcase 142]: L2 72114; PVBand 16372; EPE 60
[Testcase 143]: L2 55253; PVBand 2598; EPE 50
[Testcase 144]: L2 83498; PVBand 6351; EPE 73
[Testcase 145]: L2 78408; PVBand 0; EPE 72
[Testcase 146]: L2 85894; PVBand 3131; EPE 78
[Testcase 147]: L2 126466; PVBand 7008; EPE 115
[Testcase 148]: L2 91045; PVBand 8470; EPE 82
[Testcase 149]: L2 65340; PVBand 0; EPE 60
[Testcase 150]: L2 64033; PVBand 2467; EPE 58
[Testcase 151]: L2 69696; PVBand 501; EPE 64
[Testcase 152]: L2 101465; PVBand 2749; EPE 93
[Testcase 153]: L2 84092; PVBand 4276; EPE 74
[Testcase 154]: L2 65340; PVBand 0; EPE 60
[Testcase 155]: L2 76383; PVBand 3111; EPE 68
[Testcase 156]: L2 78297; PVBand 15543; EPE 68
[Testcase 157]: L2 91447; PVBand 992; EPE 84
[Testcase 158]: L2 91013; PVBand 3306; EPE 83
[Testcase 159]: L2 69696; PVBand 118; EPE 64
[Testcase 160]: L2 68772; PVBand 3100; EPE 63
[Testcase 161]: L2 64140; PVBand 15652; EPE 53
[Testcase 162]: L2 105211; PVBand 2103; EPE 96
[Testcase 163]: L2 82764; PVBand 0; EPE 76
[Testcase 164]: L2 79426; PVBand 12354; EPE 68
[Testcase 165]: L2 68626; PVBand 16549; EPE 58
[Initialized]: L2 81378; PVBand 4931; EPE 73.2
[Evaluation]: Finetuning the masks with pixel-based ILT method
[Testcase 1]: L2 38013; PVBand 50931; EPE 10
[Testcase 2]: L2 31520; PVBand 44915; EPE 4
[Testcase 3]: L2 19353; PVBand 35519; EPE 0
[Testcase 4]: L2 26856; PVBand 37937; EPE 5
[Testcase 5]: L2 27683; PVBand 41416; EPE 4
[Testcase 6]: L2 29149; PVBand 48232; EPE 1
[Testcase 7]: L2 17962; PVBand 32857; EPE 0
[Testcase 8]: L2 29519; PVBand 49473; EPE 0
[Testcase 9]: L2 48550; PVBand 55705; EPE 18
[Testcase 10]: L2 24692; PVBand 34099; EPE 6
[Testcase 11]: L2 38072; PVBand 37372; EPE 17
[Testcase 12]: L2 37222; PVBand 37225; EPE 15
[Testcase 13]: L2 49534; PVBand 53487; EPE 19
[Testcase 14]: L2 27452; PVBand 38332; EPE 4
[Testcase 15]: L2 25063; PVBand 36827; EPE 4
[Testcase 16]: L2 41116; PVBand 40194; EPE 17
[Testcase 17]: L2 43613; PVBand 51047; EPE 14
[Testcase 18]: L2 43813; PVBand 44555; EPE 16
[Testcase 19]: L2 22597; PVBand 33315; EPE 4
[Testcase 20]: L2 17165; PVBand 32997; EPE 0
[Testcase 21]: L2 30902; PVBand 39700; EPE 9
[Testcase 22]: L2 25139; PVBand 37997; EPE 4
[Testcase 23]: L2 24344; PVBand 42664; EPE 2
[Testcase 24]: L2 30877; PVBand 37775; EPE 8
[Testcase 25]: L2 26114; PVBand 46119; EPE 1
[Testcase 26]: L2 23438; PVBand 35045; EPE 4
[Testcase 27]: L2 24156; PVBand 34538; EPE 5
[Testcase 28]: L2 30418; PVBand 38493; EPE 9
[Testcase 29]: L2 30926; PVBand 44150; EPE 7
[Testcase 30]: L2 20578; PVBand 29580; EPE 4
[Testcase 31]: L2 26140; PVBand 36405; EPE 7
[Testcase 32]: L2 36309; PVBand 50070; EPE 8
[Testcase 33]: L2 24440; PVBand 35845; EPE 5
[Testcase 34]: L2 14757; PVBand 27245; EPE 0
[Testcase 35]: L2 61578; PVBand 64856; EPE 30
[Testcase 36]: L2 56237; PVBand 58356; EPE 23
[Testcase 37]: L2 37464; PVBand 42617; EPE 12
[Testcase 38]: L2 24513; PVBand 41928; EPE 0
[Testcase 39]: L2 24999; PVBand 38090; EPE 4
[Testcase 40]: L2 24767; PVBand 38614; EPE 5
[Testcase 41]: L2 27216; PVBand 31029; EPE 8
[Testcase 42]: L2 35086; PVBand 40288; EPE 13
[Testcase 43]: L2 20325; PVBand 36908; EPE 2
[Testcase 44]: L2 45529; PVBand 52198; EPE 18
[Testcase 45]: L2 20775; PVBand 38958; EPE 0
[Testcase 46]: L2 27405; PVBand 37947; EPE 6
[Testcase 47]: L2 17317; PVBand 32045; EPE 0
[Testcase 48]: L2 20470; PVBand 37448; EPE 0
[Testcase 49]: L2 23015; PVBand 39588; EPE 2
[Testcase 50]: L2 26700; PVBand 37836; EPE 4
[Testcase 51]: L2 23528; PVBand 46868; EPE 0
[Testcase 52]: L2 20768; PVBand 24343; EPE 8
[Testcase 53]: L2 17086; PVBand 30631; EPE 0
[Testcase 54]: L2 24600; PVBand 42013; EPE 2
[Testcase 55]: L2 24921; PVBand 42840; EPE 1
[Testcase 56]: L2 33964; PVBand 48146; EPE 6
[Testcase 57]: L2 28582; PVBand 27647; EPE 12
[Testcase 58]: L2 21133; PVBand 38142; EPE 0
[Testcase 59]: L2 44034; PVBand 54970; EPE 13
[Testcase 60]: L2 31763; PVBand 31528; EPE 12
[Testcase 61]: L2 57251; PVBand 50618; EPE 28
[Testcase 62]: L2 42981; PVBand 57714; EPE 11
[Testcase 63]: L2 16011; PVBand 29469; EPE 0
[Testcase 64]: L2 26169; PVBand 38078; EPE 4
[Testcase 65]: L2 16443; PVBand 28817; EPE 0
[Testcase 66]: L2 37542; PVBand 34029; EPE 17
[Testcase 67]: L2 37935; PVBand 50822; EPE 9
[Testcase 68]: L2 41689; PVBand 36928; EPE 19
[Testcase 69]: L2 31816; PVBand 42318; EPE 8
[Testcase 70]: L2 17744; PVBand 30710; EPE 0
[Testcase 71]: L2 29282; PVBand 40729; EPE 7
[Testcase 72]: L2 23290; PVBand 32206; EPE 4
[Testcase 73]: L2 47659; PVBand 55758; EPE 16
[Testcase 74]: L2 24193; PVBand 46687; EPE 0
[Testcase 75]: L2 30269; PVBand 47891; EPE 5
[Testcase 76]: L2 29386; PVBand 42504; EPE 5
[Testcase 77]: L2 21488; PVBand 30542; EPE 4
[Testcase 78]: L2 40179; PVBand 45887; EPE 12
[Testcase 79]: L2 34343; PVBand 46059; EPE 9
[Testcase 80]: L2 34708; PVBand 48076; EPE 8
[Testcase 81]: L2 48276; PVBand 42139; EPE 24
[Testcase 82]: L2 30714; PVBand 44131; EPE 6
[Testcase 83]: L2 35735; PVBand 45708; EPE 9
[Testcase 84]: L2 27263; PVBand 34036; EPE 8
[Testcase 85]: L2 27323; PVBand 46577; EPE 3
[Testcase 86]: L2 33370; PVBand 39416; EPE 11
[Testcase 87]: L2 39706; PVBand 59061; EPE 7
[Testcase 88]: L2 13733; PVBand 25137; EPE 0
[Testcase 89]: L2 30733; PVBand 39708; EPE 8
[Testcase 90]: L2 36744; PVBand 48593; EPE 9
[Testcase 91]: L2 30875; PVBand 43868; EPE 5
[Testcase 92]: L2 28241; PVBand 44759; EPE 3
[Testcase 93]: L2 19887; PVBand 36102; EPE 0
[Testcase 94]: L2 42843; PVBand 46348; EPE 15
[Testcase 95]: L2 21963; PVBand 39115; EPE 0
[Testcase 96]: L2 38566; PVBand 41274; EPE 13
[Testcase 97]: L2 30788; PVBand 33541; EPE 13
[Testcase 98]: L2 28015; PVBand 45462; EPE 5
[Testcase 99]: L2 24111; PVBand 42492; EPE 0
[Testcase 100]: L2 24542; PVBand 38362; EPE 4
[Testcase 101]: L2 15939; PVBand 28798; EPE 0
[Testcase 102]: L2 37204; PVBand 46762; EPE 12
[Testcase 103]: L2 51718; PVBand 57227; EPE 19
[Testcase 104]: L2 44907; PVBand 40754; EPE 20
[Testcase 105]: L2 45137; PVBand 55565; EPE 15
[Testcase 106]: L2 25562; PVBand 39635; EPE 6
[Testcase 107]: L2 22955; PVBand 37063; EPE 3
[Testcase 108]: L2 37601; PVBand 50508; EPE 10
[Testcase 109]: L2 40009; PVBand 60118; EPE 4
[Testcase 110]: L2 15149; PVBand 28560; EPE 0
[Testcase 111]: L2 49875; PVBand 61012; EPE 15
[Testcase 112]: L2 31099; PVBand 53787; EPE 2
[Testcase 113]: L2 25986; PVBand 42110; EPE 3
[Testcase 114]: L2 16810; PVBand 31068; EPE 0
[Testcase 115]: L2 17901; PVBand 33245; EPE 0
[Testcase 116]: L2 37295; PVBand 48339; EPE 8
[Testcase 117]: L2 21916; PVBand 32518; EPE 4
[Testcase 118]: L2 34583; PVBand 40004; EPE 11
[Testcase 119]: L2 46788; PVBand 53101; EPE 18
[Testcase 120]: L2 19490; PVBand 28339; EPE 4
[Testcase 121]: L2 61393; PVBand 55896; EPE 29
[Testcase 122]: L2 18752; PVBand 33552; EPE 0
[Testcase 123]: L2 17719; PVBand 33841; EPE 0
[Testcase 124]: L2 23904; PVBand 33684; EPE 4
[Testcase 125]: L2 31347; PVBand 47426; EPE 5
[Testcase 126]: L2 24169; PVBand 27685; EPE 8
[Testcase 127]: L2 40075; PVBand 57542; EPE 9
[Testcase 128]: L2 13047; PVBand 27692; EPE 0
[Testcase 129]: L2 44698; PVBand 52384; EPE 15
[Testcase 130]: L2 24166; PVBand 44164; EPE 1
[Testcase 131]: L2 42452; PVBand 48585; EPE 17
[Testcase 132]: L2 25484; PVBand 38929; EPE 4
[Testcase 133]: L2 20676; PVBand 39747; EPE 0
[Testcase 134]: L2 20152; PVBand 34101; EPE 1
[Testcase 135]: L2 31821; PVBand 42412; EPE 8
[Testcase 136]: L2 22808; PVBand 40831; EPE 0
[Testcase 137]: L2 25796; PVBand 41619; EPE 1
[Testcase 138]: L2 17519; PVBand 32261; EPE 0
[Testcase 139]: L2 22405; PVBand 33826; EPE 4
[Testcase 140]: L2 26725; PVBand 33280; EPE 8
[Testcase 141]: L2 53352; PVBand 63921; EPE 21
[Testcase 142]: L2 22799; PVBand 42196; EPE 0
[Testcase 143]: L2 24702; PVBand 21902; EPE 12
[Testcase 144]: L2 27777; PVBand 43175; EPE 4
[Testcase 145]: L2 22673; PVBand 40200; EPE 0
[Testcase 146]: L2 39433; PVBand 40642; EPE 17
[Testcase 147]: L2 54423; PVBand 65694; EPE 18
[Testcase 148]: L2 31981; PVBand 49359; EPE 5
[Testcase 149]: L2 18323; PVBand 33853; EPE 0
[Testcase 150]: L2 17024; PVBand 31917; EPE 0
[Testcase 151]: L2 21729; PVBand 36847; EPE 4
[Testcase 152]: L2 36070; PVBand 50671; EPE 8
[Testcase 153]: L2 26712; PVBand 42255; EPE 4
[Testcase 154]: L2 17877; PVBand 33550; EPE 0
[Testcase 155]: L2 29606; PVBand 37031; EPE 10
[Testcase 156]: L2 26106; PVBand 40048; EPE 4
[Testcase 157]: L2 27005; PVBand 46661; EPE 1
[Testcase 158]: L2 44314; PVBand 43010; EPE 20
[Testcase 159]: L2 23643; PVBand 33608; EPE 4
[Testcase 160]: L2 19726; PVBand 36347; EPE 0
[Testcase 161]: L2 21124; PVBand 37004; EPE 0
[Testcase 162]: L2 41637; PVBand 55312; EPE 14
[Testcase 163]: L2 25282; PVBand 42893; EPE 0
[Testcase 164]: L2 32090; PVBand 43070; EPE 8
[Testcase 165]: L2 25179; PVBand 42247; EPE 3
[Finetuned]: L2 29992; PVBand 41339; EPE 6.9
'''