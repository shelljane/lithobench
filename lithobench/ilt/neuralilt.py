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



class UNet(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = repeat2d(2, 1,   64,   kernel_size=3, stride=1, padding=1, relu=True)
        self.conv2 = repeat2d(2, 64,  128,  kernel_size=3, stride=1, padding=1, relu=True)
        self.conv3 = repeat2d(2, 128, 256,  kernel_size=3, stride=1, padding=1, relu=True)
        self.conv4 = repeat2d(2, 256, 512,  kernel_size=3, stride=1, padding=1, relu=True)
        self.deconv4 = repeat2d(2, 256+512, 256,  kernel_size=3, stride=1, padding=1, relu=True)
        self.deconv3 = repeat2d(2, 128+256, 128,  kernel_size=3, stride=1, padding=1, relu=True)
        self.deconv2 = repeat2d(2, 64+128,  64,   kernel_size=3, stride=1, padding=1, relu=True)
        self.deconv1 = conv2d(64, 1, kernel_size=3, stride=1, padding=1, norm=False, relu=False)

    def forward(self, x): 
        conv1 = self.conv1(x)
        x = self.pool(conv1)

        conv2 = self.conv2(x)
        x = self.pool(conv2)

        conv3 = self.conv3(x)
        x = self.pool(conv3)

        x = self.conv4(x)
        x = self.upscale(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.deconv4(x)
        x = self.upscale(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.deconv3(x)
        x = self.upscale(x)
        x = torch.cat([x, conv1], dim=1)
        
        x = self.deconv2(x)
        x = self.deconv1(x)
        x = self.sigmoid(x)

        return x


class NeuralILT(ModelILT): 
    def __init__(self, size=(512, 512)): 
        super().__init__(size=size, name="NeuralILT")
        self.simLitho = litho.LithoSim("./config/lithosimple.txt")
        self.net = UNet()
        if torch.cuda.is_available():
            self.net = self.net.cuda()
        self.net = nn.DataParallel(self.net)
    
    @property
    def size(self): 
        return self._size
    @property
    def name(self): 
        return self._name

    def pretrain(self, train_loader, val_loader, epochs=1): 
        opt = optim.Adam(self.net.parameters(), lr=1e-3)
        sched = lr_sched.StepLR(opt, 1, gamma=0.1)
        for epoch in range(epochs): 
            print(f"[Pre-Epoch {epoch}] Training")
            self.net.train()
            progress = tqdm(train_loader)
            for target, label in progress: 
                if torch.cuda.is_available():
                    target = target.cuda()
                    label = label.cuda()
                
                mask = self.net(target)
                loss = F.mse_loss(mask, label)
                
                opt.zero_grad()
                loss.backward()
                opt.step()

                progress.set_postfix(loss=loss.item())

            print(f"[Pre-Epoch {epoch}] Testing")
            self.net.eval()
            losses = []
            progress = tqdm(val_loader)
            for target, label in progress: 
                with torch.no_grad():
                    if torch.cuda.is_available():
                        target = target.cuda()
                        label = label.cuda()
                    
                    mask = self.net(target)
                    loss = F.mse_loss(mask, label)
                    losses.append(loss.item())

                    progress.set_postfix(loss=loss.item())
            
            print(f"[Pre-Epoch {epoch}] loss = {np.mean(losses)}")

            if epoch == epochs//2: 
                sched.step()

    def train(self, train_loader, val_loader, epochs=1): 
        opt = optim.Adam(self.net.parameters(), lr=1e-3)
        sched = lr_sched.StepLR(opt, 1, gamma=0.1)
        for epoch in range(epochs): 
            print(f"[Epoch {epoch}] Training")
            self.net.train()
            progress = tqdm(train_loader)
            for target, label in progress: 
                if torch.cuda.is_available():
                    target = target.cuda()
                    label = label.cuda()
                
                mask = self.net(target)
                mask = mask.squeeze(1)
                printedNom, printedMax, printedMin = self.simLitho(mask)
                l2loss = F.mse_loss(printedNom.unsqueeze(1), target)
                cpxloss = F.mse_loss(printedMax, printedMin)
                loss = l2loss + cpxloss
                
                opt.zero_grad()
                loss.backward()
                opt.step()

                progress.set_postfix(l2loss=l2loss.item(), cpxloss=cpxloss.item())

            print(f"[Epoch {epoch}] Testing")
            self.net.eval()
            l2losses = []
            cpxlosses = []
            progress = tqdm(val_loader)
            for target, label in progress: 
                with torch.no_grad():
                    if torch.cuda.is_available():
                        target = target.cuda()
                        label = label.cuda()
                    
                    mask = self.net(target)
                    mask = mask.squeeze(1)
                    printedNom, printedMax, printedMin = self.simLitho(mask)
                    l2loss = F.mse_loss(printedNom.unsqueeze(1), target)
                    cpxloss = F.mse_loss(printedMax, printedMin)
                    loss = l2loss + cpxloss
                    l2losses.append(l2loss.item())
                    cpxlosses.append(cpxloss.item())

                    progress.set_postfix(l2loss=l2loss.item(), cpxloss=cpxloss.item())
            
            print(f"[Epoch {epoch}] L2 loss = {np.mean(l2losses)}, cpxlosses = {np.mean(cpxlosses)}")

            if epoch == epochs//2: 
                sched.step()
    
    def save(self, filenames): 
        filename = filenames[0] if isinstance(filenames, list) else filenames
        torch.save(self.net.module.state_dict(), filename)
    
    def load(self, filenames): 
        filename = filenames[0] if isinstance(filenames, list) else filenames
        self.net.module.load_state_dict(torch.load(filename))

    def run(self, target): 
        self.net.eval()
        return self.net(target)[0, 0].detach()


if __name__ == "__main__": 
    Benchmark = "MetalSet"
    ImageSize = (512, 512)
    Epochs = 1
    BatchSize = 12
    NJobs = 8
    TrainOnly = False
    EvalOnly = False
    train_loader, val_loader = loadersILT(Benchmark, ImageSize, BatchSize, NJobs)
    targets = evaluate.getTargets(samples=None, dataset=Benchmark)
    ilt = NeuralILT(size=ImageSize)
    
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
        ilt.save("trivial/neuralilt/pretrain.pth")
    else: 
        ilt.load("trivial/neuralilt/pretrain.pth")
    if not EvalOnly: 
        ilt.train(train_loader, val_loader, epochs=Epochs)
        ilt.save("trivial/neuralilt/train.pth")
    else: 
        ilt.load("trivial/neuralilt/train.pth")
    ilt.evaluate(targets, finetune=False, folder="trivial/neuralilt")

'''
[MetalSet]
[Evaluation]: Getting masks from the model
[Testcase 1]: L2 46679; PVBand 48628; EPE 8; Shots: 483
[Testcase 2]: L2 40290; PVBand 39955; EPE 8; Shots: 392
[Testcase 3]: L2 82887; PVBand 84912; EPE 42; Shots: 520
[Testcase 4]: L2 15932; PVBand 22656; EPE 1; Shots: 327
[Testcase 5]: L2 41951; PVBand 50522; EPE 2; Shots: 581
[Testcase 6]: L2 38532; PVBand 45760; EPE 4; Shots: 658
[Testcase 7]: L2 21320; PVBand 38694; EPE 0; Shots: 494
[Testcase 8]: L2 15795; PVBand 21310; EPE 0; Shots: 424
[Testcase 9]: L2 48727; PVBand 56775; EPE 4; Shots: 618
[Testcase 10]: L2 14583; PVBand 17444; EPE 4; Shots: 261
[Initialized]: L2 36670; PVBand 42666; EPE 7.3; Runtime: 0.37s; Shots: 476
[Evaluation]: Finetuning the masks with pixel-based ILT method
[Testcase 1]: L2 39354; PVBand 48861; EPE 3; Shots: 587
[Testcase 2]: L2 30665; PVBand 39079; EPE 0; Shots: 563
[Testcase 3]: L2 67103; PVBand 71522; EPE 23; Shots: 654
[Testcase 4]: L2 8662; PVBand 23822; EPE 0; Shots: 479
[Testcase 5]: L2 29636; PVBand 53944; EPE 0; Shots: 558
[Testcase 6]: L2 30180; PVBand 48644; EPE 0; Shots: 608
[Testcase 7]: L2 15490; PVBand 41411; EPE 0; Shots: 522
[Testcase 8]: L2 11161; PVBand 21251; EPE 0; Shots: 522
[Testcase 9]: L2 34322; PVBand 62163; EPE 0; Shots: 614
[Testcase 10]: L2 7497; PVBand 16941; EPE 0; Shots: 367
[Finetuned]: L2 27407; PVBand 42764; EPE 2.6; Shots: 547

[ViaSet]
[Evaluation]: Getting masks from the model
[Testcase 1]: L2 3892; PVBand 4624; EPE 0; Shots: 126
[Testcase 2]: L2 8781; PVBand 2829; EPE 7; Shots: 156
[Testcase 3]: L2 12599; PVBand 7673; EPE 7; Shots: 249
[Testcase 4]: L2 6808; PVBand 7379; EPE 0; Shots: 140
[Testcase 5]: L2 14715; PVBand 12434; EPE 7; Shots: 338
[Testcase 6]: L2 14629; PVBand 9952; EPE 8; Shots: 333
[Testcase 7]: L2 11570; PVBand 5685; EPE 5; Shots: 216
[Testcase 8]: L2 27923; PVBand 19132; EPE 13; Shots: 526
[Testcase 9]: L2 18875; PVBand 11452; EPE 11; Shots: 384
[Testcase 10]: L2 7440; PVBand 4207; EPE 4; Shots: 164
[Initialized]: L2 12723; PVBand 8537; EPE 6.2; Runtime: 0.44s; Shots: 263
[Evaluation]: Finetuning the masks with pixel-based ILT method
[Testcase 1]: L2 2715; PVBand 4567; EPE 0; Shots: 154
[Testcase 2]: L2 2545; PVBand 4493; EPE 0; Shots: 158
[Testcase 3]: L2 4303; PVBand 8467; EPE 0; Shots: 340
[Testcase 4]: L2 3186; PVBand 6214; EPE 0; Shots: 183
[Testcase 5]: L2 6755; PVBand 12480; EPE 0; Shots: 444
[Testcase 6]: L2 5483; PVBand 10597; EPE 0; Shots: 443
[Testcase 7]: L2 3655; PVBand 6619; EPE 0; Shots: 245
[Testcase 8]: L2 10515; PVBand 20837; EPE 0; Shots: 573
[Testcase 9]: L2 8244; PVBand 14121; EPE 1; Shots: 411
[Testcase 10]: L2 3907; PVBand 5033; EPE 1; Shots: 137
[Finetuned]: L2 5131; PVBand 9343; EPE 0.2; Shots: 309

[StdMetal]
[Evaluation]: Getting masks from the model
[Testcase 1]: L2 14016; PVBand 19965; EPE 0
[Testcase 2]: L2 5764; PVBand 6748; EPE 0
[Testcase 3]: L2 10108; PVBand 11747; EPE 0
[Testcase 4]: L2 11718; PVBand 16312; EPE 0
[Testcase 5]: L2 4966; PVBand 5914; EPE 0
[Testcase 6]: L2 3555; PVBand 6250; EPE 0
[Testcase 7]: L2 12555; PVBand 12606; EPE 0
[Testcase 8]: L2 4404; PVBand 7558; EPE 0
[Testcase 9]: L2 14510; PVBand 18357; EPE 0
[Testcase 10]: L2 15807; PVBand 26871; EPE 0
[Testcase 11]: L2 24573; PVBand 30518; EPE 0
[Testcase 12]: L2 4636; PVBand 6502; EPE 0
[Testcase 13]: L2 33076; PVBand 43393; EPE 0
[Testcase 14]: L2 16839; PVBand 22415; EPE 0
[Testcase 15]: L2 20389; PVBand 23611; EPE 0
[Testcase 16]: L2 14745; PVBand 14467; EPE 1
[Testcase 17]: L2 14166; PVBand 17710; EPE 0
[Testcase 18]: L2 21454; PVBand 29034; EPE 2
[Testcase 19]: L2 4230; PVBand 9317; EPE 0
[Testcase 20]: L2 9820; PVBand 13151; EPE 0
[Testcase 21]: L2 29380; PVBand 46168; EPE 1
[Testcase 22]: L2 17591; PVBand 22097; EPE 1
[Testcase 23]: L2 55168; PVBand 36639; EPE 34
[Testcase 24]: L2 13187; PVBand 18874; EPE 0
[Testcase 25]: L2 11637; PVBand 14353; EPE 0
[Testcase 26]: L2 14654; PVBand 19542; EPE 0
[Testcase 27]: L2 19029; PVBand 20858; EPE 1
[Testcase 28]: L2 2936; PVBand 7298; EPE 0
[Testcase 29]: L2 44316; PVBand 50496; EPE 0
[Testcase 30]: L2 37921; PVBand 46349; EPE 4
[Testcase 31]: L2 51044; PVBand 36183; EPE 29
[Testcase 32]: L2 6546; PVBand 8862; EPE 0
[Testcase 33]: L2 8523; PVBand 10987; EPE 0
[Testcase 34]: L2 7169; PVBand 7853; EPE 0
[Testcase 35]: L2 12650; PVBand 19534; EPE 0
[Testcase 36]: L2 50953; PVBand 62085; EPE 12
[Testcase 37]: L2 15588; PVBand 17830; EPE 0
[Testcase 38]: L2 95290; PVBand 71408; EPE 31
[Testcase 39]: L2 6234; PVBand 7988; EPE 0
[Testcase 40]: L2 11370; PVBand 14452; EPE 1
[Testcase 41]: L2 70040; PVBand 81368; EPE 5
[Testcase 42]: L2 8821; PVBand 11092; EPE 0
[Testcase 43]: L2 29784; PVBand 38101; EPE 3
[Testcase 44]: L2 81878; PVBand 64995; EPE 27
[Testcase 45]: L2 18234; PVBand 22752; EPE 0
[Testcase 46]: L2 17150; PVBand 21709; EPE 2
[Testcase 47]: L2 18730; PVBand 22632; EPE 2
[Testcase 48]: L2 11821; PVBand 14683; EPE 1
[Testcase 49]: L2 27406; PVBand 43843; EPE 0
[Testcase 50]: L2 17361; PVBand 19128; EPE 2
[Testcase 51]: L2 4655; PVBand 6307; EPE 0
[Testcase 52]: L2 3920; PVBand 6610; EPE 0
[Testcase 53]: L2 51725; PVBand 31964; EPE 30
[Testcase 54]: L2 22434; PVBand 24499; EPE 2
[Testcase 55]: L2 14016; PVBand 19965; EPE 0
[Testcase 56]: L2 5195; PVBand 5825; EPE 0
[Testcase 57]: L2 40009; PVBand 44603; EPE 1
[Testcase 58]: L2 5367; PVBand 6930; EPE 0
[Testcase 59]: L2 17425; PVBand 23609; EPE 0
[Testcase 60]: L2 29237; PVBand 30941; EPE 3
[Testcase 61]: L2 4577; PVBand 5739; EPE 0
[Testcase 62]: L2 39731; PVBand 46059; EPE 3
[Testcase 63]: L2 40717; PVBand 47251; EPE 6
[Testcase 64]: L2 14401; PVBand 18210; EPE 0
[Testcase 65]: L2 13669; PVBand 18813; EPE 1
[Testcase 66]: L2 43866; PVBand 47379; EPE 2
[Testcase 67]: L2 17221; PVBand 19466; EPE 0
[Testcase 68]: L2 5483; PVBand 6067; EPE 0
[Testcase 69]: L2 22821; PVBand 27736; EPE 0
[Testcase 70]: L2 19574; PVBand 23163; EPE 3
[Testcase 71]: L2 17591; PVBand 22097; EPE 1
[Testcase 72]: L2 15069; PVBand 15777; EPE 0
[Testcase 73]: L2 35390; PVBand 51754; EPE 3
[Testcase 74]: L2 13095; PVBand 15483; EPE 0
[Testcase 75]: L2 13979; PVBand 15429; EPE 0
[Testcase 76]: L2 4966; PVBand 5914; EPE 0
[Testcase 77]: L2 4636; PVBand 6502; EPE 0
[Testcase 78]: L2 5278; PVBand 5733; EPE 0
[Testcase 79]: L2 29052; PVBand 37035; EPE 1
[Testcase 80]: L2 14764; PVBand 19722; EPE 0
[Testcase 81]: L2 15888; PVBand 22975; EPE 1
[Testcase 82]: L2 12760; PVBand 18972; EPE 0
[Testcase 83]: L2 10474; PVBand 12777; EPE 0
[Testcase 84]: L2 22106; PVBand 28393; EPE 0
[Testcase 85]: L2 4319; PVBand 8508; EPE 0
[Testcase 86]: L2 11821; PVBand 14683; EPE 1
[Testcase 87]: L2 16353; PVBand 17674; EPE 1
[Testcase 88]: L2 19139; PVBand 23209; EPE 0
[Testcase 89]: L2 40189; PVBand 52112; EPE 2
[Testcase 90]: L2 3748; PVBand 6437; EPE 0
[Testcase 91]: L2 52959; PVBand 62050; EPE 7
[Testcase 92]: L2 12702; PVBand 20703; EPE 0
[Testcase 93]: L2 6140; PVBand 5350; EPE 0
[Testcase 94]: L2 11103; PVBand 14203; EPE 1
[Testcase 95]: L2 69596; PVBand 53058; EPE 41
[Testcase 96]: L2 8136; PVBand 10428; EPE 0
[Testcase 97]: L2 25941; PVBand 34356; EPE 0
[Testcase 98]: L2 13669; PVBand 18813; EPE 1
[Testcase 99]: L2 3686; PVBand 6521; EPE 0
[Testcase 100]: L2 11718; PVBand 16312; EPE 0
[Testcase 101]: L2 16839; PVBand 22415; EPE 0
[Testcase 102]: L2 5346; PVBand 8381; EPE 0
[Testcase 103]: L2 40613; PVBand 46430; EPE 6
[Testcase 104]: L2 31841; PVBand 35951; EPE 1
[Testcase 105]: L2 15252; PVBand 19366; EPE 0
[Testcase 106]: L2 3555; PVBand 6250; EPE 0
[Testcase 107]: L2 15073; PVBand 19649; EPE 1
[Testcase 108]: L2 13889; PVBand 18466; EPE 0
[Testcase 109]: L2 3338; PVBand 6917; EPE 0
[Testcase 110]: L2 12728; PVBand 18592; EPE 0
[Testcase 111]: L2 15205; PVBand 19349; EPE 0
[Testcase 112]: L2 29180; PVBand 43979; EPE 1
[Testcase 113]: L2 15205; PVBand 19349; EPE 0
[Testcase 114]: L2 3754; PVBand 6501; EPE 0
[Testcase 115]: L2 20543; PVBand 26377; EPE 0
[Testcase 116]: L2 48023; PVBand 28072; EPE 22
[Testcase 117]: L2 34437; PVBand 40792; EPE 4
[Testcase 118]: L2 4355; PVBand 6539; EPE 0
[Testcase 119]: L2 15039; PVBand 17633; EPE 1
[Testcase 120]: L2 11417; PVBand 13801; EPE 0
[Testcase 121]: L2 16054; PVBand 24398; EPE 0
[Testcase 122]: L2 26108; PVBand 27910; EPE 0
[Testcase 123]: L2 3686; PVBand 6521; EPE 0
[Testcase 124]: L2 4526; PVBand 7912; EPE 0
[Testcase 125]: L2 12919; PVBand 16433; EPE 0
[Testcase 126]: L2 20174; PVBand 24591; EPE 1
[Testcase 127]: L2 34387; PVBand 41499; EPE 2
[Testcase 128]: L2 31186; PVBand 34134; EPE 0
[Testcase 129]: L2 12180; PVBand 18478; EPE 1
[Testcase 130]: L2 9432; PVBand 13816; EPE 0
[Testcase 131]: L2 4232; PVBand 10530; EPE 0
[Testcase 132]: L2 14686; PVBand 17698; EPE 0
[Testcase 133]: L2 50857; PVBand 30210; EPE 19
[Testcase 134]: L2 13492; PVBand 15823; EPE 0
[Testcase 135]: L2 49511; PVBand 29791; EPE 19
[Testcase 136]: L2 16563; PVBand 23805; EPE 0
[Testcase 137]: L2 4355; PVBand 6539; EPE 0
[Testcase 138]: L2 11860; PVBand 13770; EPE 0
[Testcase 139]: L2 47467; PVBand 30468; EPE 27
[Testcase 140]: L2 5697; PVBand 11861; EPE 0
[Testcase 141]: L2 32487; PVBand 36526; EPE 2
[Testcase 142]: L2 4103; PVBand 6751; EPE 0
[Testcase 143]: L2 16830; PVBand 18041; EPE 1
[Testcase 144]: L2 4655; PVBand 6307; EPE 0
[Testcase 145]: L2 16708; PVBand 16705; EPE 1
[Testcase 146]: L2 5278; PVBand 5733; EPE 0
[Testcase 147]: L2 14654; PVBand 19542; EPE 0
[Testcase 148]: L2 3849; PVBand 6513; EPE 0
[Testcase 149]: L2 6282; PVBand 8327; EPE 0
[Testcase 150]: L2 11103; PVBand 14203; EPE 1
[Testcase 151]: L2 3748; PVBand 6437; EPE 0
[Testcase 152]: L2 2936; PVBand 7298; EPE 0
[Testcase 153]: L2 26205; PVBand 41452; EPE 0
[Testcase 154]: L2 28663; PVBand 38223; EPE 6
[Testcase 155]: L2 36697; PVBand 35532; EPE 5
[Testcase 156]: L2 15369; PVBand 17138; EPE 0
[Testcase 157]: L2 7642; PVBand 14919; EPE 0
[Testcase 158]: L2 52855; PVBand 57959; EPE 6
[Testcase 159]: L2 14704; PVBand 19013; EPE 0
[Testcase 160]: L2 23319; PVBand 32330; EPE 0
[Testcase 161]: L2 18639; PVBand 23040; EPE 0
[Testcase 162]: L2 3849; PVBand 6513; EPE 0
[Testcase 163]: L2 12919; PVBand 16433; EPE 0
[Testcase 164]: L2 5133; PVBand 6075; EPE 0
[Testcase 165]: L2 32712; PVBand 44997; EPE 0
[Testcase 166]: L2 34079; PVBand 35132; EPE 3
[Testcase 167]: L2 56878; PVBand 39516; EPE 33
[Testcase 168]: L2 13965; PVBand 19344; EPE 1
[Testcase 169]: L2 26699; PVBand 38528; EPE 0
[Testcase 170]: L2 24125; PVBand 31383; EPE 2
[Testcase 171]: L2 9592; PVBand 12168; EPE 0
[Testcase 172]: L2 7259; PVBand 14756; EPE 0
[Testcase 173]: L2 46753; PVBand 54207; EPE 3
[Testcase 174]: L2 39615; PVBand 49679; EPE 1
[Testcase 175]: L2 46434; PVBand 51524; EPE 3
[Testcase 176]: L2 19387; PVBand 18985; EPE 0
[Testcase 177]: L2 14831; PVBand 19651; EPE 0
[Testcase 178]: L2 12555; PVBand 12606; EPE 0
[Testcase 179]: L2 11370; PVBand 14452; EPE 1
[Testcase 180]: L2 16153; PVBand 24855; EPE 0
[Testcase 181]: L2 16707; PVBand 24761; EPE 0
[Testcase 182]: L2 3555; PVBand 6250; EPE 0
[Testcase 183]: L2 9976; PVBand 21242; EPE 0
[Testcase 184]: L2 31004; PVBand 38429; EPE 6
[Testcase 185]: L2 14712; PVBand 19298; EPE 1
[Testcase 186]: L2 47489; PVBand 65415; EPE 4
[Testcase 187]: L2 9820; PVBand 13151; EPE 0
[Testcase 188]: L2 10881; PVBand 11605; EPE 0
[Testcase 189]: L2 27784; PVBand 36086; EPE 2
[Testcase 190]: L2 4842; PVBand 7384; EPE 0
[Testcase 191]: L2 16830; PVBand 18041; EPE 1
[Testcase 192]: L2 23413; PVBand 46393; EPE 1
[Testcase 193]: L2 31101; PVBand 40853; EPE 0
[Testcase 194]: L2 25941; PVBand 34356; EPE 0
[Testcase 195]: L2 19021; PVBand 23065; EPE 0
[Testcase 196]: L2 18851; PVBand 22112; EPE 0
[Testcase 197]: L2 5346; PVBand 8381; EPE 0
[Testcase 198]: L2 12760; PVBand 18972; EPE 0
[Testcase 199]: L2 19841; PVBand 28597; EPE 0
[Testcase 200]: L2 18041; PVBand 24813; EPE 0
[Testcase 201]: L2 3338; PVBand 6917; EPE 0
[Testcase 202]: L2 4230; PVBand 9317; EPE 0
[Testcase 203]: L2 30696; PVBand 36938; EPE 3
[Testcase 204]: L2 4236; PVBand 6637; EPE 0
[Testcase 205]: L2 56062; PVBand 70138; EPE 2
[Testcase 206]: L2 13192; PVBand 19053; EPE 0
[Testcase 207]: L2 36939; PVBand 54226; EPE 1
[Testcase 208]: L2 12555; PVBand 12606; EPE 0
[Testcase 209]: L2 16427; PVBand 17006; EPE 2
[Testcase 210]: L2 46541; PVBand 57438; EPE 1
[Testcase 211]: L2 7156; PVBand 9987; EPE 0
[Testcase 212]: L2 5888; PVBand 5615; EPE 0
[Testcase 213]: L2 13187; PVBand 18874; EPE 0
[Testcase 214]: L2 19289; PVBand 19533; EPE 0
[Testcase 215]: L2 41708; PVBand 51160; EPE 1
[Testcase 216]: L2 17967; PVBand 24462; EPE 0
[Testcase 217]: L2 38394; PVBand 43793; EPE 9
[Testcase 218]: L2 13889; PVBand 18466; EPE 0
[Testcase 219]: L2 12667; PVBand 18087; EPE 1
[Testcase 220]: L2 31796; PVBand 43738; EPE 2
[Testcase 221]: L2 18234; PVBand 22752; EPE 0
[Testcase 222]: L2 9424; PVBand 8787; EPE 0
[Testcase 223]: L2 4324; PVBand 6162; EPE 0
[Testcase 224]: L2 14764; PVBand 19722; EPE 0
[Testcase 225]: L2 16089; PVBand 16569; EPE 0
[Testcase 226]: L2 21219; PVBand 27280; EPE 0
[Testcase 227]: L2 22547; PVBand 26836; EPE 2
[Testcase 228]: L2 18780; PVBand 24901; EPE 0
[Testcase 229]: L2 32252; PVBand 43637; EPE 0
[Testcase 230]: L2 61250; PVBand 52399; EPE 21
[Testcase 231]: L2 17941; PVBand 25634; EPE 0
[Testcase 232]: L2 26467; PVBand 36707; EPE 0
[Testcase 233]: L2 16061; PVBand 20553; EPE 0
[Testcase 234]: L2 11821; PVBand 14683; EPE 1
[Testcase 235]: L2 52994; PVBand 34696; EPE 33
[Testcase 236]: L2 13979; PVBand 15429; EPE 0
[Testcase 237]: L2 14787; PVBand 19667; EPE 1
[Testcase 238]: L2 18361; PVBand 22944; EPE 0
[Testcase 239]: L2 38547; PVBand 43108; EPE 1
[Testcase 240]: L2 12555; PVBand 12606; EPE 0
[Testcase 241]: L2 19811; PVBand 24579; EPE 2
[Testcase 242]: L2 3920; PVBand 6610; EPE 0
[Testcase 243]: L2 5347; PVBand 6672; EPE 0
[Testcase 244]: L2 14787; PVBand 19667; EPE 1
[Testcase 245]: L2 22564; PVBand 32341; EPE 0
[Testcase 246]: L2 31064; PVBand 34876; EPE 1
[Testcase 247]: L2 22106; PVBand 28393; EPE 0
[Testcase 248]: L2 82492; PVBand 75517; EPE 34
[Testcase 249]: L2 5888; PVBand 5615; EPE 0
[Testcase 250]: L2 14831; PVBand 19651; EPE 0
[Testcase 251]: L2 11821; PVBand 14683; EPE 1
[Testcase 252]: L2 9231; PVBand 11320; EPE 0
[Testcase 253]: L2 5367; PVBand 6930; EPE 0
[Testcase 254]: L2 12180; PVBand 18478; EPE 1
[Testcase 255]: L2 25588; PVBand 27364; EPE 5
[Testcase 256]: L2 4236; PVBand 6637; EPE 0
[Testcase 257]: L2 64324; PVBand 73047; EPE 8
[Testcase 258]: L2 22742; PVBand 30381; EPE 0
[Testcase 259]: L2 29375; PVBand 41475; EPE 1
[Testcase 260]: L2 6546; PVBand 8862; EPE 0
[Testcase 261]: L2 81694; PVBand 59654; EPE 31
[Testcase 262]: L2 46125; PVBand 26962; EPE 20
[Testcase 263]: L2 11043; PVBand 16562; EPE 0
[Testcase 264]: L2 13964; PVBand 21088; EPE 0
[Testcase 265]: L2 7403; PVBand 8725; EPE 0
[Testcase 266]: L2 28809; PVBand 38236; EPE 2
[Testcase 267]: L2 15039; PVBand 17633; EPE 1
[Testcase 268]: L2 6234; PVBand 7988; EPE 0
[Testcase 269]: L2 7509; PVBand 10915; EPE 0
[Testcase 270]: L2 15384; PVBand 23456; EPE 0
[Testcase 271]: L2 6282; PVBand 8327; EPE 0
[Initialized]: L2 20045; PVBand 23548; EPE 2.4
[Evaluation]: Finetuning the masks with pixel-based ILT method
[Testcase 1]: L2 9106; PVBand 20126; EPE 0
[Testcase 2]: L2 4664; PVBand 7043; EPE 0
[Testcase 3]: L2 6458; PVBand 12404; EPE 0
[Testcase 4]: L2 8814; PVBand 16361; EPE 0
[Testcase 5]: L2 4369; PVBand 5987; EPE 0
[Testcase 6]: L2 2921; PVBand 6175; EPE 0
[Testcase 7]: L2 9135; PVBand 13167; EPE 0
[Testcase 8]: L2 3599; PVBand 7750; EPE 0
[Testcase 9]: L2 10493; PVBand 18650; EPE 0
[Testcase 10]: L2 10429; PVBand 29748; EPE 0
[Testcase 11]: L2 14119; PVBand 32726; EPE 0
[Testcase 12]: L2 3956; PVBand 6426; EPE 0
[Testcase 13]: L2 24253; PVBand 43287; EPE 0
[Testcase 14]: L2 10881; PVBand 24329; EPE 0
[Testcase 15]: L2 13442; PVBand 25329; EPE 0
[Testcase 16]: L2 10352; PVBand 15794; EPE 0
[Testcase 17]: L2 9649; PVBand 18296; EPE 0
[Testcase 18]: L2 13846; PVBand 32418; EPE 0
[Testcase 19]: L2 4560; PVBand 7127; EPE 0
[Testcase 20]: L2 7759; PVBand 13350; EPE 0
[Testcase 21]: L2 19872; PVBand 46605; EPE 0
[Testcase 22]: L2 9884; PVBand 21385; EPE 0
[Testcase 23]: L2 20428; PVBand 40260; EPE 0
[Testcase 24]: L2 9560; PVBand 19265; EPE 0
[Testcase 25]: L2 8924; PVBand 15464; EPE 0
[Testcase 26]: L2 9110; PVBand 20238; EPE 0
[Testcase 27]: L2 13545; PVBand 21828; EPE 0
[Testcase 28]: L2 2744; PVBand 6534; EPE 0
[Testcase 29]: L2 28668; PVBand 54615; EPE 0
[Testcase 30]: L2 27626; PVBand 50865; EPE 0
[Testcase 31]: L2 21838; PVBand 39348; EPE 0
[Testcase 32]: L2 5513; PVBand 8857; EPE 0
[Testcase 33]: L2 5297; PVBand 10428; EPE 0
[Testcase 34]: L2 4922; PVBand 7699; EPE 0
[Testcase 35]: L2 8331; PVBand 19909; EPE 0
[Testcase 36]: L2 27638; PVBand 62711; EPE 1
[Testcase 37]: L2 10904; PVBand 18635; EPE 0
[Testcase 38]: L2 53396; PVBand 80762; EPE 0
[Testcase 39]: L2 5073; PVBand 7478; EPE 0
[Testcase 40]: L2 7884; PVBand 14289; EPE 0
[Testcase 41]: L2 53553; PVBand 88615; EPE 3
[Testcase 42]: L2 6252; PVBand 10434; EPE 0
[Testcase 43]: L2 21247; PVBand 40222; EPE 0
[Testcase 44]: L2 40565; PVBand 75778; EPE 0
[Testcase 45]: L2 11253; PVBand 25119; EPE 0
[Testcase 46]: L2 10335; PVBand 23129; EPE 0
[Testcase 47]: L2 11022; PVBand 23561; EPE 0
[Testcase 48]: L2 7601; PVBand 14703; EPE 0
[Testcase 49]: L2 18881; PVBand 47911; EPE 0
[Testcase 50]: L2 10938; PVBand 20049; EPE 0
[Testcase 51]: L2 4718; PVBand 6942; EPE 0
[Testcase 52]: L2 3798; PVBand 6787; EPE 0
[Testcase 53]: L2 17998; PVBand 34637; EPE 0
[Testcase 54]: L2 12956; PVBand 25986; EPE 0
[Testcase 55]: L2 9106; PVBand 20126; EPE 0
[Testcase 56]: L2 4743; PVBand 6726; EPE 0
[Testcase 57]: L2 24874; PVBand 46199; EPE 0
[Testcase 58]: L2 4449; PVBand 7086; EPE 0
[Testcase 59]: L2 13852; PVBand 24165; EPE 0
[Testcase 60]: L2 18345; PVBand 33657; EPE 0
[Testcase 61]: L2 3985; PVBand 5878; EPE 0
[Testcase 62]: L2 28879; PVBand 51914; EPE 0
[Testcase 63]: L2 24148; PVBand 49023; EPE 0
[Testcase 64]: L2 10752; PVBand 19402; EPE 0
[Testcase 65]: L2 9810; PVBand 18198; EPE 0
[Testcase 66]: L2 31532; PVBand 53202; EPE 1
[Testcase 67]: L2 12469; PVBand 18787; EPE 0
[Testcase 68]: L2 4309; PVBand 6291; EPE 0
[Testcase 69]: L2 14718; PVBand 29631; EPE 0
[Testcase 70]: L2 6880; PVBand 21398; EPE 0
[Testcase 71]: L2 9884; PVBand 21385; EPE 0
[Testcase 72]: L2 10067; PVBand 15526; EPE 0
[Testcase 73]: L2 23910; PVBand 54587; EPE 0
[Testcase 74]: L2 7691; PVBand 16502; EPE 0
[Testcase 75]: L2 9120; PVBand 15814; EPE 0
[Testcase 76]: L2 4369; PVBand 5987; EPE 0
[Testcase 77]: L2 3956; PVBand 6426; EPE 0
[Testcase 78]: L2 4715; PVBand 6182; EPE 0
[Testcase 79]: L2 20310; PVBand 40196; EPE 0
[Testcase 80]: L2 9413; PVBand 19894; EPE 0
[Testcase 81]: L2 8415; PVBand 23144; EPE 0
[Testcase 82]: L2 9897; PVBand 18982; EPE 0
[Testcase 83]: L2 7650; PVBand 13119; EPE 0
[Testcase 84]: L2 14634; PVBand 30805; EPE 0
[Testcase 85]: L2 4867; PVBand 7273; EPE 0
[Testcase 86]: L2 7601; PVBand 14703; EPE 0
[Testcase 87]: L2 10126; PVBand 18982; EPE 0
[Testcase 88]: L2 9627; PVBand 24666; EPE 0
[Testcase 89]: L2 25333; PVBand 58848; EPE 0
[Testcase 90]: L2 3737; PVBand 6347; EPE 0
[Testcase 91]: L2 36439; PVBand 69628; EPE 0
[Testcase 92]: L2 7474; PVBand 20718; EPE 0
[Testcase 93]: L2 4691; PVBand 5867; EPE 0
[Testcase 94]: L2 7905; PVBand 14246; EPE 0
[Testcase 95]: L2 28152; PVBand 52289; EPE 0
[Testcase 96]: L2 6604; PVBand 10397; EPE 0
[Testcase 97]: L2 18147; PVBand 34981; EPE 0
[Testcase 98]: L2 9810; PVBand 18198; EPE 0
[Testcase 99]: L2 3641; PVBand 6295; EPE 0
[Testcase 100]: L2 8814; PVBand 16361; EPE 0
[Testcase 101]: L2 10881; PVBand 24329; EPE 0
[Testcase 102]: L2 4790; PVBand 8195; EPE 0
[Testcase 103]: L2 26558; PVBand 50096; EPE 0
[Testcase 104]: L2 20066; PVBand 37926; EPE 0
[Testcase 105]: L2 11476; PVBand 19869; EPE 0
[Testcase 106]: L2 2921; PVBand 6175; EPE 0
[Testcase 107]: L2 10747; PVBand 19846; EPE 0
[Testcase 108]: L2 8611; PVBand 18598; EPE 0
[Testcase 109]: L2 3148; PVBand 7077; EPE 0
[Testcase 110]: L2 8407; PVBand 18867; EPE 0
[Testcase 111]: L2 9452; PVBand 19992; EPE 0
[Testcase 112]: L2 21370; PVBand 47221; EPE 0
[Testcase 113]: L2 9452; PVBand 19992; EPE 0
[Testcase 114]: L2 3174; PVBand 6253; EPE 0
[Testcase 115]: L2 15744; PVBand 27514; EPE 0
[Testcase 116]: L2 17609; PVBand 33058; EPE 0
[Testcase 117]: L2 22956; PVBand 45844; EPE 0
[Testcase 118]: L2 4064; PVBand 6969; EPE 0
[Testcase 119]: L2 11244; PVBand 17825; EPE 0
[Testcase 120]: L2 7922; PVBand 13948; EPE 0
[Testcase 121]: L2 11377; PVBand 24473; EPE 0
[Testcase 122]: L2 18153; PVBand 30701; EPE 0
[Testcase 123]: L2 3641; PVBand 6295; EPE 0
[Testcase 124]: L2 4588; PVBand 6960; EPE 0
[Testcase 125]: L2 9217; PVBand 16707; EPE 0
[Testcase 126]: L2 13006; PVBand 25414; EPE 0
[Testcase 127]: L2 19921; PVBand 45447; EPE 0
[Testcase 128]: L2 17025; PVBand 39587; EPE 0
[Testcase 129]: L2 8934; PVBand 18290; EPE 0
[Testcase 130]: L2 7527; PVBand 13681; EPE 0
[Testcase 131]: L2 4798; PVBand 7892; EPE 0
[Testcase 132]: L2 10282; PVBand 18082; EPE 0
[Testcase 133]: L2 23205; PVBand 37384; EPE 0
[Testcase 134]: L2 9177; PVBand 16069; EPE 0
[Testcase 135]: L2 19503; PVBand 36732; EPE 0
[Testcase 136]: L2 12435; PVBand 25536; EPE 0
[Testcase 137]: L2 4064; PVBand 6969; EPE 0
[Testcase 138]: L2 9088; PVBand 13710; EPE 0
[Testcase 139]: L2 17474; PVBand 32389; EPE 0
[Testcase 140]: L2 4578; PVBand 13271; EPE 0
[Testcase 141]: L2 20403; PVBand 40657; EPE 0
[Testcase 142]: L2 4350; PVBand 6380; EPE 0
[Testcase 143]: L2 11639; PVBand 20077; EPE 0
[Testcase 144]: L2 4718; PVBand 6942; EPE 0
[Testcase 145]: L2 9880; PVBand 15546; EPE 0
[Testcase 146]: L2 4715; PVBand 6182; EPE 0
[Testcase 147]: L2 9110; PVBand 20238; EPE 0
[Testcase 148]: L2 4018; PVBand 6462; EPE 0
[Testcase 149]: L2 5076; PVBand 8376; EPE 0
[Testcase 150]: L2 7905; PVBand 14246; EPE 0
[Testcase 151]: L2 3737; PVBand 6347; EPE 0
[Testcase 152]: L2 2744; PVBand 6534; EPE 0
[Testcase 153]: L2 15616; PVBand 45915; EPE 0
[Testcase 154]: L2 17627; PVBand 41919; EPE 0
[Testcase 155]: L2 20344; PVBand 40709; EPE 0
[Testcase 156]: L2 9187; PVBand 17939; EPE 0
[Testcase 157]: L2 4831; PVBand 14797; EPE 0
[Testcase 158]: L2 36191; PVBand 64489; EPE 0
[Testcase 159]: L2 8818; PVBand 19565; EPE 0
[Testcase 160]: L2 16451; PVBand 33823; EPE 0
[Testcase 161]: L2 11661; PVBand 25189; EPE 0
[Testcase 162]: L2 4018; PVBand 6462; EPE 0
[Testcase 163]: L2 9217; PVBand 16707; EPE 0
[Testcase 164]: L2 4234; PVBand 6133; EPE 0
[Testcase 165]: L2 22726; PVBand 48137; EPE 0
[Testcase 166]: L2 19382; PVBand 38522; EPE 0
[Testcase 167]: L2 22965; PVBand 43609; EPE 0
[Testcase 168]: L2 8565; PVBand 19796; EPE 0
[Testcase 169]: L2 18536; PVBand 40350; EPE 0
[Testcase 170]: L2 15193; PVBand 34552; EPE 0
[Testcase 171]: L2 4134; PVBand 12415; EPE 0
[Testcase 172]: L2 4562; PVBand 14431; EPE 0
[Testcase 173]: L2 34729; PVBand 56125; EPE 0
[Testcase 174]: L2 26645; PVBand 53049; EPE 0
[Testcase 175]: L2 32317; PVBand 55126; EPE 0
[Testcase 176]: L2 14151; PVBand 20792; EPE 0
[Testcase 177]: L2 9663; PVBand 19652; EPE 0
[Testcase 178]: L2 9135; PVBand 13167; EPE 0
[Testcase 179]: L2 7884; PVBand 14289; EPE 0
[Testcase 180]: L2 12040; PVBand 25439; EPE 0
[Testcase 181]: L2 11142; PVBand 24440; EPE 0
[Testcase 182]: L2 2921; PVBand 6175; EPE 0
[Testcase 183]: L2 6867; PVBand 17338; EPE 0
[Testcase 184]: L2 18215; PVBand 38774; EPE 0
[Testcase 185]: L2 7551; PVBand 19952; EPE 0
[Testcase 186]: L2 35775; PVBand 72463; EPE 0
[Testcase 187]: L2 7759; PVBand 13350; EPE 0
[Testcase 188]: L2 7486; PVBand 11639; EPE 0
[Testcase 189]: L2 13418; PVBand 36652; EPE 0
[Testcase 190]: L2 4690; PVBand 7257; EPE 0
[Testcase 191]: L2 11639; PVBand 20077; EPE 0
[Testcase 192]: L2 14528; PVBand 45695; EPE 0
[Testcase 193]: L2 19263; PVBand 44146; EPE 0
[Testcase 194]: L2 18147; PVBand 34981; EPE 0
[Testcase 195]: L2 11318; PVBand 25027; EPE 0
[Testcase 196]: L2 12832; PVBand 23663; EPE 0
[Testcase 197]: L2 4790; PVBand 8195; EPE 0
[Testcase 198]: L2 9897; PVBand 18982; EPE 0
[Testcase 199]: L2 12439; PVBand 30415; EPE 0
[Testcase 200]: L2 11771; PVBand 24975; EPE 0
[Testcase 201]: L2 3148; PVBand 7077; EPE 0
[Testcase 202]: L2 4560; PVBand 7127; EPE 0
[Testcase 203]: L2 21908; PVBand 38166; EPE 0
[Testcase 204]: L2 4123; PVBand 6463; EPE 0
[Testcase 205]: L2 41872; PVBand 79876; EPE 1
[Testcase 206]: L2 6857; PVBand 20050; EPE 0
[Testcase 207]: L2 23638; PVBand 56777; EPE 0
[Testcase 208]: L2 9135; PVBand 13167; EPE 0
[Testcase 209]: L2 9244; PVBand 17730; EPE 0
[Testcase 210]: L2 36308; PVBand 56728; EPE 0
[Testcase 211]: L2 4468; PVBand 9660; EPE 0
[Testcase 212]: L2 4760; PVBand 6092; EPE 0
[Testcase 213]: L2 9560; PVBand 19265; EPE 0
[Testcase 214]: L2 13384; PVBand 20876; EPE 0
[Testcase 215]: L2 33018; PVBand 55466; EPE 0
[Testcase 216]: L2 11960; PVBand 25101; EPE 0
[Testcase 217]: L2 26854; PVBand 45775; EPE 4
[Testcase 218]: L2 8611; PVBand 18598; EPE 0
[Testcase 219]: L2 8275; PVBand 17837; EPE 0
[Testcase 220]: L2 19341; PVBand 48928; EPE 0
[Testcase 221]: L2 11253; PVBand 25119; EPE 0
[Testcase 222]: L2 7439; PVBand 9421; EPE 0
[Testcase 223]: L2 3778; PVBand 6154; EPE 0
[Testcase 224]: L2 9413; PVBand 19894; EPE 0
[Testcase 225]: L2 12493; PVBand 17271; EPE 0
[Testcase 226]: L2 8682; PVBand 28746; EPE 0
[Testcase 227]: L2 14085; PVBand 28206; EPE 0
[Testcase 228]: L2 5367; PVBand 22636; EPE 0
[Testcase 229]: L2 22700; PVBand 49394; EPE 0
[Testcase 230]: L2 27723; PVBand 54546; EPE 0
[Testcase 231]: L2 11897; PVBand 27308; EPE 0
[Testcase 232]: L2 14820; PVBand 37735; EPE 0
[Testcase 233]: L2 12135; PVBand 21642; EPE 0
[Testcase 234]: L2 7601; PVBand 14703; EPE 0
[Testcase 235]: L2 18964; PVBand 38337; EPE 0
[Testcase 236]: L2 9120; PVBand 15814; EPE 0
[Testcase 237]: L2 9215; PVBand 20155; EPE 0
[Testcase 238]: L2 12118; PVBand 25212; EPE 0
[Testcase 239]: L2 20636; PVBand 48169; EPE 0
[Testcase 240]: L2 9135; PVBand 13167; EPE 0
[Testcase 241]: L2 8822; PVBand 25756; EPE 0
[Testcase 242]: L2 3798; PVBand 6787; EPE 0
[Testcase 243]: L2 4503; PVBand 6642; EPE 0
[Testcase 244]: L2 9215; PVBand 20155; EPE 0
[Testcase 245]: L2 15491; PVBand 34575; EPE 0
[Testcase 246]: L2 20936; PVBand 37544; EPE 0
[Testcase 247]: L2 14634; PVBand 30805; EPE 0
[Testcase 248]: L2 40849; PVBand 71252; EPE 0
[Testcase 249]: L2 4760; PVBand 6092; EPE 0
[Testcase 250]: L2 9663; PVBand 19652; EPE 0
[Testcase 251]: L2 7601; PVBand 14703; EPE 0
[Testcase 252]: L2 6884; PVBand 11205; EPE 0
[Testcase 253]: L2 4449; PVBand 7086; EPE 0
[Testcase 254]: L2 8934; PVBand 18290; EPE 0
[Testcase 255]: L2 16586; PVBand 28481; EPE 0
[Testcase 256]: L2 4123; PVBand 6463; EPE 0
[Testcase 257]: L2 46334; PVBand 80827; EPE 2
[Testcase 258]: L2 14993; PVBand 31165; EPE 0
[Testcase 259]: L2 17766; PVBand 46768; EPE 0
[Testcase 260]: L2 5513; PVBand 8857; EPE 0
[Testcase 261]: L2 42667; PVBand 61154; EPE 0
[Testcase 262]: L2 18004; PVBand 30448; EPE 0
[Testcase 263]: L2 6917; PVBand 16054; EPE 0
[Testcase 264]: L2 9461; PVBand 22278; EPE 0
[Testcase 265]: L2 5187; PVBand 8676; EPE 0
[Testcase 266]: L2 21441; PVBand 40446; EPE 0
[Testcase 267]: L2 11244; PVBand 17825; EPE 0
[Testcase 268]: L2 5073; PVBand 7478; EPE 0
[Testcase 269]: L2 5309; PVBand 11202; EPE 0
[Testcase 270]: L2 10877; PVBand 24088; EPE 0
[Testcase 271]: L2 5076; PVBand 8376; EPE 0
[Finetuned]: L2 12700; PVBand 24773; EPE 0.0

[StdContact]
[Evaluation]: Getting masks from the model
[Testcase 1]: L2 27524; PVBand 48321; EPE 2
[Testcase 2]: L2 26819; PVBand 45194; EPE 2
[Testcase 3]: L2 20778; PVBand 34201; EPE 2
[Testcase 4]: L2 23801; PVBand 37693; EPE 2
[Testcase 5]: L2 25333; PVBand 40544; EPE 5
[Testcase 6]: L2 27060; PVBand 45269; EPE 3
[Testcase 7]: L2 18990; PVBand 31529; EPE 2
[Testcase 8]: L2 27938; PVBand 46334; EPE 2
[Testcase 9]: L2 35025; PVBand 60365; EPE 4
[Testcase 10]: L2 21389; PVBand 33041; EPE 1
[Testcase 11]: L2 23552; PVBand 40378; EPE 0
[Testcase 12]: L2 25246; PVBand 43199; EPE 2
[Testcase 13]: L2 37321; PVBand 58354; EPE 12
[Testcase 14]: L2 23391; PVBand 39633; EPE 2
[Testcase 15]: L2 21423; PVBand 35628; EPE 0
[Testcase 16]: L2 27062; PVBand 42414; EPE 6
[Testcase 17]: L2 30658; PVBand 52000; EPE 4
[Testcase 18]: L2 29103; PVBand 48392; EPE 2
[Testcase 19]: L2 20389; PVBand 33408; EPE 1
[Testcase 20]: L2 19789; PVBand 32657; EPE 2
[Testcase 21]: L2 24240; PVBand 40162; EPE 0
[Testcase 22]: L2 23930; PVBand 38543; EPE 1
[Testcase 23]: L2 27630; PVBand 38437; EPE 6
[Testcase 24]: L2 25053; PVBand 40194; EPE 3
[Testcase 25]: L2 27763; PVBand 42332; EPE 2
[Testcase 26]: L2 22975; PVBand 35466; EPE 5
[Testcase 27]: L2 20419; PVBand 33977; EPE 0
[Testcase 28]: L2 25403; PVBand 41062; EPE 3
[Testcase 29]: L2 26495; PVBand 42128; EPE 4
[Testcase 30]: L2 22226; PVBand 30987; EPE 4
[Testcase 31]: L2 20071; PVBand 35437; EPE 1
[Testcase 32]: L2 26901; PVBand 48424; EPE 3
[Testcase 33]: L2 19510; PVBand 37715; EPE 1
[Testcase 34]: L2 14077; PVBand 24569; EPE 0
[Testcase 35]: L2 34572; PVBand 67896; EPE 1
[Testcase 36]: L2 35964; PVBand 63727; EPE 4
[Testcase 37]: L2 29855; PVBand 48620; EPE 3
[Testcase 38]: L2 28421; PVBand 43307; EPE 8
[Testcase 39]: L2 24587; PVBand 38000; EPE 4
[Testcase 40]: L2 23136; PVBand 34362; EPE 2
[Testcase 41]: L2 18598; PVBand 33996; EPE 0
[Testcase 42]: L2 26801; PVBand 41379; EPE 6
[Testcase 43]: L2 21899; PVBand 34402; EPE 1
[Testcase 44]: L2 30518; PVBand 55548; EPE 1
[Testcase 45]: L2 25525; PVBand 35979; EPE 8
[Testcase 46]: L2 22596; PVBand 39288; EPE 0
[Testcase 47]: L2 22371; PVBand 31589; EPE 6
[Testcase 48]: L2 22883; PVBand 36989; EPE 2
[Testcase 49]: L2 19736; PVBand 36085; EPE 1
[Testcase 50]: L2 26020; PVBand 39484; EPE 3
[Testcase 51]: L2 26511; PVBand 44696; EPE 1
[Testcase 52]: L2 15623; PVBand 26956; EPE 0
[Testcase 53]: L2 17803; PVBand 30081; EPE 2
[Testcase 54]: L2 24220; PVBand 37313; EPE 2
[Testcase 55]: L2 23268; PVBand 40640; EPE 0
[Testcase 56]: L2 32421; PVBand 48795; EPE 4
[Testcase 57]: L2 25328; PVBand 32432; EPE 9
[Testcase 58]: L2 25896; PVBand 35272; EPE 5
[Testcase 59]: L2 38358; PVBand 57477; EPE 12
[Testcase 60]: L2 23792; PVBand 36440; EPE 3
[Testcase 61]: L2 35823; PVBand 59385; EPE 8
[Testcase 62]: L2 36119; PVBand 56159; EPE 9
[Testcase 63]: L2 15029; PVBand 27529; EPE 1
[Testcase 64]: L2 24311; PVBand 39872; EPE 2
[Testcase 65]: L2 15456; PVBand 27183; EPE 0
[Testcase 66]: L2 24328; PVBand 40275; EPE 3
[Testcase 67]: L2 28126; PVBand 48702; EPE 5
[Testcase 68]: L2 27007; PVBand 44768; EPE 2
[Testcase 69]: L2 24418; PVBand 41904; EPE 5
[Testcase 70]: L2 19811; PVBand 28600; EPE 4
[Testcase 71]: L2 22353; PVBand 38807; EPE 0
[Testcase 72]: L2 19582; PVBand 33606; EPE 0
[Testcase 73]: L2 33821; PVBand 55109; EPE 6
[Testcase 74]: L2 28307; PVBand 44801; EPE 5
[Testcase 75]: L2 27158; PVBand 43859; EPE 4
[Testcase 76]: L2 25231; PVBand 43225; EPE 1
[Testcase 77]: L2 18491; PVBand 31719; EPE 1
[Testcase 78]: L2 31933; PVBand 51479; EPE 7
[Testcase 79]: L2 27390; PVBand 45550; EPE 2
[Testcase 80]: L2 29530; PVBand 48326; EPE 1
[Testcase 81]: L2 30115; PVBand 50379; EPE 1
[Testcase 82]: L2 28237; PVBand 46527; EPE 2
[Testcase 83]: L2 29117; PVBand 46909; EPE 4
[Testcase 84]: L2 21656; PVBand 36723; EPE 1
[Testcase 85]: L2 29266; PVBand 42499; EPE 8
[Testcase 86]: L2 23769; PVBand 41920; EPE 4
[Testcase 87]: L2 34512; PVBand 54067; EPE 10
[Testcase 88]: L2 15581; PVBand 24658; EPE 2
[Testcase 89]: L2 26932; PVBand 40595; EPE 2
[Testcase 90]: L2 30838; PVBand 50937; EPE 5
[Testcase 91]: L2 26275; PVBand 45007; EPE 4
[Testcase 92]: L2 24430; PVBand 43524; EPE 5
[Testcase 93]: L2 19761; PVBand 32893; EPE 2
[Testcase 94]: L2 30802; PVBand 49956; EPE 3
[Testcase 95]: L2 21691; PVBand 36712; EPE 1
[Testcase 96]: L2 27746; PVBand 47800; EPE 3
[Testcase 97]: L2 21899; PVBand 38183; EPE 2
[Testcase 98]: L2 25306; PVBand 42044; EPE 3
[Testcase 99]: L2 27368; PVBand 42302; EPE 5
[Testcase 100]: L2 23632; PVBand 36673; EPE 3
[Testcase 101]: L2 20881; PVBand 29529; EPE 4
[Testcase 102]: L2 27394; PVBand 48319; EPE 2
[Testcase 103]: L2 37266; PVBand 59154; EPE 8
[Testcase 104]: L2 32033; PVBand 51545; EPE 5
[Testcase 105]: L2 34305; PVBand 58086; EPE 3
[Testcase 106]: L2 20166; PVBand 35433; EPE 2
[Testcase 107]: L2 21598; PVBand 34046; EPE 3
[Testcase 108]: L2 30660; PVBand 49457; EPE 2
[Testcase 109]: L2 33922; PVBand 58336; EPE 9
[Testcase 110]: L2 19241; PVBand 26879; EPE 6
[Testcase 111]: L2 35040; PVBand 61845; EPE 3
[Testcase 112]: L2 33041; PVBand 51569; EPE 6
[Testcase 113]: L2 21416; PVBand 38699; EPE 2
[Testcase 114]: L2 20077; PVBand 29582; EPE 0
[Testcase 115]: L2 23385; PVBand 30842; EPE 6
[Testcase 116]: L2 34075; PVBand 51290; EPE 8
[Testcase 117]: L2 22429; PVBand 33923; EPE 4
[Testcase 118]: L2 24391; PVBand 41657; EPE 1
[Testcase 119]: L2 33380; PVBand 56137; EPE 5
[Testcase 120]: L2 20273; PVBand 29452; EPE 3
[Testcase 121]: L2 38121; PVBand 69699; EPE 9
[Testcase 122]: L2 18278; PVBand 31626; EPE 1
[Testcase 123]: L2 16967; PVBand 31273; EPE 0
[Testcase 124]: L2 24925; PVBand 34010; EPE 7
[Testcase 125]: L2 27147; PVBand 42501; EPE 3
[Testcase 126]: L2 23347; PVBand 31603; EPE 5
[Testcase 127]: L2 35202; PVBand 57844; EPE 6
[Testcase 128]: L2 14227; PVBand 25119; EPE 0
[Testcase 129]: L2 29310; PVBand 52392; EPE 1
[Testcase 130]: L2 23951; PVBand 39939; EPE 3
[Testcase 131]: L2 27997; PVBand 48805; EPE 1
[Testcase 132]: L2 25313; PVBand 38648; EPE 5
[Testcase 133]: L2 24491; PVBand 38727; EPE 3
[Testcase 134]: L2 17392; PVBand 31233; EPE 0
[Testcase 135]: L2 23952; PVBand 41869; EPE 2
[Testcase 136]: L2 21610; PVBand 39079; EPE 2
[Testcase 137]: L2 26227; PVBand 38972; EPE 5
[Testcase 138]: L2 15661; PVBand 29067; EPE 1
[Testcase 139]: L2 20244; PVBand 33042; EPE 2
[Testcase 140]: L2 22963; PVBand 36171; EPE 3
[Testcase 141]: L2 34350; PVBand 62492; EPE 4
[Testcase 142]: L2 23545; PVBand 38359; EPE 3
[Testcase 143]: L2 16658; PVBand 27300; EPE 1
[Testcase 144]: L2 25505; PVBand 42757; EPE 1
[Testcase 145]: L2 22992; PVBand 39416; EPE 2
[Testcase 146]: L2 27780; PVBand 43695; EPE 3
[Testcase 147]: L2 32558; PVBand 64894; EPE 1
[Testcase 148]: L2 28144; PVBand 46208; EPE 4
[Testcase 149]: L2 18937; PVBand 31378; EPE 3
[Testcase 150]: L2 20054; PVBand 33150; EPE 2
[Testcase 151]: L2 20070; PVBand 33705; EPE 1
[Testcase 152]: L2 31927; PVBand 52221; EPE 8
[Testcase 153]: L2 24917; PVBand 43140; EPE 1
[Testcase 154]: L2 20749; PVBand 31128; EPE 3
[Testcase 155]: L2 23409; PVBand 37555; EPE 3
[Testcase 156]: L2 25415; PVBand 40260; EPE 2
[Testcase 157]: L2 26836; PVBand 44065; EPE 2
[Testcase 158]: L2 29242; PVBand 45547; EPE 4
[Testcase 159]: L2 24279; PVBand 34092; EPE 8
[Testcase 160]: L2 24417; PVBand 34766; EPE 5
[Testcase 161]: L2 26289; PVBand 34813; EPE 7
[Testcase 162]: L2 26855; PVBand 52940; EPE 0
[Testcase 163]: L2 25882; PVBand 41981; EPE 4
[Testcase 164]: L2 28890; PVBand 41605; EPE 4
[Testcase 165]: L2 23449; PVBand 38876; EPE 1
[Initialized]: L2 25422; PVBand 41537; EPE 3.2
[Evaluation]: Finetuning the masks with pixel-based ILT method
[Testcase 1]: L2 37997; PVBand 47425; EPE 11
[Testcase 2]: L2 27545; PVBand 46762; EPE 1
[Testcase 3]: L2 18512; PVBand 35066; EPE 0
[Testcase 4]: L2 22079; PVBand 39460; EPE 1
[Testcase 5]: L2 24228; PVBand 43343; EPE 0
[Testcase 6]: L2 27969; PVBand 47639; EPE 1
[Testcase 7]: L2 17237; PVBand 32060; EPE 0
[Testcase 8]: L2 28595; PVBand 49413; EPE 0
[Testcase 9]: L2 45901; PVBand 61886; EPE 10
[Testcase 10]: L2 21196; PVBand 35815; EPE 2
[Testcase 11]: L2 29983; PVBand 44349; EPE 8
[Testcase 12]: L2 28747; PVBand 44481; EPE 4
[Testcase 13]: L2 42935; PVBand 61284; EPE 9
[Testcase 14]: L2 23228; PVBand 40390; EPE 0
[Testcase 15]: L2 25041; PVBand 36341; EPE 5
[Testcase 16]: L2 38462; PVBand 39170; EPE 16
[Testcase 17]: L2 43274; PVBand 51013; EPE 15
[Testcase 18]: L2 39502; PVBand 46411; EPE 12
[Testcase 19]: L2 19405; PVBand 36150; EPE 0
[Testcase 20]: L2 16988; PVBand 32613; EPE 0
[Testcase 21]: L2 28005; PVBand 42156; EPE 5
[Testcase 22]: L2 24725; PVBand 38967; EPE 4
[Testcase 23]: L2 24100; PVBand 42037; EPE 2
[Testcase 24]: L2 24023; PVBand 42461; EPE 1
[Testcase 25]: L2 25246; PVBand 45124; EPE 1
[Testcase 26]: L2 19889; PVBand 37086; EPE 0
[Testcase 27]: L2 20010; PVBand 36595; EPE 2
[Testcase 28]: L2 25046; PVBand 43131; EPE 0
[Testcase 29]: L2 27068; PVBand 46768; EPE 4
[Testcase 30]: L2 17464; PVBand 31858; EPE 0
[Testcase 31]: L2 23650; PVBand 39000; EPE 3
[Testcase 32]: L2 34028; PVBand 52595; EPE 5
[Testcase 33]: L2 20968; PVBand 37937; EPE 1
[Testcase 34]: L2 13649; PVBand 26063; EPE 0
[Testcase 35]: L2 59027; PVBand 66907; EPE 26
[Testcase 36]: L2 52686; PVBand 65683; EPE 17
[Testcase 37]: L2 30006; PVBand 50454; EPE 1
[Testcase 38]: L2 23975; PVBand 41539; EPE 0
[Testcase 39]: L2 22023; PVBand 40609; EPE 0
[Testcase 40]: L2 24022; PVBand 37659; EPE 5
[Testcase 41]: L2 18093; PVBand 34909; EPE 0
[Testcase 42]: L2 28835; PVBand 44966; EPE 5
[Testcase 43]: L2 23033; PVBand 34223; EPE 6
[Testcase 44]: L2 46463; PVBand 52030; EPE 19
[Testcase 45]: L2 20550; PVBand 38472; EPE 0
[Testcase 46]: L2 25778; PVBand 37283; EPE 6
[Testcase 47]: L2 16939; PVBand 31773; EPE 0
[Testcase 48]: L2 20028; PVBand 36892; EPE 0
[Testcase 49]: L2 24945; PVBand 35673; EPE 6
[Testcase 50]: L2 23565; PVBand 40216; EPE 1
[Testcase 51]: L2 23018; PVBand 46692; EPE 0
[Testcase 52]: L2 14673; PVBand 29125; EPE 1
[Testcase 53]: L2 16296; PVBand 29946; EPE 0
[Testcase 54]: L2 24129; PVBand 41578; EPE 2
[Testcase 55]: L2 25005; PVBand 42364; EPE 1
[Testcase 56]: L2 30002; PVBand 49607; EPE 2
[Testcase 57]: L2 18778; PVBand 33835; EPE 0
[Testcase 58]: L2 21115; PVBand 37832; EPE 0
[Testcase 59]: L2 43705; PVBand 54555; EPE 13
[Testcase 60]: L2 21129; PVBand 37694; EPE 0
[Testcase 61]: L2 49606; PVBand 58476; EPE 16
[Testcase 62]: L2 42751; PVBand 60183; EPE 8
[Testcase 63]: L2 14416; PVBand 28573; EPE 0
[Testcase 64]: L2 23555; PVBand 40380; EPE 0
[Testcase 65]: L2 14929; PVBand 28047; EPE 0
[Testcase 66]: L2 26807; PVBand 40135; EPE 5
[Testcase 67]: L2 41767; PVBand 45225; EPE 16
[Testcase 68]: L2 32561; PVBand 43836; EPE 8
[Testcase 69]: L2 28712; PVBand 44363; EPE 4
[Testcase 70]: L2 17124; PVBand 30013; EPE 0
[Testcase 71]: L2 31293; PVBand 37608; EPE 11
[Testcase 72]: L2 18688; PVBand 33930; EPE 1
[Testcase 73]: L2 42295; PVBand 56285; EPE 9
[Testcase 74]: L2 24190; PVBand 46454; EPE 0
[Testcase 75]: L2 33103; PVBand 41752; EPE 11
[Testcase 76]: L2 25632; PVBand 44617; EPE 1
[Testcase 77]: L2 17301; PVBand 32408; EPE 1
[Testcase 78]: L2 39479; PVBand 52015; EPE 9
[Testcase 79]: L2 30652; PVBand 48112; EPE 7
[Testcase 80]: L2 33057; PVBand 51403; EPE 5
[Testcase 81]: L2 35050; PVBand 54913; EPE 5
[Testcase 82]: L2 30427; PVBand 44177; EPE 6
[Testcase 83]: L2 34965; PVBand 48029; EPE 7
[Testcase 84]: L2 21018; PVBand 39269; EPE 0
[Testcase 85]: L2 26343; PVBand 46267; EPE 3
[Testcase 86]: L2 35064; PVBand 36321; EPE 15
[Testcase 87]: L2 38242; PVBand 58085; EPE 6
[Testcase 88]: L2 13333; PVBand 24549; EPE 0
[Testcase 89]: L2 24388; PVBand 43596; EPE 0
[Testcase 90]: L2 31355; PVBand 54199; EPE 1
[Testcase 91]: L2 27083; PVBand 46024; EPE 1
[Testcase 92]: L2 30146; PVBand 41703; EPE 7
[Testcase 93]: L2 18216; PVBand 35396; EPE 0
[Testcase 94]: L2 40405; PVBand 49278; EPE 11
[Testcase 95]: L2 21963; PVBand 38356; EPE 0
[Testcase 96]: L2 33991; PVBand 43372; EPE 10
[Testcase 97]: L2 21742; PVBand 41211; EPE 1
[Testcase 98]: L2 26502; PVBand 44889; EPE 6
[Testcase 99]: L2 26177; PVBand 39487; EPE 4
[Testcase 100]: L2 21496; PVBand 40231; EPE 1
[Testcase 101]: L2 15674; PVBand 28308; EPE 0
[Testcase 102]: L2 33339; PVBand 52274; EPE 6
[Testcase 103]: L2 51655; PVBand 60790; EPE 18
[Testcase 104]: L2 39947; PVBand 51810; EPE 9
[Testcase 105]: L2 42270; PVBand 60821; EPE 11
[Testcase 106]: L2 25743; PVBand 39006; EPE 6
[Testcase 107]: L2 21401; PVBand 36029; EPE 4
[Testcase 108]: L2 37005; PVBand 50309; EPE 10
[Testcase 109]: L2 37290; PVBand 62675; EPE 1
[Testcase 110]: L2 13663; PVBand 27875; EPE 0
[Testcase 111]: L2 46112; PVBand 66320; EPE 9
[Testcase 112]: L2 31393; PVBand 53482; EPE 2
[Testcase 113]: L2 24124; PVBand 41335; EPE 3
[Testcase 114]: L2 16040; PVBand 30287; EPE 0
[Testcase 115]: L2 18025; PVBand 32868; EPE 0
[Testcase 116]: L2 31292; PVBand 53422; EPE 0
[Testcase 117]: L2 18419; PVBand 34396; EPE 0
[Testcase 118]: L2 33002; PVBand 40902; EPE 10
[Testcase 119]: L2 42647; PVBand 59212; EPE 10
[Testcase 120]: L2 15843; PVBand 30126; EPE 0
[Testcase 121]: L2 55844; PVBand 66375; EPE 19
[Testcase 122]: L2 18214; PVBand 32913; EPE 0
[Testcase 123]: L2 16510; PVBand 32929; EPE 0
[Testcase 124]: L2 20043; PVBand 35511; EPE 0
[Testcase 125]: L2 29195; PVBand 46320; EPE 5
[Testcase 126]: L2 17804; PVBand 32548; EPE 1
[Testcase 127]: L2 36415; PVBand 59791; EPE 5
[Testcase 128]: L2 13503; PVBand 26988; EPE 0
[Testcase 129]: L2 41816; PVBand 54954; EPE 13
[Testcase 130]: L2 22655; PVBand 43626; EPE 1
[Testcase 131]: L2 41943; PVBand 50990; EPE 14
[Testcase 132]: L2 22874; PVBand 41402; EPE 0
[Testcase 133]: L2 20406; PVBand 39370; EPE 0
[Testcase 134]: L2 19214; PVBand 33342; EPE 1
[Testcase 135]: L2 30831; PVBand 41968; EPE 7
[Testcase 136]: L2 22214; PVBand 39944; EPE 0
[Testcase 137]: L2 26758; PVBand 38489; EPE 6
[Testcase 138]: L2 16725; PVBand 31652; EPE 0
[Testcase 139]: L2 18394; PVBand 35919; EPE 0
[Testcase 140]: L2 22529; PVBand 39284; EPE 1
[Testcase 141]: L2 55917; PVBand 61391; EPE 24
[Testcase 142]: L2 22063; PVBand 41805; EPE 0
[Testcase 143]: L2 17027; PVBand 28662; EPE 2
[Testcase 144]: L2 25703; PVBand 46066; EPE 0
[Testcase 145]: L2 21963; PVBand 39418; EPE 0
[Testcase 146]: L2 30447; PVBand 47257; EPE 4
[Testcase 147]: L2 53913; PVBand 65481; EPE 18
[Testcase 148]: L2 31228; PVBand 48959; EPE 5
[Testcase 149]: L2 20368; PVBand 31109; EPE 4
[Testcase 150]: L2 16826; PVBand 31536; EPE 0
[Testcase 151]: L2 21589; PVBand 36525; EPE 4
[Testcase 152]: L2 35695; PVBand 49876; EPE 7
[Testcase 153]: L2 23436; PVBand 44468; EPE 1
[Testcase 154]: L2 17119; PVBand 32995; EPE 0
[Testcase 155]: L2 23244; PVBand 41299; EPE 2
[Testcase 156]: L2 28599; PVBand 37480; EPE 8
[Testcase 157]: L2 26793; PVBand 46554; EPE 1
[Testcase 158]: L2 37883; PVBand 47495; EPE 13
[Testcase 159]: L2 20093; PVBand 35591; EPE 1
[Testcase 160]: L2 19205; PVBand 35865; EPE 0
[Testcase 161]: L2 20485; PVBand 36439; EPE 0
[Testcase 162]: L2 41064; PVBand 54543; EPE 15
[Testcase 163]: L2 24330; PVBand 42378; EPE 0
[Testcase 164]: L2 27987; PVBand 45222; EPE 4
[Testcase 165]: L2 25068; PVBand 41703; EPE 3
[Finetuned]: L2 27559; PVBand 42819; EPE 4.4
'''