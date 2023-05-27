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

def apply_complex(fr, fi, input):
    return torch.complex(fr(input.real)-fi(input.imag), fr(input.imag)+fi(input.real))

class ComplexLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)

    def forward(self, input):
        return apply_complex(self.fc_r, self.fc_i, input)

class CFNO(nn.Module): 
    def __init__(self, c=1, d=16, k=16, s=1, size=(128, 128)): 
        super().__init__()
        self.c = c
        self.d = d
        self.k = k
        self.s = s
        self.size = size
        self.fc = ComplexLinear(self.c*(self.k**2), self.d)
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
        self.conv6 = conv2d(32, 1,  kernel_size=3, stride=1, padding=1, norm=False, relu=False)
        
        self.sigmoid = nn.Sigmoid()

        self.tail = nn.Sequential(self.deconv0a, self.deconv0b, self.deconv1a, self.deconv1b, self.deconv2a, self.deconv2b, 
                                  self.conv3, self.conv4, self.conv5, self.conv6, self.sigmoid)

    def forward(self, x): 
        
        br0 = self.cfno0(x)
        br1 = self.cfno1(x)
        br2 = self.cfno2(x)
        br3 = self.branch(x)

        feat = torch.cat([br0, br1, br2, br3], dim=1)
        result = self.tail(feat)

        return result


class CFNOILT(ModelILT): 
    def __init__(self, size=(1024, 1024)): 
        super().__init__(size=size, name="CFNOILT")
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
                printedNom, printedMax, printedMin = self.simLitho(mask.squeeze(1))
                l2loss = F.mse_loss(printedNom.unsqueeze(1), target)
                printedNom, printedMax, printedMin = self.simLitho(label.squeeze(1))
                l2lossRef = F.mse_loss(printedNom.unsqueeze(1), target)
                loss = F.mse_loss(mask, mask) if l2loss.item() < l2lossRef.item() else F.mse_loss(mask, label)
                
                opt.zero_grad()
                loss.backward()
                opt.step()

                progress.set_postfix(loss=loss.item())

            print(f"[Epoch {epoch}] Testing")
            self.net.eval()
            l2losses = []
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
                    l2losses.append(l2loss.item())

                    progress.set_postfix(l2loss=l2loss.item())
            
            print(f"[Epoch {epoch}] L2 loss = {np.mean(l2losses)}")

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
        return self.net(target)[0, 0].detach()


if __name__ == "__main__": 
    Benchmark = "MetalSet"
    ImageSize = (1024, 1024)
    Epochs = 1
    BatchSize = 4
    NJobs = 8
    TrainOnly = False
    EvalOnly = False
    train_loader, val_loader = loadersILT(Benchmark, ImageSize, BatchSize, NJobs)
    targets = evaluate.getTargets(samples=None, dataset=Benchmark)
    ilt = CFNOILT(size=ImageSize)
    
    BatchSize = 200
    train_loader, val_loader = loadersILT(Benchmark, ImageSize, BatchSize, NJobs)
    data = None
    for target, label in train_loader: 
        data = target.cuda()
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
    
    if not EvalOnly: 
        ilt.train(train_loader, val_loader, epochs=Epochs)
        ilt.save("trivial/cfnoilt/train.pth")
    else: 
        ilt.load("trivial/cfnoilt/train.pth")
    ilt.evaluate(targets, finetune=False, folder="trivial/cfnoilt")


'''
[MetalSet]
[Evaluation]: Getting masks from the model
[Testcase 1]: L2 58227; PVBand 47815; EPE 17; Shots: 317
[Testcase 2]: L2 50625; PVBand 46225; EPE 17; Shots: 268
[Testcase 3]: L2 104054; PVBand 81724; EPE 67; Shots: 368
[Testcase 4]: L2 25142; PVBand 29814; EPE 9; Shots: 169
[Testcase 5]: L2 56601; PVBand 60013; EPE 4; Shots: 362
[Testcase 6]: L2 56728; PVBand 52354; EPE 3; Shots: 372
[Testcase 7]: L2 32103; PVBand 40065; EPE 0; Shots: 311
[Testcase 8]: L2 22018; PVBand 22722; EPE 1; Shots: 268
[Testcase 9]: L2 58791; PVBand 63364; EPE 7; Shots: 395
[Testcase 10]: L2 13846; PVBand 17216; EPE 0; Shots: 187
[Initialized]: L2 47814; PVBand 46131; EPE 12.5; Runtime: 0.43s; Shots: 302
[Evaluation]: Finetuning the masks with pixel-based ILT method
[Testcase 1]: L2 39233; PVBand 48679; EPE 3; Shots: 560
[Testcase 2]: L2 30902; PVBand 38957; EPE 0; Shots: 535
[Testcase 3]: L2 66057; PVBand 71527; EPE 24; Shots: 631
[Testcase 4]: L2 8959; PVBand 24367; EPE 0; Shots: 481
[Testcase 5]: L2 30277; PVBand 54404; EPE 0; Shots: 557
[Testcase 6]: L2 30166; PVBand 48403; EPE 0; Shots: 568
[Testcase 7]: L2 16095; PVBand 41369; EPE 0; Shots: 467
[Testcase 8]: L2 11777; PVBand 21447; EPE 0; Shots: 511
[Testcase 9]: L2 34859; PVBand 62435; EPE 0; Shots: 588
[Testcase 10]: L2 7756; PVBand 17288; EPE 0; Shots: 337
[Finetuned]: L2 27608; PVBand 42888; EPE 2.7; Shots: 524

[ViaSet]
[Evaluation]: Getting masks from the model
[Testcase 1]: L2 2650; PVBand 4637; EPE 0; Shots: 78
[Testcase 2]: L2 3051; PVBand 4819; EPE 0; Shots: 111
[Testcase 3]: L2 7977; PVBand 9389; EPE 0; Shots: 186
[Testcase 4]: L2 10580; PVBand 6616; EPE 1; Shots: 95
[Testcase 5]: L2 10758; PVBand 13340; EPE 0; Shots: 256
[Testcase 6]: L2 9080; PVBand 11629; EPE 0; Shots: 203
[Testcase 7]: L2 7322; PVBand 7283; EPE 0; Shots: 137
[Testcase 8]: L2 21484; PVBand 22286; EPE 0; Shots: 411
[Testcase 9]: L2 13931; PVBand 14337; EPE 0; Shots: 256
[Testcase 10]: L2 2657; PVBand 4561; EPE 0; Shots: 109
[Initialized]: L2 8949; PVBand 9890; EPE 0.1; Runtime: 0.59s; Shots: 184
[Evaluation]: Finetuning the masks with pixel-based ILT method
[Testcase 1]: L2 2736; PVBand 4593; EPE 0; Shots: 125
[Testcase 2]: L2 2940; PVBand 4518; EPE 0; Shots: 153
[Testcase 3]: L2 4788; PVBand 8620; EPE 0; Shots: 311
[Testcase 4]: L2 3266; PVBand 6212; EPE 0; Shots: 194
[Testcase 5]: L2 7561; PVBand 12627; EPE 0; Shots: 387
[Testcase 6]: L2 6681; PVBand 10878; EPE 0; Shots: 382
[Testcase 7]: L2 3674; PVBand 6753; EPE 0; Shots: 215
[Testcase 8]: L2 11404; PVBand 21062; EPE 0; Shots: 524
[Testcase 9]: L2 8183; PVBand 14210; EPE 1; Shots: 400
[Testcase 10]: L2 3914; PVBand 5016; EPE 1; Shots: 142
[Finetuned]: L2 5515; PVBand 9449; EPE 0.2; Shots: 283

[StdMetal]
[Evaluation]: Getting masks from the model
[Testcase 1]: L2 26623; PVBand 28511; EPE 7
[Testcase 2]: L2 6720; PVBand 6755; EPE 0
[Testcase 3]: L2 8629; PVBand 11953; EPE 0
[Testcase 4]: L2 18723; PVBand 16881; EPE 0
[Testcase 5]: L2 5900; PVBand 5711; EPE 0
[Testcase 6]: L2 4271; PVBand 5905; EPE 0
[Testcase 7]: L2 14106; PVBand 12100; EPE 0
[Testcase 8]: L2 13422; PVBand 11837; EPE 11
[Testcase 9]: L2 18974; PVBand 16463; EPE 0
[Testcase 10]: L2 22129; PVBand 31177; EPE 1
[Testcase 11]: L2 33452; PVBand 31876; EPE 2
[Testcase 12]: L2 10574; PVBand 6545; EPE 0
[Testcase 13]: L2 51825; PVBand 54816; EPE 8
[Testcase 14]: L2 28741; PVBand 29752; EPE 4
[Testcase 15]: L2 30731; PVBand 31032; EPE 2
[Testcase 16]: L2 17308; PVBand 15474; EPE 0
[Testcase 17]: L2 18066; PVBand 20375; EPE 1
[Testcase 18]: L2 26721; PVBand 31127; EPE 4
[Testcase 19]: L2 11072; PVBand 7203; EPE 0
[Testcase 20]: L2 11028; PVBand 13195; EPE 0
[Testcase 21]: L2 40981; PVBand 50502; EPE 6
[Testcase 22]: L2 19485; PVBand 24178; EPE 1
[Testcase 23]: L2 55119; PVBand 38180; EPE 14
[Testcase 24]: L2 20430; PVBand 23246; EPE 2
[Testcase 25]: L2 14074; PVBand 15022; EPE 1
[Testcase 26]: L2 22787; PVBand 26243; EPE 2
[Testcase 27]: L2 27838; PVBand 23404; EPE 4
[Testcase 28]: L2 8304; PVBand 7430; EPE 0
[Testcase 29]: L2 56315; PVBand 59736; EPE 5
[Testcase 30]: L2 54772; PVBand 54326; EPE 8
[Testcase 31]: L2 53492; PVBand 37156; EPE 4
[Testcase 32]: L2 7984; PVBand 8656; EPE 0
[Testcase 33]: L2 10777; PVBand 9781; EPE 0
[Testcase 34]: L2 8625; PVBand 7502; EPE 1
[Testcase 35]: L2 20909; PVBand 24097; EPE 3
[Testcase 36]: L2 53883; PVBand 66824; EPE 11
[Testcase 37]: L2 19708; PVBand 16473; EPE 0
[Testcase 38]: L2 102813; PVBand 79973; EPE 40
[Testcase 39]: L2 12432; PVBand 7249; EPE 0
[Testcase 40]: L2 20287; PVBand 16758; EPE 4
[Testcase 41]: L2 98809; PVBand 97829; EPE 31
[Testcase 42]: L2 12443; PVBand 9734; EPE 0
[Testcase 43]: L2 38308; PVBand 42341; EPE 1
[Testcase 44]: L2 92679; PVBand 72590; EPE 23
[Testcase 45]: L2 24343; PVBand 29516; EPE 1
[Testcase 46]: L2 20311; PVBand 24206; EPE 0
[Testcase 47]: L2 23344; PVBand 23771; EPE 2
[Testcase 48]: L2 17127; PVBand 16289; EPE 1
[Testcase 49]: L2 44581; PVBand 49770; EPE 12
[Testcase 50]: L2 20739; PVBand 21078; EPE 3
[Testcase 51]: L2 6643; PVBand 6667; EPE 0
[Testcase 52]: L2 8688; PVBand 6612; EPE 0
[Testcase 53]: L2 49069; PVBand 31921; EPE 8
[Testcase 54]: L2 27118; PVBand 30106; EPE 3
[Testcase 55]: L2 26623; PVBand 28511; EPE 7
[Testcase 56]: L2 5776; PVBand 6007; EPE 0
[Testcase 57]: L2 44986; PVBand 48418; EPE 2
[Testcase 58]: L2 9137; PVBand 7082; EPE 2
[Testcase 59]: L2 25128; PVBand 26344; EPE 3
[Testcase 60]: L2 31299; PVBand 33541; EPE 2
[Testcase 61]: L2 5914; PVBand 5952; EPE 0
[Testcase 62]: L2 50343; PVBand 53285; EPE 10
[Testcase 63]: L2 44105; PVBand 50671; EPE 9
[Testcase 64]: L2 17568; PVBand 20450; EPE 0
[Testcase 65]: L2 19815; PVBand 22367; EPE 3
[Testcase 66]: L2 64531; PVBand 53477; EPE 18
[Testcase 67]: L2 21123; PVBand 19853; EPE 2
[Testcase 68]: L2 8859; PVBand 6073; EPE 0
[Testcase 69]: L2 32135; PVBand 36026; EPE 3
[Testcase 70]: L2 17602; PVBand 23466; EPE 1
[Testcase 71]: L2 19485; PVBand 24178; EPE 1
[Testcase 72]: L2 17168; PVBand 17621; EPE 0
[Testcase 73]: L2 44721; PVBand 58270; EPE 8
[Testcase 74]: L2 15494; PVBand 17124; EPE 2
[Testcase 75]: L2 16164; PVBand 17485; EPE 0
[Testcase 76]: L2 5900; PVBand 5711; EPE 0
[Testcase 77]: L2 10574; PVBand 6545; EPE 0
[Testcase 78]: L2 5600; PVBand 5866; EPE 0
[Testcase 79]: L2 53334; PVBand 50329; EPE 19
[Testcase 80]: L2 22924; PVBand 25369; EPE 1
[Testcase 81]: L2 22314; PVBand 23826; EPE 2
[Testcase 82]: L2 20793; PVBand 23571; EPE 2
[Testcase 83]: L2 14592; PVBand 15203; EPE 0
[Testcase 84]: L2 28493; PVBand 32101; EPE 4
[Testcase 85]: L2 9041; PVBand 6884; EPE 0
[Testcase 86]: L2 17127; PVBand 16289; EPE 1
[Testcase 87]: L2 20570; PVBand 20265; EPE 2
[Testcase 88]: L2 19592; PVBand 24263; EPE 1
[Testcase 89]: L2 49316; PVBand 59243; EPE 8
[Testcase 90]: L2 7691; PVBand 6426; EPE 0
[Testcase 91]: L2 76993; PVBand 72135; EPE 20
[Testcase 92]: L2 20954; PVBand 23979; EPE 3
[Testcase 93]: L2 7455; PVBand 5759; EPE 0
[Testcase 94]: L2 18575; PVBand 16919; EPE 5
[Testcase 95]: L2 69055; PVBand 55079; EPE 16
[Testcase 96]: L2 10224; PVBand 10515; EPE 0
[Testcase 97]: L2 40948; PVBand 38826; EPE 9
[Testcase 98]: L2 19815; PVBand 22367; EPE 3
[Testcase 99]: L2 7560; PVBand 6384; EPE 0
[Testcase 100]: L2 18723; PVBand 16881; EPE 0
[Testcase 101]: L2 28741; PVBand 29752; EPE 4
[Testcase 102]: L2 9069; PVBand 7532; EPE 0
[Testcase 103]: L2 54785; PVBand 54587; EPE 12
[Testcase 104]: L2 40566; PVBand 42882; EPE 5
[Testcase 105]: L2 26678; PVBand 22532; EPE 5
[Testcase 106]: L2 4271; PVBand 5905; EPE 0
[Testcase 107]: L2 21706; PVBand 22439; EPE 1
[Testcase 108]: L2 15748; PVBand 21599; EPE 0
[Testcase 109]: L2 8608; PVBand 6924; EPE 0
[Testcase 110]: L2 17751; PVBand 16807; EPE 0
[Testcase 111]: L2 23665; PVBand 26413; EPE 3
[Testcase 112]: L2 51005; PVBand 56960; EPE 10
[Testcase 113]: L2 23665; PVBand 26413; EPE 3
[Testcase 114]: L2 10217; PVBand 6513; EPE 5
[Testcase 115]: L2 28785; PVBand 27330; EPE 0
[Testcase 116]: L2 44093; PVBand 29489; EPE 12
[Testcase 117]: L2 45283; PVBand 50140; EPE 7
[Testcase 118]: L2 8745; PVBand 6429; EPE 0
[Testcase 119]: L2 19872; PVBand 21115; EPE 0
[Testcase 120]: L2 17346; PVBand 16614; EPE 4
[Testcase 121]: L2 25190; PVBand 28744; EPE 1
[Testcase 122]: L2 33201; PVBand 30359; EPE 5
[Testcase 123]: L2 7560; PVBand 6384; EPE 0
[Testcase 124]: L2 9613; PVBand 6493; EPE 0
[Testcase 125]: L2 16865; PVBand 18182; EPE 0
[Testcase 126]: L2 25349; PVBand 28926; EPE 1
[Testcase 127]: L2 42249; PVBand 48728; EPE 4
[Testcase 128]: L2 29035; PVBand 40690; EPE 0
[Testcase 129]: L2 20553; PVBand 23082; EPE 2
[Testcase 130]: L2 19952; PVBand 19784; EPE 5
[Testcase 131]: L2 9857; PVBand 7301; EPE 0
[Testcase 132]: L2 19074; PVBand 16403; EPE 0
[Testcase 133]: L2 54139; PVBand 36241; EPE 17
[Testcase 134]: L2 17127; PVBand 15468; EPE 2
[Testcase 135]: L2 47393; PVBand 32602; EPE 13
[Testcase 136]: L2 30108; PVBand 34103; EPE 6
[Testcase 137]: L2 8745; PVBand 6429; EPE 0
[Testcase 138]: L2 16848; PVBand 12921; EPE 0
[Testcase 139]: L2 49050; PVBand 33427; EPE 21
[Testcase 140]: L2 9823; PVBand 13575; EPE 0
[Testcase 141]: L2 48037; PVBand 42943; EPE 8
[Testcase 142]: L2 7119; PVBand 6294; EPE 0
[Testcase 143]: L2 22179; PVBand 20230; EPE 1
[Testcase 144]: L2 6643; PVBand 6667; EPE 0
[Testcase 145]: L2 16059; PVBand 17410; EPE 1
[Testcase 146]: L2 5600; PVBand 5866; EPE 0
[Testcase 147]: L2 22787; PVBand 26243; EPE 2
[Testcase 148]: L2 8270; PVBand 6222; EPE 0
[Testcase 149]: L2 8946; PVBand 8365; EPE 0
[Testcase 150]: L2 18575; PVBand 16919; EPE 5
[Testcase 151]: L2 7691; PVBand 6426; EPE 0
[Testcase 152]: L2 8304; PVBand 7430; EPE 0
[Testcase 153]: L2 33035; PVBand 43084; EPE 3
[Testcase 154]: L2 34888; PVBand 42105; EPE 10
[Testcase 155]: L2 43436; PVBand 41727; EPE 7
[Testcase 156]: L2 17540; PVBand 19891; EPE 1
[Testcase 157]: L2 10396; PVBand 14940; EPE 0
[Testcase 158]: L2 75516; PVBand 71412; EPE 20
[Testcase 159]: L2 18711; PVBand 18648; EPE 1
[Testcase 160]: L2 32044; PVBand 39933; EPE 3
[Testcase 161]: L2 33187; PVBand 36305; EPE 13
[Testcase 162]: L2 8270; PVBand 6222; EPE 0
[Testcase 163]: L2 16865; PVBand 18182; EPE 0
[Testcase 164]: L2 5068; PVBand 6054; EPE 0
[Testcase 165]: L2 50519; PVBand 48975; EPE 9
[Testcase 166]: L2 49234; PVBand 46660; EPE 16
[Testcase 167]: L2 71736; PVBand 46165; EPE 37
[Testcase 168]: L2 19558; PVBand 23408; EPE 3
[Testcase 169]: L2 41235; PVBand 46417; EPE 7
[Testcase 170]: L2 33260; PVBand 37718; EPE 4
[Testcase 171]: L2 15176; PVBand 15197; EPE 1
[Testcase 172]: L2 12588; PVBand 16203; EPE 1
[Testcase 173]: L2 61663; PVBand 63583; EPE 11
[Testcase 174]: L2 61318; PVBand 60050; EPE 13
[Testcase 175]: L2 57707; PVBand 56196; EPE 4
[Testcase 176]: L2 21937; PVBand 18305; EPE 0
[Testcase 177]: L2 25759; PVBand 27413; EPE 5
[Testcase 178]: L2 14106; PVBand 12100; EPE 0
[Testcase 179]: L2 20287; PVBand 16758; EPE 4
[Testcase 180]: L2 27366; PVBand 27683; EPE 1
[Testcase 181]: L2 30672; PVBand 31127; EPE 8
[Testcase 182]: L2 4271; PVBand 5905; EPE 0
[Testcase 183]: L2 15618; PVBand 17797; EPE 0
[Testcase 184]: L2 50958; PVBand 47406; EPE 19
[Testcase 185]: L2 22195; PVBand 22325; EPE 3
[Testcase 186]: L2 73338; PVBand 74919; EPE 18
[Testcase 187]: L2 11028; PVBand 13195; EPE 0
[Testcase 188]: L2 12168; PVBand 12272; EPE 0
[Testcase 189]: L2 31597; PVBand 43010; EPE 5
[Testcase 190]: L2 8194; PVBand 7504; EPE 0
[Testcase 191]: L2 22179; PVBand 20230; EPE 1
[Testcase 192]: L2 46651; PVBand 59526; EPE 19
[Testcase 193]: L2 45146; PVBand 45064; EPE 6
[Testcase 194]: L2 40948; PVBand 38826; EPE 9
[Testcase 195]: L2 25634; PVBand 30950; EPE 2
[Testcase 196]: L2 21946; PVBand 24816; EPE 0
[Testcase 197]: L2 9069; PVBand 7532; EPE 0
[Testcase 198]: L2 20793; PVBand 23571; EPE 2
[Testcase 199]: L2 28605; PVBand 34155; EPE 5
[Testcase 200]: L2 28197; PVBand 27911; EPE 5
[Testcase 201]: L2 8608; PVBand 6924; EPE 0
[Testcase 202]: L2 11072; PVBand 7203; EPE 0
[Testcase 203]: L2 39517; PVBand 40470; EPE 5
[Testcase 204]: L2 7683; PVBand 6013; EPE 0
[Testcase 205]: L2 81976; PVBand 82111; EPE 20
[Testcase 206]: L2 18452; PVBand 22032; EPE 0
[Testcase 207]: L2 45109; PVBand 64676; EPE 6
[Testcase 208]: L2 14106; PVBand 12100; EPE 0
[Testcase 209]: L2 17579; PVBand 20176; EPE 1
[Testcase 210]: L2 63664; PVBand 60202; EPE 12
[Testcase 211]: L2 9862; PVBand 11131; EPE 0
[Testcase 212]: L2 7472; PVBand 5760; EPE 0
[Testcase 213]: L2 20430; PVBand 23246; EPE 2
[Testcase 214]: L2 24598; PVBand 23689; EPE 3
[Testcase 215]: L2 55972; PVBand 57160; EPE 9
[Testcase 216]: L2 25645; PVBand 28023; EPE 4
[Testcase 217]: L2 45791; PVBand 50038; EPE 9
[Testcase 218]: L2 15748; PVBand 21599; EPE 0
[Testcase 219]: L2 18571; PVBand 20787; EPE 0
[Testcase 220]: L2 45624; PVBand 48933; EPE 7
[Testcase 221]: L2 24343; PVBand 29516; EPE 1
[Testcase 222]: L2 10556; PVBand 8917; EPE 0
[Testcase 223]: L2 7432; PVBand 5907; EPE 0
[Testcase 224]: L2 22924; PVBand 25369; EPE 1
[Testcase 225]: L2 21197; PVBand 16050; EPE 0
[Testcase 226]: L2 23984; PVBand 29722; EPE 1
[Testcase 227]: L2 26652; PVBand 30837; EPE 2
[Testcase 228]: L2 23024; PVBand 24852; EPE 0
[Testcase 229]: L2 43299; PVBand 50762; EPE 4
[Testcase 230]: L2 62076; PVBand 57003; EPE 15
[Testcase 231]: L2 29119; PVBand 31939; EPE 6
[Testcase 232]: L2 35621; PVBand 46229; EPE 10
[Testcase 233]: L2 24176; PVBand 24535; EPE 4
[Testcase 234]: L2 17127; PVBand 16289; EPE 1
[Testcase 235]: L2 51668; PVBand 38917; EPE 9
[Testcase 236]: L2 16164; PVBand 17485; EPE 0
[Testcase 237]: L2 24341; PVBand 26829; EPE 3
[Testcase 238]: L2 32815; PVBand 35211; EPE 9
[Testcase 239]: L2 49361; PVBand 51711; EPE 6
[Testcase 240]: L2 14106; PVBand 12100; EPE 0
[Testcase 241]: L2 25396; PVBand 27898; EPE 2
[Testcase 242]: L2 8688; PVBand 6612; EPE 0
[Testcase 243]: L2 7182; PVBand 6765; EPE 0
[Testcase 244]: L2 24341; PVBand 26829; EPE 3
[Testcase 245]: L2 39301; PVBand 42536; EPE 14
[Testcase 246]: L2 37762; PVBand 37565; EPE 5
[Testcase 247]: L2 28493; PVBand 32101; EPE 4
[Testcase 248]: L2 93127; PVBand 75998; EPE 25
[Testcase 249]: L2 7472; PVBand 5760; EPE 0
[Testcase 250]: L2 25759; PVBand 27413; EPE 5
[Testcase 251]: L2 17127; PVBand 16289; EPE 1
[Testcase 252]: L2 14951; PVBand 13754; EPE 3
[Testcase 253]: L2 9137; PVBand 7082; EPE 2
[Testcase 254]: L2 20553; PVBand 23082; EPE 2
[Testcase 255]: L2 33085; PVBand 31213; EPE 4
[Testcase 256]: L2 7683; PVBand 6013; EPE 0
[Testcase 257]: L2 85690; PVBand 80431; EPE 11
[Testcase 258]: L2 33011; PVBand 36570; EPE 3
[Testcase 259]: L2 48057; PVBand 46343; EPE 9
[Testcase 260]: L2 7984; PVBand 8656; EPE 0
[Testcase 261]: L2 74375; PVBand 56685; EPE 24
[Testcase 262]: L2 43434; PVBand 30175; EPE 20
[Testcase 263]: L2 23709; PVBand 20696; EPE 6
[Testcase 264]: L2 24728; PVBand 25453; EPE 6
[Testcase 265]: L2 11086; PVBand 9104; EPE 1
[Testcase 266]: L2 40979; PVBand 44630; EPE 4
[Testcase 267]: L2 19872; PVBand 21115; EPE 0
[Testcase 268]: L2 12432; PVBand 7249; EPE 0
[Testcase 269]: L2 9899; PVBand 11731; EPE 1
[Testcase 270]: L2 18931; PVBand 26603; EPE 0
[Testcase 271]: L2 8946; PVBand 8365; EPE 0
[Initialized]: L2 26809; PVBand 26814; EPE 4.2
[Evaluation]: Finetuning the masks with pixel-based ILT method
[Testcase 1]: L2 9424; PVBand 20366; EPE 0
[Testcase 2]: L2 5033; PVBand 7169; EPE 0
[Testcase 3]: L2 6633; PVBand 12447; EPE 0
[Testcase 4]: L2 8752; PVBand 16817; EPE 0
[Testcase 5]: L2 4675; PVBand 6160; EPE 0
[Testcase 6]: L2 3108; PVBand 6218; EPE 0
[Testcase 7]: L2 9308; PVBand 13311; EPE 0
[Testcase 8]: L2 3904; PVBand 7685; EPE 0
[Testcase 9]: L2 10852; PVBand 18865; EPE 0
[Testcase 10]: L2 10564; PVBand 30545; EPE 0
[Testcase 11]: L2 14576; PVBand 32749; EPE 0
[Testcase 12]: L2 5614; PVBand 6826; EPE 0
[Testcase 13]: L2 24198; PVBand 43661; EPE 0
[Testcase 14]: L2 10900; PVBand 24233; EPE 0
[Testcase 15]: L2 13529; PVBand 26346; EPE 0
[Testcase 16]: L2 10268; PVBand 15761; EPE 0
[Testcase 17]: L2 9514; PVBand 18226; EPE 0
[Testcase 18]: L2 14348; PVBand 32569; EPE 0
[Testcase 19]: L2 5299; PVBand 7250; EPE 0
[Testcase 20]: L2 8086; PVBand 13834; EPE 0
[Testcase 21]: L2 19776; PVBand 46455; EPE 0
[Testcase 22]: L2 9538; PVBand 21583; EPE 0
[Testcase 23]: L2 20252; PVBand 40403; EPE 0
[Testcase 24]: L2 9776; PVBand 19577; EPE 0
[Testcase 25]: L2 9278; PVBand 15443; EPE 0
[Testcase 26]: L2 9340; PVBand 20441; EPE 0
[Testcase 27]: L2 13786; PVBand 21867; EPE 0
[Testcase 28]: L2 3092; PVBand 6868; EPE 0
[Testcase 29]: L2 29084; PVBand 54582; EPE 0
[Testcase 30]: L2 27106; PVBand 50379; EPE 0
[Testcase 31]: L2 21527; PVBand 39812; EPE 0
[Testcase 32]: L2 6078; PVBand 9046; EPE 0
[Testcase 33]: L2 6595; PVBand 10477; EPE 0
[Testcase 34]: L2 5069; PVBand 7867; EPE 0
[Testcase 35]: L2 8691; PVBand 20216; EPE 0
[Testcase 36]: L2 28133; PVBand 63011; EPE 1
[Testcase 37]: L2 11066; PVBand 18787; EPE 0
[Testcase 38]: L2 53639; PVBand 80964; EPE 0
[Testcase 39]: L2 5463; PVBand 7598; EPE 0
[Testcase 40]: L2 8696; PVBand 14564; EPE 0
[Testcase 41]: L2 54125; PVBand 89049; EPE 3
[Testcase 42]: L2 6646; PVBand 10486; EPE 0
[Testcase 43]: L2 21505; PVBand 40102; EPE 0
[Testcase 44]: L2 40508; PVBand 75981; EPE 0
[Testcase 45]: L2 11460; PVBand 25509; EPE 0
[Testcase 46]: L2 11151; PVBand 23446; EPE 0
[Testcase 47]: L2 10921; PVBand 23522; EPE 0
[Testcase 48]: L2 8137; PVBand 14648; EPE 0
[Testcase 49]: L2 18850; PVBand 47831; EPE 0
[Testcase 50]: L2 10705; PVBand 19546; EPE 0
[Testcase 51]: L2 4939; PVBand 7044; EPE 0
[Testcase 52]: L2 4309; PVBand 7025; EPE 0
[Testcase 53]: L2 17783; PVBand 34756; EPE 0
[Testcase 54]: L2 12991; PVBand 26465; EPE 0
[Testcase 55]: L2 9424; PVBand 20366; EPE 0
[Testcase 56]: L2 4787; PVBand 6825; EPE 0
[Testcase 57]: L2 24950; PVBand 48782; EPE 0
[Testcase 58]: L2 4698; PVBand 7330; EPE 0
[Testcase 59]: L2 14299; PVBand 24306; EPE 0
[Testcase 60]: L2 18047; PVBand 33603; EPE 0
[Testcase 61]: L2 4041; PVBand 6235; EPE 0
[Testcase 62]: L2 29199; PVBand 52226; EPE 0
[Testcase 63]: L2 24016; PVBand 48878; EPE 0
[Testcase 64]: L2 10852; PVBand 19423; EPE 0
[Testcase 65]: L2 9935; PVBand 18310; EPE 0
[Testcase 66]: L2 31531; PVBand 53706; EPE 1
[Testcase 67]: L2 12157; PVBand 19208; EPE 0
[Testcase 68]: L2 4551; PVBand 6533; EPE 0
[Testcase 69]: L2 14458; PVBand 30390; EPE 0
[Testcase 70]: L2 7582; PVBand 21843; EPE 0
[Testcase 71]: L2 9538; PVBand 21583; EPE 0
[Testcase 72]: L2 9963; PVBand 16074; EPE 0
[Testcase 73]: L2 24089; PVBand 54867; EPE 0
[Testcase 74]: L2 8493; PVBand 17252; EPE 0
[Testcase 75]: L2 9021; PVBand 16075; EPE 0
[Testcase 76]: L2 4675; PVBand 6160; EPE 0
[Testcase 77]: L2 5614; PVBand 6826; EPE 0
[Testcase 78]: L2 4778; PVBand 6291; EPE 0
[Testcase 79]: L2 20918; PVBand 41517; EPE 0
[Testcase 80]: L2 9897; PVBand 20317; EPE 0
[Testcase 81]: L2 9731; PVBand 25024; EPE 0
[Testcase 82]: L2 10051; PVBand 19463; EPE 0
[Testcase 83]: L2 7578; PVBand 13334; EPE 0
[Testcase 84]: L2 15183; PVBand 30333; EPE 0
[Testcase 85]: L2 5505; PVBand 7521; EPE 0
[Testcase 86]: L2 8137; PVBand 14648; EPE 0
[Testcase 87]: L2 9829; PVBand 19209; EPE 0
[Testcase 88]: L2 10533; PVBand 24784; EPE 0
[Testcase 89]: L2 25206; PVBand 59718; EPE 0
[Testcase 90]: L2 4182; PVBand 6599; EPE 0
[Testcase 91]: L2 36598; PVBand 71096; EPE 0
[Testcase 92]: L2 7367; PVBand 21238; EPE 0
[Testcase 93]: L2 4943; PVBand 6130; EPE 0
[Testcase 94]: L2 8270; PVBand 14497; EPE 0
[Testcase 95]: L2 27754; PVBand 52281; EPE 1
[Testcase 96]: L2 6901; PVBand 10969; EPE 0
[Testcase 97]: L2 18301; PVBand 35606; EPE 0
[Testcase 98]: L2 9935; PVBand 18310; EPE 0
[Testcase 99]: L2 4125; PVBand 6586; EPE 0
[Testcase 100]: L2 8752; PVBand 16817; EPE 0
[Testcase 101]: L2 10900; PVBand 24233; EPE 0
[Testcase 102]: L2 4197; PVBand 8172; EPE 0
[Testcase 103]: L2 26416; PVBand 50070; EPE 0
[Testcase 104]: L2 20289; PVBand 38302; EPE 0
[Testcase 105]: L2 11845; PVBand 19993; EPE 0
[Testcase 106]: L2 3108; PVBand 6218; EPE 0
[Testcase 107]: L2 11292; PVBand 20252; EPE 0
[Testcase 108]: L2 8723; PVBand 18707; EPE 0
[Testcase 109]: L2 3864; PVBand 7119; EPE 0
[Testcase 110]: L2 9202; PVBand 19068; EPE 0
[Testcase 111]: L2 9693; PVBand 20211; EPE 0
[Testcase 112]: L2 21394; PVBand 47429; EPE 0
[Testcase 113]: L2 9693; PVBand 20211; EPE 0
[Testcase 114]: L2 3763; PVBand 6290; EPE 0
[Testcase 115]: L2 16007; PVBand 27212; EPE 0
[Testcase 116]: L2 17646; PVBand 33557; EPE 0
[Testcase 117]: L2 23885; PVBand 46238; EPE 0
[Testcase 118]: L2 4521; PVBand 7058; EPE 0
[Testcase 119]: L2 11293; PVBand 17943; EPE 0
[Testcase 120]: L2 7904; PVBand 13979; EPE 0
[Testcase 121]: L2 11255; PVBand 24539; EPE 0
[Testcase 122]: L2 18569; PVBand 31004; EPE 0
[Testcase 123]: L2 4125; PVBand 6586; EPE 0
[Testcase 124]: L2 5685; PVBand 7060; EPE 0
[Testcase 125]: L2 9318; PVBand 16968; EPE 0
[Testcase 126]: L2 13103; PVBand 25613; EPE 0
[Testcase 127]: L2 20526; PVBand 46593; EPE 0
[Testcase 128]: L2 16715; PVBand 40085; EPE 0
[Testcase 129]: L2 9352; PVBand 18402; EPE 0
[Testcase 130]: L2 7707; PVBand 13940; EPE 0
[Testcase 131]: L2 5534; PVBand 8037; EPE 0
[Testcase 132]: L2 10557; PVBand 18144; EPE 0
[Testcase 133]: L2 23228; PVBand 37707; EPE 0
[Testcase 134]: L2 9843; PVBand 15804; EPE 0
[Testcase 135]: L2 19781; PVBand 37432; EPE 0
[Testcase 136]: L2 12375; PVBand 26125; EPE 0
[Testcase 137]: L2 4521; PVBand 7058; EPE 0
[Testcase 138]: L2 9298; PVBand 13824; EPE 0
[Testcase 139]: L2 16804; PVBand 32715; EPE 0
[Testcase 140]: L2 4791; PVBand 13071; EPE 0
[Testcase 141]: L2 20092; PVBand 40171; EPE 0
[Testcase 142]: L2 4673; PVBand 6455; EPE 0
[Testcase 143]: L2 11665; PVBand 20348; EPE 0
[Testcase 144]: L2 4939; PVBand 7044; EPE 0
[Testcase 145]: L2 9683; PVBand 15907; EPE 0
[Testcase 146]: L2 4778; PVBand 6291; EPE 0
[Testcase 147]: L2 9340; PVBand 20441; EPE 0
[Testcase 148]: L2 4766; PVBand 6670; EPE 0
[Testcase 149]: L2 5307; PVBand 8536; EPE 0
[Testcase 150]: L2 8270; PVBand 14497; EPE 0
[Testcase 151]: L2 4182; PVBand 6599; EPE 0
[Testcase 152]: L2 3092; PVBand 6868; EPE 0
[Testcase 153]: L2 16297; PVBand 46346; EPE 0
[Testcase 154]: L2 17575; PVBand 41685; EPE 0
[Testcase 155]: L2 19688; PVBand 39670; EPE 0
[Testcase 156]: L2 9564; PVBand 17621; EPE 0
[Testcase 157]: L2 5225; PVBand 14915; EPE 0
[Testcase 158]: L2 36279; PVBand 64558; EPE 0
[Testcase 159]: L2 9542; PVBand 19395; EPE 0
[Testcase 160]: L2 16658; PVBand 33933; EPE 0
[Testcase 161]: L2 12411; PVBand 25858; EPE 0
[Testcase 162]: L2 4766; PVBand 6670; EPE 0
[Testcase 163]: L2 9318; PVBand 16968; EPE 0
[Testcase 164]: L2 4221; PVBand 6380; EPE 0
[Testcase 165]: L2 23021; PVBand 48945; EPE 0
[Testcase 166]: L2 19394; PVBand 38339; EPE 0
[Testcase 167]: L2 22938; PVBand 44002; EPE 0
[Testcase 168]: L2 8903; PVBand 19795; EPE 0
[Testcase 169]: L2 18846; PVBand 40617; EPE 0
[Testcase 170]: L2 16308; PVBand 34377; EPE 0
[Testcase 171]: L2 4786; PVBand 12625; EPE 0
[Testcase 172]: L2 5178; PVBand 14603; EPE 0
[Testcase 173]: L2 34816; PVBand 56101; EPE 0
[Testcase 174]: L2 26658; PVBand 53116; EPE 0
[Testcase 175]: L2 32344; PVBand 55269; EPE 0
[Testcase 176]: L2 14181; PVBand 21096; EPE 0
[Testcase 177]: L2 9450; PVBand 20048; EPE 0
[Testcase 178]: L2 9308; PVBand 13311; EPE 0
[Testcase 179]: L2 8696; PVBand 14564; EPE 0
[Testcase 180]: L2 12897; PVBand 25231; EPE 0
[Testcase 181]: L2 10734; PVBand 24850; EPE 0
[Testcase 182]: L2 3108; PVBand 6218; EPE 0
[Testcase 183]: L2 6941; PVBand 17941; EPE 0
[Testcase 184]: L2 18546; PVBand 39118; EPE 0
[Testcase 185]: L2 7504; PVBand 20243; EPE 0
[Testcase 186]: L2 35604; PVBand 72369; EPE 0
[Testcase 187]: L2 8086; PVBand 13834; EPE 0
[Testcase 188]: L2 8186; PVBand 11981; EPE 0
[Testcase 189]: L2 13694; PVBand 37859; EPE 0
[Testcase 190]: L2 5040; PVBand 7488; EPE 0
[Testcase 191]: L2 11665; PVBand 20348; EPE 0
[Testcase 192]: L2 15447; PVBand 46295; EPE 0
[Testcase 193]: L2 19709; PVBand 44120; EPE 0
[Testcase 194]: L2 18301; PVBand 35606; EPE 0
[Testcase 195]: L2 11276; PVBand 25721; EPE 0
[Testcase 196]: L2 13394; PVBand 23697; EPE 0
[Testcase 197]: L2 4197; PVBand 8172; EPE 0
[Testcase 198]: L2 10051; PVBand 19463; EPE 0
[Testcase 199]: L2 12909; PVBand 30560; EPE 0
[Testcase 200]: L2 12588; PVBand 25074; EPE 0
[Testcase 201]: L2 3864; PVBand 7119; EPE 0
[Testcase 202]: L2 5299; PVBand 7250; EPE 0
[Testcase 203]: L2 22246; PVBand 38011; EPE 0
[Testcase 204]: L2 5145; PVBand 6530; EPE 0
[Testcase 205]: L2 42851; PVBand 80421; EPE 1
[Testcase 206]: L2 6792; PVBand 20649; EPE 0
[Testcase 207]: L2 23837; PVBand 57045; EPE 0
[Testcase 208]: L2 9308; PVBand 13311; EPE 0
[Testcase 209]: L2 9416; PVBand 17629; EPE 0
[Testcase 210]: L2 36810; PVBand 57485; EPE 0
[Testcase 211]: L2 5322; PVBand 9889; EPE 0
[Testcase 212]: L2 4894; PVBand 6285; EPE 0
[Testcase 213]: L2 9776; PVBand 19577; EPE 0
[Testcase 214]: L2 13215; PVBand 20810; EPE 0
[Testcase 215]: L2 32946; PVBand 56380; EPE 0
[Testcase 216]: L2 12108; PVBand 25343; EPE 0
[Testcase 217]: L2 27021; PVBand 45780; EPE 4
[Testcase 218]: L2 8723; PVBand 18707; EPE 0
[Testcase 219]: L2 8297; PVBand 17935; EPE 0
[Testcase 220]: L2 19692; PVBand 48962; EPE 0
[Testcase 221]: L2 11460; PVBand 25509; EPE 0
[Testcase 222]: L2 7282; PVBand 9665; EPE 0
[Testcase 223]: L2 4511; PVBand 6239; EPE 0
[Testcase 224]: L2 9897; PVBand 20317; EPE 0
[Testcase 225]: L2 12191; PVBand 17355; EPE 0
[Testcase 226]: L2 8664; PVBand 29269; EPE 0
[Testcase 227]: L2 14050; PVBand 28482; EPE 0
[Testcase 228]: L2 6216; PVBand 22925; EPE 0
[Testcase 229]: L2 23885; PVBand 50207; EPE 0
[Testcase 230]: L2 28134; PVBand 54330; EPE 0
[Testcase 231]: L2 12597; PVBand 27481; EPE 0
[Testcase 232]: L2 14687; PVBand 37778; EPE 0
[Testcase 233]: L2 12215; PVBand 21265; EPE 0
[Testcase 234]: L2 8137; PVBand 14648; EPE 0
[Testcase 235]: L2 18865; PVBand 38799; EPE 0
[Testcase 236]: L2 9021; PVBand 16075; EPE 0
[Testcase 237]: L2 9412; PVBand 20463; EPE 0
[Testcase 238]: L2 12759; PVBand 25722; EPE 0
[Testcase 239]: L2 20994; PVBand 48277; EPE 0
[Testcase 240]: L2 9308; PVBand 13311; EPE 0
[Testcase 241]: L2 9164; PVBand 26154; EPE 0
[Testcase 242]: L2 4309; PVBand 7025; EPE 0
[Testcase 243]: L2 4219; PVBand 7097; EPE 0
[Testcase 244]: L2 9412; PVBand 20463; EPE 0
[Testcase 245]: L2 15789; PVBand 34595; EPE 0
[Testcase 246]: L2 21144; PVBand 37717; EPE 0
[Testcase 247]: L2 15183; PVBand 30333; EPE 0
[Testcase 248]: L2 40353; PVBand 72352; EPE 0
[Testcase 249]: L2 4894; PVBand 6285; EPE 0
[Testcase 250]: L2 9450; PVBand 20048; EPE 0
[Testcase 251]: L2 8137; PVBand 14648; EPE 0
[Testcase 252]: L2 7284; PVBand 11377; EPE 0
[Testcase 253]: L2 4698; PVBand 7330; EPE 0
[Testcase 254]: L2 9352; PVBand 18402; EPE 0
[Testcase 255]: L2 16626; PVBand 28679; EPE 0
[Testcase 256]: L2 5145; PVBand 6530; EPE 0
[Testcase 257]: L2 46187; PVBand 80399; EPE 3
[Testcase 258]: L2 15117; PVBand 31618; EPE 0
[Testcase 259]: L2 17903; PVBand 46464; EPE 0
[Testcase 260]: L2 6078; PVBand 9046; EPE 0
[Testcase 261]: L2 42549; PVBand 61137; EPE 0
[Testcase 262]: L2 18443; PVBand 30855; EPE 0
[Testcase 263]: L2 6766; PVBand 16228; EPE 0
[Testcase 264]: L2 9944; PVBand 22497; EPE 0
[Testcase 265]: L2 5162; PVBand 8921; EPE 0
[Testcase 266]: L2 21877; PVBand 40119; EPE 0
[Testcase 267]: L2 11293; PVBand 17943; EPE 0
[Testcase 268]: L2 5463; PVBand 7598; EPE 0
[Testcase 269]: L2 5998; PVBand 11600; EPE 0
[Testcase 270]: L2 11464; PVBand 24142; EPE 0
[Testcase 271]: L2 5307; PVBand 8536; EPE 0
[Finetuned]: L2 12957; PVBand 24999; EPE 0.1

[StdContact]
[Evaluation]: Getting masks from the model
[Testcase 1]: L2 92008; PVBand 9628; EPE 82
[Testcase 2]: L2 80341; PVBand 17191; EPE 68
[Testcase 3]: L2 55958; PVBand 18904; EPE 42
[Testcase 4]: L2 70833; PVBand 12576; EPE 58
[Testcase 5]: L2 57691; PVBand 28325; EPE 35
[Testcase 6]: L2 83753; PVBand 12984; EPE 71
[Testcase 7]: L2 57200; PVBand 13645; EPE 43
[Testcase 8]: L2 82427; PVBand 17010; EPE 68
[Testcase 9]: L2 94749; PVBand 24564; EPE 75
[Testcase 10]: L2 57853; PVBand 14007; EPE 43
[Testcase 11]: L2 71051; PVBand 17539; EPE 54
[Testcase 12]: L2 80821; PVBand 6507; EPE 73
[Testcase 13]: L2 83380; PVBand 38577; EPE 55
[Testcase 14]: L2 70077; PVBand 18855; EPE 54
[Testcase 15]: L2 67364; PVBand 12221; EPE 56
[Testcase 16]: L2 74792; PVBand 17392; EPE 57
[Testcase 17]: L2 86171; PVBand 19047; EPE 71
[Testcase 18]: L2 87937; PVBand 13283; EPE 74
[Testcase 19]: L2 64467; PVBand 9338; EPE 57
[Testcase 20]: L2 56243; PVBand 13975; EPE 44
[Testcase 21]: L2 69575; PVBand 18015; EPE 53
[Testcase 22]: L2 72180; PVBand 11075; EPE 62
[Testcase 23]: L2 71084; PVBand 12043; EPE 61
[Testcase 24]: L2 77524; PVBand 9306; EPE 69
[Testcase 25]: L2 75521; PVBand 18579; EPE 59
[Testcase 26]: L2 59666; PVBand 21495; EPE 41
[Testcase 27]: L2 55961; PVBand 16350; EPE 41
[Testcase 28]: L2 67286; PVBand 20017; EPE 51
[Testcase 29]: L2 67291; PVBand 25823; EPE 50
[Testcase 30]: L2 58752; PVBand 15576; EPE 43
[Testcase 31]: L2 66366; PVBand 11987; EPE 54
[Testcase 32]: L2 89061; PVBand 14210; EPE 74
[Testcase 33]: L2 63812; PVBand 15021; EPE 47
[Testcase 34]: L2 51628; PVBand 2374; EPE 47
[Testcase 35]: L2 97473; PVBand 37878; EPE 70
[Testcase 36]: L2 100128; PVBand 35000; EPE 69
[Testcase 37]: L2 90111; PVBand 12375; EPE 75
[Testcase 38]: L2 64889; PVBand 21749; EPE 43
[Testcase 39]: L2 67124; PVBand 13279; EPE 56
[Testcase 40]: L2 57874; PVBand 14952; EPE 43
[Testcase 41]: L2 62409; PVBand 15170; EPE 52
[Testcase 42]: L2 79351; PVBand 8002; EPE 71
[Testcase 43]: L2 52888; PVBand 22742; EPE 36
[Testcase 44]: L2 91623; PVBand 31417; EPE 63
[Testcase 45]: L2 65055; PVBand 15486; EPE 53
[Testcase 46]: L2 66172; PVBand 19364; EPE 49
[Testcase 47]: L2 50936; PVBand 18365; EPE 38
[Testcase 48]: L2 58597; PVBand 20837; EPE 40
[Testcase 49]: L2 62624; PVBand 15499; EPE 47
[Testcase 50]: L2 69810; PVBand 18294; EPE 57
[Testcase 51]: L2 65623; PVBand 30858; EPE 35
[Testcase 52]: L2 50180; PVBand 10028; EPE 42
[Testcase 53]: L2 55260; PVBand 10138; EPE 46
[Testcase 54]: L2 56861; PVBand 25456; EPE 37
[Testcase 55]: L2 69919; PVBand 18170; EPE 48
[Testcase 56]: L2 79234; PVBand 30732; EPE 58
[Testcase 57]: L2 61737; PVBand 17031; EPE 45
[Testcase 58]: L2 52236; PVBand 20455; EPE 35
[Testcase 59]: L2 88805; PVBand 32118; EPE 59
[Testcase 60]: L2 69179; PVBand 8449; EPE 59
[Testcase 61]: L2 88200; PVBand 33497; EPE 64
[Testcase 62]: L2 99289; PVBand 15786; EPE 83
[Testcase 63]: L2 52744; PVBand 7202; EPE 46
[Testcase 64]: L2 68681; PVBand 21394; EPE 53
[Testcase 65]: L2 51540; PVBand 8256; EPE 42
[Testcase 66]: L2 79087; PVBand 6745; EPE 68
[Testcase 67]: L2 93109; PVBand 10171; EPE 84
[Testcase 68]: L2 84234; PVBand 6187; EPE 76
[Testcase 69]: L2 79242; PVBand 8947; EPE 67
[Testcase 70]: L2 47946; PVBand 16891; EPE 36
[Testcase 71]: L2 75842; PVBand 7971; EPE 69
[Testcase 72]: L2 47840; PVBand 24493; EPE 32
[Testcase 73]: L2 91206; PVBand 23064; EPE 67
[Testcase 74]: L2 59855; PVBand 33940; EPE 32
[Testcase 75]: L2 80901; PVBand 12004; EPE 69
[Testcase 76]: L2 77518; PVBand 11133; EPE 68
[Testcase 77]: L2 53430; PVBand 12492; EPE 42
[Testcase 78]: L2 88553; PVBand 16090; EPE 72
[Testcase 79]: L2 79472; PVBand 18416; EPE 65
[Testcase 80]: L2 82680; PVBand 25236; EPE 61
[Testcase 81]: L2 97181; PVBand 8341; EPE 87
[Testcase 82]: L2 65495; PVBand 27445; EPE 43
[Testcase 83]: L2 88788; PVBand 8004; EPE 80
[Testcase 84]: L2 70481; PVBand 7602; EPE 61
[Testcase 85]: L2 69153; PVBand 24637; EPE 49
[Testcase 86]: L2 79351; PVBand 8113; EPE 73
[Testcase 87]: L2 83375; PVBand 31786; EPE 54
[Testcase 88]: L2 47019; PVBand 9191; EPE 38
[Testcase 89]: L2 69062; PVBand 19551; EPE 53
[Testcase 90]: L2 75755; PVBand 34607; EPE 47
[Testcase 91]: L2 63942; PVBand 34369; EPE 37
[Testcase 92]: L2 78745; PVBand 8654; EPE 70
[Testcase 93]: L2 60452; PVBand 12131; EPE 47
[Testcase 94]: L2 86219; PVBand 19074; EPE 68
[Testcase 95]: L2 56496; PVBand 21367; EPE 38
[Testcase 96]: L2 82119; PVBand 9367; EPE 71
[Testcase 97]: L2 64184; PVBand 18842; EPE 48
[Testcase 98]: L2 66622; PVBand 22821; EPE 49
[Testcase 99]: L2 74567; PVBand 12680; EPE 61
[Testcase 100]: L2 75373; PVBand 7725; EPE 68
[Testcase 101]: L2 47792; PVBand 19915; EPE 33
[Testcase 102]: L2 91778; PVBand 10030; EPE 81
[Testcase 103]: L2 101602; PVBand 24340; EPE 80
[Testcase 104]: L2 85256; PVBand 18340; EPE 67
[Testcase 105]: L2 89989; PVBand 34008; EPE 63
[Testcase 106]: L2 60675; PVBand 16317; EPE 46
[Testcase 107]: L2 51926; PVBand 22049; EPE 35
[Testcase 108]: L2 78563; PVBand 26666; EPE 58
[Testcase 109]: L2 90813; PVBand 30874; EPE 59
[Testcase 110]: L2 45371; PVBand 15038; EPE 34
[Testcase 111]: L2 98374; PVBand 27768; EPE 72
[Testcase 112]: L2 79049; PVBand 29280; EPE 56
[Testcase 113]: L2 68905; PVBand 16663; EPE 57
[Testcase 114]: L2 49977; PVBand 12522; EPE 41
[Testcase 115]: L2 53568; PVBand 13669; EPE 42
[Testcase 116]: L2 80599; PVBand 26026; EPE 61
[Testcase 117]: L2 50532; PVBand 25640; EPE 35
[Testcase 118]: L2 76938; PVBand 11612; EPE 67
[Testcase 119]: L2 100163; PVBand 8190; EPE 88
[Testcase 120]: L2 53192; PVBand 12033; EPE 44
[Testcase 121]: L2 101909; PVBand 37403; EPE 66
[Testcase 122]: L2 59646; PVBand 8677; EPE 54
[Testcase 123]: L2 54103; PVBand 11630; EPE 43
[Testcase 124]: L2 64883; PVBand 7570; EPE 57
[Testcase 125]: L2 73599; PVBand 16642; EPE 60
[Testcase 126]: L2 54466; PVBand 13025; EPE 44
[Testcase 127]: L2 78216; PVBand 34361; EPE 51
[Testcase 128]: L2 50950; PVBand 3960; EPE 47
[Testcase 129]: L2 95314; PVBand 16375; EPE 80
[Testcase 130]: L2 65456; PVBand 21681; EPE 47
[Testcase 131]: L2 90336; PVBand 22909; EPE 68
[Testcase 132]: L2 61795; PVBand 21954; EPE 49
[Testcase 133]: L2 49457; PVBand 30054; EPE 28
[Testcase 134]: L2 59668; PVBand 10053; EPE 51
[Testcase 135]: L2 77976; PVBand 10699; EPE 66
[Testcase 136]: L2 67760; PVBand 14747; EPE 54
[Testcase 137]: L2 69883; PVBand 14184; EPE 56
[Testcase 138]: L2 59093; PVBand 5921; EPE 54
[Testcase 139]: L2 58150; PVBand 14692; EPE 45
[Testcase 140]: L2 65257; PVBand 14419; EPE 51
[Testcase 141]: L2 94874; PVBand 39642; EPE 66
[Testcase 142]: L2 59504; PVBand 21548; EPE 42
[Testcase 143]: L2 53574; PVBand 3364; EPE 48
[Testcase 144]: L2 74524; PVBand 16751; EPE 59
[Testcase 145]: L2 70995; PVBand 13483; EPE 57
[Testcase 146]: L2 83540; PVBand 8190; EPE 75
[Testcase 147]: L2 88893; PVBand 41014; EPE 59
[Testcase 148]: L2 76187; PVBand 21462; EPE 61
[Testcase 149]: L2 60618; PVBand 9403; EPE 52
[Testcase 150]: L2 52097; PVBand 18608; EPE 38
[Testcase 151]: L2 61422; PVBand 12206; EPE 51
[Testcase 152]: L2 81142; PVBand 22810; EPE 61
[Testcase 153]: L2 67542; PVBand 25710; EPE 40
[Testcase 154]: L2 55944; PVBand 13819; EPE 41
[Testcase 155]: L2 59877; PVBand 21945; EPE 43
[Testcase 156]: L2 67286; PVBand 19939; EPE 53
[Testcase 157]: L2 81582; PVBand 16621; EPE 67
[Testcase 158]: L2 74802; PVBand 26315; EPE 52
[Testcase 159]: L2 64061; PVBand 9711; EPE 52
[Testcase 160]: L2 60436; PVBand 16794; EPE 48
[Testcase 161]: L2 67109; PVBand 12314; EPE 57
[Testcase 162]: L2 72737; PVBand 33675; EPE 49
[Testcase 163]: L2 67414; PVBand 17760; EPE 52
[Testcase 164]: L2 57673; PVBand 32046; EPE 33
[Testcase 165]: L2 62541; PVBand 17178; EPE 49
[Initialized]: L2 70740; PVBand 17950; EPE 55.1
[Evaluation]: Finetuning the masks with pixel-based ILT method
[Testcase 1]: L2 37862; PVBand 50716; EPE 8
[Testcase 2]: L2 27620; PVBand 47246; EPE 0
[Testcase 3]: L2 19198; PVBand 35356; EPE 0
[Testcase 4]: L2 23182; PVBand 40035; EPE 1
[Testcase 5]: L2 24379; PVBand 43740; EPE 0
[Testcase 6]: L2 29047; PVBand 48255; EPE 1
[Testcase 7]: L2 17793; PVBand 32696; EPE 0
[Testcase 8]: L2 29106; PVBand 49400; EPE 0
[Testcase 9]: L2 43082; PVBand 64287; EPE 7
[Testcase 10]: L2 22346; PVBand 36273; EPE 2
[Testcase 11]: L2 29398; PVBand 44912; EPE 8
[Testcase 12]: L2 29910; PVBand 44726; EPE 3
[Testcase 13]: L2 43442; PVBand 61480; EPE 9
[Testcase 14]: L2 24346; PVBand 40729; EPE 0
[Testcase 15]: L2 22605; PVBand 39038; EPE 1
[Testcase 16]: L2 33336; PVBand 47208; EPE 9
[Testcase 17]: L2 43582; PVBand 51134; EPE 15
[Testcase 18]: L2 37022; PVBand 48991; EPE 8
[Testcase 19]: L2 21022; PVBand 36630; EPE 0
[Testcase 20]: L2 18061; PVBand 32898; EPE 0
[Testcase 21]: L2 28810; PVBand 42339; EPE 5
[Testcase 22]: L2 23929; PVBand 41292; EPE 0
[Testcase 23]: L2 24399; PVBand 42730; EPE 2
[Testcase 24]: L2 24891; PVBand 42858; EPE 1
[Testcase 25]: L2 25849; PVBand 45861; EPE 1
[Testcase 26]: L2 20759; PVBand 37565; EPE 0
[Testcase 27]: L2 21585; PVBand 37170; EPE 2
[Testcase 28]: L2 25856; PVBand 43823; EPE 0
[Testcase 29]: L2 28461; PVBand 47176; EPE 4
[Testcase 30]: L2 17877; PVBand 32053; EPE 0
[Testcase 31]: L2 24103; PVBand 39268; EPE 3
[Testcase 32]: L2 36604; PVBand 50036; EPE 8
[Testcase 33]: L2 21516; PVBand 38305; EPE 1
[Testcase 34]: L2 14652; PVBand 26889; EPE 0
[Testcase 35]: L2 59931; PVBand 66760; EPE 27
[Testcase 36]: L2 53266; PVBand 66468; EPE 16
[Testcase 37]: L2 30263; PVBand 50484; EPE 1
[Testcase 38]: L2 24374; PVBand 41857; EPE 0
[Testcase 39]: L2 22066; PVBand 40741; EPE 0
[Testcase 40]: L2 24552; PVBand 38605; EPE 5
[Testcase 41]: L2 20036; PVBand 35713; EPE 0
[Testcase 42]: L2 30913; PVBand 45221; EPE 5
[Testcase 43]: L2 20905; PVBand 36793; EPE 2
[Testcase 44]: L2 44484; PVBand 54822; EPE 15
[Testcase 45]: L2 21177; PVBand 38913; EPE 0
[Testcase 46]: L2 24408; PVBand 40201; EPE 2
[Testcase 47]: L2 17197; PVBand 32030; EPE 0
[Testcase 48]: L2 20578; PVBand 37346; EPE 0
[Testcase 49]: L2 23116; PVBand 39551; EPE 2
[Testcase 50]: L2 24095; PVBand 40271; EPE 1
[Testcase 51]: L2 24008; PVBand 47097; EPE 0
[Testcase 52]: L2 16080; PVBand 29809; EPE 1
[Testcase 53]: L2 17427; PVBand 30600; EPE 0
[Testcase 54]: L2 24840; PVBand 41987; EPE 2
[Testcase 55]: L2 25208; PVBand 42827; EPE 1
[Testcase 56]: L2 30807; PVBand 50487; EPE 2
[Testcase 57]: L2 20050; PVBand 34305; EPE 0
[Testcase 58]: L2 21824; PVBand 37973; EPE 0
[Testcase 59]: L2 43930; PVBand 54822; EPE 13
[Testcase 60]: L2 22354; PVBand 38020; EPE 0
[Testcase 61]: L2 49818; PVBand 58591; EPE 16
[Testcase 62]: L2 43153; PVBand 60366; EPE 8
[Testcase 63]: L2 16153; PVBand 29455; EPE 0
[Testcase 64]: L2 24371; PVBand 40362; EPE 0
[Testcase 65]: L2 16911; PVBand 28775; EPE 0
[Testcase 66]: L2 25606; PVBand 43321; EPE 1
[Testcase 67]: L2 43688; PVBand 42304; EPE 20
[Testcase 68]: L2 33077; PVBand 44357; EPE 7
[Testcase 69]: L2 29293; PVBand 44551; EPE 3
[Testcase 70]: L2 17814; PVBand 30645; EPE 0
[Testcase 71]: L2 27431; PVBand 43142; EPE 3
[Testcase 72]: L2 19771; PVBand 34478; EPE 1
[Testcase 73]: L2 42124; PVBand 53278; EPE 10
[Testcase 74]: L2 24529; PVBand 47001; EPE 0
[Testcase 75]: L2 31087; PVBand 47765; EPE 3
[Testcase 76]: L2 27012; PVBand 44962; EPE 1
[Testcase 77]: L2 18335; PVBand 33053; EPE 0
[Testcase 78]: L2 41547; PVBand 50094; EPE 11
[Testcase 79]: L2 31265; PVBand 48310; EPE 6
[Testcase 80]: L2 33590; PVBand 51586; EPE 5
[Testcase 81]: L2 35437; PVBand 55000; EPE 5
[Testcase 82]: L2 28193; PVBand 46960; EPE 2
[Testcase 83]: L2 34961; PVBand 48235; EPE 6
[Testcase 84]: L2 22617; PVBand 39632; EPE 0
[Testcase 85]: L2 27761; PVBand 46539; EPE 3
[Testcase 86]: L2 29530; PVBand 44968; EPE 4
[Testcase 87]: L2 38178; PVBand 58202; EPE 6
[Testcase 88]: L2 14496; PVBand 25159; EPE 0
[Testcase 89]: L2 24550; PVBand 44049; EPE 0
[Testcase 90]: L2 33716; PVBand 51490; EPE 5
[Testcase 91]: L2 27677; PVBand 46339; EPE 1
[Testcase 92]: L2 31360; PVBand 41989; EPE 7
[Testcase 93]: L2 20503; PVBand 36155; EPE 0
[Testcase 94]: L2 37908; PVBand 51920; EPE 8
[Testcase 95]: L2 21687; PVBand 38984; EPE 0
[Testcase 96]: L2 31420; PVBand 45985; EPE 6
[Testcase 97]: L2 22418; PVBand 41694; EPE 1
[Testcase 98]: L2 27778; PVBand 45443; EPE 6
[Testcase 99]: L2 24128; PVBand 42429; EPE 0
[Testcase 100]: L2 22682; PVBand 40733; EPE 1
[Testcase 101]: L2 15697; PVBand 28624; EPE 0
[Testcase 102]: L2 36098; PVBand 49922; EPE 9
[Testcase 103]: L2 54810; PVBand 56584; EPE 22
[Testcase 104]: L2 41451; PVBand 51981; EPE 10
[Testcase 105]: L2 42596; PVBand 61164; EPE 11
[Testcase 106]: L2 25992; PVBand 39584; EPE 6
[Testcase 107]: L2 23568; PVBand 36953; EPE 4
[Testcase 108]: L2 37529; PVBand 50778; EPE 11
[Testcase 109]: L2 37304; PVBand 62735; EPE 1
[Testcase 110]: L2 16272; PVBand 28626; EPE 0
[Testcase 111]: L2 46777; PVBand 61254; EPE 11
[Testcase 112]: L2 31362; PVBand 53663; EPE 2
[Testcase 113]: L2 25606; PVBand 41853; EPE 3
[Testcase 114]: L2 16555; PVBand 30837; EPE 0
[Testcase 115]: L2 18483; PVBand 33128; EPE 0
[Testcase 116]: L2 31737; PVBand 53586; EPE 0
[Testcase 117]: L2 19238; PVBand 34791; EPE 0
[Testcase 118]: L2 34229; PVBand 39227; EPE 11
[Testcase 119]: L2 43978; PVBand 59384; EPE 10
[Testcase 120]: L2 16839; PVBand 30494; EPE 0
[Testcase 121]: L2 57498; PVBand 65762; EPE 21
[Testcase 122]: L2 18762; PVBand 33519; EPE 0
[Testcase 123]: L2 18591; PVBand 33823; EPE 0
[Testcase 124]: L2 21028; PVBand 36004; EPE 0
[Testcase 125]: L2 30814; PVBand 47295; EPE 5
[Testcase 126]: L2 18585; PVBand 32724; EPE 1
[Testcase 127]: L2 37072; PVBand 59904; EPE 5
[Testcase 128]: L2 13702; PVBand 27698; EPE 0
[Testcase 129]: L2 39574; PVBand 57250; EPE 8
[Testcase 130]: L2 23816; PVBand 44109; EPE 0
[Testcase 131]: L2 42185; PVBand 51312; EPE 14
[Testcase 132]: L2 23443; PVBand 41699; EPE 0
[Testcase 133]: L2 20991; PVBand 39866; EPE 0
[Testcase 134]: L2 20073; PVBand 34068; EPE 1
[Testcase 135]: L2 31085; PVBand 41967; EPE 7
[Testcase 136]: L2 23543; PVBand 40926; EPE 0
[Testcase 137]: L2 25802; PVBand 41581; EPE 1
[Testcase 138]: L2 17276; PVBand 32093; EPE 0
[Testcase 139]: L2 22692; PVBand 33827; EPE 4
[Testcase 140]: L2 22726; PVBand 39645; EPE 1
[Testcase 141]: L2 53278; PVBand 63763; EPE 21
[Testcase 142]: L2 22439; PVBand 42203; EPE 0
[Testcase 143]: L2 17892; PVBand 29306; EPE 2
[Testcase 144]: L2 25860; PVBand 46336; EPE 0
[Testcase 145]: L2 22059; PVBand 39714; EPE 0
[Testcase 146]: L2 30527; PVBand 47568; EPE 3
[Testcase 147]: L2 54412; PVBand 65413; EPE 19
[Testcase 148]: L2 31740; PVBand 49416; EPE 5
[Testcase 149]: L2 18733; PVBand 33767; EPE 0
[Testcase 150]: L2 17438; PVBand 31920; EPE 0
[Testcase 151]: L2 22664; PVBand 37007; EPE 4
[Testcase 152]: L2 35780; PVBand 50299; EPE 7
[Testcase 153]: L2 24441; PVBand 44704; EPE 1
[Testcase 154]: L2 18104; PVBand 33398; EPE 0
[Testcase 155]: L2 26579; PVBand 39295; EPE 6
[Testcase 156]: L2 25785; PVBand 39887; EPE 4
[Testcase 157]: L2 26507; PVBand 46574; EPE 1
[Testcase 158]: L2 41614; PVBand 45501; EPE 18
[Testcase 159]: L2 21729; PVBand 36179; EPE 1
[Testcase 160]: L2 20415; PVBand 36259; EPE 0
[Testcase 161]: L2 21206; PVBand 36932; EPE 0
[Testcase 162]: L2 41546; PVBand 55018; EPE 14
[Testcase 163]: L2 24876; PVBand 42624; EPE 0
[Testcase 164]: L2 28867; PVBand 45650; EPE 4
[Testcase 165]: L2 25739; PVBand 42061; EPE 3
[Finetuned]: L2 28016; PVBand 43431; EPE 3.9
'''


