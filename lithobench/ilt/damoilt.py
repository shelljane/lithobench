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

        self.conv_tail = nn.Conv2d(n1, out_ch, kernel_size=7, stride=1, padding=3)


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

        output = self.conv_tail(x0_1)
        return output



class Discriminator(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.conv0_0 = conv_block(1, 64, kernel_size=4, stride=2, padding=1, leaky=True)
        self.conv1_0 = conv_block(64, 128, kernel_size=4, stride=1, padding='same', leaky=True)
        self.conv2_0 = conv_block(128, 1, kernel_size=4, stride=1, padding='same', leaky=True)
        self.flatten_0 = nn.Flatten()
        self.fc0_0 = nn.Linear(512**2, 1)
        self.sigmoid_0 = nn.Sigmoid()
        self.seq0 = nn.Sequential(self.conv0_0, self.conv1_0, self.conv2_0, self.flatten_0, self.fc0_0, self.sigmoid_0)
        
        self.conv0_1 = conv_block(1, 64, kernel_size=4, stride=2, padding=1, leaky=True)
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
    


class DAMOILT(ModelILT): 
    def __init__(self, size=(256, 256)): 
        super().__init__(size=size, name="DAMOILT")
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
                l1loss = F.l1_loss(printedNom.unsqueeze(1), target)
                lossG2 = F.mse_loss(mask.unsqueeze(1), label)
                lossG = l1loss + lossG2
                
                optimPre.zero_grad()
                lossG.backward()
                optimPre.step()

                progress.set_postfix(l1loss=l1loss.item(), lossG2=lossG2.item())

            print(f"[Pre-Epoch {epoch}] Testing")
            self.netG.eval()
            l1losses = []
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
                    l1loss = F.l1_loss(printedNom.unsqueeze(1), target)
                    lossG2 = F.mse_loss(mask.unsqueeze(1), label)
                    lossG = l1loss + lossG2
                    l1losses.append(l1loss.item())
                    lossG2s.append(lossG.item())

                    progress.set_postfix(l1loss=l1loss.item(), lossG2=lossG2.item())
            
            print(f"[Pre-Epoch {epoch}] L1 loss = {np.mean(l1losses)}, lossG2 = {np.mean(lossG2s)}")

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
        l1losses = []
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
                l1loss = F.l1_loss(printedNom.unsqueeze(1), target)
                lossG2 = F.mse_loss(mask.unsqueeze(1), label)
                lossG = l1loss + lossG2
                l1losses.append(l1loss.item())
                lossG2s.append(lossG2.item())
                
                printedNom, printedMax, printedMin = self.simLitho(label.squeeze(1))
                l2ref = F.mse_loss(printedNom.unsqueeze(1), target)

                progress.set_postfix(l1loss=l1loss.item(), l2ref=l2ref.item(), lossG2=lossG2.item())
        
        print(f"[Initial] L1 loss = {np.mean(l1losses)}, lossG2 = {np.mean(lossG2s)}")

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
                printedNom, printedMax, printedMin = self.simLitho(maskG.squeeze(1))
                l1loss = F.l1_loss(printedNom.unsqueeze(1), target)
                predD = self.netD(maskG)
                lossG1 = -torch.mean(torch.log(predD))
                lossG2 = F.mse_loss(maskG, label)
                lossG = l1loss + lossG1 + lossG2
                optimG.zero_grad()
                lossG.backward()
                optimG.step()
                # Log
                progress.set_postfix(lossD=lossD.item(), l1loss=l1loss.item(), lossG1=lossG1.item(), lossG2=lossG2.item())

            print(f"[Epoch {epoch}] Testing")
            self.netG.eval()
            self.netD.eval()
            l1losses = []
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
                    l1loss = F.l1_loss(printedNom.unsqueeze(1), target)
                    lossG2 = F.mse_loss(mask.unsqueeze(1), label)
                    lossG = l1loss + lossG2
                    l1losses.append(l1loss.item())
                    lossG2s.append(lossG2.item())
                    
                    printedNom, printedMax, printedMin = self.simLitho(label.squeeze(1))
                    l2ref = F.mse_loss(printedNom.unsqueeze(1), target)
                    l2refs.append(l2ref.item())

                    progress.set_postfix(l1loss=l1loss.item(), l2ref=l2ref.item(), lossG2=lossG2.item())
            
            print(f"[Epoch {epoch}] L1 loss = {np.mean(l1losses)}, l2ref = {np.mean(l2refs)}, lossG2 = {np.mean(lossG2s)}")

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
    ImageSize = (1024, 1024)
    Epochs = 1
    BatchSize = 4
    NJobs = 8
    TrainOnly = False
    EvalOnly = False
    train_loader, val_loader = loadersILT(Benchmark, ImageSize, BatchSize, NJobs)
    targets = evaluate.getTargets(samples=None, dataset=Benchmark)
    ilt = DAMOILT(size=ImageSize)
    
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
        ilt.save(["trivial/damoilt/pretrainG.pth","trivial/damoilt/pretrainD.pth"])
    else: 
        ilt.load(["trivial/damoilt/pretrainG.pth","trivial/damoilt/pretrainD.pth"])
    if not EvalOnly: 
        ilt.train(train_loader, val_loader, epochs=Epochs)
        ilt.save(["trivial/damoilt/trainG.pth","trivial/damoilt/trainD.pth"])
    else: 
        ilt.load(["trivial/damoilt/trainG.pth","trivial/damoilt/trainD.pth"])
    ilt.evaluate(targets, finetune=False, folder="trivial/damoilt")


'''
[MetalSet]
[Evaluation]: Getting masks from the model
[Testcase 1]: L2 46874; PVBand 45487; EPE 5; Shots: 513
[Testcase 2]: L2 36897; PVBand 35208; EPE 3; Shots: 485
[Testcase 3]: L2 79415; PVBand 72717; EPE 43; Shots: 636
[Testcase 4]: L2 11319; PVBand 22036; EPE 2; Shots: 469
[Testcase 5]: L2 35139; PVBand 54833; EPE 1; Shots: 550
[Testcase 6]: L2 33814; PVBand 46857; EPE 0; Shots: 607
[Testcase 7]: L2 20359; PVBand 38907; EPE 0; Shots: 530
[Testcase 8]: L2 12634; PVBand 20374; EPE 0; Shots: 480
[Testcase 9]: L2 40328; PVBand 58728; EPE 0; Shots: 608
[Testcase 10]: L2 9015; PVBand 16579; EPE 0; Shots: 350
[Initialized]: L2 32579; PVBand 41173; EPE 5.4; Runtime: 0.19s; Shots: 523
[Evaluation]: Finetuning the masks with pixel-based ILT method
[Testcase 1]: L2 39383; PVBand 48499; EPE 3; Shots: 606
[Testcase 2]: L2 31143; PVBand 38848; EPE 0; Shots: 517
[Testcase 3]: L2 63591; PVBand 76547; EPE 15; Shots: 662
[Testcase 4]: L2 9016; PVBand 23961; EPE 0; Shots: 488
[Testcase 5]: L2 29943; PVBand 54615; EPE 0; Shots: 564
[Testcase 6]: L2 30189; PVBand 48737; EPE 0; Shots: 578
[Testcase 7]: L2 16050; PVBand 41222; EPE 0; Shots: 529
[Testcase 8]: L2 11251; PVBand 20850; EPE 0; Shots: 566
[Testcase 9]: L2 34499; PVBand 62220; EPE 0; Shots: 615
[Testcase 10]: L2 7936; PVBand 16770; EPE 0; Shots: 386
[Finetuned]: L2 27300; PVBand 43227; EPE 1.8; Shots: 551

[ViaSet]
[Evaluation]: Getting masks from the model
[Testcase 1]: L2 2422; PVBand 4732; EPE 0; Shots: 64
[Testcase 2]: L2 2964; PVBand 4993; EPE 0; Shots: 88
[Testcase 3]: L2 4347; PVBand 9323; EPE 0; Shots: 190
[Testcase 4]: L2 3169; PVBand 6434; EPE 0; Shots: 146
[Testcase 5]: L2 7000; PVBand 13352; EPE 0; Shots: 259
[Testcase 6]: L2 5835; PVBand 11989; EPE 0; Shots: 171
[Testcase 7]: L2 3225; PVBand 7261; EPE 0; Shots: 131
[Testcase 8]: L2 12041; PVBand 21738; EPE 0; Shots: 429
[Testcase 9]: L2 7059; PVBand 14654; EPE 0; Shots: 233
[Testcase 10]: L2 2752; PVBand 5149; EPE 0; Shots: 54
[Initialized]: L2 5081; PVBand 9962; EPE 0.0; Runtime: 2.04s; Shots: 176
[Evaluation]: Finetuning the masks with pixel-based ILT method
[Testcase 1]: L2 2646; PVBand 4582; EPE 0; Shots: 126
[Testcase 2]: L2 2863; PVBand 4534; EPE 0; Shots: 151
[Testcase 3]: L2 4648; PVBand 8624; EPE 0; Shots: 316
[Testcase 4]: L2 3373; PVBand 6220; EPE 0; Shots: 195
[Testcase 5]: L2 7563; PVBand 12664; EPE 0; Shots: 384
[Testcase 6]: L2 7397; PVBand 11085; EPE 0; Shots: 359
[Testcase 7]: L2 3990; PVBand 6864; EPE 0; Shots: 202
[Testcase 8]: L2 11430; PVBand 21046; EPE 0; Shots: 519
[Testcase 9]: L2 8197; PVBand 14218; EPE 1; Shots: 389
[Testcase 10]: L2 3919; PVBand 5018; EPE 1; Shots: 147
[Finetuned]: L2 5603; PVBand 9486; EPE 0.2; Shots: 279

[StdMetal]
[Evaluation]: Getting masks from the model
[Testcase 1]: L2 11623; PVBand 19297; EPE 0
[Testcase 2]: L2 7467; PVBand 7344; EPE 0
[Testcase 3]: L2 6401; PVBand 13153; EPE 0
[Testcase 4]: L2 11203; PVBand 14948; EPE 0
[Testcase 5]: L2 4311; PVBand 6002; EPE 0
[Testcase 6]: L2 3352; PVBand 5956; EPE 0
[Testcase 7]: L2 14538; PVBand 12392; EPE 0
[Testcase 8]: L2 3949; PVBand 7016; EPE 0
[Testcase 9]: L2 14072; PVBand 17793; EPE 0
[Testcase 10]: L2 12927; PVBand 26147; EPE 0
[Testcase 11]: L2 16122; PVBand 32096; EPE 0
[Testcase 12]: L2 4727; PVBand 6536; EPE 0
[Testcase 13]: L2 30296; PVBand 42887; EPE 1
[Testcase 14]: L2 14836; PVBand 22426; EPE 0
[Testcase 15]: L2 15433; PVBand 25156; EPE 0
[Testcase 16]: L2 10296; PVBand 16013; EPE 0
[Testcase 17]: L2 11006; PVBand 18547; EPE 0
[Testcase 18]: L2 16464; PVBand 29388; EPE 0
[Testcase 19]: L2 3774; PVBand 8592; EPE 0
[Testcase 20]: L2 9097; PVBand 13824; EPE 0
[Testcase 21]: L2 26927; PVBand 44439; EPE 0
[Testcase 22]: L2 12299; PVBand 20439; EPE 0
[Testcase 23]: L2 39005; PVBand 41000; EPE 3
[Testcase 24]: L2 12317; PVBand 18585; EPE 0
[Testcase 25]: L2 10154; PVBand 14496; EPE 0
[Testcase 26]: L2 11418; PVBand 19221; EPE 0
[Testcase 27]: L2 15476; PVBand 20788; EPE 0
[Testcase 28]: L2 3109; PVBand 6622; EPE 0
[Testcase 29]: L2 32857; PVBand 51787; EPE 0
[Testcase 30]: L2 35563; PVBand 48557; EPE 0
[Testcase 31]: L2 40413; PVBand 37170; EPE 0
[Testcase 32]: L2 6298; PVBand 8886; EPE 0
[Testcase 33]: L2 5773; PVBand 10609; EPE 0
[Testcase 34]: L2 7445; PVBand 7586; EPE 0
[Testcase 35]: L2 10436; PVBand 19188; EPE 0
[Testcase 36]: L2 38529; PVBand 59474; EPE 3
[Testcase 37]: L2 15257; PVBand 17363; EPE 0
[Testcase 38]: L2 64503; PVBand 76029; EPE 2
[Testcase 39]: L2 10408; PVBand 7519; EPE 0
[Testcase 40]: L2 10708; PVBand 14399; EPE 0
[Testcase 41]: L2 63359; PVBand 82646; EPE 3
[Testcase 42]: L2 7981; PVBand 10430; EPE 0
[Testcase 43]: L2 25834; PVBand 39122; EPE 1
[Testcase 44]: L2 54900; PVBand 67717; EPE 2
[Testcase 45]: L2 13965; PVBand 23604; EPE 0
[Testcase 46]: L2 14552; PVBand 21656; EPE 0
[Testcase 47]: L2 14118; PVBand 22542; EPE 0
[Testcase 48]: L2 9324; PVBand 14572; EPE 0
[Testcase 49]: L2 22982; PVBand 45031; EPE 0
[Testcase 50]: L2 12324; PVBand 19352; EPE 0
[Testcase 51]: L2 4017; PVBand 6675; EPE 0
[Testcase 52]: L2 3059; PVBand 6610; EPE 0
[Testcase 53]: L2 33665; PVBand 34186; EPE 0
[Testcase 54]: L2 16686; PVBand 25142; EPE 0
[Testcase 55]: L2 11623; PVBand 19297; EPE 0
[Testcase 56]: L2 4715; PVBand 6082; EPE 0
[Testcase 57]: L2 34769; PVBand 45617; EPE 0
[Testcase 58]: L2 4525; PVBand 6788; EPE 0
[Testcase 59]: L2 19308; PVBand 24214; EPE 0
[Testcase 60]: L2 23681; PVBand 31429; EPE 0
[Testcase 61]: L2 4556; PVBand 5880; EPE 0
[Testcase 62]: L2 34568; PVBand 48267; EPE 0
[Testcase 63]: L2 29163; PVBand 48234; EPE 1
[Testcase 64]: L2 13339; PVBand 18717; EPE 0
[Testcase 65]: L2 11817; PVBand 17881; EPE 0
[Testcase 66]: L2 40817; PVBand 49953; EPE 1
[Testcase 67]: L2 14463; PVBand 18144; EPE 0
[Testcase 68]: L2 6036; PVBand 5880; EPE 0
[Testcase 69]: L2 17926; PVBand 29561; EPE 0
[Testcase 70]: L2 10176; PVBand 21537; EPE 0
[Testcase 71]: L2 12299; PVBand 20439; EPE 0
[Testcase 72]: L2 11089; PVBand 15363; EPE 0
[Testcase 73]: L2 27634; PVBand 54158; EPE 0
[Testcase 74]: L2 9299; PVBand 15926; EPE 0
[Testcase 75]: L2 10318; PVBand 15290; EPE 0
[Testcase 76]: L2 4311; PVBand 6002; EPE 0
[Testcase 77]: L2 4727; PVBand 6536; EPE 0
[Testcase 78]: L2 4889; PVBand 6273; EPE 0
[Testcase 79]: L2 23311; PVBand 37613; EPE 0
[Testcase 80]: L2 11029; PVBand 19660; EPE 0
[Testcase 81]: L2 9783; PVBand 21829; EPE 0
[Testcase 82]: L2 12388; PVBand 18562; EPE 0
[Testcase 83]: L2 9242; PVBand 13230; EPE 0
[Testcase 84]: L2 16580; PVBand 29740; EPE 0
[Testcase 85]: L2 5129; PVBand 7900; EPE 0
[Testcase 86]: L2 9324; PVBand 14572; EPE 0
[Testcase 87]: L2 11580; PVBand 18233; EPE 0
[Testcase 88]: L2 10803; PVBand 23159; EPE 0
[Testcase 89]: L2 30854; PVBand 52817; EPE 1
[Testcase 90]: L2 3489; PVBand 6661; EPE 0
[Testcase 91]: L2 42659; PVBand 65741; EPE 0
[Testcase 92]: L2 8576; PVBand 20808; EPE 0
[Testcase 93]: L2 5113; PVBand 6047; EPE 0
[Testcase 94]: L2 9725; PVBand 14152; EPE 0
[Testcase 95]: L2 36859; PVBand 54428; EPE 0
[Testcase 96]: L2 5921; PVBand 11062; EPE 0
[Testcase 97]: L2 22666; PVBand 34053; EPE 0
[Testcase 98]: L2 11817; PVBand 17881; EPE 0
[Testcase 99]: L2 3544; PVBand 6711; EPE 0
[Testcase 100]: L2 11203; PVBand 14948; EPE 0
[Testcase 101]: L2 14836; PVBand 22426; EPE 0
[Testcase 102]: L2 4793; PVBand 8635; EPE 0
[Testcase 103]: L2 28610; PVBand 48265; EPE 0
[Testcase 104]: L2 24633; PVBand 37382; EPE 0
[Testcase 105]: L2 15645; PVBand 19415; EPE 0
[Testcase 106]: L2 3352; PVBand 5956; EPE 0
[Testcase 107]: L2 16229; PVBand 19021; EPE 0
[Testcase 108]: L2 11078; PVBand 18500; EPE 0
[Testcase 109]: L2 3745; PVBand 6434; EPE 0
[Testcase 110]: L2 10060; PVBand 17899; EPE 0
[Testcase 111]: L2 11677; PVBand 19096; EPE 0
[Testcase 112]: L2 27273; PVBand 47831; EPE 0
[Testcase 113]: L2 11677; PVBand 19096; EPE 0
[Testcase 114]: L2 4086; PVBand 6065; EPE 0
[Testcase 115]: L2 23861; PVBand 26753; EPE 0
[Testcase 116]: L2 31816; PVBand 30729; EPE 1
[Testcase 117]: L2 26752; PVBand 41838; EPE 0
[Testcase 118]: L2 3101; PVBand 6570; EPE 0
[Testcase 119]: L2 13565; PVBand 18348; EPE 0
[Testcase 120]: L2 8839; PVBand 14337; EPE 0
[Testcase 121]: L2 14096; PVBand 24136; EPE 0
[Testcase 122]: L2 20378; PVBand 29152; EPE 0
[Testcase 123]: L2 3544; PVBand 6711; EPE 0
[Testcase 124]: L2 4162; PVBand 8070; EPE 0
[Testcase 125]: L2 10996; PVBand 15936; EPE 0
[Testcase 126]: L2 14501; PVBand 25099; EPE 0
[Testcase 127]: L2 25983; PVBand 44041; EPE 0
[Testcase 128]: L2 19601; PVBand 36100; EPE 0
[Testcase 129]: L2 11516; PVBand 18325; EPE 0
[Testcase 130]: L2 8752; PVBand 13653; EPE 0
[Testcase 131]: L2 5307; PVBand 8406; EPE 0
[Testcase 132]: L2 15580; PVBand 17194; EPE 0
[Testcase 133]: L2 36369; PVBand 34543; EPE 3
[Testcase 134]: L2 11892; PVBand 16595; EPE 0
[Testcase 135]: L2 33795; PVBand 34016; EPE 1
[Testcase 136]: L2 13948; PVBand 24599; EPE 0
[Testcase 137]: L2 3101; PVBand 6570; EPE 0
[Testcase 138]: L2 14490; PVBand 13015; EPE 0
[Testcase 139]: L2 34905; PVBand 31179; EPE 0
[Testcase 140]: L2 7109; PVBand 11974; EPE 0
[Testcase 141]: L2 24711; PVBand 37305; EPE 0
[Testcase 142]: L2 4506; PVBand 6495; EPE 0
[Testcase 143]: L2 13061; PVBand 19371; EPE 0
[Testcase 144]: L2 4017; PVBand 6675; EPE 0
[Testcase 145]: L2 11214; PVBand 14923; EPE 0
[Testcase 146]: L2 4889; PVBand 6273; EPE 0
[Testcase 147]: L2 11418; PVBand 19221; EPE 0
[Testcase 148]: L2 4078; PVBand 6682; EPE 0
[Testcase 149]: L2 6270; PVBand 8476; EPE 0
[Testcase 150]: L2 9725; PVBand 14152; EPE 0
[Testcase 151]: L2 3489; PVBand 6661; EPE 0
[Testcase 152]: L2 3109; PVBand 6622; EPE 0
[Testcase 153]: L2 19721; PVBand 40934; EPE 0
[Testcase 154]: L2 22085; PVBand 40150; EPE 0
[Testcase 155]: L2 24124; PVBand 37749; EPE 0
[Testcase 156]: L2 11409; PVBand 16658; EPE 0
[Testcase 157]: L2 6236; PVBand 14330; EPE 0
[Testcase 158]: L2 42641; PVBand 60903; EPE 1
[Testcase 159]: L2 12633; PVBand 19325; EPE 0
[Testcase 160]: L2 20379; PVBand 33629; EPE 0
[Testcase 161]: L2 13735; PVBand 23770; EPE 1
[Testcase 162]: L2 4078; PVBand 6682; EPE 0
[Testcase 163]: L2 10996; PVBand 15936; EPE 0
[Testcase 164]: L2 3999; PVBand 6373; EPE 0
[Testcase 165]: L2 28370; PVBand 44888; EPE 0
[Testcase 166]: L2 22895; PVBand 36035; EPE 0
[Testcase 167]: L2 44916; PVBand 43366; EPE 1
[Testcase 168]: L2 9528; PVBand 19703; EPE 1
[Testcase 169]: L2 21662; PVBand 39391; EPE 0
[Testcase 170]: L2 19554; PVBand 32598; EPE 0
[Testcase 171]: L2 4534; PVBand 12317; EPE 0
[Testcase 172]: L2 6299; PVBand 15019; EPE 0
[Testcase 173]: L2 41225; PVBand 54211; EPE 0
[Testcase 174]: L2 31020; PVBand 51484; EPE 0
[Testcase 175]: L2 43045; PVBand 49932; EPE 0
[Testcase 176]: L2 20510; PVBand 18548; EPE 0
[Testcase 177]: L2 11853; PVBand 19108; EPE 0
[Testcase 178]: L2 14538; PVBand 12392; EPE 0
[Testcase 179]: L2 10708; PVBand 14399; EPE 0
[Testcase 180]: L2 15723; PVBand 25236; EPE 0
[Testcase 181]: L2 13455; PVBand 24044; EPE 0
[Testcase 182]: L2 3352; PVBand 5956; EPE 0
[Testcase 183]: L2 8250; PVBand 17601; EPE 0
[Testcase 184]: L2 22880; PVBand 38075; EPE 0
[Testcase 185]: L2 9854; PVBand 19792; EPE 0
[Testcase 186]: L2 46975; PVBand 67858; EPE 1
[Testcase 187]: L2 9097; PVBand 13824; EPE 0
[Testcase 188]: L2 7686; PVBand 12361; EPE 0
[Testcase 189]: L2 18632; PVBand 36369; EPE 0
[Testcase 190]: L2 6724; PVBand 8177; EPE 0
[Testcase 191]: L2 13061; PVBand 19371; EPE 0
[Testcase 192]: L2 17256; PVBand 42495; EPE 0
[Testcase 193]: L2 24449; PVBand 42367; EPE 0
[Testcase 194]: L2 22666; PVBand 34053; EPE 0
[Testcase 195]: L2 14407; PVBand 23825; EPE 0
[Testcase 196]: L2 18129; PVBand 23141; EPE 0
[Testcase 197]: L2 4793; PVBand 8635; EPE 0
[Testcase 198]: L2 12388; PVBand 18562; EPE 0
[Testcase 199]: L2 16391; PVBand 29609; EPE 0
[Testcase 200]: L2 14377; PVBand 24991; EPE 0
[Testcase 201]: L2 3745; PVBand 6434; EPE 0
[Testcase 202]: L2 3774; PVBand 8592; EPE 0
[Testcase 203]: L2 25891; PVBand 37487; EPE 1
[Testcase 204]: L2 4728; PVBand 6731; EPE 0
[Testcase 205]: L2 51323; PVBand 72751; EPE 1
[Testcase 206]: L2 9327; PVBand 19347; EPE 0
[Testcase 207]: L2 24538; PVBand 53647; EPE 0
[Testcase 208]: L2 14538; PVBand 12392; EPE 0
[Testcase 209]: L2 11428; PVBand 16341; EPE 0
[Testcase 210]: L2 40308; PVBand 53387; EPE 1
[Testcase 211]: L2 5853; PVBand 9713; EPE 0
[Testcase 212]: L2 5428; PVBand 5528; EPE 0
[Testcase 213]: L2 12317; PVBand 18585; EPE 0
[Testcase 214]: L2 16125; PVBand 20029; EPE 0
[Testcase 215]: L2 41156; PVBand 52276; EPE 2
[Testcase 216]: L2 13975; PVBand 24658; EPE 0
[Testcase 217]: L2 30730; PVBand 43757; EPE 2
[Testcase 218]: L2 11078; PVBand 18500; EPE 0
[Testcase 219]: L2 10054; PVBand 17689; EPE 0
[Testcase 220]: L2 26580; PVBand 44873; EPE 0
[Testcase 221]: L2 13965; PVBand 23604; EPE 0
[Testcase 222]: L2 8420; PVBand 8785; EPE 0
[Testcase 223]: L2 4481; PVBand 6152; EPE 0
[Testcase 224]: L2 11029; PVBand 19660; EPE 0
[Testcase 225]: L2 17486; PVBand 15952; EPE 0
[Testcase 226]: L2 12501; PVBand 27064; EPE 0
[Testcase 227]: L2 16408; PVBand 27393; EPE 0
[Testcase 228]: L2 8541; PVBand 22830; EPE 0
[Testcase 229]: L2 25319; PVBand 44813; EPE 0
[Testcase 230]: L2 35724; PVBand 51185; EPE 1
[Testcase 231]: L2 13862; PVBand 26151; EPE 0
[Testcase 232]: L2 17416; PVBand 36035; EPE 0
[Testcase 233]: L2 17272; PVBand 20970; EPE 0
[Testcase 234]: L2 9324; PVBand 14572; EPE 0
[Testcase 235]: L2 27288; PVBand 36692; EPE 0
[Testcase 236]: L2 10318; PVBand 15290; EPE 0
[Testcase 237]: L2 11699; PVBand 19297; EPE 0
[Testcase 238]: L2 14100; PVBand 23400; EPE 1
[Testcase 239]: L2 27417; PVBand 45816; EPE 0
[Testcase 240]: L2 14538; PVBand 12392; EPE 0
[Testcase 241]: L2 12752; PVBand 24503; EPE 0
[Testcase 242]: L2 3059; PVBand 6610; EPE 0
[Testcase 243]: L2 4283; PVBand 7443; EPE 0
[Testcase 244]: L2 11699; PVBand 19297; EPE 0
[Testcase 245]: L2 18069; PVBand 33311; EPE 0
[Testcase 246]: L2 24629; PVBand 35705; EPE 0
[Testcase 247]: L2 16580; PVBand 29740; EPE 0
[Testcase 248]: L2 59953; PVBand 65528; EPE 0
[Testcase 249]: L2 5428; PVBand 5528; EPE 0
[Testcase 250]: L2 11853; PVBand 19108; EPE 0
[Testcase 251]: L2 9324; PVBand 14572; EPE 0
[Testcase 252]: L2 7786; PVBand 11200; EPE 0
[Testcase 253]: L2 4525; PVBand 6788; EPE 0
[Testcase 254]: L2 11516; PVBand 18325; EPE 0
[Testcase 255]: L2 25777; PVBand 26744; EPE 0
[Testcase 256]: L2 4728; PVBand 6731; EPE 0
[Testcase 257]: L2 62556; PVBand 75576; EPE 4
[Testcase 258]: L2 17880; PVBand 30162; EPE 0
[Testcase 259]: L2 25758; PVBand 41404; EPE 0
[Testcase 260]: L2 6298; PVBand 8886; EPE 0
[Testcase 261]: L2 53887; PVBand 59754; EPE 1
[Testcase 262]: L2 33334; PVBand 28188; EPE 0
[Testcase 263]: L2 6958; PVBand 16824; EPE 0
[Testcase 264]: L2 12372; PVBand 21077; EPE 0
[Testcase 265]: L2 5739; PVBand 8630; EPE 0
[Testcase 266]: L2 25805; PVBand 38233; EPE 0
[Testcase 267]: L2 13565; PVBand 18348; EPE 0
[Testcase 268]: L2 10408; PVBand 7519; EPE 0
[Testcase 269]: L2 7273; PVBand 10795; EPE 0
[Testcase 270]: L2 14007; PVBand 23771; EPE 0
[Testcase 271]: L2 6270; PVBand 8476; EPE 0
[Initialized]: L2 16120; PVBand 23796; EPE 0.2
[Evaluation]: Finetuning the masks with pixel-based ILT method
[Testcase 1]: L2 9364; PVBand 20174; EPE 0
[Testcase 2]: L2 4983; PVBand 7338; EPE 0
[Testcase 3]: L2 6380; PVBand 12702; EPE 0
[Testcase 4]: L2 9239; PVBand 16599; EPE 0
[Testcase 5]: L2 4481; PVBand 6067; EPE 0
[Testcase 6]: L2 3130; PVBand 6046; EPE 0
[Testcase 7]: L2 9271; PVBand 13292; EPE 0
[Testcase 8]: L2 3612; PVBand 7566; EPE 0
[Testcase 9]: L2 10778; PVBand 18766; EPE 0
[Testcase 10]: L2 10274; PVBand 30206; EPE 0
[Testcase 11]: L2 14250; PVBand 33185; EPE 0
[Testcase 12]: L2 4310; PVBand 6610; EPE 0
[Testcase 13]: L2 24623; PVBand 43931; EPE 0
[Testcase 14]: L2 10762; PVBand 24179; EPE 0
[Testcase 15]: L2 13517; PVBand 25905; EPE 0
[Testcase 16]: L2 10439; PVBand 15616; EPE 0
[Testcase 17]: L2 9769; PVBand 18507; EPE 0
[Testcase 18]: L2 13920; PVBand 32574; EPE 0
[Testcase 19]: L2 4983; PVBand 7383; EPE 0
[Testcase 20]: L2 7611; PVBand 13685; EPE 0
[Testcase 21]: L2 20403; PVBand 47911; EPE 0
[Testcase 22]: L2 9631; PVBand 21409; EPE 0
[Testcase 23]: L2 20809; PVBand 40760; EPE 0
[Testcase 24]: L2 9601; PVBand 19498; EPE 0
[Testcase 25]: L2 9391; PVBand 15499; EPE 0
[Testcase 26]: L2 8868; PVBand 20042; EPE 0
[Testcase 27]: L2 13585; PVBand 21957; EPE 0
[Testcase 28]: L2 2901; PVBand 6717; EPE 0
[Testcase 29]: L2 28550; PVBand 54654; EPE 0
[Testcase 30]: L2 27814; PVBand 49824; EPE 0
[Testcase 31]: L2 20697; PVBand 39884; EPE 0
[Testcase 32]: L2 6080; PVBand 8936; EPE 0
[Testcase 33]: L2 5761; PVBand 10511; EPE 0
[Testcase 34]: L2 5420; PVBand 7712; EPE 0
[Testcase 35]: L2 8722; PVBand 20013; EPE 0
[Testcase 36]: L2 27766; PVBand 62546; EPE 1
[Testcase 37]: L2 11299; PVBand 18607; EPE 0
[Testcase 38]: L2 53197; PVBand 81741; EPE 0
[Testcase 39]: L2 5182; PVBand 7565; EPE 0
[Testcase 40]: L2 8620; PVBand 14594; EPE 0
[Testcase 41]: L2 54092; PVBand 88593; EPE 3
[Testcase 42]: L2 6749; PVBand 10395; EPE 0
[Testcase 43]: L2 21270; PVBand 39894; EPE 0
[Testcase 44]: L2 40279; PVBand 76295; EPE 0
[Testcase 45]: L2 11433; PVBand 25325; EPE 0
[Testcase 46]: L2 11136; PVBand 23469; EPE 0
[Testcase 47]: L2 11550; PVBand 23610; EPE 0
[Testcase 48]: L2 7946; PVBand 14698; EPE 0
[Testcase 49]: L2 18730; PVBand 47816; EPE 0
[Testcase 50]: L2 11063; PVBand 19556; EPE 0
[Testcase 51]: L2 4755; PVBand 6929; EPE 0
[Testcase 52]: L2 3667; PVBand 6881; EPE 0
[Testcase 53]: L2 18206; PVBand 35298; EPE 0
[Testcase 54]: L2 13147; PVBand 25798; EPE 0
[Testcase 55]: L2 9364; PVBand 20174; EPE 0
[Testcase 56]: L2 5072; PVBand 6690; EPE 0
[Testcase 57]: L2 25460; PVBand 46948; EPE 0
[Testcase 58]: L2 4536; PVBand 7194; EPE 0
[Testcase 59]: L2 14463; PVBand 24502; EPE 0
[Testcase 60]: L2 18560; PVBand 33519; EPE 0
[Testcase 61]: L2 4111; PVBand 6072; EPE 0
[Testcase 62]: L2 28863; PVBand 52419; EPE 0
[Testcase 63]: L2 24071; PVBand 49265; EPE 0
[Testcase 64]: L2 11086; PVBand 19374; EPE 0
[Testcase 65]: L2 9517; PVBand 18310; EPE 0
[Testcase 66]: L2 31374; PVBand 53825; EPE 1
[Testcase 67]: L2 12434; PVBand 18848; EPE 0
[Testcase 68]: L2 4537; PVBand 6184; EPE 0
[Testcase 69]: L2 15166; PVBand 30291; EPE 0
[Testcase 70]: L2 7771; PVBand 22105; EPE 0
[Testcase 71]: L2 9631; PVBand 21409; EPE 0
[Testcase 72]: L2 9843; PVBand 15827; EPE 0
[Testcase 73]: L2 23984; PVBand 54988; EPE 0
[Testcase 74]: L2 8437; PVBand 16873; EPE 0
[Testcase 75]: L2 9318; PVBand 15737; EPE 0
[Testcase 76]: L2 4481; PVBand 6067; EPE 0
[Testcase 77]: L2 4310; PVBand 6610; EPE 0
[Testcase 78]: L2 4811; PVBand 6229; EPE 0
[Testcase 79]: L2 20383; PVBand 40422; EPE 0
[Testcase 80]: L2 9189; PVBand 19924; EPE 0
[Testcase 81]: L2 9336; PVBand 23548; EPE 0
[Testcase 82]: L2 9984; PVBand 19482; EPE 0
[Testcase 83]: L2 8006; PVBand 13506; EPE 0
[Testcase 84]: L2 14826; PVBand 29855; EPE 0
[Testcase 85]: L2 4831; PVBand 7703; EPE 0
[Testcase 86]: L2 7946; PVBand 14698; EPE 0
[Testcase 87]: L2 10069; PVBand 18982; EPE 0
[Testcase 88]: L2 9560; PVBand 24636; EPE 0
[Testcase 89]: L2 24845; PVBand 59086; EPE 0
[Testcase 90]: L2 3861; PVBand 6507; EPE 0
[Testcase 91]: L2 36102; PVBand 70419; EPE 0
[Testcase 92]: L2 7671; PVBand 20820; EPE 0
[Testcase 93]: L2 4987; PVBand 6054; EPE 0
[Testcase 94]: L2 8188; PVBand 14485; EPE 0
[Testcase 95]: L2 28091; PVBand 52574; EPE 1
[Testcase 96]: L2 6435; PVBand 10833; EPE 0
[Testcase 97]: L2 18403; PVBand 35190; EPE 0
[Testcase 98]: L2 9517; PVBand 18310; EPE 0
[Testcase 99]: L2 3837; PVBand 6601; EPE 0
[Testcase 100]: L2 9239; PVBand 16599; EPE 0
[Testcase 101]: L2 10762; PVBand 24179; EPE 0
[Testcase 102]: L2 4376; PVBand 8064; EPE 0
[Testcase 103]: L2 26536; PVBand 50016; EPE 0
[Testcase 104]: L2 20599; PVBand 38398; EPE 0
[Testcase 105]: L2 11702; PVBand 19993; EPE 0
[Testcase 106]: L2 3130; PVBand 6046; EPE 0
[Testcase 107]: L2 11407; PVBand 19910; EPE 0
[Testcase 108]: L2 8917; PVBand 18755; EPE 0
[Testcase 109]: L2 3532; PVBand 6996; EPE 0
[Testcase 110]: L2 8725; PVBand 19039; EPE 0
[Testcase 111]: L2 8934; PVBand 19878; EPE 0
[Testcase 112]: L2 21513; PVBand 47649; EPE 0
[Testcase 113]: L2 8934; PVBand 19878; EPE 0
[Testcase 114]: L2 3638; PVBand 6209; EPE 0
[Testcase 115]: L2 16248; PVBand 27108; EPE 0
[Testcase 116]: L2 17808; PVBand 33739; EPE 0
[Testcase 117]: L2 23572; PVBand 46328; EPE 0
[Testcase 118]: L2 3988; PVBand 7019; EPE 0
[Testcase 119]: L2 11341; PVBand 18005; EPE 0
[Testcase 120]: L2 8128; PVBand 14120; EPE 0
[Testcase 121]: L2 11444; PVBand 24646; EPE 0
[Testcase 122]: L2 18325; PVBand 31280; EPE 0
[Testcase 123]: L2 3837; PVBand 6601; EPE 0
[Testcase 124]: L2 4412; PVBand 7375; EPE 0
[Testcase 125]: L2 9326; PVBand 16658; EPE 0
[Testcase 126]: L2 13139; PVBand 25538; EPE 0
[Testcase 127]: L2 20388; PVBand 46577; EPE 0
[Testcase 128]: L2 16968; PVBand 39798; EPE 0
[Testcase 129]: L2 9361; PVBand 18413; EPE 0
[Testcase 130]: L2 8030; PVBand 13903; EPE 0
[Testcase 131]: L2 4990; PVBand 8210; EPE 0
[Testcase 132]: L2 10720; PVBand 17868; EPE 0
[Testcase 133]: L2 23482; PVBand 38183; EPE 0
[Testcase 134]: L2 8945; PVBand 16148; EPE 0
[Testcase 135]: L2 20545; PVBand 37959; EPE 0
[Testcase 136]: L2 12523; PVBand 25645; EPE 0
[Testcase 137]: L2 3988; PVBand 7019; EPE 0
[Testcase 138]: L2 9250; PVBand 13549; EPE 0
[Testcase 139]: L2 17009; PVBand 32852; EPE 0
[Testcase 140]: L2 4583; PVBand 13147; EPE 0
[Testcase 141]: L2 20342; PVBand 40526; EPE 0
[Testcase 142]: L2 4654; PVBand 6336; EPE 0
[Testcase 143]: L2 11661; PVBand 20258; EPE 0
[Testcase 144]: L2 4755; PVBand 6929; EPE 0
[Testcase 145]: L2 9653; PVBand 15343; EPE 0
[Testcase 146]: L2 4811; PVBand 6229; EPE 0
[Testcase 147]: L2 8868; PVBand 20042; EPE 0
[Testcase 148]: L2 4360; PVBand 6675; EPE 0
[Testcase 149]: L2 5163; PVBand 8502; EPE 0
[Testcase 150]: L2 8188; PVBand 14485; EPE 0
[Testcase 151]: L2 3861; PVBand 6507; EPE 0
[Testcase 152]: L2 2901; PVBand 6717; EPE 0
[Testcase 153]: L2 16559; PVBand 46039; EPE 0
[Testcase 154]: L2 17824; PVBand 41861; EPE 0
[Testcase 155]: L2 19907; PVBand 39859; EPE 0
[Testcase 156]: L2 8984; PVBand 17396; EPE 0
[Testcase 157]: L2 5034; PVBand 15233; EPE 0
[Testcase 158]: L2 36761; PVBand 65242; EPE 0
[Testcase 159]: L2 9467; PVBand 19984; EPE 0
[Testcase 160]: L2 17025; PVBand 33974; EPE 0
[Testcase 161]: L2 12221; PVBand 25976; EPE 0
[Testcase 162]: L2 4360; PVBand 6675; EPE 0
[Testcase 163]: L2 9326; PVBand 16658; EPE 0
[Testcase 164]: L2 3949; PVBand 6370; EPE 0
[Testcase 165]: L2 22721; PVBand 48412; EPE 0
[Testcase 166]: L2 19622; PVBand 38299; EPE 0
[Testcase 167]: L2 23572; PVBand 44247; EPE 0
[Testcase 168]: L2 8707; PVBand 19867; EPE 0
[Testcase 169]: L2 18514; PVBand 40815; EPE 0
[Testcase 170]: L2 15612; PVBand 34378; EPE 0
[Testcase 171]: L2 4102; PVBand 12429; EPE 0
[Testcase 172]: L2 5031; PVBand 14571; EPE 0
[Testcase 173]: L2 34443; PVBand 56416; EPE 0
[Testcase 174]: L2 27056; PVBand 53666; EPE 0
[Testcase 175]: L2 33020; PVBand 56011; EPE 0
[Testcase 176]: L2 14290; PVBand 21210; EPE 0
[Testcase 177]: L2 9502; PVBand 19699; EPE 0
[Testcase 178]: L2 9271; PVBand 13292; EPE 0
[Testcase 179]: L2 8620; PVBand 14594; EPE 0
[Testcase 180]: L2 12333; PVBand 25729; EPE 0
[Testcase 181]: L2 10886; PVBand 24812; EPE 0
[Testcase 182]: L2 3130; PVBand 6046; EPE 0
[Testcase 183]: L2 6843; PVBand 17687; EPE 0
[Testcase 184]: L2 18938; PVBand 39164; EPE 0
[Testcase 185]: L2 7724; PVBand 20035; EPE 0
[Testcase 186]: L2 35726; PVBand 72283; EPE 0
[Testcase 187]: L2 7611; PVBand 13685; EPE 0
[Testcase 188]: L2 7810; PVBand 12055; EPE 0
[Testcase 189]: L2 13925; PVBand 36968; EPE 0
[Testcase 190]: L2 4756; PVBand 7644; EPE 0
[Testcase 191]: L2 11661; PVBand 20258; EPE 0
[Testcase 192]: L2 15374; PVBand 45852; EPE 0
[Testcase 193]: L2 19851; PVBand 44063; EPE 0
[Testcase 194]: L2 18403; PVBand 35190; EPE 0
[Testcase 195]: L2 11365; PVBand 25406; EPE 0
[Testcase 196]: L2 13616; PVBand 23684; EPE 0
[Testcase 197]: L2 4376; PVBand 8064; EPE 0
[Testcase 198]: L2 9984; PVBand 19482; EPE 0
[Testcase 199]: L2 12545; PVBand 30568; EPE 0
[Testcase 200]: L2 12481; PVBand 25205; EPE 0
[Testcase 201]: L2 3532; PVBand 6996; EPE 0
[Testcase 202]: L2 4983; PVBand 7383; EPE 0
[Testcase 203]: L2 22184; PVBand 38283; EPE 0
[Testcase 204]: L2 4596; PVBand 6667; EPE 0
[Testcase 205]: L2 42757; PVBand 80676; EPE 1
[Testcase 206]: L2 7094; PVBand 20589; EPE 0
[Testcase 207]: L2 23242; PVBand 56717; EPE 0
[Testcase 208]: L2 9271; PVBand 13292; EPE 0
[Testcase 209]: L2 9834; PVBand 17293; EPE 0
[Testcase 210]: L2 36155; PVBand 56714; EPE 0
[Testcase 211]: L2 5090; PVBand 9975; EPE 0
[Testcase 212]: L2 4816; PVBand 6002; EPE 0
[Testcase 213]: L2 9601; PVBand 19498; EPE 0
[Testcase 214]: L2 13330; PVBand 20911; EPE 0
[Testcase 215]: L2 32758; PVBand 55868; EPE 0
[Testcase 216]: L2 12065; PVBand 25476; EPE 0
[Testcase 217]: L2 27161; PVBand 46043; EPE 3
[Testcase 218]: L2 8917; PVBand 18755; EPE 0
[Testcase 219]: L2 8370; PVBand 17888; EPE 0
[Testcase 220]: L2 19727; PVBand 49508; EPE 0
[Testcase 221]: L2 11433; PVBand 25325; EPE 0
[Testcase 222]: L2 7471; PVBand 9379; EPE 0
[Testcase 223]: L2 4206; PVBand 6209; EPE 0
[Testcase 224]: L2 9189; PVBand 19924; EPE 0
[Testcase 225]: L2 12760; PVBand 17144; EPE 0
[Testcase 226]: L2 8408; PVBand 32019; EPE 0
[Testcase 227]: L2 13941; PVBand 28308; EPE 0
[Testcase 228]: L2 6154; PVBand 26417; EPE 0
[Testcase 229]: L2 22769; PVBand 49651; EPE 0
[Testcase 230]: L2 27958; PVBand 53954; EPE 0
[Testcase 231]: L2 13000; PVBand 27398; EPE 0
[Testcase 232]: L2 14948; PVBand 37648; EPE 0
[Testcase 233]: L2 12964; PVBand 21117; EPE 0
[Testcase 234]: L2 7946; PVBand 14698; EPE 0
[Testcase 235]: L2 19181; PVBand 38853; EPE 0
[Testcase 236]: L2 9318; PVBand 15737; EPE 0
[Testcase 237]: L2 9162; PVBand 19989; EPE 0
[Testcase 238]: L2 12764; PVBand 25588; EPE 0
[Testcase 239]: L2 21099; PVBand 48138; EPE 0
[Testcase 240]: L2 9271; PVBand 13292; EPE 0
[Testcase 241]: L2 9451; PVBand 25982; EPE 0
[Testcase 242]: L2 3667; PVBand 6881; EPE 0
[Testcase 243]: L2 4453; PVBand 6905; EPE 0
[Testcase 244]: L2 9162; PVBand 19989; EPE 0
[Testcase 245]: L2 16330; PVBand 34736; EPE 0
[Testcase 246]: L2 21477; PVBand 38132; EPE 0
[Testcase 247]: L2 14826; PVBand 29855; EPE 0
[Testcase 248]: L2 40269; PVBand 71725; EPE 0
[Testcase 249]: L2 4816; PVBand 6002; EPE 0
[Testcase 250]: L2 9502; PVBand 19699; EPE 0
[Testcase 251]: L2 7946; PVBand 14698; EPE 0
[Testcase 252]: L2 7086; PVBand 11234; EPE 0
[Testcase 253]: L2 4536; PVBand 7194; EPE 0
[Testcase 254]: L2 9361; PVBand 18413; EPE 0
[Testcase 255]: L2 16966; PVBand 28365; EPE 0
[Testcase 256]: L2 4596; PVBand 6667; EPE 0
[Testcase 257]: L2 46171; PVBand 80785; EPE 3
[Testcase 258]: L2 15361; PVBand 31548; EPE 0
[Testcase 259]: L2 17993; PVBand 46908; EPE 0
[Testcase 260]: L2 6080; PVBand 8936; EPE 0
[Testcase 261]: L2 41626; PVBand 61287; EPE 0
[Testcase 262]: L2 18090; PVBand 31047; EPE 0
[Testcase 263]: L2 6738; PVBand 16317; EPE 0
[Testcase 264]: L2 9876; PVBand 22910; EPE 0
[Testcase 265]: L2 5449; PVBand 8764; EPE 0
[Testcase 266]: L2 21545; PVBand 40252; EPE 0
[Testcase 267]: L2 11341; PVBand 18005; EPE 0
[Testcase 268]: L2 5182; PVBand 7565; EPE 0
[Testcase 269]: L2 5553; PVBand 11122; EPE 0
[Testcase 270]: L2 11676; PVBand 23986; EPE 0
[Testcase 271]: L2 5163; PVBand 8502; EPE 0
[Finetuned]: L2 12883; PVBand 24956; EPE 0.0

[StdContact]
[Evaluation]: Getting masks from the model
[Testcase 1]: L2 65833; PVBand 38923; EPE 40
[Testcase 2]: L2 50567; PVBand 44376; EPE 24
[Testcase 3]: L2 38817; PVBand 30956; EPE 19
[Testcase 4]: L2 56877; PVBand 27020; EPE 39
[Testcase 5]: L2 54323; PVBand 30839; EPE 28
[Testcase 6]: L2 53871; PVBand 43600; EPE 24
[Testcase 7]: L2 37733; PVBand 29482; EPE 18
[Testcase 8]: L2 59607; PVBand 39806; EPE 36
[Testcase 9]: L2 78002; PVBand 43804; EPE 48
[Testcase 10]: L2 43500; PVBand 28576; EPE 25
[Testcase 11]: L2 42195; PVBand 38506; EPE 18
[Testcase 12]: L2 61650; PVBand 33759; EPE 42
[Testcase 13]: L2 66154; PVBand 45525; EPE 35
[Testcase 14]: L2 48448; PVBand 37803; EPE 19
[Testcase 15]: L2 41335; PVBand 33842; EPE 19
[Testcase 16]: L2 60756; PVBand 30895; EPE 40
[Testcase 17]: L2 73158; PVBand 34731; EPE 51
[Testcase 18]: L2 67654; PVBand 40083; EPE 41
[Testcase 19]: L2 47992; PVBand 25340; EPE 30
[Testcase 20]: L2 37922; PVBand 29366; EPE 18
[Testcase 21]: L2 53960; PVBand 36574; EPE 28
[Testcase 22]: L2 50470; PVBand 35610; EPE 29
[Testcase 23]: L2 52676; PVBand 29702; EPE 32
[Testcase 24]: L2 50260; PVBand 35295; EPE 25
[Testcase 25]: L2 48514; PVBand 41191; EPE 20
[Testcase 26]: L2 49591; PVBand 31290; EPE 29
[Testcase 27]: L2 34467; PVBand 34503; EPE 11
[Testcase 28]: L2 52331; PVBand 32056; EPE 32
[Testcase 29]: L2 48695; PVBand 40677; EPE 27
[Testcase 30]: L2 48285; PVBand 21174; EPE 33
[Testcase 31]: L2 41594; PVBand 30923; EPE 17
[Testcase 32]: L2 65258; PVBand 39116; EPE 38
[Testcase 33]: L2 46922; PVBand 31015; EPE 29
[Testcase 34]: L2 32052; PVBand 24511; EPE 12
[Testcase 35]: L2 60703; PVBand 67007; EPE 22
[Testcase 36]: L2 61298; PVBand 61813; EPE 26
[Testcase 37]: L2 74463; PVBand 30518; EPE 54
[Testcase 38]: L2 60520; PVBand 30496; EPE 38
[Testcase 39]: L2 48627; PVBand 32210; EPE 28
[Testcase 40]: L2 35860; PVBand 28514; EPE 17
[Testcase 41]: L2 37267; PVBand 32934; EPE 16
[Testcase 42]: L2 63442; PVBand 31260; EPE 43
[Testcase 43]: L2 38195; PVBand 33093; EPE 17
[Testcase 44]: L2 51999; PVBand 55489; EPE 17
[Testcase 45]: L2 43183; PVBand 32194; EPE 20
[Testcase 46]: L2 49261; PVBand 34228; EPE 23
[Testcase 47]: L2 37259; PVBand 26738; EPE 19
[Testcase 48]: L2 39212; PVBand 35446; EPE 16
[Testcase 49]: L2 41914; PVBand 32970; EPE 21
[Testcase 50]: L2 51672; PVBand 33794; EPE 26
[Testcase 51]: L2 40797; PVBand 42371; EPE 16
[Testcase 52]: L2 34089; PVBand 21843; EPE 21
[Testcase 53]: L2 40229; PVBand 23635; EPE 21
[Testcase 54]: L2 38985; PVBand 36463; EPE 14
[Testcase 55]: L2 42650; PVBand 38273; EPE 16
[Testcase 56]: L2 56127; PVBand 41440; EPE 26
[Testcase 57]: L2 52120; PVBand 23096; EPE 33
[Testcase 58]: L2 41943; PVBand 30956; EPE 21
[Testcase 59]: L2 57177; PVBand 50778; EPE 28
[Testcase 60]: L2 51247; PVBand 27531; EPE 33
[Testcase 61]: L2 66645; PVBand 47468; EPE 37
[Testcase 62]: L2 74504; PVBand 41161; EPE 45
[Testcase 63]: L2 34850; PVBand 23773; EPE 17
[Testcase 64]: L2 49714; PVBand 36720; EPE 24
[Testcase 65]: L2 37794; PVBand 22518; EPE 24
[Testcase 66]: L2 59429; PVBand 29989; EPE 38
[Testcase 67]: L2 68870; PVBand 35379; EPE 47
[Testcase 68]: L2 66146; PVBand 34609; EPE 44
[Testcase 69]: L2 60534; PVBand 31339; EPE 39
[Testcase 70]: L2 31481; PVBand 28784; EPE 14
[Testcase 71]: L2 48204; PVBand 35975; EPE 20
[Testcase 72]: L2 34692; PVBand 32874; EPE 12
[Testcase 73]: L2 68076; PVBand 44296; EPE 40
[Testcase 74]: L2 41835; PVBand 45226; EPE 18
[Testcase 75]: L2 61885; PVBand 34032; EPE 40
[Testcase 76]: L2 55381; PVBand 32797; EPE 36
[Testcase 77]: L2 36651; PVBand 25919; EPE 19
[Testcase 78]: L2 64016; PVBand 41395; EPE 33
[Testcase 79]: L2 49201; PVBand 44711; EPE 16
[Testcase 80]: L2 50794; PVBand 40373; EPE 24
[Testcase 81]: L2 72352; PVBand 42212; EPE 48
[Testcase 82]: L2 52260; PVBand 41251; EPE 21
[Testcase 83]: L2 66904; PVBand 35466; EPE 44
[Testcase 84]: L2 51436; PVBand 25796; EPE 31
[Testcase 85]: L2 46188; PVBand 38194; EPE 23
[Testcase 86]: L2 56209; PVBand 34631; EPE 32
[Testcase 87]: L2 57043; PVBand 45475; EPE 25
[Testcase 88]: L2 30513; PVBand 20558; EPE 18
[Testcase 89]: L2 58813; PVBand 26071; EPE 44
[Testcase 90]: L2 56696; PVBand 47216; EPE 26
[Testcase 91]: L2 51404; PVBand 39258; EPE 25
[Testcase 92]: L2 57431; PVBand 33230; EPE 35
[Testcase 93]: L2 43294; PVBand 27450; EPE 25
[Testcase 94]: L2 63708; PVBand 42379; EPE 33
[Testcase 95]: L2 44087; PVBand 32411; EPE 21
[Testcase 96]: L2 69669; PVBand 28059; EPE 52
[Testcase 97]: L2 44997; PVBand 33578; EPE 28
[Testcase 98]: L2 43619; PVBand 43024; EPE 14
[Testcase 99]: L2 55363; PVBand 35950; EPE 27
[Testcase 100]: L2 42598; PVBand 33829; EPE 21
[Testcase 101]: L2 32354; PVBand 28413; EPE 11
[Testcase 102]: L2 64436; PVBand 40177; EPE 37
[Testcase 103]: L2 68905; PVBand 52771; EPE 29
[Testcase 104]: L2 64956; PVBand 41431; EPE 38
[Testcase 105]: L2 51431; PVBand 53455; EPE 21
[Testcase 106]: L2 39850; PVBand 33930; EPE 17
[Testcase 107]: L2 40520; PVBand 32712; EPE 21
[Testcase 108]: L2 63415; PVBand 40722; EPE 35
[Testcase 109]: L2 63673; PVBand 49589; EPE 32
[Testcase 110]: L2 27507; PVBand 28241; EPE 5
[Testcase 111]: L2 75650; PVBand 49522; EPE 44
[Testcase 112]: L2 53759; PVBand 47176; EPE 25
[Testcase 113]: L2 48958; PVBand 36146; EPE 25
[Testcase 114]: L2 35718; PVBand 25359; EPE 18
[Testcase 115]: L2 35109; PVBand 28002; EPE 18
[Testcase 116]: L2 69302; PVBand 34120; EPE 40
[Testcase 117]: L2 37242; PVBand 34221; EPE 10
[Testcase 118]: L2 58157; PVBand 32151; EPE 41
[Testcase 119]: L2 79831; PVBand 36440; EPE 51
[Testcase 120]: L2 37682; PVBand 28081; EPE 17
[Testcase 121]: L2 60141; PVBand 65581; EPE 18
[Testcase 122]: L2 44816; PVBand 24827; EPE 29
[Testcase 123]: L2 35892; PVBand 28580; EPE 17
[Testcase 124]: L2 44726; PVBand 30265; EPE 22
[Testcase 125]: L2 49144; PVBand 35616; EPE 28
[Testcase 126]: L2 41636; PVBand 26917; EPE 25
[Testcase 127]: L2 60528; PVBand 54464; EPE 33
[Testcase 128]: L2 29215; PVBand 25457; EPE 13
[Testcase 129]: L2 73283; PVBand 42633; EPE 42
[Testcase 130]: L2 43106; PVBand 36507; EPE 19
[Testcase 131]: L2 61843; PVBand 35402; EPE 39
[Testcase 132]: L2 39296; PVBand 37912; EPE 18
[Testcase 133]: L2 34139; PVBand 39634; EPE 10
[Testcase 134]: L2 38906; PVBand 28260; EPE 18
[Testcase 135]: L2 57544; PVBand 32973; EPE 35
[Testcase 136]: L2 45497; PVBand 31132; EPE 26
[Testcase 137]: L2 52849; PVBand 31600; EPE 33
[Testcase 138]: L2 35246; PVBand 27774; EPE 17
[Testcase 139]: L2 36058; PVBand 31767; EPE 18
[Testcase 140]: L2 39024; PVBand 30888; EPE 18
[Testcase 141]: L2 55600; PVBand 61061; EPE 21
[Testcase 142]: L2 41953; PVBand 35938; EPE 21
[Testcase 143]: L2 42494; PVBand 18644; EPE 31
[Testcase 144]: L2 53819; PVBand 38399; EPE 25
[Testcase 145]: L2 44619; PVBand 38396; EPE 11
[Testcase 146]: L2 65366; PVBand 33312; EPE 41
[Testcase 147]: L2 66643; PVBand 58024; EPE 30
[Testcase 148]: L2 54105; PVBand 40254; EPE 29
[Testcase 149]: L2 35690; PVBand 30510; EPE 11
[Testcase 150]: L2 36722; PVBand 30484; EPE 12
[Testcase 151]: L2 43668; PVBand 27252; EPE 25
[Testcase 152]: L2 54581; PVBand 44971; EPE 26
[Testcase 153]: L2 41605; PVBand 39726; EPE 17
[Testcase 154]: L2 32596; PVBand 29751; EPE 15
[Testcase 155]: L2 46013; PVBand 32812; EPE 26
[Testcase 156]: L2 54761; PVBand 34227; EPE 30
[Testcase 157]: L2 60250; PVBand 35499; EPE 34
[Testcase 158]: L2 59396; PVBand 33945; EPE 36
[Testcase 159]: L2 40971; PVBand 30355; EPE 21
[Testcase 160]: L2 42977; PVBand 30736; EPE 22
[Testcase 161]: L2 44551; PVBand 29301; EPE 24
[Testcase 162]: L2 61832; PVBand 46123; EPE 30
[Testcase 163]: L2 48846; PVBand 38175; EPE 22
[Testcase 164]: L2 38252; PVBand 43774; EPE 14
[Testcase 165]: L2 47721; PVBand 32243; EPE 24
[Initialized]: L2 50445; PVBand 35673; EPE 26.7
[Evaluation]: Finetuning the masks with pixel-based ILT method
[Testcase 1]: L2 35300; PVBand 53059; EPE 6
[Testcase 2]: L2 28150; PVBand 47230; EPE 0
[Testcase 3]: L2 19212; PVBand 35461; EPE 0
[Testcase 4]: L2 24083; PVBand 40151; EPE 1
[Testcase 5]: L2 24768; PVBand 43702; EPE 0
[Testcase 6]: L2 29037; PVBand 48029; EPE 1
[Testcase 7]: L2 18130; PVBand 32673; EPE 0
[Testcase 8]: L2 29732; PVBand 49456; EPE 0
[Testcase 9]: L2 44185; PVBand 61604; EPE 9
[Testcase 10]: L2 22304; PVBand 36285; EPE 2
[Testcase 11]: L2 32614; PVBand 41782; EPE 12
[Testcase 12]: L2 30063; PVBand 44756; EPE 3
[Testcase 13]: L2 42544; PVBand 60989; EPE 9
[Testcase 14]: L2 24965; PVBand 40762; EPE 0
[Testcase 15]: L2 26160; PVBand 36894; EPE 5
[Testcase 16]: L2 32995; PVBand 47310; EPE 9
[Testcase 17]: L2 40579; PVBand 53641; EPE 11
[Testcase 18]: L2 35122; PVBand 51473; EPE 5
[Testcase 19]: L2 20799; PVBand 36561; EPE 0
[Testcase 20]: L2 17330; PVBand 32715; EPE 0
[Testcase 21]: L2 29363; PVBand 42533; EPE 5
[Testcase 22]: L2 23524; PVBand 41223; EPE 0
[Testcase 23]: L2 26397; PVBand 39029; EPE 5
[Testcase 24]: L2 25093; PVBand 42806; EPE 1
[Testcase 25]: L2 26122; PVBand 45934; EPE 1
[Testcase 26]: L2 20861; PVBand 37400; EPE 0
[Testcase 27]: L2 21627; PVBand 37176; EPE 2
[Testcase 28]: L2 25688; PVBand 43667; EPE 0
[Testcase 29]: L2 28975; PVBand 47334; EPE 3
[Testcase 30]: L2 18086; PVBand 32051; EPE 0
[Testcase 31]: L2 23869; PVBand 39190; EPE 3
[Testcase 32]: L2 34728; PVBand 52706; EPE 5
[Testcase 33]: L2 22002; PVBand 38339; EPE 1
[Testcase 34]: L2 14902; PVBand 26891; EPE 0
[Testcase 35]: L2 59407; PVBand 66213; EPE 27
[Testcase 36]: L2 51824; PVBand 63010; EPE 17
[Testcase 37]: L2 30157; PVBand 50386; EPE 1
[Testcase 38]: L2 24335; PVBand 41782; EPE 0
[Testcase 39]: L2 22478; PVBand 40813; EPE 0
[Testcase 40]: L2 27572; PVBand 35657; EPE 9
[Testcase 41]: L2 20930; PVBand 35917; EPE 0
[Testcase 42]: L2 30415; PVBand 45132; EPE 4
[Testcase 43]: L2 21407; PVBand 36889; EPE 2
[Testcase 44]: L2 44188; PVBand 54674; EPE 15
[Testcase 45]: L2 21483; PVBand 38817; EPE 0
[Testcase 46]: L2 24641; PVBand 40088; EPE 2
[Testcase 47]: L2 17842; PVBand 32011; EPE 0
[Testcase 48]: L2 21431; PVBand 37595; EPE 0
[Testcase 49]: L2 23125; PVBand 39650; EPE 2
[Testcase 50]: L2 23875; PVBand 40321; EPE 1
[Testcase 51]: L2 23082; PVBand 46673; EPE 0
[Testcase 52]: L2 15748; PVBand 29554; EPE 1
[Testcase 53]: L2 17639; PVBand 30480; EPE 0
[Testcase 54]: L2 24571; PVBand 41906; EPE 2
[Testcase 55]: L2 24885; PVBand 42768; EPE 1
[Testcase 56]: L2 30322; PVBand 50406; EPE 2
[Testcase 57]: L2 20462; PVBand 34354; EPE 0
[Testcase 58]: L2 21924; PVBand 37935; EPE 0
[Testcase 59]: L2 42595; PVBand 57808; EPE 11
[Testcase 60]: L2 21974; PVBand 38000; EPE 0
[Testcase 61]: L2 47682; PVBand 61389; EPE 13
[Testcase 62]: L2 43069; PVBand 60380; EPE 8
[Testcase 63]: L2 15892; PVBand 29501; EPE 0
[Testcase 64]: L2 24255; PVBand 40409; EPE 0
[Testcase 65]: L2 17007; PVBand 28803; EPE 0
[Testcase 66]: L2 24809; PVBand 43257; EPE 1
[Testcase 67]: L2 38280; PVBand 50578; EPE 9
[Testcase 68]: L2 31250; PVBand 47187; EPE 4
[Testcase 69]: L2 28241; PVBand 44537; EPE 4
[Testcase 70]: L2 18244; PVBand 30609; EPE 0
[Testcase 71]: L2 27390; PVBand 43061; EPE 3
[Testcase 72]: L2 20419; PVBand 34540; EPE 1
[Testcase 73]: L2 42896; PVBand 56299; EPE 9
[Testcase 74]: L2 24518; PVBand 46644; EPE 0
[Testcase 75]: L2 30358; PVBand 47665; EPE 4
[Testcase 76]: L2 26828; PVBand 44766; EPE 1
[Testcase 77]: L2 18582; PVBand 32957; EPE 1
[Testcase 78]: L2 40127; PVBand 51325; EPE 9
[Testcase 79]: L2 31224; PVBand 48212; EPE 6
[Testcase 80]: L2 32681; PVBand 51503; EPE 5
[Testcase 81]: L2 35712; PVBand 55024; EPE 5
[Testcase 82]: L2 28203; PVBand 46813; EPE 2
[Testcase 83]: L2 33342; PVBand 50817; EPE 4
[Testcase 84]: L2 22976; PVBand 39697; EPE 0
[Testcase 85]: L2 27744; PVBand 46510; EPE 3
[Testcase 86]: L2 29300; PVBand 44856; EPE 4
[Testcase 87]: L2 37874; PVBand 58116; EPE 6
[Testcase 88]: L2 14425; PVBand 25107; EPE 0
[Testcase 89]: L2 24116; PVBand 43978; EPE 0
[Testcase 90]: L2 30954; PVBand 54131; EPE 1
[Testcase 91]: L2 27811; PVBand 46270; EPE 1
[Testcase 92]: L2 29105; PVBand 44711; EPE 3
[Testcase 93]: L2 20132; PVBand 36034; EPE 0
[Testcase 94]: L2 38173; PVBand 51939; EPE 8
[Testcase 95]: L2 21281; PVBand 38891; EPE 0
[Testcase 96]: L2 30160; PVBand 48938; EPE 2
[Testcase 97]: L2 21991; PVBand 41595; EPE 1
[Testcase 98]: L2 28062; PVBand 45230; EPE 5
[Testcase 99]: L2 24477; PVBand 42417; EPE 0
[Testcase 100]: L2 23001; PVBand 40711; EPE 1
[Testcase 101]: L2 16470; PVBand 28768; EPE 0
[Testcase 102]: L2 34248; PVBand 52408; EPE 6
[Testcase 103]: L2 51505; PVBand 62685; EPE 16
[Testcase 104]: L2 39655; PVBand 54011; EPE 6
[Testcase 105]: L2 42010; PVBand 60446; EPE 12
[Testcase 106]: L2 26236; PVBand 39381; EPE 6
[Testcase 107]: L2 23597; PVBand 37056; EPE 3
[Testcase 108]: L2 36902; PVBand 54053; EPE 10
[Testcase 109]: L2 41561; PVBand 57123; EPE 8
[Testcase 110]: L2 16146; PVBand 28570; EPE 0
[Testcase 111]: L2 46438; PVBand 63834; EPE 10
[Testcase 112]: L2 32040; PVBand 53835; EPE 2
[Testcase 113]: L2 26317; PVBand 42004; EPE 3
[Testcase 114]: L2 16776; PVBand 30944; EPE 0
[Testcase 115]: L2 18555; PVBand 33097; EPE 0
[Testcase 116]: L2 32089; PVBand 53416; EPE 0
[Testcase 117]: L2 19984; PVBand 34886; EPE 0
[Testcase 118]: L2 30278; PVBand 44896; EPE 4
[Testcase 119]: L2 42931; PVBand 59211; EPE 10
[Testcase 120]: L2 16946; PVBand 30504; EPE 0
[Testcase 121]: L2 56126; PVBand 65874; EPE 19
[Testcase 122]: L2 18885; PVBand 33568; EPE 0
[Testcase 123]: L2 18644; PVBand 33741; EPE 0
[Testcase 124]: L2 22016; PVBand 36077; EPE 0
[Testcase 125]: L2 30933; PVBand 47322; EPE 5
[Testcase 126]: L2 18471; PVBand 32812; EPE 1
[Testcase 127]: L2 36733; PVBand 59853; EPE 5
[Testcase 128]: L2 14183; PVBand 27580; EPE 0
[Testcase 129]: L2 39797; PVBand 57154; EPE 8
[Testcase 130]: L2 23838; PVBand 44010; EPE 0
[Testcase 131]: L2 42394; PVBand 51291; EPE 14
[Testcase 132]: L2 23380; PVBand 41566; EPE 0
[Testcase 133]: L2 20253; PVBand 39778; EPE 0
[Testcase 134]: L2 20432; PVBand 33992; EPE 1
[Testcase 135]: L2 28572; PVBand 44806; EPE 4
[Testcase 136]: L2 23386; PVBand 40703; EPE 0
[Testcase 137]: L2 25910; PVBand 41540; EPE 2
[Testcase 138]: L2 18087; PVBand 32174; EPE 0
[Testcase 139]: L2 20606; PVBand 36873; EPE 0
[Testcase 140]: L2 23032; PVBand 39627; EPE 1
[Testcase 141]: L2 61166; PVBand 57106; EPE 30
[Testcase 142]: L2 22602; PVBand 42120; EPE 0
[Testcase 143]: L2 17829; PVBand 29280; EPE 2
[Testcase 144]: L2 26060; PVBand 46283; EPE 0
[Testcase 145]: L2 22298; PVBand 39593; EPE 0
[Testcase 146]: L2 31148; PVBand 47445; EPE 4
[Testcase 147]: L2 54161; PVBand 64270; EPE 20
[Testcase 148]: L2 31683; PVBand 49114; EPE 5
[Testcase 149]: L2 18712; PVBand 33756; EPE 0
[Testcase 150]: L2 17880; PVBand 31777; EPE 0
[Testcase 151]: L2 23103; PVBand 37112; EPE 4
[Testcase 152]: L2 34802; PVBand 53065; EPE 5
[Testcase 153]: L2 23855; PVBand 44391; EPE 1
[Testcase 154]: L2 21515; PVBand 31042; EPE 4
[Testcase 155]: L2 23490; PVBand 41344; EPE 2
[Testcase 156]: L2 26647; PVBand 40037; EPE 4
[Testcase 157]: L2 27556; PVBand 46573; EPE 1
[Testcase 158]: L2 38440; PVBand 47828; EPE 13
[Testcase 159]: L2 22299; PVBand 36134; EPE 1
[Testcase 160]: L2 20248; PVBand 36079; EPE 0
[Testcase 161]: L2 21763; PVBand 36918; EPE 0
[Testcase 162]: L2 41451; PVBand 54504; EPE 15
[Testcase 163]: L2 24750; PVBand 42637; EPE 0
[Testcase 164]: L2 28919; PVBand 45455; EPE 4
[Testcase 165]: L2 26130; PVBand 42094; EPE 3
[Finetuned]: L2 27910; PVBand 43651; EPE 3.6
'''
