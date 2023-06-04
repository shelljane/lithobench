import os
import sys
sys.path.append(".")
import glob
import time
import math
import random
import pickle

import numpy as np 
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pycommon.settings import *
import lithobench.evaluate as evaluate
from lithobench.dataset import loadersLitho

class ModelILT: 
    def __init__(self, size=(512, 512), name="ModelILT"): 
        self._size = size
        self._name = name
        pass
    
    @property
    def size(self): 
        return self._size
    @property
    def name(self): 
        return self._name

    def pretrain(self, train_loader, val_loader, epochs=1, batch_size=4): 
        pass

    def train(self, train_loader, val_loader, epochs=1, batch_size=4): 
        pass
    
    def save(self, filenames): 
        pass
    
    def load(self, filenames): 
        pass

    def run(self, target): 
        pass
    
    def evaluate(self, targets, finetune=False, folder="images", shot=False): 
        if shot == True: 
            print(f"[ModelILT]: Warning, shot counting is enabled. The evaluation will be slow. ")
            
        paramsInit = []
        masksInit = []
        runtimes = []
        for target in targets: 
            target = torch.tensor(target, dtype=REALTYPE, device=DEVICE)
            x = F.interpolate(target[None, None, :, :], size=self._size)
            runtime = time.time()
            params0 = self.run(x)
            runtime = time.time() - runtime
            mask0 = params0
            mask0[mask0 > 0.5] = 1.0
            mask0[mask0 <= 0.5] = 0.0
            paramsInit.append(params0)
            masksInit.append(mask0)
            runtimes.append(runtime)
        
        if not os.path.exists(folder): 
            os.mkdir(folder)
        print(f"[Evaluation]: Getting masks from the model")
        l2s, pvbs, epes = evaluate.evalRaw(masksInit, targets)
        if shot: 
            shots = evaluate.shots(masksInit)
        else: 
            shots = [-1 for _ in range(len(masksInit))]
        for idx in range(len(targets)): 
            print(f"[Testcase {idx+1}]: L2 {l2s[idx]:.0f}; PVBand {pvbs[idx]:.0f}; EPE {epes[idx]:.0f}; Shots: {shots[idx]:.0f}")
            cv2.imwrite(f"{folder}/{self.name}_mask0_{idx+1}.png", (masksInit[idx]*255).detach().cpu().numpy())
        print(f"[Initialized]: L2 {np.mean(l2s):.0f}; PVBand {np.mean(pvbs):.0f}; EPE {np.mean(epes):.1f}; Runtime: {np.mean(runtimes):.2f}s; Shots: {np.mean(shots):.0f}")
        
        if not finetune: 
            return
        
        print(f"[Evaluation]: Finetuning the masks with pixel-based ILT method")
        # l2s, pvbs, epes, masks = evaluate.finetune(paramsInit, masksInit, targets)
        l2s, pvbs, epes, masks = evaluate.finetuneFast(paramsInit, masksInit, targets)
        if shot: 
            shots = evaluate.shots(masks)
        else: 
            shots = [-1 for _ in range(len(masks))]
        for idx in range(len(targets)): 
            cv2.imwrite(f"{folder}/{self.name}_mask1_{idx+1}.png", (masks[idx]*255).detach().cpu().numpy())
            print(f"[Testcase {idx+1}]: L2 {l2s[idx]:.0f}; PVBand {pvbs[idx]:.0f}; EPE {epes[idx]:.0f}; Shots: {shots[idx]:.0f}")
        print(f"[Finetuned]: L2 {np.mean(l2s):.0f}; PVBand {np.mean(pvbs):.0f}; EPE {np.mean(epes):.1f}; Shots: {np.mean(shots):.0f}")


class ModelLitho: 
    def __init__(self, size=(512, 512), name="ModelILT"): 
        self._size = size
        self._name = name
        pass
    
    @property
    def size(self): 
        return self._size
    @property
    def name(self): 
        return self._name

    def pretrain(self, train_loader, val_loader, epochs=1, batch_size=4): 
        pass

    def train(self, train_loader, val_loader, epochs=1, batch_size=4): 
        pass

    def run(self, target): 
        pass
    
    def evaluate(self, benchmark, image_size, batch_size=4, njobs=8, test_loader=None, folder="images", samples=0): 
        if test_loader is None: 
            _, test_loader = loadersLitho(benchmark, (2048, 2048), batch_size, njobs)
        lossesAerial = []
        lossesResist = []
        mious = []
        mpas = []
        count = 0
        progress = tqdm(test_loader)
        for mask, litho, label in progress: 
            if torch.cuda.is_available():
                mask = mask.cuda()
                litho = litho.cuda()
                label = label.cuda()
            
            mask = F.interpolate(mask, size=image_size)
            aerial, resist = self.run(mask)
            aerial, resist = aerial.detach(), resist.detach()
            
            aerial = F.interpolate(aerial, size=(2048, 2048))
            resist = F.interpolate(resist, size=(2048, 2048))
            
            resist[resist > 0.5] = 1.0
            resist[resist <= 0.5] = 0.0
            for idx in range(resist.shape[0]): 
                if count < samples: 
                    cv2.imwrite(f"{folder}/{self.name}_resist_{count+1}.png", (resist[idx, 0]*255).detach().cpu().numpy())
                    count += 1 
                    
            ored = (resist > 0.5) | (label > 0.5)
            anded = (resist > 0.5) & (label > 0.5)

            lossAerial = F.mse_loss(aerial, litho)
            lossResist = F.mse_loss(resist, label)
            miou = anded.sum() / ored.sum()
            mpa = anded.sum() / label.sum()
            lossesAerial.append(lossAerial.item())
            lossesResist.append(lossResist.item())
            mious.append(miou.item())
            mpas.append(mpa.item())

            progress.set_postfix(L2Aerial=lossAerial.item(), L2Resist=lossResist.item(), IOU=miou.item(), PA=mpa.item())
        
        print(f"[Evaluation] L2Aerial = {np.mean(lossesAerial)} L2Resist = {np.mean(lossesResist)} IOU = {np.mean(mious)} PA = {np.mean(mpas)}")



