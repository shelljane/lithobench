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


def filesMaskOpt(folder): 
    folderGLP = os.path.join(folder, "glp")
    folderTarget = os.path.join(folder, "target")
    folderPixel = os.path.join(folder, "pixelILT")
    filesGLP = glob.glob(folderGLP + "/*.glp")
    filesTarget = glob.glob(folderTarget + "/*.png")
    filesPixel = glob.glob(folderPixel + "/*.png")
    basefunc = lambda x: os.path.basename(x)[:-4]
    setGLP = set(map(basefunc, filesGLP))
    setTarget = set(map(basefunc, filesTarget))
    setPixel = set(map(basefunc, filesPixel))
    basenames = setGLP & setTarget & setPixel
    filesGLP = list(filter(lambda x: basefunc(x) in basenames, filesGLP))
    filesTarget = list(filter(lambda x: basefunc(x) in basenames, filesTarget))
    filesPixel = list(filter(lambda x: basefunc(x) in basenames, filesPixel))
    filesGLP = sorted(filesGLP, key=basefunc)
    filesTarget = sorted(filesTarget, key=basefunc)
    filesPixel = sorted(filesPixel, key=basefunc)

    return filesGLP, filesTarget, filesPixel


def filesLithoSim(folder, binarized=True): 
    folderPixel = os.path.join(folder, "pixelILT")
    folderLitho = os.path.join(folder, "litho")
    folderResist = os.path.join(folder, "printed") if binarized else os.path.join(folder, "resist")
    filesPixel = glob.glob(folderPixel + "/*.png")
    filesLitho = glob.glob(folderLitho + "/*.png")
    filesResist = glob.glob(folderResist + "/*.png")
    basefunc = lambda x: os.path.basename(x)[:-4]
    setPixel = set(map(basefunc, filesPixel))
    setLitho = set(map(basefunc, filesLitho))
    setResist = set(map(basefunc, filesResist))
    basenames = setPixel & setLitho & setResist
    filesPixel = list(filter(lambda x: basefunc(x) in basenames, filesPixel))
    filesLitho = list(filter(lambda x: basefunc(x) in basenames, filesLitho))
    filesResist = list(filter(lambda x: basefunc(x) in basenames, filesResist))
    filesPixel = sorted(filesPixel, key=basefunc)
    filesLitho = sorted(filesLitho, key=basefunc)
    filesResist = sorted(filesResist, key=basefunc)

    return filesPixel, filesLitho, filesResist


class DataILT(torch.utils.data.Dataset): 
    def __init__(self, filesTarget, filesPixel, crop=False, size=(1024, 1024), cache=False):
        super().__init__()

        self._filesTarget, self._filesPixel = filesTarget, filesPixel
        self._crop = crop
        self._size = size
        self._cache = cache
    
        self._imagesTarget = []
        self._imagesPixel = []
        if self._cache: 
            print(f"Pre-loading the target images")
            for filename in tqdm(self._filesTarget): 
                self._imagesTarget.append(self._loadImage(filename))
            print(f"Pre-loading the mask images")
            for filename in tqdm(self._filesPixel): 
                self._imagesPixel.append(self._loadImage(filename))

    def __getitem__(self, index): 
        target, mask = self._loadTarget(index), self._loadMask(index)
        target = target[None, :, :]
        mask = mask[None, :, :]
        if self._crop: 
            padX = self._size[0] // 8
            padY = self._size[1] // 8
            startX = random.randint(0, 2*padX-1)
            startY = random.randint(0, 2*padY-1)
            target = F.pad(target.unsqueeze(0), (padX, padX, padY, padY))
            mask = F.pad(mask.unsqueeze(0), (padX, padX, padY, padY))
            target = target[0, :, startX:startX+self._size[0], startY:startY+self._size[1]]
            mask = mask[0, :, startX:startX+self._size[0], startY:startY+self._size[1]]
            if random.randint(0, 1) == 1: 
                target = target.flip(1)
                mask = mask.flip(1)
            if random.randint(0, 1) == 1: 
                target = target.flip(2)
                mask = mask.flip(2)
        return target, mask

    def __len__(self): 
        return len(self._filesTarget)

    def _loadImage(self, filename): 
        image = cv2.imread(filename)
        if len(image.shape) > 2: 
            image = torch.tensor(image[:, :, 0], dtype=torch.float32, device="cpu") / 255.0
        image = F.interpolate(image[None, None, :, :], self._size)[0, 0]
        return image
    
    def _loadTarget(self, index): 
        if self._cache: 
            return self._imagesTarget[index]
        else: 
            return self._loadImage(self._filesTarget[index])

    def _loadMask(self, index): 
        if self._cache: 
            return self._imagesPixel[index]
        else: 
            return self._loadImage(self._filesPixel[index])


class DataLitho(torch.utils.data.Dataset): 
    def __init__(self, filesMask, filesLitho, filesResist, crop=False, size=(1024, 1024), cache=False):
        super().__init__()
        
        assert len(filesMask) == len(filesLitho) and len(filesMask) == len(filesResist), f"WRONG SIZE: {len(filesMask)}/{len(filesLitho)}/{len(filesResist)}"

        self._filesMask, self._filesLitho, self._filesResist = filesMask, filesLitho, filesResist
        self._crop = crop
        self._size = size
        self._cache = cache
    
        self._imagesMask = []
        self._imagesLitho = []
        self._imagesResist = []
        if self._cache: 
            print(f"Pre-loading the mask images")
            for filename in tqdm(self._filesMask): 
                self._imagesMask.append(self._loadImage(filename))
            print(f"Pre-loading the litho images")
            for filename in tqdm(self._filesLitho): 
                self._imagesLitho.append(self._loadImage(filename))
            print(f"Pre-loading the resist images")
            for filename in tqdm(self._filesResist): 
                self._imagesResist.append(self._loadImage(filename))

    def __getitem__(self, index): 
        mask, litho, resist = self._loadMask(index), self._loadLitho(index), self._loadResist(index)
        mask = mask[None, :, :]
        litho = litho[None, :, :]
        resist = resist[None, :, :]
        if self._crop: 
            padX = self._size[0] // 8
            padY = self._size[1] // 8
            startX = random.randint(0, 2*padX-1)
            startY = random.randint(0, 2*padY-1)
            mask = F.pad(mask.unsqueeze(0), (padX, padX, padY, padY))
            litho = F.pad(litho.unsqueeze(0), (padX, padX, padY, padY))
            resist = F.pad(resist.unsqueeze(0), (padX, padX, padY, padY))
            mask = mask[0, :, startX:startX+self._size[0], startY:startY+self._size[1]]
            litho = litho[0, :, startX:startX+self._size[0], startY:startY+self._size[1]]
            resist = resist[0, :, startX:startX+self._size[0], startY:startY+self._size[1]]
            if random.randint(0, 1) == 1: 
                mask = mask.flip(1)
                litho = litho.flip(1)
                resist = resist.flip(1)
            if random.randint(0, 1) == 1: 
                mask = mask.flip(2)
                litho = litho.flip(2)
                resist = resist.flip(2)
        return mask, litho, resist

    def __len__(self): 
        return len(self._filesMask)

    def _loadImage(self, filename): 
        image = cv2.imread(filename)
        if len(image.shape) > 2: 
            image = torch.tensor(image[:, :, 0], dtype=torch.float32, device="cpu") / 255.0
        image = F.interpolate(image[None, None, :, :], self._size)[0, 0]
        return image
    
    def _loadMask(self, index): 
        if self._cache: 
            return self._imagesMask[index]
        else: 
            return self._loadImage(self._filesMask[index])
    
    def _loadLitho(self, index): 
        if self._cache: 
            return self._imagesLitho[index]
        else: 
            return self._loadImage(self._filesLitho[index])

    def _loadResist(self, index): 
        if self._cache: 
            return self._imagesResist[index]
        else: 
            return self._loadImage(self._filesResist[index])

    
def maskopt(basedir, sizeImage=(512, 512), ratioTrain=0.9, cache=False): 
    filesGLP, filesTarget, filesMask = filesMaskOpt(basedir)
    numFiles = len(filesGLP)
    numTrain = round(numFiles * ratioTrain)
    numTest  = numFiles - numTrain
    trainGLP = filesGLP[:numTrain]
    trainTarget = filesTarget[:numTrain]
    trainMask = filesMask[:numTrain]
    testGLP = filesGLP[numTrain:]
    testTarget = filesTarget[numTrain:]
    testMask = filesMask[numTrain:]
    train = DataILT(trainTarget, trainMask, crop=True, size=sizeImage, cache=cache)
    test = DataILT(testTarget, testMask, crop=False, size=sizeImage, cache=cache)
    print(f"Training set: {numTrain}, Test set: {numTest}")

    return train, test

    
def lithosim(basedir, sizeImage=(512, 512), ratioTrain=0.9, cache=False): 
    filesPixel, filesLitho, filesResist = filesLithoSim(basedir)
    numFiles = len(filesPixel)
    numTrain = round(numFiles * ratioTrain)
    numTest  = numFiles - numTrain
    trainPixel = filesPixel[:numTrain]
    trainLitho = filesLitho[:numTrain]
    trainResist = filesResist[:numTrain]
    testPixel = filesPixel[numTrain:]
    testLitho = filesLitho[numTrain:]
    testResist = filesResist[numTrain:]
    train = DataLitho(trainPixel, trainLitho, trainResist, crop=True, size=sizeImage, cache=cache)
    test = DataLitho(testPixel, testLitho, testResist, crop=False, size=sizeImage, cache=cache)
    print(f"Training set: {numTrain}, Test set: {numTest}")

    return train, test

def loadersILT(benchmark, image_size, batch_size, njobs, drop_last=False): 
    trainset, valset = maskopt(f"work/{benchmark}", sizeImage=image_size, ratioTrain=0.9, cache=False)
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=njobs, shuffle=True, drop_last=drop_last)
    val_loader = DataLoader(dataset=valset, batch_size=batch_size, num_workers=njobs, shuffle=False, drop_last=False)
    return train_loader, val_loader

def loadersLitho(benchmark, image_size, batch_size, njobs, drop_last=False): 
    trainset, valset = lithosim(f"work/{benchmark}", sizeImage=image_size, ratioTrain=0.9, cache=False)
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=njobs, shuffle=True, drop_last=drop_last)
    val_loader = DataLoader(dataset=valset, batch_size=batch_size, num_workers=njobs, shuffle=False, drop_last=False)
    return train_loader, val_loader

def loadersAllILT(benchmark, image_size, batch_size, njobs, drop_last=False): 
    basedir = f"work/{benchmark}"
    filesGLP, filesTarget, filesMask = filesMaskOpt(basedir)
    data = DataILT(filesTarget, filesMask, crop=False, size=image_size, cache=False)
    loader = DataLoader(dataset=data, batch_size=batch_size, num_workers=njobs, shuffle=False, drop_last=drop_last)
    return loader

def loadersAllLitho(benchmark, image_size, batch_size, njobs, drop_last=False): 
    basedir = f"work/{benchmark}"
    filesPixel, filesLitho, filesResist = filesLithoSim(basedir)
    data = DataLitho(filesPixel, filesLitho, filesResist, crop=False, size=image_size, cache=False)
    loader = DataLoader(dataset=data, batch_size=batch_size, num_workers=njobs, shuffle=False, drop_last=drop_last)
    return loader

if __name__ == "__main__": 
    infolder = "work/StdContactTest/resist"
    outfolder = "work/StdContactTest/printed"
    infiles = glob.glob(f"{infolder}/*.png")
    for infile in tqdm(infiles): 
        mat = cv2.imread(infile)
        mat[mat > 255/2.0] = 255
        mat[mat <= 255/2.0] = 0
        outfile = os.path.join(outfolder, os.path.basename(infile))
        cv2.imwrite(outfile, mat)
    
    exit(1)
    trainset, valset = maskopt("work/DataILT", sizeImage=(512, 512), ratioTrain=0.9)
    target, mask = valset[1023]
    cv2.imwrite("tmp/target.png", target*255)
    cv2.imwrite("tmp/mask.png", mask*255)

