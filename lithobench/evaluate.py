import os 
import sys
sys.path.append(".")
import math
import time
import json
import glob
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
import pycommon.utils as common
import pycommon.glp as glp
# import pylitho.simple as litho
import pylitho.exact as litho

import pyilt.simpleilt as simpleilt
import pyilt.curvmulti as curvmulti
import pyilt.initializer as initializer
import pyilt.evaluation as evaluation


def getTargets(samples=10, dataset="MetalSet"): 
    results = []
    if dataset == "MetalSet": 
        if samples is None: 
            samples = 10
        for idx in range(1, 1+samples): 
            design = glp.Design(f"./benchmark/ICCAD2013/M1_test{idx}.glp", down=1)
            design.center(2048, 2048, 384, 384)
            target = design.mat(2048, 2048, 384, 384)
            results.append(target)
    elif dataset == "ViaSet": 
        filenames = glob.glob(f"./benchmark/OpenROAD/*.glp")
        if samples is None: 
            samples = 10
        for idx in range(1, 1+samples): 
            design = glp.Design(filenames[idx-1], down=1)
            target = design.mat(2048, 2048, 384, 384)
            results.append(target)
    elif dataset == "StdMetal": 
        filenames = glob.glob(f"./benchmark/StdMetal/*.glp")
        if samples is None: 
            samples = len(filenames)
        for idx in range(1, 1+samples): 
            design = glp.Design(filenames[idx-1], down=1)
            design.center(2048, 2048, 384, 384)
            target = design.mat(2048, 2048, 384, 384)
            results.append(target)
    elif dataset == "StdContact": 
        filenames = glob.glob(f"./benchmark/StdContact/*.glp")
        if samples is None: 
            samples = len(filenames)
        for idx in range(1, 1+samples): 
            design = glp.Design(filenames[idx-1], down=1)
            target = design.mat(2048, 2048, 384, 384)
            results.append(target)
    return results


def evalRaw(masksInit, targets): 
    sim = litho.LithoSim("./config/lithosimple.txt")
    test = evaluation.Basic(sim, 0.5)
    epeCheck = evaluation.EPEChecker(sim, 0.5)
    l2s = []
    pvbs = []
    epes = []
    for idx in range(len(targets)): 
        mask0 = masksInit[idx]
        mask0 = F.interpolate(mask0[None, None, :, :], size=(2048, 2048))[0, 0]
        mask0[mask0 > 0.5] = 1.0
        mask0[mask0 <= 0.5] = 0.0
        l20, pvb0 = test.run(mask0, targets[idx], scale=1)
        epeIn0, epeOut0 = epeCheck.run(mask0, targets[idx], scale=1)
        epe0 = epeIn0 + epeOut0
        l2s.append(l20)
        pvbs.append(pvb0)
        epes.append(epe0)
    return l2s, pvbs, epes

def shots(masks, targets=None): 
    shotCount = evaluation.ShotCounter(None, 0.5)
    shots = []
    for idx in range(len(masks)): 
        mask = masks[idx]
        mask = F.interpolate(mask[None, None, :, :], size=(2048, 2048))[0, 0]
        mask[mask > 0.5] = 1.0
        mask[mask <= 0.5] = 0.0
        shot = shotCount.run(mask, shape=(256, 256))
        shots.append(shot)
    return shots


def finetuneSimple(paramsInit, masksInit, targets): 
    cfg = simpleilt.SimpleCfg("./config/simpleilt.txt")
    sim = litho.LithoSim("./config/lithosimple.txt")
    solver = simpleilt.SimpleILT(cfg, sim, multigpu=False)
    test = evaluation.Basic(sim, 0.5)
    epeCheck = evaluation.EPEChecker(sim, 0.5)
    l2s = []
    pvbs = []
    epes = []
    for idx in range(len(targets)): 
        params0 = paramsInit[idx]
        mask0 = masksInit[idx]
        target = torch.tensor(targets[idx], dtype=REALTYPE, device=DEVICE)
        params0 = F.interpolate(params0[None, None, :, :], size=(2048, 2048))[0, 0]
        mask0 = F.interpolate(mask0[None, None, :, :], size=(2048, 2048))[0, 0]
        mask0[mask0 > 0.5] = 1.0
        mask0[mask0 <= 0.5] = 0.0
        l2, pvb, bestParams, bestMask = solver.solve(target, params0)
        bestMask[bestMask > 0.5] = 1.0
        bestMask[bestMask <= 0.5] = 0.0
        l2, pvb = test.run(bestMask, target, scale=1)
        epeIn, epeOut = epeCheck.run(bestMask, target, scale=1)
        epe = epeIn + epeOut
        l2s.append(l2)
        pvbs.append(pvb)
        epes.append(epe)
    return l2s, pvbs, epes


def finetuneFast(paramsInit, masksInit, targets): 
    cfgMid = curvmulti.CurvILTCfg("./config/curvilt512.txt")
    cfgHigh = curvmulti.CurvILTCfg("./config/curvilt1024.txt")
    cfgMid._config["Iterations"] = 200
    cfgHigh._config["Iterations"] = 25
    sim = litho.LithoSim("./config/lithosimple.txt")
    solverMid = curvmulti.CurvILT(cfgMid, sim, multigpu=False)
    solverHigh = curvmulti.CurvILT(cfgHigh, sim, multigpu=False)
    test = evaluation.Basic(sim, 0.5)
    epeCheck = evaluation.EPEChecker(sim, 0.5)
    l2s = []
    pvbs = []
    epes = []
    masks = []
    for idx in range(len(targets)): 
        params0 = paramsInit[idx]
        mask0 = masksInit[idx]
        target = torch.tensor(targets[idx], dtype=REALTYPE, device=DEVICE)

        params0Mid = F.interpolate(mask0[None, None, :, :], size=(512, 512))[0, 0]
        targetMid = F.interpolate(target[None, None, :, :], size=(512, 512))[0, 0]
        l2, pvb, bestParams, bestMask = solverMid.solve(targetMid, params0Mid)

        params0High = F.interpolate(bestParams[None, None, :, :], size=(1024, 1024))[0, 0]
        targetHigh = F.interpolate(target[None, None, :, :], size=(1024, 1024))[0, 0]
        l2, pvb, bestParams, bestMask = solverHigh.solve(targetHigh, params0High)
        bestMask[bestMask > 0.5] = 1.0
        bestMask[bestMask <= 0.5] = 0.0

        l2, pvb = test.run(bestMask, target, scale=2)
        epeIn, epeOut = epeCheck.run(bestMask, target, scale=2)
        epe = epeIn + epeOut
        l2s.append(l2)
        pvbs.append(pvb)
        epes.append(epe)
        masks.append(bestMask)
    return l2s, pvbs, epes, masks


def finetune1024(paramsInit, masksInit, targets): 
    cfgHigh = curvmulti.CurvILTCfg("./config/curvilt1024.txt")
    cfgHigh._config["StepSize"] = 1.0
    cfgHigh._config["Iterations"] = 150
    sim = litho.LithoSim("./config/lithosimple.txt")
    solverHigh = curvmulti.CurvILT(cfgHigh, sim, multigpu=False)
    test = evaluation.Basic(sim, 0.5)
    epeCheck = evaluation.EPEChecker(sim, 0.5)
    l2s = []
    pvbs = []
    epes = []
    masks = []
    for idx in range(len(targets)): 
        params0 = paramsInit[idx]
        mask0 = masksInit[idx]
        target = torch.tensor(targets[idx], dtype=REALTYPE, device=DEVICE)

        params0High = F.interpolate(mask0[None, None, :, :], size=(1024, 1024))[0, 0]
        targetHigh = F.interpolate(target[None, None, :, :], size=(1024, 1024))[0, 0]
        l2, pvb, bestParams, bestMask = solverHigh.solve(targetHigh, params0High)
        bestMask[bestMask > 0.5] = 1.0
        bestMask[bestMask <= 0.5] = 0.0

        l2, pvb = test.run(bestMask, target, scale=2)
        epeIn, epeOut = epeCheck.run(bestMask, target, scale=2)
        epe = epeIn + epeOut
        l2s.append(l2)
        pvbs.append(pvb)
        epes.append(epe)
        masks.append(bestMask)
    return l2s, pvbs, epes, masks


def finetune(paramsInit, masksInit, targets): 
    cfgLow = curvmulti.CurvILTCfg("./config/curvilt256.txt")
    cfgMid = curvmulti.CurvILTCfg("./config/curvilt512.txt")
    cfgHigh = curvmulti.CurvILTCfg("./config/curvilt1024.txt")
    sim = litho.LithoSim("./config/lithosimple.txt")
    solverLow = curvmulti.CurvILT(cfgLow, sim, multigpu=False)
    solverMid = curvmulti.CurvILT(cfgMid, sim, multigpu=False)
    solverHigh = curvmulti.CurvILT(cfgHigh, sim, multigpu=False)
    test = evaluation.Basic(sim, 0.5)
    epeCheck = evaluation.EPEChecker(sim, 0.5)
    l2s = []
    pvbs = []
    epes = []
    masks = []
    for idx in range(len(targets)): 
        params0 = paramsInit[idx]
        mask0 = masksInit[idx]
        target = torch.tensor(targets[idx], dtype=REALTYPE, device=DEVICE)

        params0Low = F.interpolate(mask0[None, None, :, :], size=(256, 256))[0, 0]
        targetLow = F.interpolate(target[None, None, :, :], size=(256, 256))[0, 0]
        l2, pvb, bestParams, bestMask = solverLow.solve(targetLow, params0Low)

        params0Mid = F.interpolate(bestParams[None, None, :, :], size=(512, 512))[0, 0]
        targetMid = F.interpolate(target[None, None, :, :], size=(512, 512))[0, 0]
        l2, pvb, bestParams, bestMask = solverMid.solve(targetMid, params0Mid)

        params0High = F.interpolate(bestParams[None, None, :, :], size=(1024, 1024))[0, 0]
        targetHigh = F.interpolate(target[None, None, :, :], size=(1024, 1024))[0, 0]
        l2, pvb, bestParams, bestMask = solverHigh.solve(targetHigh, params0High)
        bestMask[bestMask > 0.5] = 1.0
        bestMask[bestMask <= 0.5] = 0.0

        l2, pvb = test.run(bestMask, target, scale=2)
        epeIn, epeOut = epeCheck.run(bestMask, target, scale=2)
        epe = epeIn + epeOut
        l2s.append(l2)
        pvbs.append(pvb)
        epes.append(epe)
        masks.append(bestMask)
    return l2s, pvbs, epes, masks

if __name__ == "__main__": 
    infile = sys.argv[1]
    reffile = sys.argv[2]

    mask = cv2.imread(infile)
    if len(mask.shape) > 2: 
        mask = np.mean(mask, axis=-1)
    mask = torch.tensor(mask, dtype=REALTYPE, device=DEVICE)
    mask = F.interpolate(mask[None, None, :, :], size=(2048, 2048))[0, 0]

    if reffile[-4:] == ".png": 
        target = cv2.imread(reffile)
        if len(target.shape) > 2: 
            target = np.mean(target, axis=-1)
    else: 
        design = glp.Design(reffile, down=1)
        design.center(2048, 2048, 384, 384)
        target = design.mat(2048, 2048, 384, 384)
    target = torch.tensor(target, dtype=REALTYPE, device=DEVICE)
    target = F.interpolate(target[None, None, :, :], size=(2048, 2048))[0, 0]

    sim = litho.LithoSim("./config/lithosimple.txt")
    test = evaluation.Basic(sim, 0.5)
    epeCheck = evaluation.EPEChecker(sim, 0.5)
    shotCount = evaluation.ShotCounter(None, 0.5)

    l2, pvb = test.run(mask, target, scale=1)
    epeIn, epeOut = epeCheck.run(mask, target, scale=1)
    epe = epeIn + epeOut
    shot = shotCount.run(mask, shape=(256, 256))

    print(f"[{infile}]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shots {shot:.0f}")
