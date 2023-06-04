#!/bin/bash

python3 lithobench/train.py -m GANOPC -s MetalSet -n 32 -b 64 -p True
# python3 lithobench/train.py -m GANOPC -s ViaSet -n 8 -b 64 -p True

# python3 lithobench/test.py -m GANOPC -s MetalSet -g saved/MetalSet_GANOPC/netG.pth -d saved/MetalSet_GANOPC/netD.pth --shots
# python3 lithobench/test.py -m GANOPC -s ViaSet -g saved/ViaSet_GANOPC/netG.pth -d saved/ViaSet_GANOPC/netD.pth --shots
# python3 lithobench/test.py -m GANOPC -s StdMetal -g saved/MetalSet_GANOPC/netG.pth -d saved/MetalSet_GANOPC/netD.pth --shots
# python3 lithobench/test.py -m GANOPC -s StdContact -n 16 -b 16 -g saved/ViaSet_GANOPC/netG.pth -d saved/ViaSet_GANOPC/netD.pth --shots