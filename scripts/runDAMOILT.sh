#!/bin/bash

python3 lithobench/train.py -m DAMOILT -s MetalSet -n 4 -b 4 -p True
# python3 lithobench/train.py -m DAMOILT -s ViaSet -n 2 -b 4 -p True

# python3 lithobench/test.py -m DAMOILT -s MetalSet -g saved/MetalSet_DAMOILT/netG.pth -d saved/MetalSet_DAMOILT/netD.pth --shots
# python3 lithobench/test.py -m DAMOILT -s ViaSet -g saved/ViaSet_DAMOILT/netG.pth -d saved/ViaSet_DAMOILT/netD.pth --shots
# python3 lithobench/test.py -m DAMOILT -s StdMetal -b 4 -g saved/MetalSet_DAMOILT/netG.pth -d saved/MetalSet_DAMOILT/netD.pth --shots
# python3 lithobench/test.py -m DAMOILT -s StdContact -n 8 -b 4 -g saved/ViaSet_DAMOILT/netG.pth -d saved/ViaSet_DAMOILT/netD.pth --shots
