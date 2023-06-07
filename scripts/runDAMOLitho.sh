#!/bin/bash

python3 lithobench/train.py -m DAMOLitho -s MetalSet -n 4 -b 4 -p True
# python3 lithobench/train.py -m DAMOLitho -s ViaSet -n 1 -b 4 -p True

# python3 lithobench/test.py -m DAMOLitho -s MetalSet -b 4 -g saved/MetalSet_DAMOLitho/netG.pth -d saved/MetalSet_DAMOLitho/netD.pth
# python3 lithobench/test.py -m DAMOLitho -s ViaSet -b 4 -g saved/ViaSet_DAMOLitho/netG.pth -d saved/ViaSet_DAMOLitho/netD.pth
# python3 lithobench/test.py -m DAMOLitho -s StdMetal -b 4 -g saved/MetalSet_DAMOLitho/netG.pth -d saved/MetalSet_DAMOLitho/netD.pth
# python3 lithobench/test.py -m DAMOLitho -s StdContact -n 8 -b 4 -g saved/ViaSet_DAMOLitho/netG.pth -d saved/ViaSet_DAMOLitho/netD.pth
