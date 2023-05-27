#!/bin/bash

python3 lithobench/train.py -m LithoGAN -s MetalSet -n 16 -b 32 -p True
# python3 lithobench/train.py -m LithoGAN -s ViaSet -n 4 -b 32 -p True

# python3 lithobench/test.py -m LithoGAN -s MetalSet -b 32 -g saved/MetalSet_LithoGAN/netG.pth -d saved/MetalSet_LithoGAN/netD.pth
# python3 lithobench/test.py -m LithoGAN -s ViaSet -b 32 -g saved/ViaSet_LithoGAN/netG.pth -d saved/ViaSet_LithoGAN/netD.pth
# python3 lithobench/test.py -m LithoGAN -s StdMetal -b 32 -g saved/MetalSet_LithoGAN/netG.pth -d saved/MetalSet_LithoGAN/netD.pth
# python3 lithobench/test.py -m LithoGAN -s StdContact -n 0 -b 32 -g saved/ViaSet_LithoGAN/netG.pth -d saved/ViaSet_LithoGAN/netD.pth
