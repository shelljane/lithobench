#!/bin/bash

# python3 lithobench/train.py -m CFNOLitho -s MetalSet -n 8 -b 4 -p False
python3 lithobench/train.py -m CFNOLitho -s ViaSet -n 2 -b 4 -p False

# python3 lithobench/test.py -m CFNOLitho -s MetalSet -b 4 -l saved/MetalSet_CFNOLitho/net.pth
# python3 lithobench/test.py -m CFNOLitho -s ViaSet -b 4 -l saved/ViaSet_CFNOLitho/net.pth
# python3 lithobench/test.py -m CFNOLitho -s StdMetal -b 4 -l saved/MetalSet_CFNOLitho/net.pth
# python3 lithobench/test.py -m CFNOLitho -s StdContact -n 16 -b 4 -l saved/ViaSet_CFNOLitho/net.pth
