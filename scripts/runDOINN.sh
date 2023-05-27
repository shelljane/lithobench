#!/bin/bash

# python3 lithobench/train.py -m DOINN -s MetalSet -n 32 -b 16 -p False
python3 lithobench/train.py -m DOINN -s ViaSet -n 8 -b 16 -p False

# python3 lithobench/test.py -m DOINN -s MetalSet -b 16 -l saved/MetalSet_DOINN/net.pth
# python3 lithobench/test.py -m DOINN -s ViaSet -b 16 -l saved/ViaSet_DOINN/net.pth
# python3 lithobench/test.py -m DOINN -s StdMetal -b 16 -l saved/MetalSet_DOINN/net.pth
# python3 lithobench/test.py -m DOINN -s StdContact -n 16 -b 16 -l saved/ViaSet_DOINN/net.pth
