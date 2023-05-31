#!/bin/bash

python3 lithobench/train.py -m dev/swin.py -a SwinILT -i 512 -t ILT -o dev -s MetalSet -n 4 -b 4 -p True

