#!/bin/bash

python3 lithobench/train.py -m CFNOILT -s MetalSet -n 8 -b 4 -p False
# python3 lithobench/train.py -m CFNOILT -s ViaSet -n 2 -b 4 -p False

# python3 lithobench/test.py -m CFNOILT -s MetalSet -l saved/MetalSet_CFNOILT/net.pth --shots
# python3 lithobench/test.py -m CFNOILT -s ViaSet -l saved/ViaSet_CFNOILT/net.pth --shots
# python3 lithobench/test.py -m CFNOILT -s StdMetal -l saved/MetalSet_CFNOILT/net.pth --shots
# python3 lithobench/test.py -m CFNOILT -s StdContact -n 16 -b 4 -l saved/ViaSet_CFNOILT/net.pth --shots
