#!/bin/bash

python3 lithobench/train.py -m NeuralILT -s MetalSet -n 8 -b 12 -p True
# python3 lithobench/train.py -m NeuralILT -s ViaSet -n 2 -b 12 -p True

# python3 lithobench/test.py -m NeuralILT -s MetalSet -l saved/MetalSet_NeuralILT/net.pth --shots
# python3 lithobench/test.py -m NeuralILT -s ViaSet -l saved/ViaSet_NeuralILT/net.pth --shots
# python3 lithobench/test.py -m NeuralILT -s StdMetal -l saved/MetalSet_NeuralILT/net.pth --shots
# python3 lithobench/test.py -m NeuralILT -s StdContact -n 16 -b 12 -l saved/ViaSet_NeuralILT/net.pth --shots
