```text
    __     _    __     __             ____                          __  
   / /    (_)  / /_   / /_   ____    / __ )  ___    ____   _____   / /_ 
  / /    / /  / __/  / __ \ / __ \  / __  | / _ \  / __ \ / ___/  / __ \
 / /___ / /  / /_   / / / // /_/ / / /_/ / /  __/ / / / // /__   / / / /
/_____//_/   \__/  /_/ /_/ \____/ /_____/  \___/ /_/ /_/ \___/  /_/ /_/ 
                                                                        
```

# LithoBench: Benchmarking AI Computational Lithography for Semiconductor Manufacturing 

## Installation

### Install Basic Dependencies

If you manage your python environments with anaconda, you can create a new environment with
```bash
conda create -n lithobench python=3.8
conda activate lithobench
```
To install the dependencies with pip, you can use
```bash
pip3 install -r requirements_pip.txt
```

You may install the dependencies with conda:
```bash
conda install --file requirements_conda.txt -c pytorch -c conda-forge
```
However, due to the complex environment solving, the process may be slow and the installed packages may be unsatisfactory. 
For example, you may get a CPU version of pytorch. 
Thus, if you want to use conda, you may install a GPU version of pytorch before you install other dependencies. 

Note that we develop LithoBench with python 3.8 and pytorch 1.10. 
We also tested LithoBench with pytorch 2.0. 
The system we use is Ubuntu 18 with Intel Xeon CPUs and NVIDIA GPUs. We also tested the program on CentOS 7. 

### Install adaptive-boxes

The python package *adaptive-boxes* is needed for shot counting. 
You can install the package in the *thirdparty/adaptive-boxes* folder. 
```
cd thirdparty/adaptive-boxes
pip3 install -e .
```

### Run ILT algorithms

You can test the ILT method in LithoBench with the following commands: 

*CurvMulti*
```
CUDA_VISIBLE_DEVICES=0 python3 pyilt/curvmulti.py
```

## Download the dataset

### LithoBench Data

Please download the data from

```
https://drive.google.com/file/d/1MzYiRRxi8Eu2L6WHCfZ1DtRnjVNOl4vu/view?usp=sharing
```

Put the lithomodels.tar.gz into the work/ directory and unzip it with: 

```
tar xvfz lithodata.tar.gz
```

### Pre-trained Models

Please download the pre-trained models from

```
https://drive.google.com/file/d/1N-VCv0gX49zzVWlwSs0yDqq2zKNQHKNB/view?usp=sharing
```

Put the lithomodels.tar.gz into the work/ directory and unzip it with: 

```
tar xvfz lithomodels.tar.gz
```


## Train and Test the Models

Please refer to scripts/runNeuralILT.sh. 

To train a model on MetalSet: 

```
python3 lithobench/train.py -m NeuralILT -s MetalSet -n 8 -b 12 -p True
```

>* "-m NeuralILT" specifies the NeuralILT model to train. 
>* "-s MetalSet" means training on MetalSet.
>* "-n" and "-b" decide the number of epochs and the batch size. 
>* "-p True" indicates that it needs pre-training. 
>* Replacing "MetalSet" with "ViaSet" can train the model on ViaSet. 


To evaluate the model on MetalSet: 

```
python3 lithobench/test.py -m NeuralILT -s MetalSet -l saved/MetalSet_NeuralILT/net.pth
```

>* By default, the trained model will be saved in saved/\<training set\>_\<model name\>/.
>* Note that when evaluting the model on StdMetal, the trained model saved/MetalSet_NeuralILT/net.pth should also be used.
>* Replacing "MetalSet" with "ViaSet" can evaluate the model on ViaSet. 
>* Note that when evaluting the model on StdContact, the trained model saved/ViaSet_NeuralILT/net.pth should also be used.


## Train and Test a New Model

Please refer to dev/swin.py and dev/run.sh. 

To train the new model "dev/swin.py" on MetalSet: 

```
python3 lithobench/train.py -m dev/swin.py -a SwinILT -i 512 -t ILT -o dev -s MetalSet -n 4 -b 4 -p True
```

>* "-m dev/swin.py" specifies path of the model. 
>* The model should inherit the "lithobench.model.ModelILT" class
>* "-a SwinILT" indicated the alias and also the class name of the model
>* "-o dev" specifies the output directory of the training process

