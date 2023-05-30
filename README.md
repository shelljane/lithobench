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

Please download the pre-trained models from

```
https://mycuhk-my.sharepoint.com/:u:/g/personal/1155186650_link_cuhk_edu_hk/EclDi3AoXlpKjWP4zUIf2uQBbiWR9YJGh3l9GSHponumhQ?e=F1K9xd
```

Put the lithomodels.tar.gz into the work/ directory and unzip it with: 

```
tar xvfz lithodata.tar.gz
```

### Pre-trained Models

Please download the pre-trained models from

```
https://mycuhk-my.sharepoint.com/:u:/g/personal/1155186650_link_cuhk_edu_hk/EZ54weC7YNdLqzxPAa--OpsBrcupd78KzRNICp2P0ggALQ?e=J9N4MJ
```

Put the lithomodels.tar.gz into the work/ directory and unzip it with: 

```
tar xvfz lithomodels.tar.gz
```

## Train and Test the Models

Please refer to scripts/runNeuralILT.sh

