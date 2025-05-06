# Feature_generation_for_SegAnyNeuron

## Hardware requirements

This code requires acceleration provided by NVIDIA GPU. The computer must have a NVIDIA GPU that supports CUDA version 11.6 or higher.

## Environmental requirements

* Operating system: windows

* CUDA >= 11.6, CUDNN >= 9.7

* Visual Studio 2017

  

  **Notes : If you don't want to compile the source code, the released executable program can be found** in [release](https://github.com/FateUBW0227/Feature_generation_for_SegAnyNeuron/releases/tag/release).

## Usage

Enter the following  command in command line.

```
generate_data.exe src_dir save_dir 0.4
```

Here src_dir represents the address of the input images, save_dir represents the address of produced feature maps, 0.4 is a default parameters. For example:

```
generate_data.exe ./fMost_dataset1/raw ./fMost_dataset1 0.4
```

**Notes : The address format must be the same as in the example above, both absolute or relative paths are acceptable.**