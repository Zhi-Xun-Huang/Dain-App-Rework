#!/usr/bin/env python3
import os
import torch

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++17']

#CUDA_HOME = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0"

major, minor = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (12, 0)
cap = f"{major}{minor}"
nvcc_args = ["-gencode", f"arch=compute_{cap},code=sm_{cap}"]

setup(
    name='correlation_cuda',
    ext_modules=[
        CUDAExtension('correlation_cuda', [
            'correlation_cuda.cc',
            'correlation_cuda_kernel.cu'
        ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
