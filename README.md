# DAIN-APP-Rework
This is the source code for the video interpolation application **Dain-App**, developed on top of the source code of **DAIN**
[Dain-APP](https://github.com/BurguerJohn/Dain-App) [Dain GIT Project](https://github.com/baowenbo/DAIN)

## DAIN-App Python 3.9 + CUDA 12.8 Setup

This guide prepares the patched DAIN-App source for Python 3.9, PyTorch CUDA 12.8, and NVIDIA Blackwell GPUs such as RTX PRO 6000 Blackwell.

## System Dependencies

Install the required system packages and CUDA Toolkit 12.8:

```bash
sudo apt update
sudo apt install libffi-dev libssl-dev openssl ffmpeg cuda-toolkit-12-8 -y
```

CUDA Toolkit is required when compiling the custom CUDA extensions. For inference-only machines, it is not required if the compiled extension modules are already installed and compatible with the Python/PyTorch environment.

## Build Python 3.9

Download and build Python 3.9 from source:

```bash
wget https://www.python.org/ftp/python/3.9.23/Python-3.9.23.tgz
tar -xf Python-3.9.23.tgz
cd Python-3.9.23
./configure --enable-optimizations
make -j"$(nproc)"
sudo make altinstall
```

Verify:

```bash
python3.9 --version
```

## Create Virtual Environment

From the DAIN-App project directory:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-py39-cu128.txt
```

## CUDA Environment

Export CUDA Toolkit 12.8 paths before compiling:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST=12.0
```

Verify:

```bash
which nvcc
nvcc --version
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
PY
```

Expected PyTorch target:

```text
torch==2.7.1+cu128
torchvision==0.22.1+cu128
torchaudio==2.7.1+cu128
```

## Compile DAIN-App Extensions

From the DAIN-App project directory:

```bash
./build-app.sh
```

This compiles and installs the patched custom CUDA extensions used by DAIN and PWCNet. Recompile if Python, PyTorch, CUDA Toolkit, GPU architecture, or source files change.

## Run Inference

Example command:

```bash
python my_design.py -cli \
  --input source.mp4 \
  -o output \
  -on output.mp4 \
  -m "model_weights/best.pth" \
  -fh 1 \
  --interpolations 2 \
  --check_scene_change 10 \
  --png_compress 0 \
  --crf 18 \
  --batch_size 1
```

Single-line version:

```bash
python my_design.py -cli --input source.mp4 -o output -on output.mp4 -m "model_weights/best.pth" -fh 1 --interpolations 2 --check_scene_change 10 --png_compress 0 --crf 18 --batch_size 1
```

## Runtime Notes

For inference after successful compilation, CUDA Toolkit and `nvcc` are not required. Runtime still requires:

- Python 3.9
- PyTorch CUDA 12.8 packages
- NVIDIA driver compatible with CUDA 12.8 and the target GPU
- ffmpeg
- DAIN model weights
- compiled DAIN custom extension modules compatible with the current Python and PyTorch environment

The compiled extensions are tied to the Python ABI, PyTorch ABI, Linux platform, and GPU architecture. The Blackwell patch targets `sm_120` through `TORCH_CUDA_ARCH_LIST=12.0`.

### License
See [MIT License](https://github.com/HeylonNHP/Dain-App/blob/master/LICENSE)
