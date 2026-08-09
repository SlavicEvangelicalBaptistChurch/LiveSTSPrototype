# Russian Speech to English Speech Prototype program

## Overview

A program that recieves Russian audio and outputs english audio all done through local al models with exception of translation method, where it uses Google Translate

## Pipline

Audio to bit data with 24KHz sample rate -> Faster Whisper (large-v3) -> Translate text to English (via Google Translate) -> Pykokoro to Output with 24KHz sample rate

## Requirements

1. Nvidia GPU with minimum 5GB VRAM available (mostly for whisper model)
2. Internet connection for Google Translate
3. Mic Input
4. Audio Output

## Setup 

### Install in venv

Inside the directory run:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you cannot install cuda version of torch after the installation of all requirements run:

```
pip uninstall torch torchvision -y
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

### Launch

Activate local virtual environment and then run:

```
python main.py
```

Pay attention to warning messages and debugs to make sure you are using cuda instead of cpu for whisper and kokoro.

## Why CUDA

Our church's streaming computer has Nvidia GPU and CUDA fits out needs best. Most local AI models run on Nvidia GPUs in enterprise environments as well.
