# Speech to Speech Prototype

## Overview

A program that recieves audio, supported by Whisper and outputs english audio all done through local models. This program can optionally also use Google Translate in between.

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

### Install CUDNN DLL

For arm64 download here: https://developer.nvidia.com/cudnn-downloads?target_os=Windows&target_arch=arm64

For x86_64 download here: https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/

Download the CUDNN matching your CUDA version, on modern NVIDIA cards you can download the latest cudnn with cuda13 and put the dlls into your NVIDIA GPU Computing Toolkit\CUDA, usually located in Program Files on Windows. This library is needed for ONNX runtime that Kokoro uses.

## Launch

Activate local virtual environment and then run:

```
python main.py
```

Pay attention to warning messages and debugs to make sure you are using cuda instead of cpu for whisper and kokoro.

## Why CUDA

Our church's streaming computer has Nvidia GPU and CUDA fits out needs best. Most local AI models run on Nvidia GPUs in enterprise environments as well.

## Future Development

Whisper has the following languages available for translations and the ability to transcribe/translate audio without knowing the config for the language.

The spoken languages for input are all languages supported by OpenAI/Whisper:

| ISO Code | Language Name        |
|----------|---------------------|
| en       | English             |
| zh       | Chinese             |
| de       | German              |
| es       | Spanish             |
| ru       | Russian             |
| ko       | Korean              |
| fr       | French              |
| ja       | Japanese            |
| pt       | Portuguese          |
| tr       | Turkish             |
| pl       | Polish              |
| ca       | Catalan             |
| nl       | Dutch               |
| ar       | Arabic              |
| sv       | Swedish             |
| it       | Italian             |
| id       | Indonesian          |
| hi       | Hindi               |
| fi       | Finnish             |
| vi       | Vietnamese          |
| he       | Hebrew              |
| uk       | Ukrainian           |
| el       | Greek               |
| ms       | Malay               |
| cs       | Czech               |
| ro       | Romanian            |
| da       | Danish              |
| hu       | Hungarian           |
| ta       | Tamil               |
| no       | Norwegian           |
| th       | Thai                |
| ur       | Urdu                |
| hr       | Croatian            |
| bg       | Bulgarian           |
| lt       | Lithuanian          |
| la       | Latin               |
| mi       | Maori               |
| ml       | Malayalam           |
| cy       | Welsh               |
| sk       | Slovak              |
| te       | Telugu              |
| fa       | Persian             |
| lv       | Latvian             |
| bn       | Bengali             |
| sr       | Serbian             |
| az       | Azerbaijani         |
| sl       | Slovenian           |
| kn       | Kannada             |
| et       | Estonian            |
| mk       | Macedonian          |
| br       | Breton              |
| eu       | Basque              |
| is       | Icelandic           |
| hy       | Armenian            |
| ne       | Nepali              |
| mn       | Mongolian           |
| bs       | Bosnian             |
| kk       | Kazakh              |
| sq       | Albanian            |
| sw       | Swahili             |
| gl       | Galician            |
| mr       | Marathi             |
| pa       | Punjabi             |
| si       | Sinhala             |
| km       | Khmer               |
| sn       | Shona               |
| yo       | Yoruba              |
| so       | Somali              |
| af       | Afrikaans           |
| oc       | Occitan             |
| ka       | Georgian            |
| be       | Belarusian          |
| tg       | Tajik               |
| sd       | Sindhi              |
| gu       | Gujarati            |
| am       | Amharic             |
| yi       | Yiddish             |
| lo       | Lao                 |
| uz       | Uzbek               |
| fo       | Faroese             |
| ht       | Haitian Creole      |
| ps       | Pashto              |
| tk       | Turkmen             |
| nn       | Nynorsk             |
| mt       | Maltese             |
| sa       | Sanskrit            |
| lb       | Luxembourgish       |
| my       | Myanmar             |
| bo       | Tibetan             |
| tl       | Tagalog             |
| mg       | Malagasy            |
| as       | Assamese            |
| tt       | Tatar               |
| haw      | Hawaiian            |
| ln       | Lingala             |
| ha       | Hausa               |
| ba       | Bashkir             |
| jw       | Javanese            |
| su       | Sundanese           |
| yue      | Cantonese           |

Kokoro model has only 9 languages with multiple voices per language, and ability to blend voices to create a more clear and coherent voice:

https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md

This project can include more TTS open source models for a wider range of voices, however for now t