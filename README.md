# Speech to Speech Prototype (CUDA)

## Overview

A program that recieves audio, supported by Whisper and outputs english audio all done through local models. This program can optionally also use Google Translate in between. The tutorial for installing and launching this app is written for Windows with NVIDIA but it's possible to launch on MacOS and Linux, with different instructions, that are not written here.

## Pipline

Audio to bit data with 24KHz sample rate -> Faster Whisper (large-v3) -> Translate text to English (via Google Translate) -> Pykokoro to Output with 24KHz sample rate

## Requirements

1. Nvidia GPU with minimum 5GB VRAM available (mostly for whisper model)
2. Internet connection for Google Translate
3. Mic Input
4. Audio Output

## Setup 

### Install in venv

For Cuda to work with the models, faster-whisper requires Nvidia libraries for CUDA 12, while ONNX runtime relies on cudNN and can be used with libraries for CUDA 13.

Inside the directory run:

```
python -m venv .venv
.venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
pip install -r requirements.txt
```

*best to install torch and torchvision separately first*

### Install CUDNN DLL

For arm64 download here: https://developer.nvidia.com/cudnn-downloads?target_os=Windows&target_arch=arm64

For x86_64 download here: https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/

Download the CUDNN 9.X matching your CUDA version, on modern NVIDIA cards you can download the latest cudnn with cuda13 and put the dlls into your NVIDIA GPU Computing Toolkit\CUDA, usually located in Program Files on Windows. This way you can use cudnn globally for multiple projects. This library is needed for ONNX runtime that Kokoro uses.

## Launch

Activate local virtual environment and then run:

```
python main.py
```

### First Launch

You will wait for a while for the Whisper model available in CTranslate2 and ONNX Runtime to download locally. The models are stored in %USERPROFILE%\.cache\huggingface\hub

Pay attention to warning messages and debugs to make sure you are using cuda instead of cpu for whisper and kokoro.

If you have errors regarding onnxruntime-gpu and cudnn consider downgrading to 1.20.2

## Understanding config.toml

config.toml provides all configuration needed for this application. Make config.toml file, copying contents of config-example.toml

### [whisper]

###### compute_type: float32, float16, int8, int8_float16

Changes precision types during calculations, which impacts speed and memory of whisper.

###### language: auto, {ISOCode supported by whisper}

Declares input language for whisper

###### size: large-v3, large-v3-turbo, etc. all available whisper models in ctranslate2 interface 

Sets the type of whisper model

*If you have your own whisper model you can convert it for ctranslate2 and replace the model name with local pathname. It's best done with try_to_load_from_cache hf function but this feature hasn't been implemented yet*

### [translate]

###### method: google, google-byline, whisper

The method of translating input. It can be done through whisper itself, or rely on external google translation server. Whisper cannot do byline, because it already returns a collection of text chunks anyway, while for google you can either pass in all whisper chunks and translate at once or translate line by line, changing the text translation with less context for each line.

###### target: en, es, ja, zh, fr, hi, it, pt

These are the languages you should only stick with as kokoro has voice models for only these. Refer to kokoro voices for selecting available languages.

https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md

### [synthesize]

###### method: chunk, byline

This method passes full text of audio segment or passes individual whisper lines to kokoro to generate audio from.

###### kokoro_voice: af_heart, am_michael, ef_dora ...

This is a config to select the voice model to produce audio for output. Look at Voices of kokoro to match the model with the target language. If your target language is "en" then you must only use kokoro voice models designated for english speech generation, for example. The program did not map target languages to the models so that users can choose preferred voice models as well as blend multiple models together. You can also blend the voices to create unique combinations. I highly recommend to use ```kokoro_voice="am_fenrir:30,am_michael:70"``` for a natural male voice.

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

This project can include more TTS models for a wider range of target languages. For now the output languages supported are the ones Kokoro provides.

## Why CUDA

Most consumer computers have an Nvidia GPU. Most local AI models run on Nvidia GPUs in enterprise environments as well.
