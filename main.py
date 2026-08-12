import queue
import threading
import tomllib

import numpy as np
import sounddevice as sd

from audio_segmenter import live_audio_segments

from google_translate import Translator
from whisper import WhisperSegment
from speech import Speech

def selectIODevices() -> tuple[int, int]:
    selected_input_device: int = 0
    selected_output_device: int = 0

    print("Available input devices:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            print(f"  [{i}] {dev['name']} (default_sr={dev['default_samplerate']})")

    selected_input_device = int(input("Select input device eg. 1: "))

    print("\nAvailable output devices:")
    for i, dev in enumerate(devices):
        if dev["max_output_channels"] > 0:
            print(f"  [{i}] {dev['name']} (default_sr={dev['default_samplerate']})")

    selected_output_device = int(input("Select output device eg. 1: "))



    input_name = sd.query_devices(selected_input_device)['name'] if selected_input_device is not None else 'system default'
    output_name = sd.query_devices(selected_output_device)['name'] if selected_output_device is not None else 'system default'

    print(f"\nInput device: {input_name}")
    print(f"Output device: {output_name}")

    print("Listening for speech segments... (Ctrl+C to stop)")

    return selected_input_device, selected_output_device

def _playback_worker(
    segment_queue: queue.Queue[np.ndarray | None],
    stop_event: threading.Event,
    output_device: int | None = None,
    output_sample_rate: int = 24000,
) -> None:
    try:
        with sd.OutputStream(samplerate=output_sample_rate, channels=1, dtype="float32", device=output_device) as out_stream:
            out_stream.start()
            while not stop_event.is_set():
                try:
                    segment = segment_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if segment is None:
                    break 

                chunk_size = 1024
                pos = 0
                while pos < len(segment) and not stop_event.is_set():
                    end = min(pos + chunk_size, len(segment))
                    out_stream.write(segment[pos:end])
                    pos = end
    except Exception:
        pass

sample_rate=24000

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

model_size = config['whisper']['size'] or "large-v3"
compute_type = config['whisper']['compute_type'] or "float16"
whisper_language = config['whisper']['language'] or "auto"
lang_target = config['translate']['target'] or "en"
translate_method = config['translate']['method'] or "google"

synth_mode = config["synthesize"]["method"] or "chunk"
voice = config["synthesize"]["kokoro_voice"] or "af_heart"

if(translate_method == "whisper" and lang_target != "en"):
    translate_method = "google"

try:
    whisperSegment = WhisperSegment(model_size, compute_type, whisper_language, translate_method)
    speech = Speech(voice)
    translator = Translator(whisper_language, lang_target)

    input_device, output_device = selectIODevices()

    segment_queue: queue.Queue[np.ndarray | None] = queue.Queue()
    stop_playback = threading.Event()

    playback_thread = threading.Thread(
        target=_playback_worker,
        args=(segment_queue, stop_playback, output_device, sample_rate),
        daemon=True,
    )
    playback_thread.start()

    for segment in live_audio_segments(device=input_device, sample_rate=sample_rate):
        segments = whisperSegment.transcribe(segment)
        if(len(segments) == 0):
            continue

        translated_list = []
        final_text = ""
        if(translate_method == "google"):
            for seg in segments:
                translated_list.append(seg[1])
            final_text = " ".join(translated_list)
            final_text = translator.translate(final_text)
        elif(translate_method == "google-byline"):
            for seg in segments:
                translated = translator.translate(seg[1])
                translated_list.append(translated)
        elif(translate_method == "whisper"):
            for seg in segments:
                translated_list.append(seg[1])

        if(len(final_text) > 0):
            print("[%.2f] %s" % (segments[0][0], final_text), flush=True)
            audio_segment = speech.toVoice(final_text)
            segment_queue.put(audio_segment)
        elif(synth_mode == "chunk"):
            final_text = " ".join(translated_list)
            print("[%.2f] %s" % (segments[0][0], final_text), flush=True)
            audio_segment = speech.toVoice(final_text)
            segment_queue.put(audio_segment)
        elif(synth_mode == "byline"):
            for line in translated_list:
                print("[%.2f] %s" % (segments[0][0], line), flush=True)
                audio_segment = speech.toVoice(line)
                segment_queue.put(audio_segment)
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    segment_queue.put(None)
    stop_playback.set()
    playback_thread.join(timeout=2.0)
    print("Stopped.")
