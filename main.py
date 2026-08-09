import queue
import threading

import numpy as np
import sounddevice as sd

from audio_segmenter import live_audio_segments

from translate import Translator
from whisper import WhisperSegment
from speech import Speech

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

sample_rate = 24000

input_name = sd.query_devices(selected_input_device)['name'] if selected_input_device is not None else 'system default'
output_name = sd.query_devices(selected_output_device)['name'] if selected_output_device is not None else 'system default'

print(f"\nInput device: {input_name}")
print(f"Output device: {output_name}")

print("Listening for speech segments... (Ctrl+C to stop)")

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


try:
    speech = Speech()
    whisperSegment = WhisperSegment()
    translator = Translator()

    segment_queue: queue.Queue[np.ndarray | None] = queue.Queue()
    stop_playback = threading.Event()

    playback_thread = threading.Thread(
        target=_playback_worker,
        args=(segment_queue, stop_playback, selected_output_device, sample_rate),
        daemon=True,
    )
    playback_thread.start()

    for segment in live_audio_segments(device=selected_input_device, sample_rate=sample_rate):
        # print(f"Got speech segment: {len(segment)} samples, {len(segment) / sample_rate:.2f}s")

        segments = whisperSegment.transcribe(segment)
        if(len(segments) == 0):
            continue

        for seg in segments:
            #translated = translator.translate(comb, lang_dest="en")
            print("[%.2f] %s" % (segments[0][0], seg[1]), flush=True)
            audio_segment = speech.toVoice(seg[1].strip())
            segment_queue.put(audio_segment)
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    segment_queue.put(None)  # sentinel to stop playback worker
    stop_playback.set()
    playback_thread.join(timeout=2.0)
    print("Stopped.")
