import gc
import os
from dotenv import load_dotenv
import torch
import tomllib
import time
from faster_whisper import WhisperModel
from huggingface_hub import login

class WhisperSegment:
    def __init__(self):
        load_dotenv()
        HF_TOKEN = os.getenv("HF_TOKEN")

        # faster model download time
        if (HF_TOKEN):
            login(HF_TOKEN)

        with open("config.toml", "rb") as f:
            config = tomllib.load(f)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(device)

        model_size = config['whisper']['size'] or "large-v3"
        compute_type = config['whisper']['compute_type'] or "float16"
        print("Using model size (SYSTRAN):", model_size)

        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

        # Load banned phrases (common whisper hallucinations / unwanted transcriptions)
        self._banned_phrases: set[str] = set()
        banned_path = "whisper_banned_phrases_ru.txt"
        if os.path.isfile(banned_path):
            with open(banned_path, encoding="utf-8") as bf:
                for line in bf:
                    stripped = line.strip()
                    if stripped:
                        self._banned_phrases.add(stripped)
        print(f"Loaded {len(self._banned_phrases)} banned phrase(s)")

    def unload(self):
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def transcribe(self, segment):
        gen_segments, info = self.model.transcribe(
            segment,
            language="ru",
            beam_size=5,
            word_timestamps=False,
            task="transcribe",
            condition_on_previous_text=False,
            repetition_penalty=1.00
        )

        #start = time.perf_counter()

        #print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

        segments = []
        for gen_segment in gen_segments:
            text = gen_segment.text.strip()
            if text in self._banned_phrases:
                continue
            segments.append((gen_segment.start, text))

        return segments

        # for gen_segment in gen_segments:
        #     for word in gen_segment.words:
        #         print("[%.2f] %s" % (word.start, word.word.strip()), flush=True)

        elapsed = time.perf_counter() - start
        print(f"Elapsed: {elapsed:.2f}s ({elapsed * 1000:.0f}ms)")
        print("written transcripts in autodetected language")
