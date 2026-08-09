import gc
import os
from dotenv import load_dotenv
import torch
import tomllib
import time
from faster_whisper import WhisperModel
from huggingface_hub import login, try_to_load_from_cache
from pathlib import Path

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
        
        # cached_file = try_to_load_from_cache(
        #     repo_id="bzikst/faster-whisper-large-v3-russian", 
        #     filename="config.json"
        # )
        # model_folder = Path(cached_file).parent
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
            language=None,
            beam_size=8,
            word_timestamps=False,
            task="translate",
            condition_on_previous_text=False,
            repetition_penalty=1.2,
            no_speech_threshold=0.7,      # Increase to ignore more quiet/ambient sounds (default is 0.6)
            log_prob_threshold=-0.5,      # Stricter confidence filter (default is -1.0; closer to 0 = stricter)
            compression_ratio_threshold=2.2, # Drops repetitive hallucinations common in quiet audio (default is 2.4)
            vad_filter=True               # Cuts out non-speech segments entirely prior to processing
        )

        #start = time.perf_counter()

        #print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

        segments = []
        for gen_segment in gen_segments:
            text = gen_segment.text.strip()
            cont = False
            if(text.lower() in self._banned_phrases):
                continue
            
            segments.append((gen_segment.start, text))

        return segments

        # for gen_segment in gen_segments:
        #     for word in gen_segment.words:
        #         print("[%.2f] %s" % (word.start, word.word.strip()), flush=True)

        # elapsed = time.perf_counter() - start
        # print(f"Elapsed: {elapsed:.2f}s ({elapsed * 1000:.0f}ms)")
        # print("written transcripts in autodetected language")
