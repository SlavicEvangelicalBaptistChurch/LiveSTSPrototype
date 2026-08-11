import gc
import os
from dotenv import load_dotenv
import torch
import tomllib
from faster_whisper import WhisperModel
from huggingface_hub import login, try_to_load_from_cache


class WhisperSegment:
    def __init__(self, model_size, compute_type, language, translate_method):
        load_dotenv()
        HF_TOKEN = os.getenv("HF_TOKEN")

        # faster model download time
        if (HF_TOKEN):
            login(HF_TOKEN)

        with open("config.toml", "rb") as f:
            config = tomllib.load(f)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.language = language
        self.translate_method = translate_method

        if(self.language == "auto"):
            self.language = None

        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print(f'Using model size (SYSTRAN): {model_size} ({compute_type}) on {device}')

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
        task = "translate"
        if(self.translate_method != "whisper"):
            task = "transcribe"

        gen_segments, info = self.model.transcribe(
            segment,
            language=self.language,
            beam_size=8,
            word_timestamps=False,
            task=task,
            condition_on_previous_text=False
            ,repetition_penalty=1.2
            ,no_speech_threshold=0.4
            # ,log_prob_threshold=-0.5
            # ,compression_ratio_threshold=2.2
            ,vad_filter=True
        )

        segments = []
        for gen_segment in gen_segments:
            text = gen_segment.text.strip()
            if(text.lower() in self._banned_phrases):
                continue

            segments.append((gen_segment.start, text))

        return segments
