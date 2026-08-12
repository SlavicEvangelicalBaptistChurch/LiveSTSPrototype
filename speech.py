from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.onnx_backend import VoiceBlend

class Speech:
    _instance = None
    def __new__(cls, voice):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init(voice)
        return cls._instance

    def _init(self, voice):
        if("," in voice):
            self.voice = VoiceBlend.parse(voice)
        else:
            self.voice = voice

        self.pipe = KokoroPipeline(PipelineConfig(
                provider="cuda",
                voice=self.voice))
        self.pipe.run("Warming up.")

    def toVoice(self, text):
        res = self.pipe.run(text)
        return res.audio