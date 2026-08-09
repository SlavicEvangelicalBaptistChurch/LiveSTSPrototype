# import os
#
# cudnn_dir = r"D:\Development\SEBC\TTSTest\.venv\Lib\site-packages\nvidia\cudnn\bin"
# if os.path.isdir(cudnn_dir):
#     os.environ["PATH"] = cudnn_dir + os.pathsep + os.environ["PATH"]
# else:
#     print("cudnn bin folder not found at expected path — check actual location")

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.onnx_backend import VoiceBlend

class Speech:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.blend = VoiceBlend.parse("af_heart")
        self.pipe = KokoroPipeline(PipelineConfig(
                provider="cuda",
                voice=self.blend))
        self.pipe.run("Warming up.")

    def toVoice(self, text):
        res = self.pipe.run(text)
        return res.audio