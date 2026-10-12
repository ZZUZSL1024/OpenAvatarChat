"""CPU contract tests: fake only optional UI/GPU dependencies, never pipeline code.

Real ChatSession, RTC stream/delegate, TTS callbacks and all MuseTalk worker
functions run here. Inference/provider I/O are controlled by per-test fakes.
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))


def module(name, **attributes):
    mod = types.ModuleType(name)
    mod.__dict__.update(attributes)
    sys.modules[name] = mod
    return mod


class StreamHandler:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def set_channel(self, channel):
        pass


module('gradio', blocks=types.SimpleNamespace(Block=object))
module('fastapi', FastAPI=object)
module('fastrtc', AsyncAudioVideoStreamHandler=StreamHandler,
       AudioEmitType=object, VideoEmitType=object)
module('torch', cuda=types.SimpleNamespace(is_available=lambda: False), Tensor=type('Tensor', (), {}))
# Tests use identical input/output sample rates; DSP itself is outside the contract.
module('librosa', resample=lambda audio, **kwargs: audio)
module('soundfile')
