import queue
import threading
from types import SimpleNamespace

import av
import numpy as np
import pytest

from chat_engine.contexts.session_context import SharedStates
from chat_engine.data_models.chat_data_type import ChatDataType as Kind
from handlers.avatar.liteavatar.model.algo_model import AudioResult, VideoResult, AvatarStatus
from handlers.avatar.liteavatar.model.audio_input import SpeechAudio
from handlers.avatar.musetalk.avatar_handler_musetalk import HandlerAvatarMusetalk
from handlers.avatar.musetalk.avatar_musetalk_config import AvatarMuseTalkConfig
from handlers.avatar.musetalk.avatar_musetalk_processor import AvatarMuseTalkProcessor
from test_pipeline import session, sink, register, definition, interrupt, wait_until


class FakeAvatar:
    """Deterministic inference with a barrier inside a selected expensive stage."""
    def __init__(self, stage=None):
        self.stage = stage
        self.entered = threading.Event()
        self.release = threading.Event()
        self.blocked = False

    def barrier(self, stage):
        if stage == self.stage and not self.blocked:
            self.blocked = True
            self.entered.set()
            assert self.release.wait(3), 'test did not release inference'

    def extract_whisper_feature(self, audio, rate):
        self.barrier('feature')
        return np.full((25, 1, 1), audio[0], np.float32)

    def generate_frames_unet(self, chunks, start, count):
        self.barrier('unet')
        return chunks, list(range(start, start + count))

    def generate_frames_vae(self, latents, indices, count):
        self.barrier('vae')
        return [(np.full((2, 2, 3), int(latents[i].flat[0]), np.uint8), indices[i]) for i in range(count)]

    def generate_frames(self, chunks, start, count):
        self.barrier('combined')
        return [(np.full((2, 2, 3), int(chunks[i].flat[0]), np.uint8), start + i) for i in range(count)]

    def res2combined(self, image, index):
        self.barrier('compose')
        return image

    def generate_idle_frame(self, index):
        return np.zeros((2, 2, 3), np.uint8)


def processor(avatar=None, split=True, states=None):
    config = AvatarMuseTalkConfig(batch_size=2, fps=25, output_audio_sample_rate=16000,
                                 algo_audio_sample_rate=16000, multi_thread_inference=split)
    p = AvatarMuseTalkProcessor(avatar or FakeAvatar(), config, states or SharedStates())
    p.audio_output_queue = queue.Queue()
    p.video_output_queue = queue.Queue()
    p.event_out_queue = queue.Queue()
    return p


def speech(value=1, end=True, size=1280):
    return SpeechAudio(speech_id='reused', audio_data=np.full(size, value, np.float32).tobytes(),
                       sample_rate=16000, end_of_speech=end)


@pytest.mark.parametrize('split,stage', [(True, 'feature'), (True, 'unet'), (True, 'vae'),
                                        (True, 'compose'), (False, 'combined'), (False, 'compose')])
def test_interrupt_during_each_inference_stage_drops_old_audio_video_and_end(split, stage):
    avatar = FakeAvatar(stage)
    p = processor(avatar, split)
    p._states.interruption_enabled = True
    p.start()
    try:
        p.add_audio(speech(11), 0)
        assert avatar.entered.wait(3)
        with p._states.lock:
            p._states.generation = 1
            p.interrupt()
        p.add_audio(speech(22), 1)
        avatar.release.set()
        generation, audio = p.audio_output_queue.get(timeout=3)
        assert generation == 1 and np.all(audio == 22)
        wait_until(lambda: not p.event_out_queue.empty())
        assert p.event_out_queue.get_nowait()[0] == 1
        while not p.video_output_queue.empty():
            epoch, video = p.video_output_queue.get_nowait()
            assert epoch == 1 and not np.any(video == 11)
        while not p.audio_output_queue.empty():
            epoch, audio = p.audio_output_queue.get_nowait()
            assert epoch == 1 and np.all(audio == 22)
    finally:
        avatar.release.set()
        p.stop()
        for name in ('_feature_thread', '_frame_gen_thread', '_frame_gen_unet_thread',
                     '_frame_gen_vae_thread', '_compose_thread', '_frame_collect_thread'):
            thread = getattr(p, name)
            assert thread is None or not thread.is_alive()


@pytest.mark.parametrize('split', [True, False])
def test_partial_batch_is_not_mixed_with_new_generation(split):
    p = processor(split=split)
    p.start()
    try:
        p._whisper_queue.put({'generation': 0, 'whisper_chunks': np.ones((1, 1, 1), np.float32),
                              'audio_data': np.full(640, 11, np.float32),
                              'speech_id': 'same', 'end_of_speech': False})
        wait_until(p._whisper_queue.empty)
        with p._states.lock:
            p._states.generation += 1
            p.interrupt()
        p._whisper_queue.put({'generation': 1, 'whisper_chunks': np.full((1, 1, 1), 22, np.float32),
                              'audio_data': np.full(640, 22, np.float32),
                              'speech_id': 'same', 'end_of_speech': True})
        epoch, audio = p.audio_output_queue.get(timeout=3)
        assert epoch == 1 and np.all(audio == 22)
        wait_until(lambda: not p.event_out_queue.empty())
        assert p.audio_output_queue.empty()
    finally:
        p.stop()


def test_interrupt_clears_all_processor_queues_and_late_notifications():
    p = processor()
    queues = [p._audio_queue, p._whisper_queue, p._unet_queue, p._frame_queue,
              p._frame_id_queue, p._compose_queue, p._output_queue,
              p.audio_output_queue, p.video_output_queue, p.event_out_queue]
    for q in queues:
        q.put(object())
    p._states.generation = 1
    p.interrupt()
    assert all(q.empty() for q in queues)
    af = av.AudioFrame.from_ndarray(np.ones((1, 8), np.float32), format='flt', layout='mono')
    vf = av.VideoFrame.from_ndarray(np.zeros((2, 2, 3), np.uint8), format='bgr24')
    p._notify_audio(AudioResult(audio_frame=af, speech_id='same'), 0)
    p._notify_video(VideoResult(video_frame=vf, speech_id='same', avatar_status=AvatarStatus.SPEAKING,
                               end_of_speech=True, bg_frame_id=-1), 0)
    p._notify_status_change('same', AvatarStatus.LISTENING, 0)
    assert all(q.empty() for q in queues)
    p._notify_audio(AudioResult(audio_frame=af, speech_id='same'), 1)
    assert p.audio_output_queue.get_nowait()[0] == 1


def test_contexts_have_independent_processors_slicers_queues_and_lifecycle():
    h = HandlerAvatarMusetalk()
    h.avatar = FakeAvatar()
    h.output_data_definitions = {Kind.AVATAR_AUDIO: definition(Kind.AVATAR_AUDIO),
                                 Kind.AVATAR_VIDEO: definition(Kind.AVATAR_VIDEO)}
    a, b = session('a'), session('b')
    ca, cb = h.create_context(a.session_context), h.create_context(b.session_context)
    try:
        register(a, h, ca, 'avatar', Kind.AVATAR_AUDIO)
        register(b, h, cb, 'avatar', Kind.AVATAR_AUDIO)
        assert ca.processor is not cb.processor
        for name in ('audio_out_queue', 'video_out_queue', 'event_out_queue', 'input_slice_context'):
            assert getattr(ca, name) is not getattr(cb, name)
        ca.processor._audio_queue.put({'old': 1})
        cb.processor._audio_queue.put({'other': 1})
        interrupt(a)
        assert ca.processor._audio_queue.empty()
        assert cb.processor._audio_queue.qsize() == 1
        ca.input_slice_context.last_remainder = np.ones(32)
        h.reset_generation(ca, 1)
        assert ca.input_slice_context.last_remainder is None
        # Delayed output event must not reopen VAD for the new round.
        a.session_context.shared_states.enable_vad = False
        from handlers.avatar.liteavatar.model.tts2face_event import Tts2FaceEvent
        ca.event_out_queue.put((0, Tts2FaceEvent.SPEAKING_TO_LISTENING))
        wait_until(ca.event_out_queue.empty)
        assert not a.session_context.shared_states.enable_vad
        ca.event_out_queue.put((1, Tts2FaceEvent.SPEAKING_TO_LISTENING))
        wait_until(lambda: a.session_context.shared_states.enable_vad)
    finally:
        h.destroy_context(ca)
        h.destroy_context(cb)


def test_tts_avatar_rtc_pipeline_rejects_late_old_round_after_stop(monkeypatch):
    """Actual engine pumps + callback + avatar media thread + RTC output consumer."""
    import asyncio
    from handlers.tts.bailian_tts import tts_handler_cosyvoice_bailian as tts
    from chat_engine.common.engine_channel_type import EngineChannelType as Channel
    from chat_engine.core.chat_session import DataSink
    from test_pipeline import data, rtc
    s = session()
    stream, delegate = rtc(s)
    h = HandlerAvatarMusetalk()
    h.avatar = FakeAvatar()
    h.output_data_definitions = {Kind.AVATAR_AUDIO: definition(Kind.AVATAR_AUDIO),
                                 Kind.AVATAR_VIDEO: definition(Kind.AVATAR_VIDEO)}
    cfg = AvatarMuseTalkConfig(batch_size=2, output_audio_sample_rate=24000,
                              algo_audio_sample_rate=24000)
    avatar_context = h.create_context(s.session_context, cfg)
    avatar_env = register(s, h, avatar_context, 'avatar', Kind.AVATAR_AUDIO)
    s.data_sinks[Kind.AVATAR_AUDIO] = [DataSink('avatar', avatar_env.input_queue,
                                              SimpleNamespace(input_consume_mode=None))]
    for kind in (Kind.AVATAR_AUDIO, Kind.AVATAR_VIDEO):
        s.outputs[('avatar', kind)] = DataSink('rtc', delegate.output_queues[kind.channel_type])
    th = tts.HandlerTTS()
    tc = th.create_context(s.session_context)
    te = register(s, th, tc, 'tts', Kind.AVATAR_AUDIO)
    instances = []
    class Synth:
        def __init__(self, **kwargs):
            self.callback = kwargs['callback']
            self.cancelled = threading.Event()
            instances.append(self)
        def streaming_call(self, text):
            pass
        def streaming_complete(self):
            self.callback.on_complete()
        def streaming_cancel(self):
            self.cancelled.set()
    monkeypatch.setattr(tts, 'SpeechSynthesizer', Synth)
    s.start()
    try:
        te.input_queue.put(data(speech_id='same'))
        wait_until(lambda: len(instances) == 1)
        old = instances[0]
        old.callback.on_data(np.full(24000, 1100, np.int16).tobytes())
        wait_until(lambda: not delegate.output_queues[Channel.AUDIO].empty())
        stream.handle_message('{"type":"stop_chat"}', stream.chat_channel)
        assert old.cancelled.wait(3)
        assert s.session_context.shared_states.enable_vad
        te.input_queue.put(data(generation=1, speech_id='same'))
        wait_until(lambda: len(instances) == 2)
        # The cancelled provider keeps delivering after new synthesis starts.
        old.callback.on_data(np.full(24000, 1100, np.int16).tobytes())
        old.callback.on_complete()
        new = instances[1]
        new.callback.on_data(np.full(24000, 2200, np.int16).tobytes())
        new.callback.on_complete()
        emitted = asyncio.run(asyncio.wait_for(stream.emit(), 3))
        assert np.allclose(emitted[1], 2200 / 32767)
        # A second interruption cannot resurrect either prior callback.
        interrupt(s)
        old.callback.on_complete()
        new.callback.on_complete()
        assert delegate.output_queues[Channel.AUDIO].empty()
    finally:
        s.stop()
