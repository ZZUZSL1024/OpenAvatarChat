import asyncio
import json
import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from chat_engine.common.engine_channel_type import EngineChannelType as Channel
from chat_engine.common.handler_base import HandlerBaseInfo, HandlerDataInfo
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.contexts.session_context import SessionContext
from chat_engine.core.chat_session import ChatSession, ChatDataSubmitter, DataSink, HandlerEnv, HandlerRecord
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType as Kind
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel
from chat_engine.data_models.chat_signal import ChatSignal
from chat_engine.data_models.chat_signal_type import ChatSignalType, ChatSignalSourceType
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry
from chat_engine.data_models.session_info_data import SessionInfoData
from handlers.client.rtc_client.rtc_session_delegate import RtcClientSessionDelegate
from handlers.tts.bailian_tts import tts_handler_cosyvoice_bailian as tts
from service.rtc_service.rtc_stream import RtcStream


def session(name='test'):
    context = SessionContext(SessionInfoData(session_id=name), {}, {})
    context.shared_states.interruption_enabled = True
    return ChatSession(context, ChatEngineConfigModel())


def interrupt(s):
    s.emit_signal(ChatSignal(type=ChatSignalType.INTERRUPT, source_type=ChatSignalSourceType.CLIENT))


def definition(kind=Kind.AVATAR_TEXT):
    d = DataBundleDefinition()
    if kind.channel_type == Channel.AUDIO:
        d.add_entry(DataBundleEntry.create_audio_entry('audio', 1, 24000))
    elif kind.channel_type == Channel.VIDEO:
        d.add_entry(DataBundleEntry.create_framed_entry('video', [1, 2, 2, 3], 0, 25))
    else:
        d.add_entry(DataBundleEntry.create_text_entry('text'))
    d.lockdown()
    return d


def data(kind=Kind.AVATAR_TEXT, generation=0, value='old', **meta):
    b = DataBundle(definition(kind))
    b.set_main_data(value)
    for key, val in meta.items():
        b.add_meta(key, val)
    return ChatData(type=kind, data=b, generation=generation)


def sink(s, kind=Kind.AVATAR_TEXT, owner='sink'):
    q = queue.Queue()
    s.data_sinks.setdefault(kind, []).append(DataSink(owner, q, HandlerDataInfo(type=kind)))
    return q


def register(s, handler, context, name, kind=Kind.AVATAR_TEXT):
    env = HandlerEnv(HandlerBaseInfo(name=name), handler, None, context, queue.Queue(),
                     {kind: HandlerDataInfo(type=kind, definition=definition(kind))})
    s.handlers[name] = HandlerRecord(env)
    context.data_submitter = ChatDataSubmitter(name, env.output_info, s.session_context, s.data_sinks, s.outputs)
    return env


def wait_until(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline, 'worker did not reach expected state'
        time.sleep(.005)


def rtc(s):
    delegate = RtcClientSessionDelegate()
    delegate.shared_states = s.session_context.shared_states
    delegate.timestamp_generator = lambda: (16000, 16000)
    delegate.set_signal_handler(s.emit_signal)
    delegate.input_data_definitions = {Channel.TEXT: definition(Kind.HUMAN_TEXT),
                                       Channel.AUDIO: definition(Kind.MIC_AUDIO)}
    delegate.data_submitter = ChatDataSubmitter('rtc', {}, s.session_context, s.data_sinks, s.outputs)
    stream = RtcStream('test', stream_start_delay=0)
    stream.client_session_delegate = delegate
    stream.chat_channel = SimpleNamespace(send=lambda message: None)
    return stream, delegate


def test_interrupt_clears_every_engine_queue_and_notifies_each_handler():
    s = session()
    observed = []
    for name in ('tts', 'avatar'):
        h = SimpleNamespace(on_interrupt=lambda ctx, gen: observed.append((ctx.session_id, gen)))
        env = register(s, h, HandlerContext(name), name)
        env.input_queue.put(data())
    out = sink(s)
    out.put(data())
    external = queue.Queue()
    external.put(data())
    s.outputs[('producer', Kind.AVATAR_TEXT)] = DataSink('out', external)
    s.session_context.input_queues[Channel.AUDIO] = queue.Queue()
    s.session_context.input_queues[Channel.AUDIO].put('stale mic')
    s.session_context.shared_states.enable_vad = False
    interrupt(s)
    assert s.session_context.shared_states.generation == 1
    assert s.session_context.shared_states.enable_vad
    assert observed == [('tts', 1), ('avatar', 1)]
    assert out.empty() and external.empty()
    assert all(r.env.input_queue.empty() for r in s.handlers.values())
    assert s.session_context.input_queues[Channel.AUDIO].empty()


def test_running_generator_and_sync_submit_cannot_repopulate_after_interrupt():
    s = session()
    q = sink(s)
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    class Handler:
        def reset_generation(self, context, generation):
            pass
        def on_interrupt(self, context, generation):
            pass
        def handle(self, context, inputs, outputs):
            try:
                if inputs.generation == 0:
                    entered.set()
                    assert release.wait(3)
                    context.submit_data(data().data)  # ContextVar must retain 0.
                yield data(value=str(inputs.generation)).data
            finally:
                closed.set()
    env = register(s, Handler(), HandlerContext('test'), 'producer')
    states = s.session_context.shared_states
    states.active = True
    worker = threading.Thread(target=s.handler_pumper,
                              args=(s.session_context, env, s.data_sinks, s.outputs))
    env.input_queue.put(data())
    worker.start()
    try:
        assert entered.wait(3)
        interrupt(s)
        env.input_queue.put(data(generation=1))
        release.set()
        received = q.get(timeout=3)
        assert received.generation == 1 and received.data.get_main_data() == '1'
        assert q.empty() and closed.is_set()
    finally:
        release.set()
        states.active = False
        worker.join(3)
        assert not worker.is_alive()


def test_rapid_interrupts_and_cross_session_isolation():
    a, b = session('a'), session('b')
    qa, qb = sink(a), sink(b)
    for _ in range(40):
        interrupt(a)
    for s in (a, b):
        for gen in range(41):
            s.submit_data(data(generation=gen), 'producer', {}, s.session_context, s.data_sinks, {})
    assert qa.get_nowait().generation == 40 and qa.empty()
    assert qb.get_nowait().generation == 0 and qb.empty()
    assert b.session_context.shared_states.generation == 0


def test_stale_end_cannot_change_new_round_vad():
    s = session()
    interrupt(s)
    s.session_context.shared_states.enable_vad = False
    for gen in (None, 0):
        s.emit_signal(ChatSignal(type=ChatSignalType.END, source_type=ChatSignalSourceType.CLIENT, generation=gen))
        assert not s.session_context.shared_states.enable_vad
    s.emit_signal(ChatSignal(type=ChatSignalType.END, source_type=ChatSignalSourceType.CLIENT, generation=1))
    assert s.session_context.shared_states.enable_vad


def test_rtc_text_stop_invalid_messages_and_new_text_not_dropped():
    s = session()
    q = sink(s, Kind.HUMAN_TEXT)
    stream, delegate = rtc(s)
    for invalid in ('{', '{}', '[]', '{"type":"chat"}', '{"type":"chat","data":" "}'):
        stream.handle_message(invalid, stream.chat_channel)
    assert s.session_context.shared_states.generation == 0
    for i in range(10):
        delegate.output_queues[Channel.AUDIO].put(data())
        stream.handle_message(json.dumps({'type': 'chat', 'data': str(i)}), stream.chat_channel)
        result = q.get_nowait()
        assert result.generation == i + 1 and result.data.get_main_data() == str(i)
        assert not s.session_context.shared_states.enable_vad
        assert delegate.output_queues[Channel.AUDIO].empty()
    stream.handle_message('{"type":"stop_chat"}', stream.chat_channel)
    assert s.session_context.shared_states.enable_vad and delegate.output_queues[Channel.TEXT].empty()


@pytest.mark.parametrize('dtype,level', [(np.int16, -32768), (np.float32, -.5)])
def test_voice_barge_in_normalization_debounce_and_onset_preserved(dtype, level):
    async def run():
        s = session()
        q = sink(s, Kind.MIC_AUDIO)
        stream, _ = rtc(s)
        states = s.session_context.shared_states
        states.enable_vad = False
        # One loud spike followed by silence is insufficient.
        await stream.receive((16000, np.full((1, 512), level, dtype=dtype)))
        await stream.receive((16000, np.zeros((1, 512), dtype=dtype)))
        assert states.generation == 0
        while not q.empty():
            q.get_nowait()
        for _ in range(3):
            await stream.receive((16000, np.full((1, 512), level, dtype=dtype)))
        assert states.generation == 1 and states.enable_vad
        assert q.qsize() == 3
        assert all(q.get_nowait().generation == 1 for _ in range(3))
        await stream.receive((16000, np.full((1, 512), level, dtype=dtype)))
        assert states.generation == 1
    asyncio.run(run())


def test_rtc_filters_audio_video_and_text_stale_queue_entries():
    async def run():
        s = session()
        stream, delegate = rtc(s)
        interrupt(s)
        for channel in (Channel.AUDIO, Channel.VIDEO, Channel.TEXT):
            delegate.output_queues[channel].put(data(generation=0))
            delegate.output_queues[channel].put(data(generation=1))
            assert (await delegate.get_data(channel)).generation == 1
        delegate.output_queues[Channel.AUDIO].put(data(Kind.AVATAR_AUDIO, 0, np.ones((1, 8), np.float32)))
        delegate.output_queues[Channel.AUDIO].put(data(Kind.AVATAR_AUDIO, 1, np.zeros((1, 8), np.float32)))
        assert not (await stream.emit())[1].any()
        delegate.output_queues[Channel.VIDEO].put(data(Kind.AVATAR_VIDEO, 0, np.ones((1, 2, 2, 3), np.uint8)))
        delegate.output_queues[Channel.VIDEO].put(data(Kind.AVATAR_VIDEO, 1, np.zeros((1, 2, 2, 3), np.uint8)))
        assert not (await stream.video_emit()).any()
    asyncio.run(run())


def test_tts_cancel_and_late_callbacks_including_reused_speech_id(monkeypatch):
    s = session()
    q = sink(s, Kind.AVATAR_AUDIO)
    h = tts.HandlerTTS()
    context = h.create_context(s.session_context)
    env = register(s, h, context, 'tts', Kind.AVATAR_AUDIO)
    instances = []
    class Synth:
        def __init__(self, **kwargs):
            self.callback = kwargs['callback']
            self.cancelled = threading.Event()
            self.completed = False
            instances.append(self)
        def streaming_call(self, text):
            self.callback.on_data(bytes(24002))
        def streaming_cancel(self):
            self.cancelled.set()
        def streaming_complete(self):
            self.completed = True
            self.callback.on_complete()
    monkeypatch.setattr(tts, 'SpeechSynthesizer', Synth)
    h.handle(context, data(speech_id='same'), env.output_info)
    old = instances[0]
    assert q.qsize() == 1
    interrupt(s)
    assert old.cancelled.wait(3) and not old.completed
    assert context.synthesizer is None and q.empty()
    h.handle(context, data(generation=1, speech_id='same'), env.output_info)
    for callback in (lambda: old.callback.on_data(bytes(30000)), old.callback.on_complete,
                     lambda: old.callback.on_error('late')):
        callback()
    assert q.qsize() == 1 and q.get_nowait().generation == 1
    h.handle(context, data(generation=1, value='end', speech_id='same', avatar_text_end=True), env.output_info)
    assert instances[1].completed and not instances[1].cancelled.is_set()
    assert all(q.get_nowait().generation == 1 for _ in range(q.qsize()))


def test_legacy_pipeline_without_interrupt_opt_in_keeps_async_output():
    s = session()
    s.session_context.shared_states.interruption_enabled = False
    q = sink(s)
    s.submit_data(data(generation=None), 'producer', {}, s.session_context, s.data_sinks, {})
    assert q.get_nowait().generation == 0
    interrupt(s)
    assert s.session_context.shared_states.generation == 0


def test_vad_reset_and_stale_disable_do_not_override_interrupt():
    from handlers.vad.silerovad.vad_handler_silero import HandlerAudioVAD, SpeakingStatus
    s = session()
    h = HandlerAudioVAD()
    context = h.create_context(s.session_context)
    context.audio_history = [np.ones(512)]
    context.speaking_status = SpeakingStatus.START
    context.model_state.fill(1)
    context.slice_context.last_remainder = np.ones(20)
    interrupt(s)
    s.session_context.shared_states.set_vad(False, 0)
    assert s.session_context.shared_states.enable_vad
    h.reset_generation(context, 1)
    assert context.speaking_status == SpeakingStatus.END and context.audio_history == []
    assert not context.model_state.any() and context.slice_context.last_remainder is None


def test_tts_interrupt_between_callback_check_and_submission(monkeypatch):
    s = session()
    q = sink(s, Kind.AVATAR_AUDIO)
    h = tts.HandlerTTS()
    c = h.create_context(s.session_context)
    register(s, h, c, 'tts', Kind.AVATAR_AUDIO)
    c.active_speech_id = 'old'
    cb = tts.CosyvoiceCallBack(c, definition(Kind.AVATAR_AUDIO), 'old', 0)
    def racing_check():
        interrupt(s)
        return False  # callback validated immediately before interruption
    monkeypatch.setattr(cb, '_should_drop', racing_check)
    cb.on_data(bytes(30000))
    assert q.empty()


def test_interrupt_is_atomic_with_distribution():
    s = session()
    entered, release, interrupted = threading.Event(), threading.Event(), threading.Event()
    class DelayedQueue(queue.Queue):
        def put_nowait(self, value):
            entered.set()
            assert release.wait(3)
            super().put_nowait(value)
    q = DelayedQueue()
    s.data_sinks[Kind.AVATAR_TEXT] = [DataSink('sink', q, HandlerDataInfo(type=Kind.AVATAR_TEXT))]
    submitter = threading.Thread(target=s.submit_data,
        args=(data(), 'producer', {}, s.session_context, s.data_sinks, {}))
    def interrupt_worker():
        interrupt(s)
        interrupted.set()
    submitter.start()
    assert entered.wait(3)
    interrupter = threading.Thread(target=interrupt_worker)
    interrupter.start()
    release.set()
    submitter.join(3)
    interrupter.join(3)
    assert interrupted.is_set() and q.empty()
    assert not submitter.is_alive() and not interrupter.is_alive()


def test_voice_interrupts_audio_backlog_even_after_vad_resumes():
    async def run():
        s = session()
        stream, delegate = rtc(s)
        delegate.output_queues[Channel.AUDIO].put(data(Kind.AVATAR_AUDIO, 0, np.ones((1, 8), np.float32)))
        assert s.session_context.shared_states.enable_vad
        for _ in range(3):
            await stream.receive((16000, np.full((1, 512), .5, np.float32)))
        assert s.session_context.shared_states.generation == 1
        assert delegate.output_queues[Channel.AUDIO].empty()
    asyncio.run(run())


def test_interrupt_cancels_inflight_tts_completion_without_waiting_for_it(monkeypatch):
    s = session()
    q = sink(s, Kind.AVATAR_AUDIO)
    h = tts.HandlerTTS()
    c = h.create_context(s.session_context)
    env = register(s, h, c, 'tts', Kind.AVATAR_AUDIO)
    entered, cancelled, finished = threading.Event(), threading.Event(), threading.Event()
    class Synth:
        def __init__(self, **kwargs):
            self.callback = kwargs['callback']
        def streaming_call(self, text):
            pass
        def streaming_complete(self):
            entered.set()
            assert cancelled.wait(3)
            self.callback.on_complete()
            finished.set()
        def streaming_cancel(self):
            cancelled.set()
    monkeypatch.setattr(tts, 'SpeechSynthesizer', Synth)
    worker = threading.Thread(target=h.handle, args=(c, data(avatar_text_end=True, speech_id='old'), env.output_info))
    worker.start()
    try:
        assert entered.wait(3)
        interrupt(s)
        assert finished.wait(3)
        assert q.empty() and c.synthesizer is None
        assert s.session_context.shared_states.enable_vad
    finally:
        cancelled.set()
        worker.join(3)
        assert not worker.is_alive()
