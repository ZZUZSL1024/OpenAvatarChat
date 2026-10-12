import asyncio
import json
import time
import uuid
import weakref
from typing import Optional, Dict

import numpy as np
# noinspection PyPackageRequirements
from fastrtc import AsyncAudioVideoStreamHandler, AudioEmitType, VideoEmitType
from loguru import logger

from chat_engine.common.client_handler_base import ClientHandlerDelegate, ClientSessionDelegate
from chat_engine.common.engine_channel_type import EngineChannelType
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_signal import ChatSignal
from chat_engine.data_models.chat_signal_type import ChatSignalType, ChatSignalSourceType
from engine_utils.interval_counter import IntervalCounter

def _get_h264_encoder_info():
    """Get H.264 encoder info dynamically to avoid circular imports"""
    try:
        from handlers.client.rtc_client import client_handler_rtc
        return client_handler_rtc._selected_h264_encoder, client_handler_rtc._actual_h264_encoder
    except Exception:
        return "unknown", None


class RtcStream(AsyncAudioVideoStreamHandler):
    def __init__(self,
                 session_id: Optional[str],
                 expected_layout="mono",
                 input_sample_rate=16000,
                 output_sample_rate=24000,
                 output_frame_size=480,
                 fps=30,
                 stream_start_delay = 0.5,
                 interrupt_threshold = 0.01,
                 interrupt_duration = 0.096,
                 ):
        super().__init__(
            expected_layout=expected_layout,
            input_sample_rate=input_sample_rate,
            output_sample_rate=output_sample_rate,
            output_frame_size=output_frame_size,
            fps=fps
        )
        self.client_handler_delegate: Optional[ClientHandlerDelegate] = None
        self.client_session_delegate: Optional[ClientSessionDelegate] = None

        self.weak_factory: Optional[weakref.ReferenceType[RtcStream]] = None

        self.session_id = session_id
        self.stream_start_delay = stream_start_delay

        self.interrupt_threshold = interrupt_threshold
        self.interrupt_duration = interrupt_duration
        self._loud_samples = 0
        self._voice_preroll = []
        self.chat_channel = None
        self.first_audio_emitted = False

        self.quit = asyncio.Event()
        self.last_frame_time = 0

        self.emit_counter = IntervalCounter("emit counter")

        self.start_time = None
        self.timestamp_base = self.input_sample_rate

        self.streams: Dict[str, RtcStream] = {}


    # copy is used as create_instance in fastrtc
    def copy(self, **kwargs) -> AsyncAudioVideoStreamHandler:
        try:
            if self.client_handler_delegate is None:
                raise Exception("ClientHandlerDelegate is not set.")
            session_id = kwargs.get("webrtc_id", None)
            if session_id is None:
                session_id = uuid.uuid4().hex
            
            # Log codec information
            selected_encoder, _ = _get_h264_encoder_info()
            logger.debug(f"[{session_id}] H.264 encoder: {selected_encoder}")
            
            new_stream = RtcStream(
                session_id,
                expected_layout=self.expected_layout,
                input_sample_rate=self.input_sample_rate,
                output_sample_rate=self.output_sample_rate,
                output_frame_size=self.output_frame_size,
                fps=self.fps,
                stream_start_delay=self.stream_start_delay,
                interrupt_threshold=self.interrupt_threshold,
                interrupt_duration=self.interrupt_duration,
            )
            new_stream.weak_factory = weakref.ref(self)
            new_session_delegate = self.client_handler_delegate.start_session(
                session_id=session_id,
                timestamp_base=self.input_sample_rate,
            )
            new_stream.client_session_delegate = new_session_delegate
            if session_id in self.streams:
                msg = f"Stream {session_id} already exists."
                raise RuntimeError(msg)
            self.streams[session_id] = new_stream
            return new_stream
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to create stream: {e}")
            raise

    async def emit(self) -> AudioEmitType:
        try:
            # if not self.args_set.is_set():
            # await self.wait_for_args()

            if not self.first_audio_emitted:
                self.first_audio_emitted = True

            while not self.quit.is_set():
                chat_data = await self.client_session_delegate.get_data(EngineChannelType.AUDIO)
                if chat_data is None or chat_data.data is None:
                    continue
                audio_array = chat_data.data.get_main_data()
                if audio_array is None:
                    continue
                sample_num = audio_array.shape[-1]
                self.emit_counter.add_property("audio_emit", sample_num / self.output_sample_rate)
                states = self.client_session_delegate.shared_states
                with states.lock:
                    if states.is_current(chat_data.generation):
                        return self.output_sample_rate, audio_array
        except Exception as e:
            logger.opt(exception=e).error("Error in emit: ")
            raise

    async def video_emit(self) -> VideoEmitType:
        try:
            if not self.first_audio_emitted:
                await asyncio.sleep(0.1)
            
            # Log actual encoder being used (for verification)
            selected_encoder, actual_encoder = _get_h264_encoder_info()
            if actual_encoder:
                logger.debug(f"[{self.session_id}] Using H.264 encoder: {actual_encoder}")
            
            self.emit_counter.add_property("video_emit")
            
            while not self.quit.is_set():
                get_data_start = time.perf_counter()
                video_frame_data: ChatData = await self.client_session_delegate.get_data(EngineChannelType.VIDEO)
                get_data_wait_time = time.perf_counter() - get_data_start
                
                # Log slow data retrieval
                if get_data_wait_time > 0.05:
                    logger.debug(f"[{self.session_id}] Slow video data retrieval: {get_data_wait_time:.3f}s")
                
                if video_frame_data is None or video_frame_data.data is None:
                    continue
                
                frame_data = video_frame_data.data.get_main_data().squeeze()
                if frame_data is None:
                    continue
                
                states = self.client_session_delegate.shared_states
                with states.lock:
                    if states.is_current(video_frame_data.generation):
                        return frame_data
        except Exception as e:
            logger.opt(exception=e).error("Error in video_emit")
            raise

    async def receive(self, frame: tuple[int, np.ndarray]):
        if self.client_session_delegate is None:
            return
        timestamp = self.client_session_delegate.get_timestamp()
        if timestamp[0] / timestamp[1] < self.stream_start_delay:
            return
        sample_rate, array = frame
        states = self.client_session_delegate.shared_states
        audio_queue = self.client_session_delegate.output_queues.get(EngineChannelType.AUDIO)
        audio_pending = audio_queue is not None and not audio_queue.empty()
        if states.interruption_enabled and (not states.enable_vad or audio_pending) and array.size:
            # PCM may be int16 or normalized float. Convert before abs to avoid
            # int16 overflow and measure sustained RMS rather than one spike.
            normalized = array.astype(np.float32)
            if np.issubdtype(array.dtype, np.integer):
                normalized /= max(abs(np.iinfo(array.dtype).min), np.iinfo(array.dtype).max)
            rms = float(np.sqrt(np.mean(normalized ** 2)))
            if rms >= self.interrupt_threshold:
                self._loud_samples += array.shape[-1]
                self._voice_preroll.append((array.copy(), timestamp))
                if self._loud_samples < sample_rate * self.interrupt_duration:
                    return
                self._interrupt()
                # Keep the onset that triggered barge-in for ASR/VAD.
                for buffered, buffered_timestamp in self._voice_preroll[:-1]:
                    self.client_session_delegate.put_data(
                        EngineChannelType.AUDIO, buffered, buffered_timestamp, sample_rate)
            self._loud_samples = 0
            self._voice_preroll.clear()
        else:
            self._loud_samples = 0
            self._voice_preroll.clear()
        self.client_session_delegate.put_data(
            EngineChannelType.AUDIO,
            array,
            timestamp,
            self.input_sample_rate,
        )

    async def video_receive(self, frame):
        if self.client_session_delegate is None:
            return
        timestamp = self.client_session_delegate.get_timestamp()
        if timestamp[0] / timestamp[1] < self.stream_start_delay:
            return
        self.client_session_delegate.put_data(
            EngineChannelType.VIDEO,
            frame,
            timestamp,
            self.fps,
        )

    def set_channel(self, channel):
            super().set_channel(channel)
            self.chat_channel = channel
            
            async def process_chat_history():
                role = None
                chat_id = None
                while not self.quit.is_set():
                    chat_data = await self.client_session_delegate.get_data(EngineChannelType.TEXT)
                    if chat_data is None or chat_data.data is None:
                        continue
                    logger.debug(f"Got chat data {str(chat_data)}")
                    current_role = 'human' if chat_data.type == ChatDataType.HUMAN_TEXT else 'avatar'
                    chat_id = uuid.uuid4().hex if current_role != role else chat_id
                    role = current_role
                    states = self.client_session_delegate.shared_states
                    with states.lock:
                        if states.is_current(chat_data.generation):
                            self.chat_channel.send(json.dumps({
                                'type': 'chat', 'message': chat_data.data.get_main_data(),
                                'id': chat_id, 'role': current_role}))
            asyncio.create_task(process_chat_history())
                
            @channel.on("message")
            def on_message(message):
                self.handle_message(message, channel)

    def _interrupt(self):
        if not self.client_session_delegate.shared_states.interruption_enabled:
            return
        self.client_session_delegate.emit_signal(ChatSignal(
            type=ChatSignalType.INTERRUPT,
            source_type=ChatSignalSourceType.CLIENT,
            source_name="rtc",
        ))
        if self.chat_channel is not None:
            self.chat_channel.send(json.dumps({'type': 'avatar_end'}))

    def handle_message(self, raw_message, channel):
        try:
            message = json.loads(raw_message)
        except (ValueError, TypeError):
            return
        if not isinstance(message, dict) or self.client_session_delegate is None:
            return
        timestamp = self.client_session_delegate.get_timestamp()
        if timestamp[1] <= 0 or timestamp[0] / timestamp[1] < self.stream_start_delay:
            return
        kind = message.get('type')
        if kind == 'stop_chat':
            self._interrupt()
        elif kind == 'chat' and isinstance(message.get('data'), str) and message['data'].strip():
            states = self.client_session_delegate.shared_states
            if not states.interruption_enabled and not states.enable_vad:
                return
            self._interrupt()
            if not states.interruption_enabled:
                channel.send(json.dumps({'type': 'avatar_end'}))
            self._loud_samples = 0
            self._voice_preroll.clear()
            self.client_session_delegate.shared_states.enable_vad = False
            self.client_session_delegate.emit_signal(ChatSignal(
                type=ChatSignalType.BEGIN,
                stream_type=ChatDataType.AVATAR_AUDIO,
                source_type=ChatSignalSourceType.CLIENT,
                source_name="rtc",
            ))
            self.client_session_delegate.put_data(
                EngineChannelType.TEXT, message['data'], loopback=True)

    async def on_chat_datachannel(self, message: Dict, channel):
        # {"type":"chat",id:"标识属于同一段话", "message":"Hello, world!"}
        # unique_id = uuid.uuid4().hex
        pass
    def shutdown(self):
        self.quit.set()
        factory = None
        if self.weak_factory is not None:
            factory = self.weak_factory()
        if factory is None:
            factory = self
        self.client_session_delegate = None
        if factory.client_handler_delegate is not None:
            factory.client_handler_delegate.stop_session(self.session_id)
