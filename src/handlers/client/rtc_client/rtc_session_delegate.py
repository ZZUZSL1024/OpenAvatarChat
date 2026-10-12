"""Thread-safe session transport, independent of RTC codec/UI initialization."""
import asyncio
import queue
import time
from typing import Dict, Optional, Tuple, Union
from uuid import uuid4
import numpy as np
from chat_engine.common.client_handler_base import ClientSessionDelegate
from chat_engine.common.engine_channel_type import EngineChannelType
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_signal import ChatSignal
from chat_engine.data_models.chat_signal_type import ChatSignalType
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition


class RtcClientSessionDelegate(ClientSessionDelegate):
    def __init__(self):
        self.timestamp_generator = None
        self.data_submitter = None
        self.shared_states = None
        self.signal_handler = None
        self.output_queues = {
            EngineChannelType.AUDIO: queue.Queue(),
            EngineChannelType.VIDEO: queue.Queue(),
            EngineChannelType.TEXT: queue.Queue(),
        }
        self.input_data_definitions: Dict[EngineChannelType, DataBundleDefinition] = {}
        self.modality_mapping = {
            EngineChannelType.AUDIO: ChatDataType.MIC_AUDIO,
            EngineChannelType.VIDEO: ChatDataType.CAMERA_VIDEO,
            EngineChannelType.TEXT: ChatDataType.HUMAN_TEXT,
        }

    async def get_data(self, modality: EngineChannelType, timeout: Optional[float] = 0.1) -> Optional[ChatData]:
        data_queue = self.output_queues.get(modality)
        if data_queue is None:
            return None
        deadline = time.monotonic() + timeout if timeout and timeout > 0 else None
        while deadline is None or time.monotonic() < deadline:
            try:
                data = data_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.005)
                continue
            if self.shared_states.is_current(data.generation):
                return data
        return None

    def put_data(self, modality: EngineChannelType, data: Union[np.ndarray, str],
                 timestamp: Optional[Tuple[int, int]] = None, samplerate: Optional[int] = None, loopback: bool = False):
        if timestamp is None:
            timestamp = self.get_timestamp()
        if self.data_submitter is None:
            return
        definition = self.input_data_definitions.get(modality)
        chat_data_type = self.modality_mapping.get(modality)
        if chat_data_type is None or definition is None:
            return
        data_bundle = DataBundle(definition)
        if modality == EngineChannelType.AUDIO:
            data_bundle.set_main_data(data.squeeze()[np.newaxis, ...])
        elif modality == EngineChannelType.VIDEO:
            data_bundle.set_main_data(data[np.newaxis, ...])
        elif modality == EngineChannelType.TEXT:
            data_bundle.add_meta('human_text_end', True)
            data_bundle.add_meta('speech_id', str(uuid4()))
            data_bundle.set_main_data(data)
        else:
            return
        chat_data = ChatData(
            source="client",
            type=chat_data_type,
            data=data_bundle,
            timestamp=timestamp,
        )
        with self.shared_states.lock:
            chat_data.generation = self.shared_states.generation
            self.data_submitter.submit(chat_data, generation=chat_data.generation)
            if loopback:
                self.output_queues[modality].put_nowait(chat_data)

    def get_timestamp(self):
        return self.timestamp_generator()

    def emit_signal(self, signal: ChatSignal):
        if self.signal_handler is not None:
            self.signal_handler(signal)
        if signal.type == ChatSignalType.INTERRUPT and self.shared_states.interruption_enabled:
            self.clear_data()

    def clear_data(self):
        for data_queue in self.output_queues.values():
            while True:
                try:
                    data_queue.get_nowait()
                except queue.Empty:
                    break


