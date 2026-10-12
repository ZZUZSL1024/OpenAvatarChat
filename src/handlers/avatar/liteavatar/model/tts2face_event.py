from enum import Enum


class Tts2FaceEvent(Enum):
    START = 1001
    STOP = 1002

    LISTENING_TO_SPEAKING = 2001
    SPEAKING_TO_LISTENING = 2002

