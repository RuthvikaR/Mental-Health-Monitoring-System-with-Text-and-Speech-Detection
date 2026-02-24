import streamlit as st
import av
import numpy as np
import io
import wave
import speech_recognition as sr
from textblob import TextBlob
from streamlit_webrtc import AudioProcessorBase

# -------------------------------
# 🎧 AUDIO PROCESSOR
# -------------------------------
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        self.audio_frames.append(audio)
        return frame

    def get_audio(self):
        if len(self.audio_frames) == 0:
            return None

        audio = np.concatenate(self.audio_frames, axis=0)

        # 🔥 Clear buffer after reading
        self.audio_frames = []

        # Convert stereo → mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        return audio

# -------------------------------
# 🔊 NUMPY → WAV CONVERSION
# -------------------------------
def numpy_to_wav_bytes(audio_array, sample_rate=48000):
    buffer = io.BytesIO()

    # Normalize and convert to int16
    if audio_array.dtype != np.int16:
        audio_array = np.clip(audio_array, -1, 1)
        audio_array = (audio_array * 32767).astype(np.int16)

    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)       # mono
        wf.setsampwidth(2)       # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_array.tobytes())

    buffer.seek(0)
    return buffer

# -------------------------------
# 🎤 SPEECH TO TEXT
# -------------------------------
def speech_to_text(audio_array):
    recognizer = sr.Recognizer()

    try:
        if audio_array is None:
            return None

        # Reject too-short audio (~0.5 sec at 16kHz)
        if len(audio_array) < 8000:
            st.warning("Audio too short")
            return None

        wav_buffer = numpy_to_wav_bytes(audio_array)

        with sr.AudioFile(wav_buffer) as source:
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio)
        return text.lower()

    except sr.UnknownValueError:
        st.error("Could not understand audio")
        return None
    except sr.RequestError as e:
        st.error(f"Speech recognition service error: {e}")
        return None
    except Exception as e:
        st.error(f"Speech recognition error: {e}")
        return None


