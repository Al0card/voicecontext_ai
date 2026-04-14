import wave
from piper import PiperVoice
import soundfile as sf

fs = 16000
voice = PiperVoice.load("en_US-lessac-medium.onnx")
with wave.open("audios/test.wav", "wb") as wav_file:
    voice.synthesize_wav("Write a short professional email to my manager explaining that the project will be delayed by two days due to technical issues and propose a new deadline", wav_file)
    