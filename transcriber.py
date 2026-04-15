from pynput import keyboard
import sounddevice as sd
import tempfile
import soundfile as sf
from faster_whisper import WhisperModel
import numpy as np
import time

class KeyHandler:
    def __init__(self):
        self.audio_chunks = []
        self.audio_data = None
        self.model = WhisperModel("base", device="cpu", compute_type="float32")
        self.fs = 16000
        self.channels = 1
        self.chunk_size = 3200
        self.recording = False
        self.running = True
        self.should_transcribe = False
    def on_press(self, key):
        
        try:
            if key == keyboard.Key.f9 and not self.recording:
                print("Recording ...")
                self.recording = True
            if key == keyboard.Key.esc:
                 self.running = False
                 self.recording = False
                 return False
        except AttributeError:
            pass
    def on_release(self, key):
        try:
            if key == keyboard.Key.f9:
                self.recording = False
                self.should_transcribe = True
                print("Recording completed!")
            
        except AttributeError:
            pass
    def record_audio(self):
        with sd.InputStream(samplerate=self.fs, channels=self.channels, dtype='float32') as stream:
            while self.recording:
                data, _ = stream.read(self.chunk_size) 
                self.audio_chunks.append(data)
    def transcribe_audio(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, self.audio_data, self.fs)
            segments, info = self.model.transcribe(tmp.name, beam_size=5, language="en")
            has_output = False
            for segment in segments:
                print(segment.text)
                has_output = True
            if not has_output:
                print("Nothing was recorded")
            self.audio_chunks = []
            self.should_transcribe = False
            self.audio_data = None
handler = KeyHandler()
listener = keyboard.Listener(on_press=handler.on_press, on_release=handler.on_release)
listener.start()
while handler.running:
    if handler.recording:
        handler.record_audio()
    elif handler.should_transcribe:
        if handler.audio_chunks:
            handler.audio_data = np.concatenate(handler.audio_chunks, axis=0)
        else:
             print("Nothing was recorded")
             handler.should_transcribe = False
             continue
        handler.transcribe_audio()
    else:
        time.sleep(0.01)
    