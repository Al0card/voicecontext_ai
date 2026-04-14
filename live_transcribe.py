import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import soundfile as sf
import tempfile
from faster_whisper import WhisperModel
import time

fs = 16000
seconds = 10
model = WhisperModel("base", device="cpu", compute_type="float32")
while True:
    print("Assistant ready. Model loaded.")
    user_input = input("Press Enter to start recording or type 'quit' to exit")
    if user_input == "quit":
        break
    else:
        print("Recording...")
        start = time.perf_counter()
        
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels = 1)
        sd.wait()
        print("Recording complete")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:           
            sf.write(tmp.name, myrecording, fs)
            segments , info = model.transcribe(tmp.name, beam_size=5, language="en")
            print("Result:")
            for segment in segments:
                print(segment.text)
            end = time.perf_counter()
        
        print(f"Execution time: {end - start:.3} seconds")