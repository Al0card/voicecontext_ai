import sounddevice as sd
import tempfile
import soundfile as sf
from faster_whisper import WhisperModel


class Transcriber:
    def __init__(self):
        self.audio_chunks = []
        self.audio_data = None
        self.model = WhisperModel("base", device="cpu", compute_type="float32")
        self.fs = 16000
        self.channels = 1
        self.chunk_size = 3200
        
    def open_stream(self):
         return sd.InputStream(samplerate=self.fs, channels=self.channels, dtype='float32')
    def record_chunk(self, stream):
        data, _ = stream.read(self.chunk_size)
        return data
    def transcribe_audio(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, self.audio_data, self.fs)
            segments, info = self.model.transcribe(tmp.name, beam_size=5, language="en")
            # has_output = False
            for segment in segments:
                print(segment.text)
                # has_output = True
            # if not has_output:
                # print("Nothing was recorded")
            self.audio_chunks = []
            self.audio_data = None
    