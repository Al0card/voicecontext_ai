from faster_whisper import WhisperModel
import time



model = WhisperModel("small", device="cpu", compute_type="float32")
start = time.perf_counter()
segments, info = model.transcribe("audios/audio1.flac", beam_size=5, language="en")
end = time.perf_counter()



for i, segment in enumerate(segments):
    # print(f"{i}: {segment.text}")
    print(segment.text)
print(f"Execution time: {end - start:.6f} seconds")