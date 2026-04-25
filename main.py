import numpy as np
import time
from transcriber import Transcriber
from keyhandler import KeyHandler



def main():
    transcriber = Transcriber()
    handler = KeyHandler(transcriber)
    handler.listener.start()
    while handler.running:
        if handler.recording:
            with transcriber.open_stream() as stream:
                while handler.recording:
                    data = transcriber.record_chunk(stream)
                    transcriber.audio_chunks.append(data)
             
        elif handler.should_transcribe:
            if transcriber.audio_chunks:
                transcriber.audio_data = np.concatenate(transcriber.audio_chunks, axis=0)
            else:
                print("Nothing was recorded")
                handler.should_transcribe = False
                continue
            transcriber.transcribe_audio()
            handler.should_transcribe = False
        else:
            time.sleep(0.01)
if __name__ == "__main__":
    main()