import numpy as np
import time
from transcriber import Transcriber
from keyhandler import KeyHandler
from ai_handler import AI_Handler
import os 
from dotenv import load_dotenv
from google import genai
from google.genai import types


def main():
    load_dotenv()
    transcriber = Transcriber()
    handler = KeyHandler()
    handler.listener.start()
    model = "gemini-2.5-flash-lite"
    client = genai.Client()
    config=types.GenerateContentConfig(system_instruction="You are a helpful writing assistant. The user will give you a voice command. Respond with only the output text, no explanations.")
    #  def __init__(self, client, model, system_instruction, config):
    ai_handler = AI_Handler(client, model, config)
    while handler.running:
        
        if handler.recording:
            with transcriber.open_stream() as stream:
                while handler.recording:
                    data = transcriber.record_chunk(stream)
                    transcriber.audio_chunks.append(data)
             
        elif handler.should_transcribe:
            transcriber.full_text = ""
            if transcriber.audio_chunks:
                transcriber.audio_data = np.concatenate(transcriber.audio_chunks, axis=0)
            else:
                print("Nothing was recorded")
                handler.should_transcribe = False
                continue
            text = transcriber.transcribe_audio()
            if handler.pass_toai:
                text = ai_handler.ask_llm(text)
                handler.pass_toai = False
            handler.type_on_cursor(text, transcriber.typing_delay)
            handler.should_transcribe = False
        else:
            time.sleep(0.01)
if __name__ == "__main__":
    main()