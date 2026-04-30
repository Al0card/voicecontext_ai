from pynput import keyboard
import time
from pynput.keyboard import Controller
import random

class KeyHandler:
    def __init__(self):
        self.recording = False
        self.running = True
        self.should_transcribe = False
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.keyboard = Controller()
        self.pass_toai = False
    def on_press(self, key):
        
        try:
            if ((key == keyboard.Key.f9 or key == keyboard.Key.f8) and not self.recording):
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
            if key == keyboard.Key.f9 or key == keyboard.Key.f8:
                self.recording = False
                self.should_transcribe = True
                print("Recording completed!")
            if key == keyboard.Key.f8:
                self.pass_toai = True
        except AttributeError:
            pass
    def type_on_cursor(self, text, typing_delay):

        for char in text:
            self.keyboard.press(char)
            self.keyboard.release(char)
            time.sleep(typing_delay + random.uniform(0, 0.02))
