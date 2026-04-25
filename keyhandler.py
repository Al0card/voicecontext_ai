from pynput import keyboard
class KeyHandler:
    def __init__(self):
        self.recording = False
        self.running = True
        self.should_transcribe = False
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
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