from pynput.keyboard import Key, Controller
import time

# Initialize the keyboard controller
keyboard = Controller()

# Wait 5 seconds to switch to the target window
time.sleep(5)

# Method 1: Type a whole string at once
keyboard.type("Hello, this is automated text!")