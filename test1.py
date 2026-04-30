import pyperclip
# import time
# history = []
# pyperclip.copy('phd')
# pyperclip.copy('scienco')
# pyperclip.copy('waht the fuck i am doing with my life')
# print(pyperclip.paste())
import time

from pynput.keyboard import Key, Controller

clipboard = []
keyboard = Controller()


def add_to_clipboard():
    keyboard.press(Key.ctrl)
    keyboard.press(Key.shift)
    keyboard.press('c')
    keyboard.release('c')
    keyboard.release(Key.shift)
    keyboard.release(Key.ctrl)
    clipboard.append(pyperclip.paste())

def get_from_clipboard():
    for text in clipboard:
        print(text)



HotKeys_Listener =  keyboard.GlobalHotKeys({
        '<ctrl>+<shift>+c': add_to_clipboard})
HotKeys_Listener.start()

time.sleep(30)
get_from_clipboard()