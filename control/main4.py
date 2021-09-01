# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import time
os.system("/home/chase/uartapp /dev/ttyTHS0 0 2")
time.sleep(0.4)
os.system("/home/chase/uartapp /dev/ttyTHS0 0 0")
