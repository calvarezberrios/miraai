"""Standalone launcher for the avatar renderer server (dev/preview use).

Starts the motor_cortex web server and blocks. Handy for opening the avatar in a
browser without running the whole brain. The real app starts/stops the server
through the adapter lifecycle in main.py.

    python avatar/serve.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.forebrain.cerebrum.frontal_lobe import motor_cortex as mc

if __name__ == "__main__":
    mc.start(open_browser=False)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        mc.stop()
