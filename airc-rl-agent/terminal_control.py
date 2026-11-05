"""Terminal START/STOP for `racer train` (replaces user_interface_without_gamepad.ipynb).

Run in a second SSH window, BEFORE `racer train`:
    python3 terminal_control.py
Keys: g = START (go)   SPACE or s = STOP (crash / end episode)   q = STOP and quit
"""
import json
import select
import sys
import termios
import threading
import tty

import posix_ipc

TO_AGENT = "/ctr_status"     # racer reads commands here: True = STOP, False = START
FROM_AGENT = "/ctr_message"  # racer reports here: True = waiting for START, False = crash registered


def drain(mq):
    """Throw away messages left over from earlier runs."""
    n = 0
    while True:
        try:
            mq.receive(timeout=0)
            n += 1
        except posix_ipc.BusyError:
            return n


def say(text):
    print("\r" + text + "\r", flush=True)


def main():
    to_agent = posix_ipc.MessageQueue(TO_AGENT, posix_ipc.O_CREAT)
    from_agent = posix_ipc.MessageQueue(FROM_AGENT, posix_ipc.O_CREAT)
    say("cleared %d stale message(s)" % (drain(to_agent) + drain(from_agent)))

    def send(stop):
        try:
            to_agent.send(json.dumps({"status": stop}), timeout=1)
        except posix_ipc.BusyError:
            say("!! racer is not reading commands (not running yet, or stuck)")

    def listen():
        while True:
            data, _ = from_agent.receive()
            if isinstance(data, bytes):
                data = data.decode()
            if json.loads(data)["status"]:
                say(">>> READY: put the car on the track, press g to START")
            else:
                say(">>> CRASH registered: car stopped, learning... wait for READY")

    threading.Thread(target=listen, daemon=True).start()
    send(True)  # start in STOP, like the notebook
    say("Sent STOP. Now start `racer train` in the other window.")
    say("Keys: g = START   SPACE/s = STOP   q = quit")

    old = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        while True:
            if not select.select([sys.stdin], [], [], 0.1)[0]:
                continue
            key = sys.stdin.read(1).lower()
            if key == "g":
                send(False)
                say("--> START sent")
            elif key in (" ", "s"):
                send(True)
                say("--> STOP sent")
            elif key == "q":
                break
    except KeyboardInterrupt:
        pass
    finally:
        send(True)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        say("STOP sent, bye.")


if __name__ == "__main__":
    main()
