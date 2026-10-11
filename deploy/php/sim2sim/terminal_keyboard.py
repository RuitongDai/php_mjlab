"""Read immediate key presses from a POSIX terminal without blocking simulation."""

import os
import select
import sys
import termios
import tty


class TerminalKeyboard:
    def __init__(self, controller, stream=None):
        self.controller = controller
        self.stream = sys.stdin if stream is None else stream
        self._saved = None
        self._escape = None

    def __enter__(self):
        if not self.stream.isatty():
            raise ValueError("--command-source=terminal requires an interactive terminal")
        self.fd = self.stream.fileno()
        self._saved = termios.tcgetattr(self.fd)
        try:
            # Keep Ctrl+C signal handling while disabling line buffering and echo.
            tty.setcbreak(self.fd, termios.TCSANOW)
        except BaseException:
            termios.tcsetattr(self.fd, termios.TCSANOW, self._saved)
            raise
        return self

    def __exit__(self, *_):
        if self._saved is not None:
            termios.tcsetattr(self.fd, termios.TCSANOW, self._saved)
            self._saved = None

    def poll(self):
        """Drain available keys; ignore escape sequences such as arrow keys."""
        while select.select([self.fd], [], [], 0)[0]:
            data = os.read(self.fd, 64)
            if not data:
                raise EOFError("Terminal input closed")
            for key in data:
                if key == 27:
                    self._escape = "start"
                elif self._escape == "start":
                    self._escape = "sequence" if key in (ord("["), ord("O")) else None
                elif self._escape == "sequence":
                    if 0x40 <= key <= 0x7E:
                        self._escape = None
                elif key in (8, 127):
                    self.controller.key_callback(259)
                elif key < 128:
                    self.controller.key_callback(ord(chr(key).upper()))
