"""Interface for reading commands from keyboard."""

from __future__ import annotations

import time
from absl import app
from pynput import keyboard
from pynput.keyboard import Key

class Keyboard:

    def __init__(self, vel_scale_x: float = .4, vel_scale_y: float = .4, vel_scale_rot: float = .4):
        self._vel_scale_x = vel_scale_x
        self._vel_scale_y = vel_scale_y
        self._vel_scale_rot = vel_scale_rot

        # Key states 
        self.vx, self.vy, self.wz = 0., 0., 0.
        self.estop_flagged = False
        self.is_running = True

        self.pressed_keys = set()

        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
            suppress=True
        )
        self.listener.start()

    def _on_press(self, key):
        try:
            k = key.char if hasattr(key, 'char') else key
            if k not in self.pressed_keys:
                self.pressed_keys.add(k)
                self._update_commands() 
        except Exception:
            pass


    def _on_release(self, key):
        try:
            k = key.char if hasattr(key, 'char') else key
            if k in self.pressed_keys:
                self.pressed_keys.remove(k)
            self._update_commands() 
        except Exception:
            pass

        if key == Key.esc:
            self.stop()

    def _update_commands(self):
        new_vx, new_vy, new_wz = 0., 0., 0.

        # Press 'Ctrl + C' to exit
        if (Key.ctrl_l or Key.ctrl_r) and 'c' in self.pressed_keys:
            self.stop()

        # E-Stop: 'q'
        if 'q' in self.pressed_keys:
            self.estop_flagged = True
            
        if self.estop_flagged:
            self.vx, self.vy, self.wz = 0., 0., 0.
            self.estop_flagged = False
            vx, _, wz = None

        if any(k in self.pressed_keys for k in ['w', Key.up]):
            new_vx = self._vel_scale_x
        elif any(k in self.pressed_keys for k in ['s', Key.down]):
            new_vx = -self._vel_scale_x 

        if any(k in self.pressed_keys for k in ['a', Key.left]):
            new_wz = self._vel_scale_rot
        elif any(k in self.pressed_keys for k in ['d', Key.right]):
            new_wz = -self._vel_scale_rot

        self.vx, self.vy, self.wz = new_vx, new_vy, new_wz

    def get_command(self):
        return self.vx, self.vy, self.wz

    def stop(self):
        self.is_running = False
        self.listener.stop()

def main(_):
    controller = Keyboard()
    print("Commands: W/S (Forward/Backward), A/D (Rotate), Q (E-Stop), ESC/Ctrl + C (Exit)")
    
    try:
        while controller.is_running:
            status = f"Vx: {controller.vx:>5.2f} | Vy: {controller.vy:>5.2f} | Wz: {controller.wz:>5.2f} | Estop: {controller.estop_flagged}"
            print(f"\r{status}", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()

if __name__ == "__main__":
    app.run(main)