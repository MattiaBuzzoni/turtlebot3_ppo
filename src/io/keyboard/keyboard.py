from absl import app

import threading
import time
from pynput import keyboard

class Keyboard:
    """Interface for reading commands for the keyboard."""

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
                self._update_commands() # Chiamata corretta
        except Exception:
            pass

    def _on_release(self, key):
        try:
            k = key.char if hasattr(key, 'char') else key
            if k in self.pressed_keys:
                self.pressed_keys.remove(k)
            self._update_commands() # Chiamata corretta
        except Exception:
            pass

        if key == keyboard.Key.esc:
            self.stop()

    def _update_commands(self):
        new_vx, new_vy, new_wz = 0., 0., 0.

        # E-Stop: 'q'
        if 'q' in self.pressed_keys:
            self.estop_flagged = True
            
        if self.estop_flagged:
            self.vx, self.vy, self.wz = 0., 0., 0.
            return

        if 'w' in self.pressed_keys:
            new_vx = self._vel_scale_x
        elif 's' in self.pressed_keys:
            new_vx = -self._vel_scale_x 

        # Rotazione (A/D) -> Yaw (wz)
        if 'a' in self.pressed_keys:
            new_wz = self._vel_scale_rot
        elif 'd' in self.pressed_keys:
            new_wz = -self._vel_scale_rot

        self.vx, self.vy, self.wz = new_vx, new_vy, new_wz

    def stop(self):
        self.is_running = False
        self.listener.stop()

def main(_):
    controller = Keyboard()
    print("Comandi: W/S (Avanti/Indietro), A/D (Rotazione), Q (E-Stop), ESC (Esci)")
    
    try:
        while controller.is_running:
            status = f"Vx: {controller.vx:>5.2f} | Vy: {controller.vy:>5.2f} | Wz: {controller.wz:>5.2f} | Estop: {controller.estop_flagged}"
            print(f"\r{status}", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        print("\nScript terminato.")

if __name__ == "__main__":
    app.run(main)