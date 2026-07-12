import asyncio
import os
import select
import sys
import termios
import tty
import cv2
import websockets

# Add Server directory to sys.path to import encrypt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Server")))
import encrypt as e

KEYSEED = 2342344554623453242345234634577674354634563456567467564674567456
key = e.buildkey(KEYSEED)

async def send_image(path):
    try:
        encrypted_bytes = e.encrypt_image(path, key)
        async with websockets.connect("ws://127.0.0.1:8765") as ws:
            await ws.send(encrypted_bytes)
            response = await ws.recv()
            print(f"\n[Server Response]: {response}\n")
    except Exception as err:
        print(f"\n[Error sending image]: {err}\n")

class NonBlockingTTY:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, type, value, traceback):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def get_key(self):
        if select.select([sys.stdin], [], [], 0.001)[0]:
            return sys.stdin.read(1)
        return None

def main():
    print("Starting clcam (Headless Mode on Pi 5)...")
    print("Press 's' in the terminal to capture and send an image to the server.")
    print("Press 'q' in the terminal to exit.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    temp_img_path = "captured_frame.jpg"

    try:
        with NonBlockingTTY() as tty_input:
            while True:
                # Grab the frame to keep the buffer fresh
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to grab frame.")
                    break

                # Check keypress from Terminal
                term_key = tty_input.get_key()

                action_s = (term_key == 's')
                action_q = (term_key == 'q')

                if action_s:
                    print("\n's' pressed! Capturing frame and sending...")
                    # Save the frame
                    cv2.imwrite(temp_img_path, frame)
                    # Run the client send image script synchronously (blocks until done)
                    asyncio.run(send_image(temp_img_path))
                    # Clean up temp image
                    if os.path.exists(temp_img_path):
                        try:
                            os.remove(temp_img_path)
                        except OSError:
                            pass
                    print("Resuming camera capture...")

                if action_q:
                    print("\nExiting clcam...")
                    break
    finally:
        cap.release()

if __name__ == "__main__":
    main()
