import asyncio

import websockets

import encrypt as e

SHARED_KEY_SEED = 2342344554623453242345234634577674354634563456567467564674567456
key = e.buildkey(SHARED_KEY_SEED)


async def send_image(path):
    async with websockets.connect("ws://127.0.0.1:8765") as ws:
        with open(path, "rb") as f:
            image_bytes = f.read()
        await ws.send(image_bytes)
        response = await ws.recv()
        print(response)


e.encrypt_image("newt.jpg", key, "photo.enc")
asyncio.run(send_image("photo.enc"))
