import asyncio

import encrypt as e
import websockets

KEYSEED = 2342344554623453242345234634577674354634563456567467564674567456
key = e.buildkey(KEYSEED)


async def send_image(path):
    encrypted_bytes = e.encrypt_image(path, key)
    async with websockets.connect("ws://127.0.0.1:8765") as ws:
        await ws.send(encrypted_bytes)
        response = await ws.recv()
        print(response)


asyncio.run(send_image("images/newt.jpg"))
