import asyncio

import websockets

import encrypt as e

SHARED_KEY_SEED = 2342344554623453242345234634577674354634563456567467564674567456
key = e.buildkey(SHARED_KEY_SEED)


async def handler(websocket):
    async for message in websocket:
        if isinstance(message, bytes):
            with open("photo.enc", "wb") as f:
                f.write(message)
            e.decrypt_image("photo.enc", key, "photo_decrypted.jpg")
            print(f"Saved and decrypted image, {len(message)} bytes")
            await websocket.send("received")
        else:
            print(f"Text message: {message}")


async def main():
    async with websockets.serve(handler, "127.0.0.1", 8765):
        print("Listening on ws://127.0.0.1:8765")
        await asyncio.Future()


asyncio.run(main())
