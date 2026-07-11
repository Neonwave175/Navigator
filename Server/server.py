import asyncio

import encrypt as e
import mlx_infer
import websockets

KEYSEED = 2342344554623453242345234634577674354634563456567467564674567456
key = e.buildkey(KEYSEED)

MODEL_PATH = "mlx-community/gemma-4-26b-a4b-it-mxfp4"

# Loads Model
model, processor, config = mlx_infer.load_model(MODEL_PATH)


async def handler(websocket):
    async for message in websocket:
        if isinstance(message, bytes):
            with open("photo.enc", "wb") as f:
                f.write(message)

            e.decrypt_image("photo.enc", key, "photo_decrypted.jpg")
            print(f"Saved and decrypted image, {len(message)} bytes")

            result = mlx_infer.run(model, processor, config, "photo_decrypted.jpg")
            print(f"Inference result: {result}")
            await websocket.send(result)
        else:
            print(f"Text message: {message}")


async def main():
    async with websockets.serve(handler, "127.0.0.1", 8765):
        print("Listening on ws://127.0.0.1:8765")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt():
        print("Bye")
