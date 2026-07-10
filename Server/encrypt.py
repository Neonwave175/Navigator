import os

from argon2 import low_level

# https://github.com/P-H-C/phc-winner-argon2.git
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def keygen(secret: bytes, salt: bytes) -> bytes:
    return low_level.hash_secret_raw(
        secret=secret,
        salt=salt,
        time_cost=3,
        memory_cost=65536,  # 64 MB
        parallelism=4,
        hash_len=32,
        type=low_level.Type.ID,
    )


def buildkey(skey: int) -> bytes:
    secret = str(skey).encode("utf-8")
    salt = b"fixed-shared-salt-16b"[:16].ljust(16, b"0")  # exactly 16 bytes, fixed
    return keygen(secret, salt)


def encrypt_image(image_path: str, key: bytes, output_path: str):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    nonce = os.urandom(12)
    ciphertext = AESGCM(key).encrypt(nonce, image_bytes, None)
    with open(output_path, "wb") as f:
        f.write(nonce)
        f.write(ciphertext)


def decrypt_image(encrypted_path: str, key: bytes, output_path: str):
    with open(encrypted_path, "rb") as f:
        data = f.read()
    nonce = data[:12]
    ciphertext = data[12:]
    image_bytes = AESGCM(key).decrypt(nonce, ciphertext, None)
    with open(output_path, "wb") as f:
        f.write(image_bytes)
