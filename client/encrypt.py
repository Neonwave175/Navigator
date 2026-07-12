import os

from argon2 import low_level

# Argon2 and AESGCMSIV unbreakable
from cryptography.hazmat.primitives.ciphers.aead import AESGCMSIV


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


def encrypt_image(image_path: str, key: bytes) -> bytes:
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    nonce = os.urandom(12)
    # Using SIV mode ensures that even if os.urandom fails/repeats on the Pi, key safety holds
    ciphertext = AESGCMSIV(key).encrypt(nonce, image_bytes, None)
    return nonce + ciphertext


def decrypt_image(encrypted_bytes: bytes, key: bytes) -> bytes:
    nonce = encrypted_bytes[:12]
    ciphertext = encrypted_bytes[12:]
    return AESGCMSIV(key).decrypt(nonce, ciphertext, None)
