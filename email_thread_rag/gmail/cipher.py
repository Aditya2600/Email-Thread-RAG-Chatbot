"""Refresh-token encryption behind a small ``TokenCipher`` interface.

The store only ever sees ciphertext. Swapping AES-GCM-with-a-local-key for a
KMS/HSM later means writing one more class with these two methods; nothing
else in the package changes.
"""

from __future__ import annotations

import base64
import os
from typing import Protocol


class TokenCipherError(RuntimeError):
    """Raised on missing/invalid key material or a failed decrypt.

    Never carries plaintext, ciphertext, or key material in its message: this
    exception text reaches logs.
    """


class TokenCipher(Protocol):
    key_id: str

    def encrypt(self, plaintext: str) -> bytes: ...

    def decrypt(self, ciphertext: bytes) -> str: ...


class AesGcmTokenCipher:
    """AES-256-GCM with a local 32-byte key. Nonce is random per encrypt and
    prefixed to the ciphertext, so the same token encrypts differently each
    time and ciphertext equality never leaks token equality.
    """

    NONCE_BYTES = 12

    def __init__(self, key: bytes, *, key_id: str = "local"):
        if len(key) != 32:
            raise TokenCipherError("Token encryption key must be exactly 32 bytes (AES-256).")
        self._key = key
        self.key_id = key_id

    @classmethod
    def from_base64_key(cls, encoded: str | None, *, key_id: str = "local") -> "AesGcmTokenCipher":
        if not encoded:
            raise TokenCipherError(
                "GMAIL_TOKEN_ENCRYPTION_KEY is not set. Generate one with: "
                "python -c \"import base64,os;print(base64.b64encode(os.urandom(32)).decode())\""
            )
        try:
            key = base64.b64decode(encoded, validate=True)
        except Exception as exc:  # noqa: BLE001 - never echo the value back
            raise TokenCipherError("GMAIL_TOKEN_ENCRYPTION_KEY is not valid base64.") from exc
        return cls(key, key_id=key_id)

    def _aead(self):
        # Lazy: keeps `cryptography` out of any import path the memory backend
        # walks, and out of installs that never connect a mailbox.
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise TokenCipherError(
                "Refresh-token encryption requires the 'gmail' extra: pip install -e '.[gmail]'"
            ) from exc
        return AESGCM(self._key)

    def encrypt(self, plaintext: str) -> bytes:
        nonce = os.urandom(self.NONCE_BYTES)
        return nonce + self._aead().encrypt(nonce, plaintext.encode("utf-8"), None)

    def decrypt(self, ciphertext: bytes) -> str:
        if len(ciphertext) <= self.NONCE_BYTES:
            raise TokenCipherError("Stored refresh token ciphertext is malformed.")
        nonce, body = ciphertext[: self.NONCE_BYTES], ciphertext[self.NONCE_BYTES :]
        try:
            return self._aead().decrypt(nonce, body, None).decode("utf-8")
        except TokenCipherError:
            raise
        except Exception as exc:  # noqa: BLE001 - authentication failure detail is not loggable
            raise TokenCipherError("Refresh token could not be decrypted (wrong key or tampered data).") from exc


def build_token_cipher(settings) -> TokenCipher:
    return AesGcmTokenCipher.from_base64_key(
        settings.gmail_token_encryption_key, key_id=settings.gmail_token_key_id
    )
