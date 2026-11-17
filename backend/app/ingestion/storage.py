"""
Object storage for ORIGINAL uploaded files, separate from the database (which keeps metadata, extracted
text, chunks and vectors). Keys are generated server-side; client filenames never reach a path.

  local  a folder on disk (default; fine locally, EPHEMERAL on Render's free tier: originals vanish on
         redeploy, but extracted text and chunks are in the database so search keeps working)
  s3     any S3-compatible bucket (AWS S3, Cloudflare R2, Backblaze B2). Needs `pip install boto3`.
"""
import os
import re
from functools import lru_cache
from typing import Protocol

from backend.app.config import settings


class Storage(Protocol):
    name: str

    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes: ...
    def delete(self, key: str) -> None: ...
    def exists(self, key: str) -> bool: ...


class LocalStorage:
    name = "local"

    def __init__(self, root: str):
        self.root = os.path.abspath(root)

    def _path(self, key: str) -> str:
        path = os.path.abspath(os.path.join(self.root, key))
        if not path.startswith(self.root + os.sep):
            raise ValueError("Invalid storage key")  # defence in depth: keys are server-generated anyway
        return path

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)

    def get(self, key: str) -> bytes:
        with open(self._path(key), "rb") as f:
            return f.read()

    def delete(self, key: str) -> None:
        try:
            os.remove(self._path(key))
        except FileNotFoundError:
            pass

    def exists(self, key: str) -> bool:
        return os.path.exists(self._path(key))


class S3Storage:
    name = "s3"

    def __init__(self, bucket: str, endpoint_url: str = "", prefix: str = ""):
        import boto3  # optional dependency

        self.bucket, self.prefix = bucket, prefix
        self.client = boto3.client("s3", endpoint_url=endpoint_url or None)

    def put(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self.prefix + key, Body=data, ServerSideEncryption="AES256")

    def get(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=self.prefix + key)["Body"].read()

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=self.prefix + key)

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self.prefix + key)
            return True
        except Exception:
            return False


@lru_cache(maxsize=1)
def get_storage() -> Storage:
    inner = _raw_storage()
    if settings.encryption_key:  # originals encrypted at rest (Fernet); objects written before stay readable
        from backend.app.ops.security import EncryptedStorage

        return EncryptedStorage(inner)
    return inner


def _raw_storage() -> Storage:
    if settings.storage_backend == "s3":
        if not settings.s3_bucket:
            raise RuntimeError("STORAGE_BACKEND=s3 requires S3_BUCKET")
        return S3Storage(settings.s3_bucket, settings.s3_endpoint_url, settings.s3_prefix)
    return LocalStorage(settings.storage_path)


def _safe(part: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", part)  # ':' etc. are invalid on Windows and awkward in S3 consoles


def storage_key(patient_id: str, document_id: str, version: int, checksum: str) -> str:
    return f"{_safe(patient_id)}/{_safe(document_id)}/v{version}-{checksum[:16]}"
