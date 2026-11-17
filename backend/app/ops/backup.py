"""
Encrypted logical backups, rotation, restore and a restore DRILL (a backup that was never restored is not a backup).

  python -m backend.app.ops.backup create             # dump -> Fernet-encrypt -> BACKUP_DIR (and storage, if S3) -> rotate
  python -m backend.app.ops.backup list
  python -m backend.app.ops.backup verify [file]      # restore into a scratch database and compare row counts
  python -m backend.app.ops.backup restore FILE --target-url URL --yes

SQLite: online backup API. Postgres: pg_dump --format=custom / pg_restore (client tools must be installed; on Render use
the Neon point-in-time restore as the primary mechanism and this as the off-site, provider-independent copy).
Retention: BACKUP_KEEP newest files are kept. That number, times the backup interval, is how long DELETED patient data
can still exist in backups; document it in the privacy notice. Restoring an old backup can resurrect deleted
records, so after any restore re-run retention/erasure (`python -m backend.app.ops.retention`).
"""
import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from backend.app.config import settings
from backend.app.ops.security import decrypt, encrypt

TABLES = ["patients", "visits", "audit_logs", "access_logs", "users", "clinical_documents", "document_versions",
          "document_chunks", "ingestion_jobs", "intake_captures", "vitals_readings"]


def _sqlite_path(url: str) -> str:
    return url.split("sqlite:///", 1)[1]


def dump_bytes(url: Optional[str] = None) -> bytes:
    url = url or settings.database_url
    if url.startswith("sqlite"):
        src = sqlite3.connect(_sqlite_path(url))
        tmp = tempfile.mktemp(suffix=".db")
        dst = sqlite3.connect(tmp)
        try:
            src.backup(dst)  # consistent online copy
        finally:
            dst.close()
            src.close()
        data = Path(tmp).read_bytes()
        os.remove(tmp)
        return data
    if not shutil.which("pg_dump"):
        raise RuntimeError("pg_dump not found; install the PostgreSQL client tools or rely on provider PITR.")
    return subprocess.run(["pg_dump", "--format=custom", "--no-owner", url], check=True, capture_output=True, timeout=3600).stdout


def row_counts(url: str) -> Dict[str, int]:
    from sqlalchemy import create_engine, inspect, text

    eng = create_engine(url)
    try:
        with eng.connect() as c:
            present = set(inspect(c).get_table_names())
            return {t: c.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar() for t in TABLES if t in present}
    finally:
        eng.dispose()


def create_backup(url: Optional[str] = None, out_dir: Optional[str] = None) -> Dict:
    url, out_dir = url or settings.database_url, out_dir or settings.backup_dir
    raw = dump_bytes(url)
    kind = "sqlite" if url.startswith("sqlite") else "pgdump"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    payload = encrypt(raw)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"triage-{stamp}.{kind}.bak")
    Path(path).write_bytes(payload)
    manifest = {"file": os.path.basename(path), "created_at": stamp, "kind": kind, "encrypted": payload != raw,
                "sha256_plain": hashlib.sha256(raw).hexdigest(), "bytes": len(payload), "row_counts": row_counts(url)}
    Path(path + ".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if settings.storage_backend == "s3":  # off-box copy
        from backend.app.ingestion.storage import _raw_storage

        _raw_storage().put(f"backups/{os.path.basename(path)}", payload)  # already ciphertext
    rotate(out_dir)
    return manifest


def list_backups(out_dir: Optional[str] = None) -> List[str]:
    out_dir = out_dir or settings.backup_dir
    if not os.path.isdir(out_dir):
        return []
    return sorted(f for f in os.listdir(out_dir) if f.endswith(".bak"))


def rotate(out_dir: str) -> List[str]:
    files = list_backups(out_dir)
    removed = files[: max(0, len(files) - settings.backup_keep)]
    for f in removed:
        for p in (os.path.join(out_dir, f), os.path.join(out_dir, f + ".json")):
            if os.path.exists(p):
                os.remove(p)
    return removed


def restore(path: str, target_url: str) -> None:
    raw = decrypt(Path(path).read_bytes())
    if target_url.startswith("sqlite"):
        Path(_sqlite_path(target_url)).write_bytes(raw)
        return
    if not shutil.which("pg_restore"):
        raise RuntimeError("pg_restore not found.")
    subprocess.run(["pg_restore", "--clean", "--if-exists", "--no-owner", "--dbname", target_url], input=raw, check=True, timeout=3600)


def verify(path: Optional[str] = None, out_dir: Optional[str] = None) -> Dict:
    """Restore drill: decrypt, restore into a scratch SQLite database, compare row counts with the manifest."""
    out_dir = out_dir or settings.backup_dir
    path = path or os.path.join(out_dir, list_backups(out_dir)[-1])
    manifest = json.loads(Path(path + ".json").read_text(encoding="utf-8"))
    raw = decrypt(Path(path).read_bytes())
    if hashlib.sha256(raw).hexdigest() != manifest["sha256_plain"]:
        return {"ok": False, "reason": "checksum mismatch"}
    if manifest["kind"] != "sqlite":
        return {"ok": True, "note": "checksum verified; restore pg dumps into a scratch Postgres with `restore --target-url`"}
    scratch = tempfile.mktemp(suffix=".db")
    restore(path, f"sqlite:///{scratch}")
    counts = row_counts(f"sqlite:///{scratch}")
    os.remove(scratch)
    return {"ok": counts == manifest["row_counts"], "restored_counts": counts, "manifest_counts": manifest["row_counts"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["create", "list", "verify", "restore"])
    ap.add_argument("file", nargs="?")
    ap.add_argument("--target-url")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    if a.command == "create":
        print(json.dumps(create_backup(), indent=2))
    elif a.command == "list":
        print("\n".join(list_backups()))
    elif a.command == "verify":
        r = verify(a.file)
        print(json.dumps(r, indent=2))
        sys.exit(0 if r["ok"] else 1)
    else:
        if not (a.file and a.target_url and a.yes):
            sys.exit("restore needs FILE --target-url URL --yes (it overwrites the target database)")
        restore(a.file, a.target_url)
        print("restored; now re-run erasure/retention so deleted records stay deleted: python -m backend.app.ops.retention")
