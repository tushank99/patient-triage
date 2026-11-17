"""
Authentication (bearer JWT), roles and per-patient authorization.

Roles and what they may do are defined in ONE table (PERMISSIONS). Patient-level data (names, history
documents, vitals trends, audit trails) additionally requires `patient:read` AND that the visit belongs
to the user's facility; every such access, allowed or denied, is written to access_logs.
Passwords: scrypt (stdlib). Tokens: HS256, short-lived, secret from JWT_SECRET (see docs/SECURITY.md).
"""
import base64
import hashlib
import hmac
import secrets as pysecrets
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.database import get_db
from backend.app.models import AccessLog, Patient, User, Visit

CLINICAL = ("triage_nurse", "charge_nurse", "physician")

PERMISSIONS: Dict[str, tuple] = {
    "queue:read": ("triage_nurse", "charge_nurse", "physician", "admin", "viewer"),  # viewers see the queue without names
    "patient:read": CLINICAL,
    "triage:write": CLINICAL,
    "override:write": CLINICAL,
    "override:below_floor": ("charge_nurse", "physician"),  # lowering ESI under a safety-rule floor
    "archive:read": CLINICAL + ("admin",),  # admin sees MRNs (to anonymise) but never names
    "patient:anonymize": ("admin",),
    "demo:surge": ("admin", "charge_nurse"),
    "document:write": CLINICAL,
    "document:delete": ("charge_nurse", "physician", "admin"),  # deleting part of a medical record is restricted
    "ingestion:admin": ("charge_nurse", "admin"),  # job monitoring, retry, re-index
}


def permissions_for(role: str) -> List[str]:
    return sorted(p for p, roles in PERMISSIONS.items() if role in roles)


# ---------------------------------------------------------------- passwords

_N, _R, _P = 2**14, 8, 1


def hash_password(password: str) -> str:
    salt = pysecrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return f"scrypt${_N}${_R}${_P}${base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt_b64, digest_b64 = stored.split("$")
        expected = base64.b64decode(digest_b64)
        actual = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt_b64), n=int(n), r=int(r), p=int(p), dklen=len(expected))
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


# ---------------------------------------------------------------- tokens


def create_access_token(user: User) -> tuple[str, int]:
    expires = settings.jwt_expire_minutes * 60
    now = datetime.now(timezone.utc)
    payload = {"sub": user.username, "role": user.role, "fac": user.facility_id, "iat": now, "exp": now + timedelta(seconds=expires)}
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256"), expires


def _unauthorized(detail: str = "Not authenticated.") -> HTTPException:
    return HTTPException(status_code=401, detail=detail, headers={"WWW-Authenticate": "Bearer"})


_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer), db: Session = Depends(get_db)
) -> User:
    if creds is None:
        raise _unauthorized()
    try:
        claims = jwt.decode(creds.credentials, settings.jwt_secret, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise _unauthorized("Session expired. Please sign in again.")
    except jwt.PyJWTError:
        raise _unauthorized("Invalid token.")
    user = db.query(User).filter(User.username == claims.get("sub")).first()
    if user is None or not user.is_active:
        raise _unauthorized("Account is disabled.")
    return user  # role and facility come from the DB, not the token, so revocation is immediate


def require(permission: str) -> Callable[..., User]:
    def dep(request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> User:
        if user.role not in PERMISSIONS[permission]:
            log_access(db, user, f"DENIED:{permission}", allowed=False, detail=f"{request.method} {request.url.path}")
            db.commit()
            raise HTTPException(status_code=403, detail=f"Your role ({user.role}) is not permitted to do this.")
        return user

    return dep


# ---------------------------------------------------------------- per-patient authorization & access log


def log_access(
    db: Session, user: Optional[User], action: str, *, patient: Optional[Patient] = None,
    visit_id: Optional[str] = None, allowed: bool = True, detail: Optional[str] = None, username: Optional[str] = None,
) -> None:
    db.add(AccessLog(
        username=user.username if user else (username or "anonymous"), role=user.role if user else None, action=action,
        patient_id=patient.id if patient else None, visit_id=visit_id, allowed=allowed, detail=detail,
    ))


def authorize_visit(db: Session, user: User, visit: Optional[Visit], action: str) -> Visit:
    """Per-patient policy for anything that touches one visit: clinical role (checked by `require`)
    plus facility scope. Other facilities' visits look like they do not exist (404, not 403)."""
    if visit is None:
        raise HTTPException(status_code=404, detail="Visit not found.")
    if visit.facility_id != user.facility_id:
        log_access(db, user, f"DENIED:{action}", patient=visit.patient, visit_id=visit.id, allowed=False, detail="facility mismatch")
        db.commit()
        raise HTTPException(status_code=404, detail="Visit not found.")
    log_access(db, user, action, patient=visit.patient, visit_id=visit.id)
    return visit


def authorize_patient(db: Session, user: User, patient: Optional[Patient], action: str) -> Patient:
    """Lookup of a patient record by MRN (clinical roles only; anonymised records are gone)."""
    if patient is None or patient.is_anonymized:
        log_access(db, user, f"{action}:NOT_FOUND", allowed=False)
        db.commit()
        raise HTTPException(status_code=404, detail="No such patient.")
    log_access(db, user, action, patient=patient)
    return patient


# ---------------------------------------------------------------- login throttle (per username, in-memory)

_failures: Dict[str, List[float]] = {}


def login_locked(username: str) -> bool:
    window = settings.login_lockout_minutes * 60
    recent = [t for t in _failures.get(username, []) if time.time() - t < window]
    _failures[username] = recent
    return len(recent) >= settings.login_max_failures


def record_login_failure(username: str) -> None:
    _failures.setdefault(username, []).append(time.time())


def clear_login_failures(username: str) -> None:
    _failures.pop(username, None)
