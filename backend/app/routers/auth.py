from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.auth import (
    clear_login_failures,
    create_access_token,
    get_current_user,
    log_access,
    login_locked,
    permissions_for,
    record_login_failure,
    verify_password,
)
from backend.app.config import settings
from backend.app.database import get_db
from backend.app.models import User
from backend.app.schemas import LoginRequest, TokenResponse, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])

# Verified against when the username does not exist, so response time does not reveal valid usernames.
_DUMMY_HASH = "scrypt$16384$8$1$AAAAAAAAAAAAAAAAAAAAAA==$AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="


def _user_out(user: User) -> UserOut:
    return UserOut(
        username=user.username, display_name=user.display_name, role=user.role,  # type: ignore[arg-type]
        facility_id=user.facility_id, permissions=permissions_for(user.role),
    )


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    if login_locked(req.username):
        log_access(db, None, "LOGIN_LOCKED", allowed=False, username=req.username)
        db.commit()
        raise HTTPException(status_code=429, detail=f"Too many failed attempts. Try again in {settings.login_lockout_minutes} minutes.")
    user = db.query(User).filter(User.username == req.username).first()
    ok = verify_password(req.password, user.password_hash if user else _DUMMY_HASH)
    if not (user and user.is_active and ok):
        record_login_failure(req.username)
        log_access(db, None, "LOGIN_FAILED", allowed=False, username=req.username)
        db.commit()
        raise HTTPException(status_code=401, detail="Incorrect username or password.")
    clear_login_failures(req.username)
    log_access(db, user, "LOGIN")
    db.commit()
    token, expires = create_access_token(user)
    return TokenResponse(access_token=token, expires_in=expires, user=_user_out(user))


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return _user_out(user)
