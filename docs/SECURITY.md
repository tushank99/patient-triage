# Security model (Phase 1)

## Authentication
* `POST /api/auth/login` returns a short-lived (30 min) HS256 JWT. The frontend keeps it in
  `sessionStorage` (cleared on tab close) and sends it as `Authorization: Bearer`. No cookies, so no CSRF surface.
* Passwords are hashed with scrypt (stdlib). Failed logins lock a username for 15 minutes after 5 tries.
* Role and facility are read from the database on every request, so disabling a user takes effect immediately.

## Roles (`backend/app/auth.py: PERMISSIONS` is the single source of truth)
| Role | Queue | Patient data (names, history, vitals, audit) | Triage / override / disposition | Override *below* a safety-rule floor | Anonymise | Demo surge |
|---|---|---|---|---|---|---|
| triage_nurse | yes | yes | yes | no | no | no |
| charge_nurse | yes | yes | yes | **yes** | no | yes |
| physician | yes | yes | yes | **yes** | no | no |
| admin | yes (no names) | **no** | no | no | **yes** | yes |
| viewer | yes (no names) | no | no | no | no | no |

## Per-patient authorization
Every endpoint touching one visit goes through `authorize_visit`: clinical role **and** the visit's
`facility_id` must equal the user's. Another facility's visit returns 404 (existence is not revealed).
Every patient-level access, allowed or denied, is written to `access_logs` (who, role, action, patient, visit).

## Secrets
* Nothing secret is committed. `backend/.env` is git-ignored; `.env.example` holds placeholders only.
* Deployed values live in **Render environment variables** (mark them secret): `JWT_SECRET`,
  `DATABASE_URL`, `OPENAI_API_KEY`, `DEMO_USER_PASSWORD`.
* `JWT_SECRET` must be >= 32 random characters in staging/production; the app **refuses to start** otherwise.
  Generate one: `python -c "import secrets; print(secrets.token_urlsafe(48))"`.
* Rotation: change `JWT_SECRET` (signs out everyone), rotate the LLM key at the provider, rotate the Neon
  role password. In protected environments demo accounts are only created if `DEMO_USER_PASSWORD` is >= 12 chars.
* The frontend has **no** secrets: `VITE_API_URL` is public by design.

## Not done yet (tracked)
Rate limiting beyond login, refresh tokens/logout revocation, MFA, Postgres row-level security, encryption
at rest beyond what Neon provides, and a retention policy (Phase 6).
