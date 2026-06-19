"""
security.py
------------
Password hashing & policy enforcement.

We intentionally use Python's built-in `hashlib.pbkdf2_hmac` (PBKDF2-HMAC-SHA256)
instead of an extra third-party dependency such as bcrypt/passlib. This keeps
the security primitive in the standard library while still being a slow,
salted, industry-accepted KDF suitable for password storage.
"""

import hashlib
import hmac
import os
import re
import secrets

PBKDF2_ITERATIONS = 260_000
SALT_BYTES = 16


def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    """Return (hash_hex, salt_hex) for a plaintext password."""
    if salt is None:
        salt = secrets.token_bytes(SALT_BYTES)
    derived = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return derived.hex(), salt.hex()


def verify_password(password: str, hash_hex: str, salt_hex: str) -> bool:
    """Constant-time comparison of a plaintext password against a stored hash."""
    salt = bytes.fromhex(salt_hex)
    derived = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return hmac.compare_digest(derived.hex(), hash_hex)


def validate_password_policy(password: str, policy: dict) -> list[str]:
    """
    Validate a password against a policy dict (see config.DEFAULT_PASSWORD_POLICY).
    Returns a list of human-readable violation messages (empty list = valid).
    """
    errors = []
    min_len = policy.get("min_length", 8)
    if len(password) < min_len:
        errors.append(f"Password must be at least {min_len} characters long.")
    if policy.get("require_upper", True) and not re.search(r"[A-Z]", password):
        errors.append("Password must contain at least one uppercase letter.")
    if policy.get("require_lower", True) and not re.search(r"[a-z]", password):
        errors.append("Password must contain at least one lowercase letter.")
    if policy.get("require_number", True) and not re.search(r"[0-9]", password):
        errors.append("Password must contain at least one number.")
    if policy.get("require_special", True) and not re.search(
        r"[!@#$%^&*()\-_=+\[\]{};:,.<>/?|~`]", password
    ):
        errors.append("Password must contain at least one special character.")
    return errors


def is_valid_email(email: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email.strip()))
