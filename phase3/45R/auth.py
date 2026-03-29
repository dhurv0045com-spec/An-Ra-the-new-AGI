"""
sovereignty/auth.py
===================
API bearer-token generation and validation.

The token is a 32-byte (64-character hex) secret generated once at install
time and stored in token.key. Every authenticated API request must supply it
as:  Authorization: Bearer <token>

No token is ever logged. Invalid tokens return 401 with no body (no info leak).

Relationship to other modules:
    api.py calls validate_token() on every authenticated endpoint.
    install.py calls generate_token() during the install sequence.
"""

import hmac
import os
import pathlib
import secrets
import stat
import sys

from sovereignty.config import Config
from sovereignty.logger import get_logger

log = get_logger(__name__)


def generate_token(config: Config) -> str:
    """
    Generate a cryptographically strong bearer token and write it to disk.

    The token file is created with restricted permissions (owner read-only on
    POSIX; on Windows the caller should set ACLs after this call).

    Parameters:
        config: Active Config instance (provides TOKEN_FILE, TOKEN_BYTES).

    Returns:
        The hex-encoded token string (64 characters for 32 bytes).

    Raises:
        OSError: If the token file cannot be written.
    """
    token = secrets.token_hex(config.TOKEN_BYTES)
    token_path = config.TOKEN_FILE
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(token, encoding="utf-8")

    # Restrict permissions on POSIX systems
    if sys.platform != "win32":
        try:
            os.chmod(token_path, stat.S_IRUSR | stat.S_IWUSR)
        except Exception:
            pass  # Best-effort; Windows handles this differently

    log.info("Bearer token generated and written to %s", token_path)
    return token


def load_token(config: Config) -> str:
    """
    Load the bearer token from disk.

    Parameters:
        config: Active Config instance (provides TOKEN_FILE).

    Returns:
        The token string.

    Raises:
        FileNotFoundError: If token.key does not exist (run install first).
        ValueError: If the token file is empty or malformed.
    """
    token_path = config.TOKEN_FILE
    if not token_path.exists():
        raise FileNotFoundError(
            f"Token file not found: {token_path}. Run install.py first."
        )
    token = token_path.read_text(encoding="utf-8").strip()
    if not token or len(token) < 16:
        raise ValueError(f"Token file is empty or too short: {token_path}")
    return token


def validate_token(provided: str, config: Config) -> bool:
    """
    Validate a bearer token from an API request using constant-time comparison.

    Uses hmac.compare_digest to prevent timing attacks — an attacker cannot
    measure the response time to learn how many characters matched.

    Parameters:
        provided: The token string from the Authorization header (may be empty).
        config: Active Config instance (provides TOKEN_FILE).

    Returns:
        True if the token matches, False otherwise.
    """
    try:
        expected = load_token(config)
        return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))
    except Exception:
        # Any error (missing file, bad encoding) → reject
        return False


def extract_bearer(authorization_header: str) -> str:
    """
    Extract the token from an Authorization header value.

    Parameters:
        authorization_header: Raw header value, e.g. "Bearer abc123".

    Returns:
        The token string, or empty string if the header is malformed.
    """
    if not authorization_header:
        return ""
    parts = authorization_header.strip().split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return ""
    return parts[1].strip()
