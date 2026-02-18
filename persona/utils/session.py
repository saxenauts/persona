"""
Session utilities for Persona.

Provides canonical session ID generation and validation across all providers.
"""

from uuid import uuid4
from typing import Optional


def get_session_id(provider: str, provider_session_id: Optional[str] = None) -> str:
    """
    Create a canonical session ID from any source.

    Format: "{provider}:{id}" ensures uniqueness across providers.

    Args:
        provider: Source system ("persona", "claude", "chatgpt", "slack", etc.)
        provider_session_id: The ID from that system (optional - generates UUID if not provided)

    Returns:
        Canonical ID like "claude:conv_xyz" or "persona:550e8400-e29b-..."

    Examples:
        >>> get_session_id("claude", "conv_xyz")
        'claude:conv_xyz'
        >>> get_session_id("persona")  # doctest: +ELLIPSIS
        'persona:...'
        >>> get_session_id("slack", "C123_12345")
        'slack:C123_12345'
    """
    if provider_session_id:
        return f"{provider}:{provider_session_id}"
    return f"{provider}:{str(uuid4())}"


def parse_session_id(session_id: str) -> tuple[str, str]:
    """
    Parse a canonical session ID back into provider and ID.

    Args:
        session_id: Canonical session ID like "claude:conv_xyz"

    Returns:
        Tuple of (provider, provider_session_id)

    Examples:
        >>> parse_session_id("claude:conv_xyz")
        ('claude', 'conv_xyz')
        >>> parse_session_id("slack:C123_12345")
        ('slack', 'C123_12345')
    """
    if ":" not in session_id:
        return ("unknown", session_id)

    parts = session_id.split(":", 1)
    return (parts[0], parts[1])


def is_transcript_source_type(source_type: str) -> bool:
    """Check if a source_type indicates a raw transcript."""
    return source_type == "transcript"
