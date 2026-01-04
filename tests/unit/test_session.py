"""Tests for session ID utilities and transcript handling."""

import pytest
from persona.utils.session import (
    get_session_id,
    parse_session_id,
    is_transcript_source_type,
)


class TestGetSessionId:
    def test_with_provider_session_id(self):
        result = get_session_id("claude", "conv_xyz")
        assert result == "claude:conv_xyz"

    def test_with_different_providers(self):
        assert get_session_id("chatgpt", "abc123") == "chatgpt:abc123"
        assert get_session_id("slack", "C123_12345") == "slack:C123_12345"
        assert get_session_id("persona", "user-session") == "persona:user-session"

    def test_without_provider_session_id_generates_uuid(self):
        result = get_session_id("persona")
        assert result.startswith("persona:")
        assert len(result) > len("persona:")

    def test_different_calls_generate_different_uuids(self):
        result1 = get_session_id("persona")
        result2 = get_session_id("persona")
        assert result1 != result2

    def test_empty_provider_session_id_generates_uuid(self):
        result = get_session_id("claude", None)
        assert result.startswith("claude:")


class TestParseSessionId:
    def test_parse_standard_format(self):
        provider, session_id = parse_session_id("claude:conv_xyz")
        assert provider == "claude"
        assert session_id == "conv_xyz"

    def test_parse_with_colons_in_id(self):
        provider, session_id = parse_session_id("slack:C123:thread:12345")
        assert provider == "slack"
        assert session_id == "C123:thread:12345"

    def test_parse_without_colon(self):
        provider, session_id = parse_session_id("legacy_session_id")
        assert provider == "unknown"
        assert session_id == "legacy_session_id"


class TestIsTranscriptSourceType:
    def test_transcript_source_type(self):
        assert is_transcript_source_type("transcript") is True

    def test_non_transcript_source_types(self):
        assert is_transcript_source_type("conversation") is False
        assert is_transcript_source_type("claude") is False
        assert is_transcript_source_type("persona") is False
        assert is_transcript_source_type("") is False
