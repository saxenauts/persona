import pytest
from persona.llm.prompts import PERSONAL_AI_SYSTEM_PROMPT, GENERATE_STRUCTURED_INSIGHTS


class TestPersonalAIPrompt:
    def test_contains_personal_ai_framing(self):
        assert "Personal AI" in PERSONAL_AI_SYSTEM_PROMPT

    def test_contains_all_tool_names(self):
        assert "recall" in PERSONAL_AI_SYSTEM_PROMPT
        assert "record" in PERSONAL_AI_SYSTEM_PROMPT
        assert "expand_neighbors" in PERSONAL_AI_SYSTEM_PROMPT
        assert "follow_relationship" in PERSONAL_AI_SYSTEM_PROMPT

    def test_contains_user_context_placeholder(self):
        assert "{user_context}" in PERSONAL_AI_SYSTEM_PROMPT

    def test_formats_with_user_context(self):
        test_context = "## Who They Are\n- Test user identity"
        formatted = PERSONAL_AI_SYSTEM_PROMPT.format(
            user_context=test_context, world_model=""
        )
        assert test_context in formatted
        assert "{user_context}" not in formatted
        assert "{world_model}" not in formatted

    def test_formats_with_empty_context(self):
        formatted = PERSONAL_AI_SYSTEM_PROMPT.format(user_context="", world_model="")
        assert "Personal AI" in formatted

    def test_contains_context_edge_messaging(self):
        assert "context" in PERSONAL_AI_SYSTEM_PROMPT.lower()
        assert "edge" in PERSONAL_AI_SYSTEM_PROMPT.lower()

    def test_contains_recording_guidance(self):
        assert "record" in PERSONAL_AI_SYSTEM_PROMPT.lower()
        assert (
            "remind" in PERSONAL_AI_SYSTEM_PROMPT.lower()
            or "task" in PERSONAL_AI_SYSTEM_PROMPT.lower()
        )


class TestStructuredInsightsPrompt:
    def test_exists_and_not_empty(self):
        assert GENERATE_STRUCTURED_INSIGHTS
        assert len(GENERATE_STRUCTURED_INSIGHTS) > 50
