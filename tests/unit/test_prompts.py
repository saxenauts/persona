import pytest
from persona.llm.prompts import PERSONAL_AI_SYSTEM_PROMPT, GENERATE_STRUCTURED_INSIGHTS


class TestPersonalAIPrompt:
    """
    Tests for PERSONAL_AI_SYSTEM_PROMPT.

    The prompt follows a minimal design philosophy (Jan 2026):
    - Trust capable models (GPT-5.2), don't over-specify
    - Tool details live in tool schemas, not in the system prompt
    - Keep only: role, clock, evidence contract, tool reminder
    """

    def test_contains_personal_ai_framing(self):
        assert "Personal AI" in PERSONAL_AI_SYSTEM_PROMPT

    def test_contains_core_tool_references(self):
        # The minimal prompt mentions tools by category, not individual names
        # Tool details are in tool schemas, not duplicated in prompt
        assert "recall" in PERSONAL_AI_SYSTEM_PROMPT  # Part of evidence priority list
        assert "tool" in PERSONAL_AI_SYSTEM_PROMPT.lower()  # tool_use, tool-retrieved

    def test_contains_user_context_placeholder(self):
        assert "{user_context}" in PERSONAL_AI_SYSTEM_PROMPT

    def test_formats_with_all_placeholders(self):
        # The minimal prompt has 2 placeholders: world_model, user_context
        test_context = "## Who They Are\n- Test user identity"
        formatted = PERSONAL_AI_SYSTEM_PROMPT.format(
            user_context=test_context,
            world_model="",
        )
        assert test_context in formatted
        assert "{user_context}" not in formatted
        assert "{world_model}" not in formatted

    def test_formats_with_empty_context(self):
        formatted = PERSONAL_AI_SYSTEM_PROMPT.format(
            user_context="",
            world_model="",
        )
        assert "Personal AI" in formatted

    def test_contains_evidence_based_answering(self):
        # The new prompt emphasizes evidence-based answers
        assert "evidence" in PERSONAL_AI_SYSTEM_PROMPT.lower()

    def test_contains_tool_use_section(self):
        # Tools are listed in a dedicated section
        assert "tool_use" in PERSONAL_AI_SYSTEM_PROMPT.lower()
        assert "recall" in PERSONAL_AI_SYSTEM_PROMPT
        assert "browse" in PERSONAL_AI_SYSTEM_PROMPT


class TestStructuredInsightsPrompt:
    def test_exists_and_not_empty(self):
        assert GENERATE_STRUCTURED_INSIGHTS
        assert len(GENERATE_STRUCTURED_INSIGHTS) > 50
