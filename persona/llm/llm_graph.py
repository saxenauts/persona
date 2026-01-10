import json
from typing import Dict, Any

from persona.llm.prompts import GENERATE_STRUCTURED_INSIGHTS
from persona.models.schema import AskRequest
from persona.llm.client_factory import get_chat_client
from persona.llm.providers.base import ChatMessage
from server.logging_config import get_logger

logger = get_logger(__name__)


async def generate_structured_insights(
    ask_request: AskRequest, context: str
) -> Dict[str, Any]:
    prompt = f"""
    Based on this context from the knowledge graph:
    {context}

    Answer this query about the user: {ask_request.query}

    Provide your response following the example structure:
    {json.dumps(ask_request.output_schema, indent=2)}
    """

    logger.debug(f"Structured insights prompt: {prompt}")

    try:
        messages = [
            ChatMessage(role="system", content=GENERATE_STRUCTURED_INSIGHTS),
            ChatMessage(role="user", content=prompt),
        ]

        client = get_chat_client()
        response = await client.chat(
            messages=messages, response_format={"type": "json_object"}
        )

        if not response.content:
            raise ValueError("Empty response content")
        return json.loads(response.content)

    except Exception as e:
        logger.error(f"Error in generate_structured_insights: {e}")
        return {
            k: [] if isinstance(v, list) else {}
            for k, v in ask_request.output_schema.items()
        }
