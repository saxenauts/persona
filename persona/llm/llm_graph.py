"""
LLM functions for persona operations.

These are the core LLM-powered functions used by the Persona system:
- generate_response_with_context: For RAG queries
- generate_structured_insights: For structured Ask queries
"""

import json
from typing import Dict, Any, Tuple
from persona.llm.prompts import GENERATE_STRUCTURED_INSIGHTS
from persona.models.schema import AskRequest
from persona.llm.client_factory import get_chat_client
from persona.llm.providers.base import ChatMessage
from server.logging_config import get_logger

logger = get_logger(__name__)


def _build_rag_messages(query: str, context: str) -> list[ChatMessage]:
    prompt = f"""
    Given the following context from a knowledge graph and a query, provide a detailed answer:

    Context:
    {context}

    Query: {query}

    Please provide a comprehensive answer based on the given context:
    """

    return [
        ChatMessage(
            role="system",
            content="You are a helpful assistant that answers queries about a user based on the provided context from their graph."
        ),
        ChatMessage(role="user", content=prompt)
    ]


async def generate_response_with_context(query: str, context: str) -> str:
    """Generate a response based on query and context using the configured LLM service."""
    try:
        messages = _build_rag_messages(query, context)
        client = get_chat_client()
        response = await client.chat(messages=messages, temperature=0.7)
        return response.content
    except Exception as e:
        logger.error(f"Error generating response with context: {e}")
        return "I apologize, but I encountered an error while processing your request."


async def generate_response_with_context_with_stats(
    query: str, context: str
) -> Tuple[str, Dict[str, Any]]:
    """Generate a response and return model usage stats for logging."""
    try:
        messages = _build_rag_messages(query, context)
        client = get_chat_client()
        response = await client.chat(messages=messages, temperature=0.7)
        usage = response.usage or {}
        stats = {
            "model": response.model,
            "usage": usage,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "temperature": 0.7
        }
        return response.content, stats
    except Exception as e:
        logger.error(f"Error generating response with context (stats): {e}")
        return (
            "I apologize, but I encountered an error while processing your request.",
            {"model": "error", "usage": {}, "prompt_tokens": None, "completion_tokens": None, "temperature": 0.7}
        )


async def generate_structured_insights(ask_request: AskRequest, context: str) -> Dict[str, Any]:
    """
    Generate structured insights based on the provided context and query using the configured LLM service
    """
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
            ChatMessage(role="user", content=prompt)
        ]
        
        client = get_chat_client()
        response = await client.chat(
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.content)
        
    except Exception as e:
        logger.error(f"Error in generate_structured_insights: {e}")
        return {k: [] if isinstance(v, list) else {} for k, v in ask_request.output_schema.items()}
