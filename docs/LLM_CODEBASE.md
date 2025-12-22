# LLM Codebase Overview

This document describes the LLM integration modules under `persona/llm/`.

## Architecture

```
llm/
├── client_factory.py      # Provider-agnostic client creation
├── embeddings.py          # Embedding generation
├── llm_graph.py           # LLM-powered functions
├── prompts.py             # Prompt templates
└── providers/
    ├── base.py            # Abstract base class
    ├── openai_client.py   # OpenAI implementation
    ├── azure_openai_client.py
    ├── anthropic_client.py
    └── gemini_client.py
```

## Client Factory

The factory creates LLM clients based on environment configuration:

```python
from persona.llm.client_factory import get_chat_client, get_embedding_client

# Get clients (cached singletons)
chat_client = get_chat_client()
embedding_client = get_embedding_client()

# Use for chat
response = await chat_client.chat(messages=[
    ChatMessage(role="user", content="Hello")
])
print(response.content)

# Use for embeddings
embeddings = await embedding_client.embeddings(["text to embed"])
```

## Configuration

Set via environment variables:

```env
# Format: provider/model
LLM_SERVICE=openai/gpt-4o-mini
EMBEDDING_SERVICE=openai/text-embedding-3-small

# Provider-specific keys
OPENAI_API_KEY=sk-...
AZURE_API_KEY=...
ANTHROPIC_API_KEY=...
```

## Core Functions

### `generate_response_with_context`

Generates RAG responses using retrieved context:

```python
from persona.llm.llm_graph import generate_response_with_context

response = await generate_response_with_context(
    query="What projects am I working on?",
    context="<memory_context>...</memory_context>"
)
```

### `generate_structured_insights`

Generates JSON-structured answers matching a schema:

```python
from persona.llm.llm_graph import generate_structured_insights
from persona.models.schema import AskRequest

request = AskRequest(
    query="What are my preferences?",
    output_schema={"preferences": ["example"], "summary": "string"}
)

result = await generate_structured_insights(request, context)
# Returns: {"preferences": ["remote work"], "summary": "..."}
```

## Embeddings

```python
from persona.llm.embeddings import generate_embeddings_async

# Returns list of 1536-dimensional vectors
embeddings = await generate_embeddings_async(["text 1", "text 2"])
```

## Adding a New Provider

1. Create `providers/new_provider_client.py`
2. Extend `BaseLLMClient`
3. Implement `chat()` and `embeddings()` methods
4. Add to `client_factory.py`
