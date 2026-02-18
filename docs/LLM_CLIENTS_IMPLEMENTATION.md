# LLM Clients Implementation

## Overview

This document describes the implementation of the new multi-provider LLM client system in Persona, which replaces the previous OpenAI-only implementation with a flexible abstraction layer supporting multiple LLM providers.

## Architecture

### Core Components

1. **Base Interface** (`persona/llm/providers/base.py`)
   - `BaseLLMClient`: Abstract base class defining the interface all providers must implement
   - `ChatMessage` and `ChatResponse`: Standard message formats for consistency
   - Support for chat completions, embeddings, and provider capabilities

2. **Provider Implementations**
   - `OpenAIClient`: Native OpenAI API integration
   - `AzureOpenAIClient`: Azure OpenAI service integration
   - `AnthropicClient`: Anthropic Claude integration
   - `GeminiClient`: Google Gemini integration

3. **Client Factory** (`persona/llm/client_factory.py`)
   - Centralized client creation and management
   - Configuration-based provider selection
   - Client caching and lifecycle management
   - Automatic fallback for providers without embedding support

4. **Refactored Core Modules**
   - `persona/llm/llm_graph.py`: Updated to use the new client system
   - `persona/llm/embeddings.py`: Abstracted to work with any provider
   - Removed dependency on `instructor` library

## Configuration

### Environment Variables

The system uses the following configuration format:

```bash
# LLM Service Configuration
LLM_SERVICE=azure/gpt-4o-mini  # Format: provider/model

# OpenAI Configuration
OPENAI_API_KEY=your_openai_key
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Azure OpenAI Configuration
AZURE_API_KEY=your_azure_key
AZURE_API_BASE=https://your-resource.openai.azure.com/
AZURE_API_VERSION=2024-02-01
AZURE_CHAT_DEPLOYMENT=gpt-4o-mini
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-small

# Anthropic Configuration
ANTHROPIC_API_KEY=your_anthropic_key
ANTHROPIC_CHAT_MODEL=claude-3-5-sonnet-20241022

# Google Gemini Configuration
GEMINI_API_KEY=your_gemini_key
GEMINI_CHAT_MODEL=gemini-1.5-flash
```

### Configuration Structure

The configuration is managed through `server/config.py` with the `MachineLearning` class containing all LLM-related settings.

## Provider Features

| Provider | Chat | Embeddings | JSON Mode | Notes |
|----------|------|------------|-----------|-------|
| OpenAI | ✅ | ✅ | ✅ | Full feature support |
| Azure OpenAI | ✅ | ✅ | ✅ | Requires deployment names |
| Anthropic | ✅ | ❌ | ✅ | No embeddings (falls back to OpenAI) |
| Google Gemini | ✅ | ❌ | ✅ | No embeddings (falls back to OpenAI) |

## Usage Examples

### Basic Chat

```python
from persona.llm.client_factory import get_chat_client
from persona.llm.providers.base import ChatMessage

client = get_chat_client()
messages = [
    ChatMessage(role="system", content="You are a helpful assistant"),
    ChatMessage(role="user", content="Hello!")
]

response = await client.chat(messages, temperature=0.7)
print(response.content)
```

### Embeddings

```python
from persona.llm.client_factory import get_embedding_client

client = get_embedding_client()
texts = ["Hello world", "How are you?"]
embeddings = await client.embeddings(texts)
```

### JSON Mode

```python
response = await client.chat(
    messages, 
    temperature=0.5,
    response_format={"type": "json_object"}
)
```

## Migration from Previous System

### Key Changes

1. **Removed Dependencies**
   - `instructor` library removed
   - Direct OpenAI client usage replaced with abstraction

2. **Updated Imports**
   ```python
   # Old
   from persona.llm.llm_graph import openai_client
   
   # New
   from persona.llm.client_factory import get_chat_client
   ```

3. **Function Signatures**
   - All LLM functions now use `async/await`
   - Consistent `ChatMessage` format across providers
   - JSON parsing handled natively instead of through instructor

4. **Configuration**
   - New `LLM_SERVICE` format: `provider/model`
   - Provider-specific configuration sections

### Backward Compatibility

The system maintains backward compatibility with existing code through:
- Same function names in `llm_graph.py`
- Consistent response formats
- Automatic mocking in tests

## Testing

### Test Structure

1. **Unit Tests** (`tests/unit/test_llm_clients.py`)
   - Client factory functionality
   - Provider capabilities
   - Configuration parsing
   - Error handling

2. **Integration Tests** (`tests/integration/test_llm_integration.py`)
   - Real API calls with Azure/OpenAI (when credentials available)
   - Provider switching
   - Fallback behavior

3. **End-to-End Tests** (`tests/integration/test_llm_e2e.py`)
   - Complete application workflows
   - Node extraction and relationship generation
   - RAG response generation

### Running Tests

```bash
# Unit tests
python -m pytest tests/unit/test_llm_clients.py -v

# Integration tests (requires API keys)
python -m pytest tests/integration/test_llm_integration.py -v

# End-to-end tests
python -m pytest tests/integration/test_llm_e2e.py -v

# All tests
python -m pytest tests/unit/test_services.py tests/unit/test_llm_clients.py -v
```

## Error Handling

The system includes comprehensive error handling:

1. **Provider Errors**: API errors are caught and logged with provider context
2. **Configuration Errors**: Missing API keys result in clear error messages
3. **Fallback Logic**: Embedding requests automatically fall back to OpenAI for providers without embedding support
4. **JSON Parsing**: Robust JSON parsing with error recovery

## Performance Considerations

1. **Client Caching**: Clients are instantiated once and reused
2. **Async Operations**: All LLM calls are asynchronous
3. **Provider Selection**: Optimal provider selection based on capabilities
4. **Connection Pooling**: Native client libraries handle connection pooling

## Future Enhancements

1. **Additional Providers**: Easy to add new providers (Cohere, Mistral, etc.)
2. **Load Balancing**: Round-robin or weighted provider selection
3. **Retry Logic**: Automatic retries with exponential backoff
4. **Cost Tracking**: Per-provider usage and cost monitoring
5. **Model Routing**: Automatic model selection based on task requirements

## Security

1. **API Key Management**: Secure environment variable handling
2. **Request Logging**: Sensitive data excluded from logs
3. **Error Sanitization**: API keys removed from error messages
4. **Provider Isolation**: Each provider client is isolated

## Monitoring and Observability

The system includes comprehensive logging:

```python
# Client initialization
logger.info(f"Initialized chat client: {provider}/{model}")

# API errors
logger.error(f"Azure OpenAI chat error: {error}")

# Fallback behavior
logger.info(f"Falling back to OpenAI for embeddings")
```

All logs include provider context for easier debugging and monitoring. 