# LLM Codebase Overview

This document describes the modules under `persona/llm` that implement the integration with OpenAI models. These utilities are used throughout the service layer to build and query the user's knowledge graph. If you are completely new to the repository, first skim [CODEBASE_OVERVIEW.md](CODEBASE_OVERVIEW.md) to see how the rest of the project is organised.

## Modules

### `embeddings.py`
- Provides `generate_embeddings` which calls the OpenAI embedding API.
- Returns 1536 dimensional vectors used for Neo4j's vector index.

```python
openai_client = openai.Client(api_key=config.MACHINE_LEARNING.OPENAI_KEY)

def generate_embeddings(texts, model="text-embedding-3-small"):
    response = openai_client.embeddings.create(input=texts, model=model, dimensions=1536)
    embeddings = [data.embedding for data in response.data]
```

### `prompts.py`
- Stores all prompt templates that instruct the LLM.
- Includes templates for node extraction, relationship extraction, community detection and structured Q&A.

### `llm_graph.py`
- Central async helper for calling the chat completion API.
- Defines `Node` and `Relationship` schema objects and functions that drive graph construction and querying.

```python
openai_client = openai.AsyncOpenAI(api_key=config.MACHINE_LEARNING.OPENAI_KEY)
client = instructor.from_openai(openai_client)

class Node(OpenAISchema):
    name: str = Field(..., description="The node content - can be a simple label ...")
```

Key functions include:
- `get_nodes(text, graph_context)` – extracts important nodes from raw text.
- `get_relationships(nodes, graph_context)` – infers relationships between nodes.
- `generate_response_with_context(query, context)` – answers questions about a user using provided graph context.
- `detect_communities(subgraphs_text)` – organizes subgraphs into meaningful communities.
- `generate_structured_insights(ask_request, context)` – returns JSON formatted answers.

```python
response = await client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[{"role": "system", "content": combined_instructions}, {"role": "user", "content": text}],
    response_model=List[Node]
)
```

## Configuration

The OpenAI API key is loaded from the environment via `server.config`. Ensure `OPENAI_KEY` is defined in `.env` before running the application.

## Summary

These modules abstract all LLM interactions. Services such as the `GraphConstructor` or `RAGInterface` call into them to enrich the user's knowledge graph and to generate answers using Retrieval Augmented Generation.
