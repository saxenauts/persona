# Persona Roadmap

  ---
  🚀 v0.2 Release (Immediate - Do First)

  - [ ] Push persona website
  - [ ] Optimize eval system for faster runs
  - [ ] Run Graphiti benchmark on golden set
  - [ ] Check graphiti plan (Review evaluation configurations and comparison strategy)
  - [ ] Add analysis results to README
  - [ ] Push to GitHub
  - [ ] Release v0.2 tag

  ---
  🔥 Critical: Core Intelligence (Main Work)

  1. Agentic System
  - Agentic Ingestion — AI-driven memory extraction and linking
  - Agentic Retrieval — Multi-step reasoning loops for context
  - Agentic Update ("Pulse") — Async daily background process that reviews graph, generates questions/links, deletes stale connections

  2. Causal Intelligence
  - Backlinking & Retrofitting — Reverse causal chain discovery
  - Causal Chain Development — Smarter causal link extraction
  - Connection Weights — "Firing together, wiring together" weights on memories and relationships

  3. Search & Retrieval
  - BM25 — Keyword/exact term matching for proper nouns
  - Date-Based Retrieval — Query through date ranges
  - Schema Variable Search — Index/retrieve by any field
  - Reasoning Model Support — o1-style models for query planning

  4. Prompts Overhaul
  - Context prompt redesign
  - Ingestion pipeline prompts
  - Retrieval pipeline prompts
  - Causal link discovery prompts
  - Retrofitting pipeline prompts

  ---
  🏗️ Platform & Infrastructure

  - More vector stores (Qdrant, Weaviate, pgvector)
  - More graph stores (PostgreSQL, RedisGraph)
  - FastAPI server
  - Default chat interface
  - MCP Integration — Two-way sync with ChatGPT, Claude
  - Screen-level presence (browser extension? OS-level?)

  ---
  🎯 Data Model & Features

  - Goal System — Proper goal model (not just tasks/projects)
  - Psyche Refinement — Better distinction between psyche vs episodes

  ---
  📊 Eval (In Progress)

  - Eval design — Current benchmark runs
  - Result analysis — Graphiti + Persona comparison
  - README documentation — Add benchmark results

  ---
  Priority Phases

  Phase 1 (Now): Agentic Ingestion → BM25/Date Retrieval → Prompts → MCP → Connection Weights

  Phase 2 (Next): Agentic Retrieval → Backlinking → Causal Chains → Pulse

  Phase 3 (Later): More backends → Chat UI → Screen presence → Goals/psyche

  ---
  Last updated: 2025-12-24
