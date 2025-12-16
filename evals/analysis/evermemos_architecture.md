# EverMemOS Architecture Analysis

> [!NOTE]
> This document analyzes the internal mechanics of EverMemOS (specifically the `locomo` implementation) based on code inspection of their evaluation framework. It serves as a reference for their memory management strategies.

## 1. Stage 1: Ingestion (The "One-by-One" Bottleneck)
**File**: `stage1_memcells_extraction.py`, `memcell_extractor/conv_memcell_extractor.py`

The core ingestion logic is strictly sequential, which explains the high latency.

### Logic Flow
1. **Stream Processing**: The system feeds messages one by one into a buffer.
2. **Boundary Detection**: For *every single message*, it calls the LLM with the current buffer and the new message.
   - Prompt: "Does this new message belong to the current event, or start a new one?"
   - This prevents parallel batching because the context of Message $N$ depends on the cut decisions made at $N-1$.
3. **Hard Limits**: To prevent infinite buffering, it has "safety valves":
   - `DEFAULT_HARD_MESSAGE_LIMIT = 50`: If 50 messages accumulate without a boundary, it forces a split (no LLM call).
   - `DEFAULT_HARD_TOKEN_LIMIT = 8192`.
4. **Extraction**: Once a boundary is found (or forced), it triggers an "Event Extraction" process (Event Summary, Atomic Facts, etc.) for that segment.

> [!IMPORTANT]
> **Performance Impact**: This design prioritizes segmentation accuracy over speed. On restricted tiers (Azure S0), the inability to parallelize "conversation scanning" makes it extremely slow (~10 mins/conversation).

---

## 2. Stage 2: Indexing (Weighted Hybrid)
**File**: `stage2_index_building.py`

EverMemOS builds two separate indices for every conversation.

### A. BM25 Index (Lexical)
- **Library**: `rank_bm25` (BM25Okapi).
- **Strategy**: 
  - Tokenizes text with NLTK + PorterStemmer.
  - **Heuristic Weighting**: It artificially repeats certain fields to boost their importance:
    - `Subject` (Title) × 3
    - `Summary` × 2
    - `Episode` (Content) × 1

### B. Vector Index (Semantic)
- **Model**: `text-embedding-3-small` (via `vectorize_service`).
- **Strategy ("MaxSim")**:
  - If `atomic_facts` exist (extracted in Stage 1), it embeds *each fact individually*.
  - During search, it computes the similarity for *all* facts and takes the **Maximum** score (not average).
  - This prevents "semantic dilution" where a long document's average vector might miss a specific detail.

---

## 3. Stage 3: Retrieval (Agentic + RRF)
**File**: `stage3_memory_retrivel.py`

This is the most sophisticated part of the system, employing a "Reflexive" multi-step pipeline.

### Step 1: Initial Hybrid Search
It runs two searches in parallel:
1. **Vector Search**: Uses the "MaxSim" strategy (query vs. all atomic facts).
2. **BM25 Search**: Standard keyword matching.
3. **Fusion**: Merges results using **Reciprocal Rank Fusion (RRF)** ($k=60$).
   - Returns Top 20 candidates.

### Step 2: Reranking (Optional)
- Reranks the Top 20 to Top 10 using a Cross-Encoder (if enabled).

### Step 3: The "Agentic" Check
The LLM inspects the Top 10 results:
- **Prompt**: "Is this information sufficient to answer the query: '{Query}'?"
- **If YES**: Returns results immediately.
- **If NO**: Triggers "Round 2".

### Step 4: Round 2 (Refinement)
1. **Query Expansion**: LLM generates **3 refined/complementary queries** based on what was missing.
2. **Parallel Search**: Executes the Hybrid Search (Vector + BM25) for *each* of the 3 new queries.
3. **Multi-Way Fusion**: Fuses all results (Original + 3 Refined) using Multi-way RRF.
4. **Final Result**: Returns Top 40 unique events.

---

## 4. Stage 4: Generation
**File**: `stage4_response.py`

1. **Context Construction**:
   - Takes the Top $K$ Event IDs from Stage 3.
   - Concatenates them: `Subject: {subject}\nEpisode: {content}\n---`.
2. **Final Answer**:
   - Feeds the context + original question to the LLM.
   - Uses `temperature=0` for factual consistency.

---

## Summary of Key Learnings

1. **Precision > Speed**: The "One-by-One" ingestion is a deliberate choice for high-fidelity segmentation, but it kills performance on low-concurrency connections.
2. **MaxSim Embedding**: Embedding individual facts and taking the *max* score is a smart way to handle granular details in long memories.
3. **Reflexive Retrieval**: The "Is this sufficient?" check prevents answering with hallucinations. If the memory is missing, it tries harder (refining queries) before giving up.
