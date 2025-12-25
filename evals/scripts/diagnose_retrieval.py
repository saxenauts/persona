#!/usr/bin/env python3
"""
Graphiti Retrieval Timing Diagnostic

This script instruments the graphiti_core search pipeline to measure timing
at each sub-stage: BM25 search, cosine similarity, reranker, etc.

Usage:
    poetry run python evals/scripts/diagnose_retrieval.py

Output: Detailed timing breakdown showing where time is spent in retrieval.
"""

import os
import sys
import json
import time
import asyncio
from pathlib import Path
from datetime import datetime
from functools import wraps

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Import adapter first to apply patches
from evals.adapters.zep_adapter import GraphitiAdapter

# Timing storage
_timings = []
_call_counts = {}

def log_timing(stage: str, duration_s: float, details: dict = None):
    """Log a timing event."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "stage": stage,
        "duration_s": round(duration_s, 3),
        "details": details or {},
    }
    _timings.append(entry)
    
    details_str = ""
    if details:
        details_str = " | " + ", ".join(f"{k}={v}" for k, v in details.items())
    print(f"⏱️  {stage}: {duration_s:.3f}s{details_str}")

def patch_graphiti_search():
    """Monkey-patch graphiti_core.search to add timing instrumentation."""
    
    from graphiti_core.search import search as search_module
    
    # Store original functions
    original_edge_search = search_module.edge_search
    original_node_search = search_module.node_search
    original_edge_fulltext = search_module.edge_fulltext_search
    original_edge_similarity = search_module.edge_similarity_search
    original_node_fulltext = search_module.node_fulltext_search
    original_node_similarity = search_module.node_similarity_search
    
    # Get reference to cross_encoder rank method from OpenAIRerankerClient
    from graphiti_core.cross_encoder import OpenAIRerankerClient
    original_cross_encoder_rank = OpenAIRerankerClient.rank
    
    # Patch edge_search
    async def timed_edge_search(*args, **kwargs):
        start = time.time()
        result = await original_edge_search(*args, **kwargs)
        duration = time.time() - start
        edges, scores = result
        log_timing("edge_search", duration, {"edges_returned": len(edges)})
        return result
    
    # Patch node_search
    async def timed_node_search(*args, **kwargs):
        start = time.time()
        result = await original_node_search(*args, **kwargs)
        duration = time.time() - start
        nodes, scores = result
        log_timing("node_search", duration, {"nodes_returned": len(nodes)})
        return result
    
    # Patch edge_fulltext_search
    async def timed_edge_fulltext(*args, **kwargs):
        start = time.time()
        result = await original_edge_fulltext(*args, **kwargs)
        duration = time.time() - start
        log_timing("edge_fulltext_search (BM25)", duration, {"results": len(result)})
        return result
    
    # Patch edge_similarity_search  
    async def timed_edge_similarity(*args, **kwargs):
        start = time.time()
        result = await original_edge_similarity(*args, **kwargs)
        duration = time.time() - start
        log_timing("edge_similarity_search (cosine)", duration, {"results": len(result)})
        return result
    
    # Patch node_fulltext_search
    async def timed_node_fulltext(*args, **kwargs):
        start = time.time()
        result = await original_node_fulltext(*args, **kwargs)
        duration = time.time() - start
        log_timing("node_fulltext_search (BM25)", duration, {"results": len(result)})
        return result
    
    # Patch node_similarity_search
    async def timed_node_similarity(*args, **kwargs):
        start = time.time()
        result = await original_node_similarity(*args, **kwargs)
        duration = time.time() - start
        log_timing("node_similarity_search (cosine)", duration, {"results": len(result)})
        return result
    
    # Patch cross_encoder.rank - this is likely the bottleneck!
    async def timed_cross_encoder_rank(self, query: str, passages: list[str]):
        start = time.time()
        _call_counts["reranker_calls"] = _call_counts.get("reranker_calls", 0) + 1
        _call_counts["reranker_passages"] = _call_counts.get("reranker_passages", 0) + len(passages)
        
        result = await original_cross_encoder_rank(self, query, passages)
        
        duration = time.time() - start
        log_timing("cross_encoder.rank", duration, {
            "passages": len(passages),
            "call_num": _call_counts["reranker_calls"],
            "avg_per_passage": round(duration / len(passages), 3) if passages else 0,
        })
        return result
    
    # Apply patches
    search_module.edge_search = timed_edge_search
    search_module.node_search = timed_node_search
    search_module.edge_fulltext_search = timed_edge_fulltext
    search_module.edge_similarity_search = timed_edge_similarity
    search_module.node_fulltext_search = timed_node_fulltext
    search_module.node_similarity_search = timed_node_similarity
    OpenAIRerankerClient.rank = timed_cross_encoder_rank
    
    print("✅ Applied timing patches to graphiti_core.search")

def run_retrieval_test():
    """Run a controlled retrieval test with timing."""
    
    print("\n" + "="*70)
    print("🔬 GRAPHITI RETRIEVAL TIMING DIAGNOSTIC")
    print("="*70)
    
    # Apply patches before creating adapter
    patch_graphiti_search()
    
    # Create adapter
    adapter = GraphitiAdapter()
    provider = os.getenv("GRAPHITI_PROVIDER", "openai")
    search_limit = adapter.search_limit
    print(f"\nProvider: {provider}")
    print(f"Search limit: {search_limit}")
    print(f"Reranker timeout: {adapter.reranker_timeout_s}s")
    print(f"Retrieval timeout: {adapter.retrieval_timeout_s}s")
    
    # Load a complex question with known graph size
    with open("evals/data/golden_sets/longmemeval_golden_set.json") as f:
        questions = json.load(f)
    
    multi_session_qs = [q for q in questions if q["question_type"] == "multi-session"]
    test_q = multi_session_qs[0]
    
    # Prepare sessions
    sessions = []
    for date, turns in zip(test_q["haystack_dates"], test_q["haystack_sessions"]):
        content_parts = [f"{t.get('role', 'user').capitalize()}: {t.get('content', '')}" for t in turns]
        sessions.append({"date": date, "content": "\n".join(content_parts)})
    
    user_id = f"retrieval_diag_{int(time.time())}"
    
    print(f"\nQuestion: {test_q['question'][:60]}...")
    print(f"Sessions: {len(sessions)}")
    print(f"Total chars: {sum(len(s['content']) for s in sessions)}")
    
    # Ingest first
    print("\n" + "-"*70)
    print("📥 INGESTION PHASE")
    print("-"*70)
    
    start_ingest = time.time()
    adapter.add_sessions(user_id, sessions)
    ingest_duration = time.time() - start_ingest
    log_timing("TOTAL_INGESTION", ingest_duration, {"sessions": len(sessions)})
    
    # Now run retrieval with detailed timing
    print("\n" + "-"*70)
    print("🔍 RETRIEVAL PHASE (with sub-stage timing)")
    print("-"*70)
    
    _timings.clear()
    _call_counts.clear()
    
    start_retrieval = time.time()
    answer = adapter.query(user_id, test_q["question"])
    retrieval_duration = time.time() - start_retrieval
    
    print("\n" + "-"*70)
    print("📊 TIMING SUMMARY")
    print("-"*70)
    
    log_timing("TOTAL_RETRIEVAL", retrieval_duration)
    
    # Aggregate timings by stage
    stage_totals = {}
    for entry in _timings:
        stage = entry["stage"]
        if stage not in stage_totals:
            stage_totals[stage] = {"total_s": 0, "calls": 0}
        stage_totals[stage]["total_s"] += entry["duration_s"]
        stage_totals[stage]["calls"] += 1
    
    print("\nBy stage (sorted by total time):")
    for stage, data in sorted(stage_totals.items(), key=lambda x: x[1]["total_s"], reverse=True):
        pct = (data["total_s"] / retrieval_duration) * 100 if retrieval_duration > 0 else 0
        print(f"  {stage}: {data['total_s']:.3f}s ({data['calls']} calls) [{pct:.1f}%]")
    
    print(f"\nReranker stats:")
    print(f"  Total passages ranked: {_call_counts.get('reranker_passages', 0)}")
    print(f"  Total rank calls: {_call_counts.get('reranker_calls', 0)}")
    
    print(f"\nAnswer preview: {answer[:150]}...")
    
    # Save detailed log
    log_path = f"evals/results/retrieval_timing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("evals/results", exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({
            "config": {
                "provider": provider,
                "search_limit": search_limit,
                "reranker_timeout_s": adapter.reranker_timeout_s,
                "retrieval_timeout_s": adapter.retrieval_timeout_s,
            },
            "timings": _timings,
            "call_counts": _call_counts,
            "totals": {
                "ingest_s": ingest_duration,
                "retrieval_s": retrieval_duration,
            }
        }, f, indent=2)
    print(f"\nFull log saved to: {log_path}")
    
    # Cleanup
    adapter.reset(user_id)

if __name__ == "__main__":
    run_retrieval_test()
