#!/usr/bin/env python3
"""
Graphiti Diagnostic Test Script

This script runs a minimal reproduction of the eval flow with comprehensive
timing instrumentation and signal handling to diagnose hangs.

Usage:
    poetry run python evals/scripts/diagnose_graphiti.py

Expected output: Granular timing data showing exactly where the hang occurs.
"""

import os
import sys
import json
import time
import signal
import asyncio
import faulthandler
import traceback
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Enable faulthandler for SIGUSR1 stack dumps
faulthandler.enable()


# Track active async tasks for diagnosis
_active_tasks = {}
_timing_log = []

def log_timing(stage: str, duration_s: float = None, note: str = None):
    """Log a timing event with timestamp."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "stage": stage,
        "duration_s": duration_s,
        "note": note,
        "thread": threading.current_thread().name,
    }
    _timing_log.append(entry)
    symbol = "⏱️ " if duration_s else "🔄"
    dur_str = f" [{duration_s:.3f}s]" if duration_s else ""
    note_str = f" - {note}" if note else ""
    print(f"{symbol} {stage}{dur_str}{note_str}", flush=True)

def signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM - dump diagnostics and exit."""
    print("\n\n" + "="*60)
    print("⚠️  INTERRUPTED - Dumping diagnostics...")
    print("="*60)
    
    # Dump timing log
    print("\n📊 TIMING LOG:")
    for entry in _timing_log[-20:]:
        print(f"  {entry['timestamp']} | {entry['stage']} | {entry.get('duration_s', 'N/A')}")
    
    # Dump stack traces for all threads
    print("\n🔍 THREAD STACK TRACES:")
    faulthandler.dump_traceback()
    
    # Try to dump async tasks
    try:
        loop = asyncio.get_running_loop()
        tasks = asyncio.all_tasks(loop)
        print(f"\n⚡ ASYNC TASKS ({len(tasks)}):")
        for task in tasks:
            print(f"  {task.get_name()}: {task.get_coro()}")
    except:
        pass
    
    print("\n" + "="*60)
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# =========================================================================
# DIAGNOSTIC TEST SUITE
# =========================================================================

def test_neo4j_connection():
    """Test 1: Verify Neo4j is reachable."""
    log_timing("neo4j_connection_test", note="Starting")
    start = time.time()
    
    try:
        from neo4j import GraphDatabase
        uri = os.getenv("URI_NEO4J", "bolt://127.0.0.1:7687")
        # Replace docker hostname with localhost
        if "neo4j:7687" in uri:
            uri = uri.replace("neo4j", "127.0.0.1")
        user = os.getenv("USER_NEO4J", "neo4j")
        password = os.getenv("PASSWORD_NEO4J", "password")
        
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            value = result.single()["test"]
            assert value == 1
        driver.close()
        
        duration = time.time() - start
        log_timing("neo4j_connection_test", duration, note="SUCCESS")
        return True
    except Exception as e:
        duration = time.time() - start
        log_timing("neo4j_connection_test", duration, note=f"FAILED: {e}")
        return False

def test_neo4j_pool_status():
    """Test 2: Check Neo4j async pool behavior."""
    log_timing("neo4j_pool_test", note="Starting")
    start = time.time()
    
    async def check_pool():
        from neo4j import AsyncGraphDatabase
        uri = os.getenv("URI_NEO4J", "bolt://127.0.0.1:7687")
        if "neo4j:7687" in uri:
            uri = uri.replace("neo4j", "127.0.0.1")
        user = os.getenv("USER_NEO4J", "neo4j")
        password = os.getenv("PASSWORD_NEO4J", "password")
        
        driver = AsyncGraphDatabase.driver(
            uri, 
            auth=(user, password),
            max_connection_pool_size=50,
        )
        
        # Run 10 concurrent queries
        async def query(i):
            async with driver.session() as session:
                result = await session.run(f"RETURN {i} as n")
                await result.consume()
            return i
        
        log_timing("neo4j_pool_concurrent_queries", note="Running 10 concurrent queries")
        sub_start = time.time()
        tasks = [query(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        sub_duration = time.time() - sub_start
        log_timing("neo4j_pool_concurrent_queries", sub_duration, note=f"Got {len(results)} results")
        
        await driver.close()
        return True
    
    try:
        result = asyncio.run(check_pool())
        duration = time.time() - start
        log_timing("neo4j_pool_test", duration, note="SUCCESS")
        return result
    except Exception as e:
        duration = time.time() - start
        log_timing("neo4j_pool_test", duration, note=f"FAILED: {e}")
        traceback.print_exc()
        return False

def test_graphiti_init():
    """Test 3: Initialize Graphiti adapter."""
    log_timing("graphiti_init", note="Starting")
    start = time.time()
    
    try:
        # Use whatever provider is set in environment
        # Note: The zep_adapter module applies the reasoning.effort bugfix at import time
        from evals.adapters.zep_adapter import GraphitiAdapter
        adapter = GraphitiAdapter()
        
        duration = time.time() - start
        log_timing("graphiti_init", duration, note=f"SUCCESS - search_limit={adapter.search_limit}")
        return adapter
    except Exception as e:
        duration = time.time() - start
        log_timing("graphiti_init", duration, note=f"FAILED: {e}")
        traceback.print_exc()
        return None

def test_graphiti_ingest(adapter, user_id: str, sessions: list):
    """Test 4: Ingest sessions into Graphiti."""
    log_timing("graphiti_ingest", note=f"Starting - {len(sessions)} sessions")
    start = time.time()
    
    try:
        # Reset first
        adapter.reset(user_id)
        log_timing("graphiti_reset", time.time() - start, note="Reset complete")
        
        ingest_start = time.time()
        adapter.add_sessions(user_id, sessions)
        ingest_duration = time.time() - ingest_start
        
        ingest_stats = getattr(adapter, "last_ingest_stats", {})
        log_timing("graphiti_ingest", ingest_duration, 
                   note=f"SUCCESS - nodes={ingest_stats.get('memories_created', '?')}")
        return True
    except Exception as e:
        duration = time.time() - start
        log_timing("graphiti_ingest", duration, note=f"FAILED: {e}")
        traceback.print_exc()
        return False

def test_graphiti_query(adapter, user_id: str, query: str, timeout_s: float = 120):
    """Test 5: Query Graphiti with hard timeout."""
    log_timing("graphiti_query", note=f"Starting - query='{query[:50]}...'")
    
    def sync_query():
        log_timing("graphiti_query_inner", note="Inside sync wrapper")
        start = time.time()
        result = adapter.query(user_id, query)
        duration = time.time() - start
        log_timing("graphiti_query_inner", duration, note=f"Got answer: {result[:50]}...")
        return result
    
    start = time.time()
    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="graphiti_query") as executor:
            future = executor.submit(sync_query)
            log_timing("graphiti_query_submitted", note=f"Waiting with {timeout_s}s timeout")
            result = future.result(timeout=timeout_s)
            duration = time.time() - start
            log_timing("graphiti_query", duration, note="SUCCESS")
            return result
    except FuturesTimeout:
        duration = time.time() - start
        log_timing("graphiti_query", duration, note=f"TIMEOUT after {timeout_s}s")
        print("\n⚠️  QUERY TIMED OUT - triggering stack dump...")
        faulthandler.dump_traceback()
        return None
    except Exception as e:
        duration = time.time() - start
        log_timing("graphiti_query", duration, note=f"FAILED: {e}")
        traceback.print_exc()
        return None

def load_test_question():
    """Load a sample question from golden set."""
    log_timing("load_test_question", note="Starting")
    try:
        with open("evals/data/golden_sets/longmemeval_golden_set.json") as f:
            questions = json.load(f)
        
        # Find a multi-session question (most likely to trigger the hang)
        multi_session_qs = [q for q in questions if q["question_type"] == "multi-session"]
        single_session_qs = [q for q in questions if q["question_type"] == "single-session-user"]
        
        log_timing("load_test_question", note=f"Found {len(multi_session_qs)} multi-session, {len(single_session_qs)} single-session")
        
        # Return both a simple and complex question
        simple_q = single_session_qs[0] if single_session_qs else questions[0]
        complex_q = multi_session_qs[0] if multi_session_qs else questions[1]
        
        return simple_q, complex_q
    except Exception as e:
        log_timing("load_test_question", note=f"FAILED: {e}")
        return None, None

def prepare_sessions(question):
    """Convert question haystack to sessions."""
    sessions = []
    for date, turns in zip(question["haystack_dates"], question["haystack_sessions"]):
        content_parts = [f"{t.get('role', 'user').capitalize()}: {t.get('content', '')}" for t in turns]
        sessions.append({"date": date, "content": "\n".join(content_parts)})
    return sessions

# =========================================================================
# MAIN DIAGNOSTIC FLOW
# =========================================================================

def run_diagnostics():
    """Run all diagnostics."""
    print("\n" + "="*60)
    print("🔬 GRAPHITI DIAGNOSTIC TEST SUITE")
    print("="*60)
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"Press Ctrl+C anytime for stack dump\n")
    
    # Test 1: Neo4j connection
    if not test_neo4j_connection():
        print("\n❌ Neo4j connection failed - aborting")
        return
    
    # Test 2: Neo4j pool
    if not test_neo4j_pool_status():
        print("\n⚠️  Neo4j pool test failed - continuing anyway")
    
    # Test 3: Initialize adapter
    adapter = test_graphiti_init()
    if not adapter:
        print("\n❌ Graphiti init failed - aborting")
        return
    
    # Load test questions
    simple_q, complex_q = load_test_question()
    if not simple_q:
        print("\n❌ Failed to load test questions")
        return
    
    # Test 4 & 5: Run simple question first
    print("\n" + "-"*60)
    print("📝 TEST A: Simple single-session question")
    print("-"*60)
    
    user_id_a = f"diag_simple_{int(time.time())}"
    sessions_a = prepare_sessions(simple_q)
    print(f"Question: {simple_q['question'][:80]}...")
    print(f"Sessions: {len(sessions_a)}, chars: {sum(len(s['content']) for s in sessions_a)}")
    
    if test_graphiti_ingest(adapter, user_id_a, sessions_a):
        answer = test_graphiti_query(adapter, user_id_a, simple_q["question"], timeout_s=60)
        if answer:
            print(f"✅ Answer: {answer[:100]}...")
    
    # Clean up
    adapter.reset(user_id_a)
    
    # Test 4 & 5: Run complex question
    print("\n" + "-"*60)
    print("📝 TEST B: Complex multi-session question")
    print("-"*60)
    
    user_id_b = f"diag_complex_{int(time.time())}"
    sessions_b = prepare_sessions(complex_q)
    print(f"Question: {complex_q['question'][:80]}...")
    print(f"Sessions: {len(sessions_b)}, chars: {sum(len(s['content']) for s in sessions_b)}")
    
    if test_graphiti_ingest(adapter, user_id_b, sessions_b):
        answer = test_graphiti_query(adapter, user_id_b, complex_q["question"], timeout_s=120)
        if answer:
            print(f"✅ Answer: {answer[:100]}...")
        else:
            print("❌ Query failed or timed out - this is the problem area!")
    
    # Clean up
    adapter.reset(user_id_b)
    
    # Final summary
    print("\n" + "="*60)
    print("📊 DIAGNOSTIC SUMMARY")
    print("="*60)
    print(f"\nTotal timing entries: {len(_timing_log)}")
    
    # Find slowest stages
    timed_entries = [e for e in _timing_log if e.get("duration_s")]
    if timed_entries:
        timed_entries.sort(key=lambda x: x["duration_s"], reverse=True)
        print("\nSlowest stages:")
        for entry in timed_entries[:5]:
            print(f"  {entry['duration_s']:.3f}s - {entry['stage']}")
    
    # Save full log
    log_path = f"evals/results/diag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("evals/results", exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(_timing_log, f, indent=2)
    print(f"\nFull log saved to: {log_path}")

if __name__ == "__main__":
    run_diagnostics()
