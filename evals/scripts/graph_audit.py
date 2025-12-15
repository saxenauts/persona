#!/usr/bin/env python3
"""
Graph audit across ingested LongMemEval users.

Reads evals/results/ingest_manifest_*.json to discover user_ids, then connects
to Neo4j using the repo's connection manager to compute per-user graph stats:
- node_count, relationship_count
- type distribution (from n.type)
- embedding coverage (nodes with n.embedding set)
- orphan ratio (nodes with degree 0)
- simple degree stats (avg, max)
- top relationship labels (type(r))

Outputs:
- evals/results/graph_audit.json (per-user and aggregate summary)

Note: This script only uses local project code and .env for config.
It will gracefully handle DB connectivity issues and still write an error summary.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Any


# Lightweight async helpers using the project Neo4j manager
async def _init_manager():
    # Import inside to avoid import cost for --help
    from persona.core.neo4j_database import Neo4jConnectionManager
    mgr = Neo4jConnectionManager()
    await mgr.initialize()
    return mgr


async def _count_nodes_with_embeddings(mgr, user_id: str) -> int:
    query = """
    MATCH (n:NodeName {UserId: $user_id})
    WHERE exists(n.embedding)
    RETURN count(n) AS c
    """
    async with mgr.driver.session() as sess:
        rec = await (await sess.run(query, user_id=user_id)).single()
        return int(rec["c"]) if rec else 0


def _degree_stats(relationships: List[Dict[str, Any]]):
    deg = Counter()
    for rel in relationships:
        deg[rel["source"]] += 1
        deg[rel["target"]] += 1
    if not deg:
        return {"avg_degree": 0.0, "max_degree": 0, "orphan_ratio": 1.0}
    values = list(deg.values())
    return {
        "avg_degree": float(mean(values)) if values else 0.0,
        "max_degree": int(max(values)) if values else 0,
        # orphan ratio computed later given node_count
    }


async def audit_user(mgr, user_id: str) -> Dict[str, Any]:
    # Import GraphOps for convenience wrappers
    from persona.core.graph_ops import GraphOps

    ops = GraphOps(mgr)

    nodes = await ops.get_all_nodes(user_id)
    rels = await ops.get_all_relationships(user_id)

    node_count = len(nodes)
    rel_count = len(rels)

    type_counts = Counter((n.type or "") for n in nodes)
    rel_type_counts = Counter((r.relation or "") for r in rels)

    deg = Counter()
    for r in rels:
        deg[r.source] += 1
        deg[r.target] += 1
    orphan_nodes = [n.name for n in nodes if deg[n.name] == 0]

    try:
        with_embeddings = await _count_nodes_with_embeddings(mgr, user_id)
    except Exception:
        with_embeddings = None

    stats = _degree_stats([{"source": r.source, "target": r.target} for r in rels])
    orphan_ratio = (len(orphan_nodes) / node_count) if node_count else 0.0

    return {
        "user_id": user_id,
        "nodes": node_count,
        "relationships": rel_count,
        "types": dict(type_counts),
        "rel_types": dict(rel_type_counts.most_common(15)),
        "avg_degree": stats.get("avg_degree", 0.0),
        "max_degree": stats.get("max_degree", 0),
        "orphan_ratio": orphan_ratio,
        "with_embeddings": with_embeddings,
    }


def _load_manifest(manifest_path: Path) -> Dict[str, Dict[str, Any]]:
    with open(manifest_path, "r") as f:
        return json.load(f)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="evals/results/ingest_manifest_hybrid.json",
                    help="Path to ingest manifest JSON")
    ap.add_argument("--out", default="evals/results/graph_audit.json",
                    help="Where to write JSON results")
    ap.add_argument("--limit", type=int, default=0, help="Optional user limit for quick runs")
    args = ap.parse_args()

    manifest = _load_manifest(Path(args.manifest))
    # question_id -> entry containing user_id
    user_ids = [entry["user_id"] for entry in manifest.values()]
    if args.limit and args.limit > 0:
        user_ids = user_ids[: args.limit]

    out = {
        "users": [],
        "aggregate": {}
    }

    try:
        mgr = await _init_manager()
    except Exception as e:
        out["error"] = f"Failed to connect to Neo4j: {e}"
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(out["error"])  # basic feedback
        return

    # Gather per-user stats
    for uid in user_ids:
        try:
            stats = await audit_user(mgr, uid)
            out["users"].append(stats)
        except Exception as e:
            out["users"].append({"user_id": uid, "error": str(e)})

    # Build aggregates
    total_nodes = sum(u.get("nodes", 0) for u in out["users"] if "nodes" in u)
    total_rels = sum(u.get("relationships", 0) for u in out["users"] if "relationships" in u)
    type_counts = Counter()
    rel_type_counts = Counter()
    orphan_sum = 0.0
    deg_vals = []
    with_embed_known = [u["with_embeddings"] for u in out["users"] if isinstance(u.get("with_embeddings"), int)]
    with_embed_sum = sum(with_embed_known) if with_embed_known else 0

    for u in out["users"]:
        if "types" in u:
            type_counts.update(u["types"])
        if "rel_types" in u:
            rel_type_counts.update(u["rel_types"])
        if "orphan_ratio" in u:
            orphan_sum += u["orphan_ratio"]
        if "avg_degree" in u:
            deg_vals.append(u["avg_degree"])

    n_users = len([u for u in out["users"] if "nodes" in u])
    out["aggregate"] = {
        "users": n_users,
        "total_nodes": total_nodes,
        "total_relationships": total_rels,
        "type_counts": dict(type_counts.most_common(25)),
        "rel_type_counts": dict(rel_type_counts.most_common(25)),
        "mean_orphan_ratio": (orphan_sum / n_users) if n_users else 0.0,
        "mean_avg_degree": (sum(deg_vals) / len(deg_vals)) if deg_vals else 0.0,
        "nodes_with_embeddings": with_embed_sum,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote graph audit to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())

