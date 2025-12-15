#!/usr/bin/env python3
"""
Quick Neo4j probe utilities:
- Show vector indexes
- Global counts
- Type distributions global or per-user

Usage examples:
  python evals/scripts/neo4j_probe.py --show-indexes
  python evals/scripts/neo4j_probe.py --counts
  python evals/scripts/neo4j_probe.py --user lme_oracle_hybrid_XXXX --types
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter


async def _mgr():
    from persona.core.neo4j_database import Neo4jConnectionManager
    m = Neo4jConnectionManager()
    await m.initialize()
    return m


async def show_indexes():
    mgr = await _mgr()
    async with mgr.driver.session() as s:
        res = await s.run("SHOW VECTOR INDEXES")
        data = await res.data()
        for row in data:
            print(row)


async def show_counts(user: str | None):
    mgr = await _mgr()
    async with mgr.driver.session() as s:
        if user:
            resn = await s.run("MATCH (n:NodeName {UserId: $u}) RETURN count(n) AS c", u=user)
            resr = await s.run("MATCH (:NodeName {UserId: $u})-[r]->(:NodeName {UserId: $u}) RETURN count(r) AS c", u=user)
        else:
            resn = await s.run("MATCH (n:NodeName) RETURN count(n) AS c")
            resr = await s.run("MATCH ()-[r]->() RETURN count(r) AS c")
        n = (await resn.single())["c"]
        r = (await resr.single())["c"]
        print({"nodes": n, "relationships": r})


async def show_types(user: str | None):
    mgr = await _mgr()
    async with mgr.driver.session() as s:
        if user:
            res = await s.run("MATCH (n:NodeName {UserId: $u}) RETURN n.type AS t, count(*) AS c", u=user)
        else:
            res = await s.run("MATCH (n:NodeName) RETURN n.type AS t, count(*) AS c")
        rows = await res.data()
        counts = Counter()
        for row in rows:
            counts[row.get("t") or ""] += row.get("c", 0)
        for k, v in counts.most_common():
            print(f"{k or '(empty type)'}: {v}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-indexes", action="store_true")
    ap.add_argument("--counts", action="store_true")
    ap.add_argument("--types", action="store_true")
    ap.add_argument("--user", help="Optional user scope")
    args = ap.parse_args()

    if args.show_indexes:
        await show_indexes()
    if args.counts:
        await show_counts(args.user)
    if args.types:
        await show_types(args.user)


if __name__ == "__main__":
    asyncio.run(main())

