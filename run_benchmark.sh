#!/bin/bash
set -e

# Load .env but verify we are running locally
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# OVERRIDE Neo4j URI for local execution
# The .env usually contains 'bolt://neo4j:7687' which is for Docker internal networking.
# When running this script from the host, we must use 'localhost'.
export URI_NEO4J="bolt://localhost:7687"

echo "🚀 Running Benchmark with URI_NEO4J=$URI_NEO4J"
echo "---------------------------------------------------"

# Run the benchmark runner using poetry to ensure dependencies are loaded
PYTHONPATH=. poetry run python3 evals/benchmark_runner.py

# Automatically generate the report for the latest run
LATEST_JSON=$(ls -t evals/results/benchmark_run_*.json | head -n 1)
if [ -n "$LATEST_JSON" ]; then
    echo ""
    echo "📊 Generating Report for $LATEST_JSON..."
    python3 evals/scripts/generate_report.py "$LATEST_JSON"
    echo "Done!"
else
    echo "⚠️ No results found to generate report."
fi
