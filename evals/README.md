# LongMemEval for Persona

This directory contains the LongMemEval evaluation framework for testing Persona's memory capabilities.

## Quick Start

### Prerequisites
1. Ensure Docker containers are running: `docker compose up -d`
2. Set your OpenAI API key: `export OPENAI_API_KEY="your-key-here"`
3. Install dependencies: `poetry install`

### Demo (3 instances)
```bash
poetry run python evals/run_demo.py
```

## Full Evaluation Commands

### 1. Clean Slate (Recommended before full eval)
```bash
# Clean the Neo4j database completely
URI_NEO4J="bolt://localhost:7687" USER_NEO4J="neo4j" PASSWORD_NEO4J="password" poetry run python evals/clean_graph.py --force

# Restart containers to reinitialize vector index
docker compose restart

# Wait for containers to start (about 15 seconds)
sleep 15
```

### 2. Run Full Evaluation (500 instances)
```bash
# Full oracle dataset evaluation
poetry run python -m evals.longmemeval.pipeline --dataset oracle --strategy hybrid --backend hybrid --limit 500
```

### 3. Run Specific Batch Ranges
```bash
# First 50 instances (single progress bar: 50/50)
poetry run python -m evals.longmemeval.pipeline --dataset oracle --strategy hybrid --backend hybrid --batch "1-50"

# Next 50 instances (single progress bar: 50/50)
poetry run python -m evals.longmemeval.pipeline --dataset oracle --strategy hybrid --backend hybrid --batch "51-100"

# Any range (single progress bar shows exact count)
poetry run python -m evals.longmemeval.pipeline --dataset oracle --strategy hybrid --backend hybrid --batch "101-150"
```

### 4. Alternative: Smaller Test Run
```bash
# Test with 50 instances
poetry run python -m evals.longmemeval.pipeline --dataset oracle --strategy hybrid --backend hybrid --limit 50
```

## One-Command Full Evaluation

For convenience, here's a complete sequence:

```bash
# Complete clean slate + full evaluation
URI_NEO4J="bolt://localhost:7687" USER_NEO4J="neo4j" PASSWORD_NEO4J="password" poetry run python evals/clean_graph.py --force && \
docker compose restart && \
sleep 15 && \
poetry run python -m evals.longmemeval.pipeline --dataset oracle --strategy hybrid --backend hybrid --limit 500
```

## Batch Processing for Large Datasets

For testing stability and managing results in chunks:

```bash
# Process in batches of 50 (single progress bar: 50/50)
URI_NEO4J="bolt://localhost:7687" USER_NEO4J="neo4j" PASSWORD_NEO4J="password" poetry run python evals/clean_graph.py --force && \
docker compose restart && \
sleep 15 && \
poetry run python -m evals.longmemeval.pipeline --dataset oracle --strategy hybrid --backend hybrid --batch "1-50"

# Then copy results and run next batch (single progress bar: 50/50)
cp -r evals/results evals/results_batch_1-50
poetry run python -m evals.longmemeval.pipeline --dataset oracle --strategy hybrid --backend hybrid --batch "51-100"
```

### Benefits of Batch Processing

- ✅ **Clean Progress Tracking**: Single progress bar per batch (e.g., 50/50, 100/100)
- ✅ **Stability Testing**: Test system performance over extended periods
- ✅ **Result Management**: Manually save and compare results between batches
- ✅ **Resource Control**: Avoid overwhelming system with 500 instances at once
- ✅ **Incremental Progress**: Resume from any point if interruptions occur

## Results

Results are saved to `evals/results/`:
- `evaluation_hybrid_hybrid.json` - Final accuracy scores
- `detailed_results_hybrid_hybrid.json` - Detailed per-question results
- `hypotheses_hybrid_hybrid.jsonl` - Generated answers
- `ingest_manifest_hybrid.json` - Ingestion tracking

## Troubleshooting

### Server Connection Issues
- Ensure containers are running: `docker ps`
- Check server health: `curl http://localhost:8000/api/v1/version`
- Restart if needed: `docker compose restart`

### Vector Index Errors
- The clean slate procedure fixes most vector index issues
- If problems persist, try: `docker compose down && docker compose up -d`

### API Key Issues
- Verify key is set: `echo $OPENAI_API_KEY`
- Make sure it has sufficient quota
- For Docker environments, the key should be set in both host and container

## Performance Notes

- **Demo (3 instances)**: ~2 minutes
- **Small batch (50 instances)**: ~30 minutes  
- **Medium batch (100 instances)**: ~1 hour
- **Full evaluation (500 instances)**: ~5-6 hours
- Each instance processes multiple conversation sessions with embeddings
- **Single progress bars**: Batches are processed concurrently for clean progress tracking

## Batch Range Examples

- `--batch "1-50"` - Process instances 1 through 50 (shows single progress bar: 50/50)
- `--batch "51-100"` - Process instances 51 through 100 (shows single progress bar: 50/50)
- `--batch "101-150"` - Process instances 101 through 150 (shows single progress bar: 50/50)
- `--batch "401-500"` - Process final instances 401 through 500 (shows single progress bar: 100/100)

## Architecture

The evaluation pipeline consists of:
1. **Download**: Fetch LongMemEval dataset from HuggingFace
2. **Ingest**: Store conversations in Persona's memory system
3. **Answer**: Generate responses using RAG
4. **Evaluate**: Score answers against ground truth using GPT-4o-mini

## Overview

The pipeline consists of four main stages:

1. **Download** - Fetches LongMemEval datasets from HuggingFace
2. **Ingest** - Loads conversation data into the Persona memory system
3. **Answer** - Generates responses using Persona's hybrid or vector-only strategies
4. **Evaluate** - Scores results using the official LongMemEval methodology

## Full Pipeline Usage

### Command Line Interface

```bash
# Run on oracle dataset with hybrid strategy (default)
python -m longmemeval.pipeline --dataset oracle --strategy hybrid --limit 50

# Run with vector-only strategy
python -m longmemeval.pipeline --dataset oracle --strategy vector-only --limit 50

# Run on full oracle dataset (500 instances)
python -m longmemeval.pipeline --dataset oracle --strategy hybrid
```

### Available Options

- `--dataset`: Dataset type (`oracle`, `s`, `m`)
- `--strategy`: Answer strategy (`hybrid`, `vector-only`)  
- `--backend`: Memory backend (`hybrid`, `vector`)
- `--limit`: Number of instances to process (default: 3 for testing)

### Dataset Types

- **oracle**: Contains only evidence sessions (~300MB, 500 questions)
- **s**: Short version with ~115k tokens per history
- **m**: Medium version with ~500 sessions per history

## Output Files

Results are saved to `evals/results/`:

- `hypotheses_<strategy>_<backend>.jsonl` - Official format for evaluation
- `detailed_results_<strategy>_<backend>.json` - Detailed timing and metadata
- `evaluation_<strategy>_<backend>.json` - Final evaluation scores
- `ingest_manifest_<backend>.json` - Ingestion tracking

## Pipeline Components

### Individual Components

You can also run individual stages:

```bash
# Download dataset
python -m longmemeval.fetch_data --dataset oracle --subset 10

# Ingest data
python -m longmemeval.ingest path/to/dataset.json --backend hybrid --limit 10

# Generate answers
python -m longmemeval.answer path/to/dataset.json path/to/manifest.json --strategy hybrid

# Evaluate results
python -m longmemeval.evaluate_qa path/to/hypotheses.jsonl path/to/dataset.json
```

### Configuration

Edit `longmemeval/config.py` to customize:

- API endpoints
- Model settings (currently uses `gpt-4o-mini`)
- Concurrency limits
- File paths

## Results Interpretation

The evaluation produces several key metrics:

- **Overall Accuracy**: Percentage of questions answered correctly
- **Task-Averaged Accuracy**: Average accuracy across different question types
- **Task-Specific Accuracy**: Performance on individual question types:
  - `temporal-reasoning`: Questions requiring time-based reasoning
  - `knowledge-update`: Questions about updated information
  - `single-session-*`: Questions from single conversation sessions
  - `multi-session`: Questions spanning multiple sessions

### Performance Metrics

- **Retrieval Time**: Time spent retrieving relevant information
- **Total Time**: End-to-end response generation time
- **Token Usage**: Estimated costs for evaluation

## Question Types

LongMemEval includes several question types:

1. **single-session-user**: User information from one session
2. **single-session-assistant**: Assistant information from one session  
3. **single-session-preference**: User preferences from one session
4. **temporal-reasoning**: Time-based reasoning across sessions
5. **knowledge-update**: Updated information over time
6. **multi-session**: Information spanning multiple sessions

## Comparison with Published Results

This pipeline follows the exact methodology used in published evaluations, enabling direct comparison with other memory systems. The oracle dataset is specifically designed for controlled evaluation where retrieval quality can be isolated from reading comprehension.

## Troubleshooting

### Common Issues

1. **Server Connection**: Ensure Persona server is running on `localhost:8000`
2. **API Key**: Verify `OPENAI_API_KEY` environment variable is set
3. **Memory**: Large datasets may require significant memory for ingestion
4. **Rate Limits**: OpenAI evaluation may hit rate limits with large datasets

### Debug Mode

Add `--verbose` flag for detailed logging:

```bash
python -m longmemeval.pipeline --dataset oracle --limit 5 --verbose
```

## Extending the Pipeline

To add new evaluation benchmarks:

1. Create a new loader in the style of `loader.py`
2. Implement data format conversion
3. Add evaluation methodology in the style of `evaluate_qa.py`
4. Update the pipeline orchestrator

The modular design makes it easy to swap in different datasets or evaluation approaches. 