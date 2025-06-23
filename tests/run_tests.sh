#!/bin/sh
set -e

echo "--- Running formatter check with black ---"
poetry run black --check .

echo "\n--- Running linter with flake8 ---"
poetry run flake8 .

echo "\n--- Running type checker with mypy ---"
poetry run mypy .

echo "\n--- Running test suite with pytest ---"
poetry run pytest 