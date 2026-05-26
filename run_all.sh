#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for fw in langgraph maf crewai; do
    echo ""
    echo "=========================================="
    echo "  Running $fw"
    echo "=========================================="
    cd "$SCRIPT_DIR/$fw"
    uv run python run_benchmark.py "$@" --log-dir "$SCRIPT_DIR/results"
done

echo ""
echo "All benchmarks complete. Results in $SCRIPT_DIR/results/"
