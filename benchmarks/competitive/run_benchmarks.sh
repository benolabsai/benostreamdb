#!/bin/bash
#
# Quick-start script for running competitive benchmarks
#

set -e

echo "=================================="
echo "HyperStreamDB Competitive Benchmarks"
echo "=================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Install HyperStreamDB from repo root
echo "Installing HyperStreamDB..."
pip install -q -e ../../

echo ""
echo "=================================="
echo "Running Benchmarks"
echo "=================================="
echo ""

# Parse arguments
QUICK=false
SIZES="10000 100000"

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK=true
            shift
            ;;
        --sizes)
            SIZES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--quick] [--sizes \"10000 100000 1000000\"]"
            exit 1
            ;;
    esac
done

# Run benchmarks
if [ "$QUICK" = true ]; then
    echo "Running quick benchmark (1K vectors)..."
    python benchmark_suite.py --quick
else
    echo "Running full benchmark (sizes: $SIZES)..."
    python benchmark_suite.py --sizes $SIZES
fi

echo ""
echo "=================================="
echo "Benchmark Complete!"
echo "=================================="
echo ""
echo "Results saved to: benchmark_results/"
echo ""
echo "View results:"
echo "  - CSV: benchmark_results/benchmark_results.csv"
echo "  - Report: benchmark_results/BENCHMARK_REPORT.md"
echo "  - Charts: benchmark_results/benchmark_charts.png"
echo ""
echo "To view the report:"
echo "  cat benchmark_results/BENCHMARK_REPORT.md"
echo ""
