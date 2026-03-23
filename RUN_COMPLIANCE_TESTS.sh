#!/bin/bash
# HyperStreamDB Iceberg Compliance Test Runner
# This script runs the full compliance check, regression tests, and performance benchmarks

set -e  # Exit on error

echo "================================================================================"
echo "HyperStreamDB Iceberg Spec Compliance Test Suite"
echo "================================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Run compliance checker
echo "Step 1: Checking Iceberg Spec Compliance..."
echo "--------------------------------------------------------------------------------"
python3 check_iceberg_compliance.py > compliance_output.txt 2>&1
COMPLIANCE_EXIT=$?

# Display compliance results
cat compliance_output.txt

if [ $COMPLIANCE_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ Compliance check completed${NC}"
else
    echo -e "${RED}✗ Compliance check failed${NC}"
    exit 1
fi

# Extract compliance percentage
COMPLIANCE_PCT=$(grep "Compliance:" compliance_output.txt | awk '{print $2}' | tr -d '%')

echo ""
echo "Compliance: ${COMPLIANCE_PCT}%"

# Check if we should proceed with tests
if (( $(echo "$COMPLIANCE_PCT < 90" | bc -l) )); then
    echo -e "${YELLOW}⚠ Compliance below 90%, skipping regression tests${NC}"
    echo "Please address compliance issues first."
    exit 1
fi

# Step 2: Run Rust tests
echo ""
echo "Step 2: Running Rust Regression Tests..."
echo "--------------------------------------------------------------------------------"
cargo test --release 2>&1 | tee rust_test_output.txt
RUST_EXIT=${PIPESTATUS[0]}

if [ $RUST_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ Rust tests passed${NC}"
else
    echo -e "${RED}✗ Rust tests failed${NC}"
    exit 2
fi

# Step 3: Run Python tests (if pytest is available)
echo ""
echo "Step 3: Running Python Tests..."
echo "--------------------------------------------------------------------------------"
if command -v pytest &> /dev/null; then
    pytest tests/ -v 2>&1 | tee python_test_output.txt
    PYTHON_EXIT=${PIPESTATUS[0]}
    
    if [ $PYTHON_EXIT -eq 0 ]; then
        echo -e "${GREEN}✓ Python tests passed${NC}"
    else
        echo -e "${YELLOW}⚠ Python tests had failures${NC}"
        # Don't exit, continue to benchmarks
    fi
else
    echo -e "${YELLOW}⚠ pytest not found, skipping Python tests${NC}"
fi

# Step 4: Run performance benchmarks
echo ""
echo "Step 4: Running Performance Benchmarks..."
echo "--------------------------------------------------------------------------------"
cargo bench 2>&1 | tee benchmark_output.txt
BENCH_EXIT=${PIPESTATUS[0]}

if [ $BENCH_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ Benchmarks completed${NC}"
else
    echo -e "${YELLOW}⚠ Benchmarks had issues${NC}"
fi

# Summary
echo ""
echo "================================================================================"
echo "TEST SUITE SUMMARY"
echo "================================================================================"
echo "Compliance:        ${COMPLIANCE_PCT}%"
echo "Rust Tests:        $([ $RUST_EXIT -eq 0 ] && echo -e "${GREEN}PASSED${NC}" || echo -e "${RED}FAILED${NC}")"
if command -v pytest &> /dev/null; then
    echo "Python Tests:      $([ $PYTHON_EXIT -eq 0 ] && echo -e "${GREEN}PASSED${NC}" || echo -e "${YELLOW}PARTIAL${NC}")"
fi
echo "Benchmarks:        $([ $BENCH_EXIT -eq 0 ] && echo -e "${GREEN}COMPLETED${NC}" || echo -e "${YELLOW}PARTIAL${NC}")"
echo ""
echo "Output files:"
echo "  - compliance_output.txt"
echo "  - rust_test_output.txt"
echo "  - python_test_output.txt (if pytest available)"
echo "  - benchmark_output.txt"
echo ""

if [ $RUST_EXIT -eq 0 ]; then
    echo -e "${GREEN}✅ SUCCESS: All critical tests passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ FAILURE: Some tests failed${NC}"
    exit 1
fi
