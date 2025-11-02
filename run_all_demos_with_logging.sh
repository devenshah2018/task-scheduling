#!/bin/bash

# Comprehensive logging test script for all scheduling demonstrations
# This script runs all demos and saves organized logs

echo "🚀 Running Comprehensive Scheduling Demonstration with Logging"
echo "=============================================================="

cd /Users/devenshah/repos/cpu-scheduling

echo ""
echo "📊 1. Running CPU Scheduling Demo..."
cargo run --bin cpu_scheduling_demo

echo ""
echo "🎮 2. Running GPU Scheduling Demo..."
timeout 30s cargo run --bin gpu_scheduling_demo

echo ""
echo "🔄 3. Running Hybrid Scheduling Demo..."
timeout 30s cargo run --bin hybrid_scheduling_demo

echo ""
echo "🏆 4. Running Comprehensive Benchmark Demo..."
timeout 45s cargo run --bin benchmark_demo

echo ""
echo "📁 Log Directory Summary:"
echo "========================"
ls -la logs/

echo ""
echo "📄 Latest Log Files:"
echo "==================="
for dir in logs/*/; do
    echo "Directory: $dir"
    echo "Files:"
    ls -la "$dir"
    echo ""
done

echo "✅ All demonstrations completed with logging!"
echo "💾 Check the logs/ directory for timestamped results"
