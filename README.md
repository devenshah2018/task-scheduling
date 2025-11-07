# CPU-GPU Scheduling Research Code

This repository contains supplemental code for research on CPU and GPU scheduling algorithms, with a focus on AI workloads and hybrid CPU-GPU scheduling strategies.

## Overview

The rapid growth of artificial intelligence and deep learning has fundamentally changed how modern hardware and operating systems handle computation. This project implements and demonstrates:

1. **Traditional CPU Scheduling Algorithms** - FCFS, Round Robin, SJF/SRTF, Priority Scheduling, MLFQ
2. **GPU Scheduling for AI Workloads** - FIFO, Warp Scheduling, Dynamic Priority
3. **Hybrid CPU-GPU Scheduling** - Intelligent placement, dynamic migration, cooperative scheduling

## Project Structure

```
cpu-scheduling/
├── src/
│   ├── lib.rs              # Main library exports and core traits
│   ├── cpu.rs              # Traditional CPU scheduling algorithms
│   ├── gpu.rs              # GPU scheduling implementations  
│   ├── hybrid.rs           # Hybrid CPU-GPU scheduling
│   ├── logger.rs           # Dual logging system (simple + full logs)
│   └── bin/
│       ├── cpu_scheduling.rs    # CPU scheduling demonstration
│       ├── gpu_scheduling.rs    # GPU scheduling demonstration
│       ├── hybrid_scheduling.rs # Hybrid scheduling demonstration
│       └── benchmark.rs         # Comprehensive benchmarks
├── logs/                   # Generated log directories (created at runtime)
│   ├── cpu_YYYYMMDD_HHMMSS/     # CPU scheduling logs
│   ├── gpu_YYYYMMDD_HHMMSS/     # GPU scheduling logs
│   ├── hybrid_YYYYMMDD_HHMMSS/  # Hybrid scheduling logs
│   └── benchmark_YYYYMMDD_HHMMSS/ # Benchmark logs
├── Cargo.toml              # Project dependencies and configuration
├── .gitignore              # Git ignore patterns
└── README.md               # This file
```

## Key Features

### CPU Scheduling Algorithms

- **First Come First Serve (FCFS)**: Simple queue-based scheduling
- **Round Robin**: Time-sliced fair scheduling with configurable quantum
- **Shortest Job First (SJF)**: Non-preemptive shortest job scheduling
- **Shortest Remaining Time First (SRTF)**: Preemptive SJF variant
- **Priority Scheduling**: Both preemptive and non-preemptive variants
- **Multi-level Feedback Queue (MLFQ)**: Adaptive multi-queue scheduling

### GPU Scheduling Implementations

- **FIFO GPU Scheduler**: Traditional first-in-first-out kernel scheduling
- **Warp Scheduler**: SIMT-aware scheduling with multiple policies:
  - Greedy-Then-Oldest (GTO)
  - Round Robin (RR)
  - Loosest First (LRR)
- **Dynamic Priority GPU Scheduler**: AI workload-aware priority adjustment

### Hybrid CPU-GPU Features

- **Intelligent Task Placement**: Workload prediction and optimal unit selection
- **Dynamic Load Balancing**: Real-time migration based on utilization
- **Memory Locality Optimization**: Reduced data transfer overhead
- **Cooperative Scheduling**: Large task splitting and coordination

## Installation and Usage

### Prerequisites

- Rust 1.70+ 
- Cargo package manager

### Build and Run

```bash
# Clone the repository
git clone https://github.com/devenshah2018/task-scheduling.git
cd cpu-scheduling

# Build the project
cargo build --release

# Run individual demonstrations
cargo run --bin cpu_scheduling_demo
cargo run --bin gpu_scheduling_demo
cargo run --bin hybrid_scheduling_demo

# Run comprehensive benchmarks
cargo run --bin benchmark_demo
```

## Generated Outputs

Each demo creates timestamped log directories in `logs/`:
- `logs/cpu_YYYYMMDD_HHMMSS/` - CPU scheduling analysis
- `logs/gpu_YYYYMMDD_HHMMSS/` - GPU scheduling results
- `logs/hybrid_YYYYMMDD_HHMMSS/` - Hybrid scheduling insights
- `logs/benchmark_YYYYMMDD_HHMMSS/` - Performance comparisons

Each directory contains:
- `simple.log` - Clean analysis results (research-ready, no color codes)
- `full.log` - Detailed execution traces with all system output

### Log Structure Example
```
logs/
├── cpu_20241102_173445/
│   ├── simple.log          # Research analysis only
│   └── full.log            # Complete execution details
├── gpu_20241102_173733/
│   ├── simple.log          # GPU scheduling analysis
│   └── full.log            # Full GPU execution trace
└── hybrid_20241102_174012/
    ├── simple.log          # Hybrid performance insights
    └── full.log            # Detailed hybrid execution log
```

## Dependencies

- `rand`: Random number generation for workload simulation
- `tokio`: Asynchronous runtime for concurrent operations
- `serde`: Serialization for metrics and configuration
- `chrono`: Date and time handling for timestamped logs
- `colored`: Terminal output formatting
- `rayon`: Data parallelism
- `crossbeam`: Lock-free data structures
