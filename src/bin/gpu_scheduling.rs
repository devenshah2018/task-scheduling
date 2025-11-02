use scheduling_lib::*;
use std::time::Duration;
use colored::*;

fn main() {
    // Initialize logging
    let logger = DualLogger::new("gpu").expect("Failed to initialize logger");
    
    logger.log_both(&format!("{}\n", "🎮 GPU SCHEDULING DEMONSTRATION".green().bold()));
    logger.log_both(&format!("{}\n", "================================".green()));
    logger.log_both("Demonstrating GPU scheduling for AI workloads and parallel computation\n\n");

    let resources = ResourceConstraints::default();
    
    // Create GPU kernels representing different AI workloads
    let kernels = create_ai_workload_kernels();
    
    logger.log_both("🧠 AI Workload Kernels:\n");
    for kernel in &kernels {
        logger.log_both(&format!("  • Kernel {}: {:?}\n", kernel.id, kernel.workload_type));
        logger.log_both(&format!("    Grid: {:?}, Block: {:?}, Threads: {}\n", 
            kernel.grid_size, kernel.block_size, kernel.total_threads()));
        logger.log_both(&format!("    Memory: {:.1}MB, Est. Time: {:.2}s\n", 
            kernel.memory_requirements as f64 / (1024.0 * 1024.0), 
            kernel.estimated_execution_time.as_secs_f64()));
    }

    // Test different GPU scheduling approaches
    run_gpu_scheduler_comparison(kernels, resources, &logger);
    
    let log_path = logger.finish();
    println!("\n📄 Logs saved to: {}", log_path.display());
}

fn create_ai_workload_kernels() -> Vec<Kernel> {
    vec![
        Kernel {
            id: 1,
            name: "GEMM Operation".to_string(),
            task_id: 101,
            grid_size: (64, 64, 1),
            block_size: (16, 16, 1),
            shared_memory_size: 48 * 1024,
            registers_per_thread: 32,
            priority: 5,
            estimated_execution_time: Duration::from_millis(800),
            memory_requirements: 512 * 1024 * 1024, // 512MB
            workload_type: WorkloadType::TensorOperation,
        },
        Kernel {
            id: 2,
            name: "CNN Forward Pass".to_string(),
            task_id: 102,
            grid_size: (128, 128, 1),
            block_size: (256, 1, 1),
            shared_memory_size: 32 * 1024,
            registers_per_thread: 64,
            priority: 7,
            estimated_execution_time: Duration::from_millis(1200),
            memory_requirements: 1024 * 1024 * 1024, // 1GB
            workload_type: WorkloadType::AITraining,
        },
        Kernel {
            id: 3,
            name: "Transformer Attention".to_string(),
            task_id: 103,
            grid_size: (32, 32, 8),
            block_size: (128, 1, 1),
            shared_memory_size: 64 * 1024,
            registers_per_thread: 48,
            priority: 8,
            estimated_execution_time: Duration::from_millis(600),
            memory_requirements: 768 * 1024 * 1024, // 768MB
            workload_type: WorkloadType::AIInference,
        },
        Kernel {
            id: 4,
            name: "RNN Backprop".to_string(),
            task_id: 104,
            grid_size: (96, 32, 1),
            block_size: (192, 1, 1),
            shared_memory_size: 40 * 1024,
            registers_per_thread: 56,
            priority: 6,
            estimated_execution_time: Duration::from_millis(1500),
            memory_requirements: 896 * 1024 * 1024, // 896MB
            workload_type: WorkloadType::AITraining,
        },
        Kernel {
            id: 5,
            name: "Batch Normalization".to_string(),
            task_id: 105,
            grid_size: (64, 32, 1),
            block_size: (64, 4, 1),
            shared_memory_size: 16 * 1024,
            registers_per_thread: 24,
            priority: 4,
            estimated_execution_time: Duration::from_millis(300),
            memory_requirements: 256 * 1024 * 1024, // 256MB
            workload_type: WorkloadType::TensorOperation,
        },
        Kernel {
            id: 6,
            name: "Adam Optimizer".to_string(),
            task_id: 106,
            grid_size: (48, 48, 1),
            block_size: (128, 2, 1),
            shared_memory_size: 24 * 1024,
            registers_per_thread: 40,
            priority: 5,
            estimated_execution_time: Duration::from_millis(400),
            memory_requirements: 320 * 1024 * 1024, // 320MB
            workload_type: WorkloadType::AITraining,
        },
        Kernel {
            id: 7,
            name: "Image Preprocessing".to_string(),
            task_id: 107,
            grid_size: (128, 64, 1),
            block_size: (32, 8, 1),
            shared_memory_size: 12 * 1024,
            registers_per_thread: 16,
            priority: 3,
            estimated_execution_time: Duration::from_millis(200),
            memory_requirements: 128 * 1024 * 1024, // 128MB
            workload_type: WorkloadType::GeneralCompute,
        },
        Kernel {
            id: 8,
            name: "Gradient Computation".to_string(),
            task_id: 108,
            grid_size: (80, 80, 1),
            block_size: (256, 1, 1),
            shared_memory_size: 56 * 1024,
            registers_per_thread: 52,
            priority: 7,
            estimated_execution_time: Duration::from_millis(900),
            memory_requirements: 640 * 1024 * 1024, // 640MB
            workload_type: WorkloadType::AITraining,
        },
    ]
}

fn run_gpu_scheduler_comparison(kernels: Vec<Kernel>, resources: ResourceConstraints, logger: &DualLogger) {
    logger.log_analysis("GPU SCHEDULING ALGORITHM COMPARISON", "");

    let schedulers: Vec<Box<dyn GPUScheduler>> = vec![
        Box::new(FIFOGPUScheduler::new(resources.clone(), 4)),
        Box::new(WarpScheduler::new(resources.clone(), 4, WarpSchedulingPolicy::GreedyThenOldest)),
        Box::new(WarpScheduler::new(resources.clone(), 4, WarpSchedulingPolicy::RoundRobin)),
        Box::new(WarpScheduler::new(resources.clone(), 4, WarpSchedulingPolicy::LoosestFirst)),
        Box::new(DynamicPriorityGPUScheduler::new(resources.clone(), 4)),
    ];

    let mut all_metrics = Vec::new();

    for mut scheduler in schedulers {
        let scheduler_name = scheduler.name().to_string();
        
        logger.log_both(&format!("\n{} Results:\n", scheduler_name.cyan().bold()));
        
        let metrics = scheduler.schedule_kernels_with_logger(kernels.clone(), Some(logger));
        
        // Capture metrics summary for logging
        let summary = format!(
            "📊 GPU Scheduling Metrics Summary\n\
             ═══════════════════════════════════\n\
             Total Kernels: {}\n\
             Completed Kernels: {}\n\
             Average Turnaround Time: {:.2}s\n\
             Throughput: {:.2} kernels/sec\n\
             GPU Utilization: {:.1}%\n\
             Total Execution Time: {:.2}s\n",
            metrics.total_tasks,
            metrics.completed_tasks,
            metrics.average_turnaround_time.as_secs_f64(),
            metrics.throughput,
            metrics.gpu_utilization.unwrap_or(0.0) * 100.0,
            metrics.total_execution_time.as_secs_f64()
        );
        
        logger.log_both(&summary);
        all_metrics.push((scheduler_name, metrics));
    }

    // GPU-specific performance analysis
    logger.log_analysis("GPU SCHEDULING PERFORMANCE ANALYSIS", "");
    
    let best_gpu_utilization = all_metrics.iter()
        .filter(|(_, m)| m.gpu_utilization.is_some())
        .max_by(|a, b| a.1.gpu_utilization.unwrap().partial_cmp(&b.1.gpu_utilization.unwrap()).unwrap());
    
    let best_throughput = all_metrics.iter()
        .max_by(|a, b| a.1.throughput.partial_cmp(&b.1.throughput).unwrap());

    let mut performance_summary = String::new();
    
    if let Some((name, metrics)) = best_gpu_utilization {
        let line = format!("🎮 Best GPU Utilization: {} ({:.1}%)\n", 
            name.green(), metrics.gpu_utilization.unwrap() * 100.0);
        performance_summary.push_str(&line);
    }
    if let Some((name, metrics)) = best_throughput {
        let line = format!("🚀 Best Kernel Throughput: {} ({:.2} kernels/sec)\n", 
            name.green(), metrics.throughput);
        performance_summary.push_str(&line);
    }
    
    logger.log_both(&performance_summary);

    // Analyze warp scheduling efficiency
    analyze_warp_scheduling_impact(&all_metrics, logger);
    
    // GPU architecture insights
    let insights = "\
🔬 GPU ARCHITECTURE INSIGHTS\n\
============================\n\
• SIMT execution model requires careful warp scheduling for optimal utilization\n\
• Dynamic priority scheduling adapts well to heterogeneous AI workloads\n\
• Warp scheduling policies significantly impact performance:\n\
  - Greedy-Then-Oldest (GTO): Good for memory coalescing\n\
  - Round Robin: Better fairness but more context switching\n\
  - Loosest First: Optimizes for thread utilization within warps\n\
• Memory bandwidth often becomes the bottleneck for AI workloads\n\
• Kernel occupancy and resource utilization are key performance factors\n";
    
    logger.log_analysis("GPU ARCHITECTURE INSIGHTS", insights);

    // AI workload specific insights
    let ai_insights = "\
🧠 AI WORKLOAD SCHEDULING INSIGHTS\n\
==================================\n\
• Training workloads benefit from priority scheduling due to iterative nature\n\
• Inference workloads require low-latency scheduling for real-time applications\n\
• Tensor operations show high parallelism and benefit from warp-level optimization\n\
• Memory-intensive operations require careful resource allocation\n\
• Gradient computations can be pipeline-optimized with proper scheduling\n";
    
    logger.log_analysis("AI WORKLOAD SCHEDULING INSIGHTS", ai_insights);
}

fn analyze_warp_scheduling_impact(metrics: &[(String, SchedulingMetrics)], logger: &DualLogger) {
    let warp_schedulers: Vec<_> = metrics.iter()
        .filter(|(name, _)| name.contains("Warp Scheduler"))
        .collect();
    
    let mut analysis = String::new();
    analysis.push_str("🔀 WARP SCHEDULING IMPACT ANALYSIS\n");
    analysis.push_str("==================================\n");
    
    if warp_schedulers.len() >= 2 {
        analysis.push_str("Comparing warp scheduling policies:\n");
        for (name, metrics) in warp_schedulers {
            let policy = if name.contains("Greedy") {
                "GTO"
            } else if name.contains("Round Robin") {
                "RR"
            } else if name.contains("Loosest") {
                "LRR"
            } else {
                "Unknown"
            };
            
            analysis.push_str(&format!("  • {}: Throughput = {:.2}, GPU Util = {:.1}%\n", 
                policy, 
                metrics.throughput,
                metrics.gpu_utilization.unwrap_or(0.0) * 100.0
            ));
        }
    }
    
    analysis.push_str("\nWarp scheduling directly impacts:\n");
    analysis.push_str("  • Memory access patterns and coalescing efficiency\n");
    analysis.push_str("  • SM occupancy and resource utilization\n");
    analysis.push_str("  • Thread divergence handling\n");
    analysis.push_str("  • Overall kernel execution latency\n");
    
    logger.log_analysis("WARP SCHEDULING IMPACT ANALYSIS", &analysis);
}
