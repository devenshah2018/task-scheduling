use scheduling_lib::*;
use std::time::{Duration, Instant};
use colored::*;
use rand::Rng;

fn main() {
    let logger = DualLogger::new("benchmark").expect("Failed to initialize logger");
    
    logger.log_both(&format!("{}\n", "📊 COMPREHENSIVE SCHEDULING BENCHMARK".purple().bold()));
    logger.log_both(&format!("{}\n", "=====================================".purple()));
    logger.log_both("Comprehensive performance analysis of CPU, GPU, and Hybrid scheduling\n\n");

    let resources = ResourceConstraints::default();
    
    run_comprehensive_benchmark(resources, &logger);
    
    let log_path = logger.finish();
    println!("\n📄 Logs saved to: {}", log_path.display());
}

fn run_comprehensive_benchmark(resources: ResourceConstraints, logger: &DualLogger) {
    logger.log_analysis("BENCHMARK TEST SUITES", "");
    
    benchmark_workload_scaling(resources.clone(), logger);
    benchmark_workload_composition(resources.clone(), logger);
    benchmark_resource_constraints(resources.clone(), logger);
    benchmark_ai_workloads(resources, logger);
}

fn benchmark_workload_scaling(resources: ResourceConstraints, logger: &DualLogger) {
    logger.log_analysis("WORKLOAD SCALING BENCHMARK", "");
    
    let workload_sizes = vec![5, 10, 20, 50];
    
    for size in workload_sizes {
        logger.log_both(&format!("\n📊 Testing with {} tasks:\n", size));
        
        let tasks = generate_random_tasks(size);
        
        let mut cpu_scheduler = RoundRobinScheduler::new(resources.clone(), Duration::from_millis(300));
        let start_time = Instant::now();
        let cpu_metrics = cpu_scheduler.schedule_with_logger(tasks.clone(), Some(logger));
        let cpu_duration = start_time.elapsed();
        
        let mut hybrid_scheduler = HybridScheduler::new(resources.clone());
        let start_time = Instant::now();
        let hybrid_metrics = hybrid_scheduler.schedule_hybrid_tasks(tasks);
        let hybrid_duration = start_time.elapsed();
        
        let throughput_improvement = ((hybrid_metrics.throughput - cpu_metrics.throughput) 
            / cpu_metrics.throughput) * 100.0;
        let scheduling_overhead = hybrid_duration.as_secs_f64() / cpu_duration.as_secs_f64();
        
        let results = format!(
            "  CPU-only:    {:.2} tasks/sec\n\
             Hybrid:      {:.2} tasks/sec ({:.1}% improvement)\n\
             Overhead:    {:.2}x scheduling time\n",
            cpu_metrics.throughput,
            hybrid_metrics.throughput, throughput_improvement,
            scheduling_overhead
        );
        logger.log_both(&results);
        
        if let Some(gpu_util) = hybrid_metrics.gpu_utilization {
            logger.log_both(&format!("  GPU Usage:   {:.1}%\n", gpu_util * 100.0));
        }
    }
}

fn benchmark_workload_composition(resources: ResourceConstraints, logger: &DualLogger) {
    logger.log_analysis("WORKLOAD COMPOSITION BENCHMARK", "");
    
    let compositions = vec![
        ("AI-Heavy", 0.7, 0.2, 0.1),      // 70% AI, 20% tensor, 10% general
        ("Balanced", 0.3, 0.3, 0.4),     // Balanced workload
        ("General-Heavy", 0.1, 0.2, 0.7), // 70% general compute
        ("Mixed-ML", 0.4, 0.4, 0.2),     // Mixed machine learning
    ];
    
    for (name, ai_ratio, tensor_ratio, general_ratio) in compositions {
        logger.log_both(&format!("\n🎯 {} Workload Composition:\n", name));
        
        let tasks = generate_composed_workload(20, ai_ratio, tensor_ratio, general_ratio);
        
        let schedulers_results = test_multiple_schedulers(tasks, resources.clone(), logger);
        
        let best_scheduler = schedulers_results.iter()
            .max_by(|a, b| a.1.throughput.partial_cmp(&b.1.throughput).unwrap())
            .unwrap();
        
        logger.log_both(&format!("  Best Scheduler: {} ({:.2} tasks/sec)\n", 
            best_scheduler.0.green().bold(), best_scheduler.1.throughput));
        
        analyze_composition_performance(name, &schedulers_results, logger);
    }
}

fn benchmark_resource_constraints(resources: ResourceConstraints, logger: &DualLogger) {
    logger.log_analysis("RESOURCE CONSTRAINT BENCHMARK", "");
    
    let resource_configs = vec![
        ("High-End", ResourceConstraints {
            cpu_cores: 16,
            total_memory: 32768,
            gpu_cores: Some(4096),
            gpu_memory: Some(16384),
        }),
        ("Mid-Range", ResourceConstraints {
            cpu_cores: 8,
            total_memory: 16384,
            gpu_cores: Some(2048),
            gpu_memory: Some(8192),
        }),
        ("Budget", ResourceConstraints {
            cpu_cores: 4,
            total_memory: 8192,
            gpu_cores: Some(1024),
            gpu_memory: Some(4096),
        }),
        ("CPU-Only", ResourceConstraints {
            cpu_cores: 8,
            total_memory: 16384,
            gpu_cores: None,
            gpu_memory: None,
        }),
    ];
    
    let test_tasks = generate_ai_heavy_workload(15);
    
    for (config_name, config) in resource_configs {
        logger.log_both(&format!("\n🖥️  {} Configuration:\n", config_name));
        logger.log_both(&format!("  CPU Cores: {}, Memory: {}MB\n", config.cpu_cores, config.total_memory));
        if let Some(gpu_cores) = config.gpu_cores {
            logger.log_both(&format!("  GPU Cores: {}, GPU Memory: {}MB\n", 
                gpu_cores, config.gpu_memory.unwrap_or(0)));
        } else {
            logger.log_both("  GPU: None\n");
        }
        
        if config.gpu_cores.is_some() {
            let mut hybrid_scheduler = HybridScheduler::new(config.clone());
            let metrics = hybrid_scheduler.schedule_hybrid_tasks(test_tasks.clone());
            logger.log_both(&format!("  Performance: {:.2} tasks/sec, {:.1}% GPU utilization\n",
                metrics.throughput, 
                metrics.gpu_utilization.unwrap_or(0.0) * 100.0));
        } else {
            let mut cpu_scheduler = MLFQScheduler::new(config);
            let metrics = cpu_scheduler.schedule_with_logger(test_tasks.clone(), Some(logger));
            logger.log_both(&format!("  Performance: {:.2} tasks/sec (CPU-only)\n", metrics.throughput));
        }
    }
}

fn benchmark_ai_workloads(resources: ResourceConstraints, logger: &DualLogger) {
    logger.log_analysis("AI WORKLOAD SPECIFIC BENCHMARK", "");
    
    let ai_scenarios = vec![
        ("Training Pipeline", generate_training_pipeline()),
        ("Inference Serving", generate_inference_workload()),
        ("Research Experiment", generate_research_workload()),
        ("Production MLOps", generate_mlops_workload()),
    ];
    
    for (scenario_name, tasks) in ai_scenarios {
        logger.log_both(&format!("\n🎯 {} Scenario:\n", scenario_name));
        
        let mut hybrid_scheduler = HybridScheduler::new(resources.clone());
        let hybrid_metrics = hybrid_scheduler.schedule_hybrid_tasks(tasks.clone());
        
        let gpu_metrics = test_gpu_priority_approach(tasks.clone(), resources.clone(), logger);
        
        let mut cpu_scheduler = PriorityScheduler::new(resources.clone(), true);
        let cpu_metrics = cpu_scheduler.schedule_with_logger(tasks, Some(logger));
        
        let results = format!(
            "  Hybrid Scheduling:    {:.2} tasks/sec\n\
             GPU-Priority:         {:.2} tasks/sec\n\
             CPU Baseline:         {:.2} tasks/sec\n",
            hybrid_metrics.throughput,
            gpu_metrics,
            cpu_metrics.throughput
        );
        logger.log_both(&results);
        
        let improvement = ((hybrid_metrics.throughput - cpu_metrics.throughput) 
            / cpu_metrics.throughput) * 100.0;
        logger.log_both(&format!("  Improvement:          {:.1}% over CPU-only\n", improvement));
        
        provide_scenario_insights(scenario_name, improvement, logger);
    }
}

fn generate_random_tasks(count: usize) -> Vec<Task> {
    let mut rng = rand::thread_rng();
    let mut tasks = Vec::new();
    
    let workload_types = vec![
        WorkloadType::AITraining,
        WorkloadType::AIInference,
        WorkloadType::TensorOperation,
        WorkloadType::GeneralCompute,
        WorkloadType::MemoryIntensive,
        WorkloadType::IOBound,
    ];
    
    for i in 0..count {
        let workload_type = workload_types[rng.gen_range(0..workload_types.len())].clone();
        let burst_time = Duration::from_millis(rng.gen_range(500..5000));
        let priority = rng.gen_range(1..10);
        let memory = rng.gen_range(128..4096);
        let parallelism = rng.gen::<f32>();
        let gpu_compat = rng.gen_bool(0.6); // 60% GPU compatible
        
        tasks.push(Task::new(
            i as u32 + 1,
            format!("RandomTask_{}", i + 1),
            workload_type,
            burst_time,
            priority,
            memory,
            parallelism,
            gpu_compat,
        ));
    }
    
    tasks
}

fn generate_composed_workload(count: usize, ai_ratio: f32, tensor_ratio: f32, general_ratio: f32) -> Vec<Task> {
    let mut tasks = Vec::new();
    let mut rng = rand::thread_rng();
    
    let ai_count = (count as f32 * ai_ratio) as usize;
    let tensor_count = (count as f32 * tensor_ratio) as usize;
    let general_count = count - ai_count - tensor_count;
    
    for i in 0..ai_count {
        let workload_type = if rng.gen_bool(0.6) { 
            WorkloadType::AITraining 
        } else { 
            WorkloadType::AIInference 
        };
        tasks.push(Task::new(
            i as u32 + 1,
            format!("AI_Task_{}", i + 1),
            workload_type,
            Duration::from_millis(rng.gen_range(1000..8000)),
            rng.gen_range(6..10),
            rng.gen_range(1024..4096),
            rng.gen_range(0.7..1.0),
            true,
        ));
    }
    
    for i in 0..tensor_count {
        tasks.push(Task::new(
            (ai_count + i) as u32 + 1,
            format!("Tensor_Task_{}", i + 1),
            WorkloadType::TensorOperation,
            Duration::from_millis(rng.gen_range(500..3000)),
            rng.gen_range(4..8),
            rng.gen_range(512..2048),
            rng.gen_range(0.5..0.9),
            true,
        ));
    }
    
    for i in 0..general_count {
        let workload_type = if rng.gen_bool(0.5) { 
            WorkloadType::GeneralCompute 
        } else if rng.gen_bool(0.3) {
            WorkloadType::MemoryIntensive
        } else {
            WorkloadType::IOBound
        };
        tasks.push(Task::new(
            (ai_count + tensor_count + i) as u32 + 1,
            format!("General_Task_{}", i + 1),
            workload_type,
            Duration::from_millis(rng.gen_range(200..2000)),
            rng.gen_range(1..6),
            rng.gen_range(128..1024),
            rng.gen_range(0.1..0.5),
            rng.gen_bool(0.2),
        ));
    }
    
    tasks
}

fn generate_ai_heavy_workload(count: usize) -> Vec<Task> {
    generate_composed_workload(count, 0.8, 0.15, 0.05)
}

fn generate_training_pipeline() -> Vec<Task> {
    vec![
        Task::new(1, "Data Loading".to_string(), WorkloadType::IOBound, 
                 Duration::from_secs(2), 3, 1024, 0.2, false),
        Task::new(2, "Preprocessing".to_string(), WorkloadType::TensorOperation, 
                 Duration::from_secs(4), 5, 2048, 0.8, true),
        Task::new(3, "Forward Pass".to_string(), WorkloadType::AITraining, 
                 Duration::from_secs(10), 9, 4096, 0.95, true),
        Task::new(4, "Backward Pass".to_string(), WorkloadType::AITraining, 
                 Duration::from_secs(15), 9, 4096, 0.95, true),
        Task::new(5, "Optimizer Step".to_string(), WorkloadType::AITraining, 
                 Duration::from_secs(3), 7, 1024, 0.6, true),
    ]
}

fn generate_inference_workload() -> Vec<Task> {
    vec![
        Task::new(1, "Input Processing".to_string(), WorkloadType::TensorOperation, 
                 Duration::from_millis(100), 8, 256, 0.7, true),
        Task::new(2, "Model Inference".to_string(), WorkloadType::AIInference, 
                 Duration::from_millis(500), 9, 1024, 0.9, true),
        Task::new(3, "Output Processing".to_string(), WorkloadType::TensorOperation, 
                 Duration::from_millis(50), 6, 128, 0.5, true),
        Task::new(4, "Response Formatting".to_string(), WorkloadType::GeneralCompute, 
                 Duration::from_millis(20), 4, 64, 0.2, false),
    ]
}

fn generate_research_workload() -> Vec<Task> {
    let mut rng = rand::thread_rng();
    let mut tasks = Vec::new();
    
    for i in 0..12 {
        let workload_type = match i % 4 {
            0 => WorkloadType::AITraining,
            1 => WorkloadType::TensorOperation,
            2 => WorkloadType::AIInference,
            _ => WorkloadType::GeneralCompute,
        };
        
        tasks.push(Task::new(
            i as u32 + 1,
            format!("Research_Experiment_{}", i + 1),
            workload_type,
            Duration::from_millis(rng.gen_range(500..6000)),
            rng.gen_range(4..9),
            rng.gen_range(512..3072),
            rng.gen_range(0.6..0.95),
            i % 4 != 3, // Most tasks GPU compatible except general compute
        ));
    }
    
    tasks
}

fn generate_mlops_workload() -> Vec<Task> {
    vec![
        Task::new(1, "Data Validation".to_string(), WorkloadType::GeneralCompute, 
                 Duration::from_secs(1), 4, 512, 0.3, false),
        Task::new(2, "Feature Engineering".to_string(), WorkloadType::TensorOperation, 
                 Duration::from_secs(3), 6, 1536, 0.7, true),
        Task::new(3, "Model Training".to_string(), WorkloadType::AITraining, 
                 Duration::from_secs(20), 9, 6144, 0.95, true),
        Task::new(4, "Model Validation".to_string(), WorkloadType::AIInference, 
                 Duration::from_secs(2), 7, 1024, 0.8, true),
        Task::new(5, "Model Deployment".to_string(), WorkloadType::IOBound, 
                 Duration::from_secs(3), 5, 256, 0.1, false),
        Task::new(6, "Monitoring Setup".to_string(), WorkloadType::GeneralCompute, 
                 Duration::from_secs(1), 3, 128, 0.2, false),
    ]
}

fn test_multiple_schedulers(tasks: Vec<Task>, resources: ResourceConstraints, logger: &DualLogger) -> Vec<(String, SchedulingMetrics)> {
    let mut results = Vec::new();
    
    let mut fcfs = FCFSScheduler::new(resources.clone());
    results.push(("FCFS".to_string(), fcfs.schedule_with_logger(tasks.clone(), Some(logger))));
    
    let mut rr = RoundRobinScheduler::new(resources.clone(), Duration::from_millis(400));
    results.push(("Round Robin".to_string(), rr.schedule_with_logger(tasks.clone(), Some(logger))));
    
    let mut priority = PriorityScheduler::new(resources.clone(), true);
    results.push(("Priority".to_string(), priority.schedule_with_logger(tasks.clone(), Some(logger))));
    
    let mut hybrid = HybridScheduler::new(resources);
    results.push(("Hybrid".to_string(), hybrid.schedule_hybrid_tasks(tasks)));
    
    results
}

fn test_gpu_priority_approach(tasks: Vec<Task>, resources: ResourceConstraints, logger: &DualLogger) -> f64 {
    let gpu_compatible_count = tasks.iter().filter(|t| t.gpu_compatibility).count();
    logger.log_full_only(&format!("GPU-compatible tasks: {}/{}", gpu_compatible_count, tasks.len()));
    
    let total_time: Duration = tasks.iter()
        .map(|t| if t.gpu_compatibility { 
            Duration::from_secs_f64(t.burst_time.as_secs_f64() / 3.0)
        } else { 
            t.burst_time 
        })
        .sum();
    
    tasks.len() as f64 / total_time.as_secs_f64()
}

fn analyze_composition_performance(composition: &str, results: &[(String, SchedulingMetrics)], logger: &DualLogger) {
    let hybrid_result = results.iter().find(|(name, _)| name == "Hybrid");
    let cpu_baseline = results.iter().find(|(name, _)| name == "Round Robin");
    
    if let (Some((_, hybrid)), Some((_, baseline))) = (hybrid_result, cpu_baseline) {
        let improvement = ((hybrid.throughput - baseline.throughput) / baseline.throughput) * 100.0;
        
        let analysis = match composition {
            "AI-Heavy" => format!(
                "    → AI-heavy workloads show excellent GPU utilization\n\
                 → Improvement: {:.1}% (expected: high due to GPU acceleration)\n", improvement
            ),
            "Balanced" => format!(
                "    → Balanced workload demonstrates adaptive scheduling\n\
                 → Improvement: {:.1}% (moderate, optimal resource distribution)\n", improvement
            ),
            "General-Heavy" => format!(
                "    → General compute benefits from CPU optimization\n\
                 → Improvement: {:.1}% (lower, CPU-bound workload)\n", improvement
            ),
            "Mixed-ML" => format!(
                "    → Mixed ML workload shows strong hybrid performance\n\
                 → Improvement: {:.1}% (high, good CPU-GPU coordination)\n", improvement
            ),
            _ => String::new()
        };
        
        if !analysis.is_empty() {
            logger.log_both(&analysis);
        }
    }
}

fn provide_scenario_insights(scenario: &str, improvement: f64, logger: &DualLogger) {
    let insights = match scenario {
        "Training Pipeline" => {
            let mut insight = "    💡 Training benefits from GPU acceleration and CPU-GPU coordination\n".to_string();
            if improvement > 200.0 {
                insight.push_str("    ✨ Excellent improvement indicates optimal GPU utilization\n");
            }
            insight
        },
        "Inference Serving" => {
            let mut insight = "    💡 Inference requires low-latency scheduling and fast task switching\n".to_string();
            if improvement > 150.0 {
                insight.push_str("    ✨ Strong improvement shows effective latency optimization\n");
            }
            insight
        },
        "Research Experiment" => {
            let mut insight = "    💡 Research workloads benefit from flexible resource allocation\n".to_string();
            if improvement > 100.0 {
                insight.push_str("    ✨ Good improvement demonstrates adaptive scheduling effectiveness\n");
            }
            insight
        },
        "Production MLOps" => {
            let mut insight = "    💡 Production requires balanced performance and resource efficiency\n".to_string();
            if improvement > 175.0 {
                insight.push_str("    ✨ Strong improvement indicates production-ready performance\n");
            }
            insight
        },
        _ => String::new()
    };
    
    if !insights.is_empty() {
        logger.log_both(&insights);
    }
}
