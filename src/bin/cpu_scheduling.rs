use scheduling_lib::*;
use std::time::Duration;
use colored::*;

fn main() {
    // Initialize logging
    let logger = DualLogger::new("cpu").expect("Failed to initialize logger");
    
    logger.log_both(&format!("{}\n", "🖥️  CPU SCHEDULING DEMONSTRATION".blue().bold()));
    logger.log_both(&format!("{}\n", "=====================================".blue()));
    logger.log_both("Demonstrating traditional CPU scheduling algorithms for research analysis\n\n");

    let resources = ResourceConstraints::default();
    
    // Create diverse workload tasks
    let tasks = create_sample_tasks();
    
    logger.log_both("📋 Sample Workload Tasks:\n");
    for task in &tasks {
        logger.log_both(&format!("  • Task {}: {:?} - {:.1}s, Priority: {}, Memory: {}MB\n", 
            task.id, task.workload_type, task.burst_time.as_secs_f64(), 
            task.priority, task.memory_requirement));
    }

    // Test different CPU scheduling algorithms
    run_cpu_scheduler_comparison(tasks, resources, &logger);
    
    let log_path = logger.finish();
    println!("\n📄 Logs saved to: {}", log_path.display());
}

fn create_sample_tasks() -> Vec<Task> {
    vec![
        Task::new(1, "General Computation".to_string(), WorkloadType::GeneralCompute, 
                 Duration::from_secs(3), 2, 512, 0.3, false),
        Task::new(2, "AI Model Training".to_string(), WorkloadType::AITraining, 
                 Duration::from_secs(8), 5, 2048, 0.9, true),
        Task::new(3, "Database Query".to_string(), WorkloadType::IOBound, 
                 Duration::from_secs(2), 1, 256, 0.1, false),
        Task::new(4, "Matrix Multiplication".to_string(), WorkloadType::TensorOperation, 
                 Duration::from_secs(4), 4, 1024, 0.8, true),
        Task::new(5, "File Processing".to_string(), WorkloadType::MemoryIntensive, 
                 Duration::from_secs(6), 3, 1536, 0.4, false),
        Task::new(6, "Neural Network Inference".to_string(), WorkloadType::AIInference, 
                 Duration::from_secs(1), 6, 512, 0.7, true),
        Task::new(7, "System Monitoring".to_string(), WorkloadType::GeneralCompute, 
                 Duration::from_secs(2), 1, 128, 0.2, false),
        Task::new(8, "Image Processing".to_string(), WorkloadType::TensorOperation, 
                 Duration::from_secs(5), 4, 1024, 0.9, true),
    ]
}

fn run_cpu_scheduler_comparison(tasks: Vec<Task>, resources: ResourceConstraints, logger: &DualLogger) {
    let schedulers: Vec<Box<dyn CPUScheduler>> = vec![
        Box::new(FCFSScheduler::new(resources.clone())),
        Box::new(RoundRobinScheduler::new(resources.clone(), Duration::from_millis(500))),
        Box::new(SJFScheduler::new(resources.clone(), false)),
        Box::new(SJFScheduler::new(resources.clone(), true)), // SRTF
        Box::new(PriorityScheduler::new(resources.clone(), false)),
        Box::new(PriorityScheduler::new(resources.clone(), true)),
        Box::new(MLFQScheduler::new(resources.clone())),
    ];

    logger.log_analysis("CPU SCHEDULING ALGORITHM COMPARISON", "");

    let mut all_metrics = Vec::new();

    for mut scheduler in schedulers {
        let scheduler_name = scheduler.name().to_string();
        
        // Log scheduler start
        logger.log_both(&format!("\n{} Results:\n", scheduler_name.cyan().bold()));
        
        let metrics = scheduler.schedule_with_logger(tasks.clone(), Some(logger));
        
        // Capture metrics summary for logging
        let summary = format!(
            "📊 Scheduling Metrics Summary\n\
             ════════════════════════════════\n\
             Total Tasks: {}\n\
             Completed Tasks: {}\n\
             Average Turnaround Time: {:.2}s\n\
             Average Waiting Time: {:.2}s\n\
             Throughput: {:.2} tasks/sec\n\
             CPU Utilization: {:.1}%\n\
             Total Execution Time: {:.2}s\n",
            metrics.total_tasks,
            metrics.completed_tasks,
            metrics.average_turnaround_time.as_secs_f64(),
            metrics.average_waiting_time.as_secs_f64(),
            metrics.throughput,
            metrics.cpu_utilization * 100.0,
            metrics.total_execution_time.as_secs_f64()
        );
        
        logger.log_both(&summary);
        
        all_metrics.push((scheduler_name, metrics));
    }

    // Performance comparison
    logger.log_analysis("PERFORMANCE COMPARISON SUMMARY", "");
    
    // Find best performers
    let best_throughput = all_metrics.iter()
        .max_by(|a, b| a.1.throughput.partial_cmp(&b.1.throughput).unwrap());
    let best_turnaround = all_metrics.iter()
        .min_by(|a, b| a.1.average_turnaround_time.partial_cmp(&b.1.average_turnaround_time).unwrap());
    let best_waiting = all_metrics.iter()
        .min_by(|a, b| a.1.average_waiting_time.partial_cmp(&b.1.average_waiting_time).unwrap());
    let best_utilization = all_metrics.iter()
        .max_by(|a, b| a.1.cpu_utilization.partial_cmp(&b.1.cpu_utilization).unwrap());

    let mut comparison_summary = String::new();
    
    if let Some((name, metrics)) = best_throughput {
        let line = format!("🚀 Best Throughput: {} ({:.2} tasks/sec)\n", name.green(), metrics.throughput);
        comparison_summary.push_str(&line);
    }
    if let Some((name, metrics)) = best_turnaround {
        let line = format!("⏱️  Best Turnaround Time: {} ({:.2}s)\n", name.green(), metrics.average_turnaround_time.as_secs_f64());
        comparison_summary.push_str(&line);
    }
    if let Some((name, metrics)) = best_waiting {
        let line = format!("⏳ Best Waiting Time: {} ({:.2}s)\n", name.green(), metrics.average_waiting_time.as_secs_f64());
        comparison_summary.push_str(&line);
    }
    if let Some((name, metrics)) = best_utilization {
        let line = format!("💪 Best CPU Utilization: {} ({:.1}%)\n", name.green(), metrics.cpu_utilization * 100.0);
        comparison_summary.push_str(&line);
    }
    
    logger.log_both(&comparison_summary);

    // Analysis insights
    let insights = "\
📈 RESEARCH INSIGHTS\n\
===================\n\
• FCFS shows simplicity but poor performance for varied workloads\n\
• Round Robin provides fairness at the cost of context switching overhead\n\
• SJF optimizes average waiting time but may cause starvation\n\
• SRTF (preemptive SJF) reduces turnaround time for short tasks\n\
• Priority scheduling handles different task importance levels\n\
• MLFQ adapts to changing task behavior and provides good interactivity\n\
• AI/ML workloads benefit from priority-based scheduling due to their resource intensity\n";
    
    logger.log_analysis("RESEARCH INSIGHTS", insights);
}
