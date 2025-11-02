use std::time::{Duration, Instant};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Represents different types of computational workloads
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum WorkloadType {
    GeneralCompute,
    AITraining,
    AIInference,
    TensorOperation,
    MemoryIntensive,
    IOBound,
}

/// Process/Task representation for scheduling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: u32,
    pub name: String,
    pub workload_type: WorkloadType,
    pub arrival_time: DateTime<Utc>,
    pub burst_time: Duration,
    pub priority: u8,
    pub memory_requirement: usize, // MB
    pub parallelism_factor: f32,   // How well it can be parallelized (0.0 - 1.0)
    pub gpu_compatibility: bool,
    pub remaining_time: Duration,
    pub start_time: Option<DateTime<Utc>>,
    pub completion_time: Option<DateTime<Utc>>,
}

impl Task {
    pub fn new(
        id: u32,
        name: String,
        workload_type: WorkloadType,
        burst_time: Duration,
        priority: u8,
        memory_requirement: usize,
        parallelism_factor: f32,
        gpu_compatibility: bool,
    ) -> Self {
        Self {
            id,
            name,
            workload_type,
            arrival_time: Utc::now(),
            burst_time,
            priority,
            memory_requirement,
            parallelism_factor,
            gpu_compatibility,
            remaining_time: burst_time,
            start_time: None,
            completion_time: None,
        }
    }

    pub fn turnaround_time(&self) -> Option<Duration> {
        if let (Some(completion), arrival) = (self.completion_time, self.arrival_time) {
            Some(completion.signed_duration_since(arrival).to_std().unwrap_or_default())
        } else {
            None
        }
    }

    pub fn waiting_time(&self) -> Option<Duration> {
        if let Some(turnaround) = self.turnaround_time() {
            Some(turnaround.saturating_sub(self.burst_time))
        } else {
            None
        }
    }
}

/// Scheduling metrics for performance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingMetrics {
    pub total_tasks: usize,
    pub completed_tasks: usize,
    pub average_turnaround_time: Duration,
    pub average_waiting_time: Duration,
    pub throughput: f64, // tasks per second
    pub cpu_utilization: f64,
    pub gpu_utilization: Option<f64>,
    pub total_execution_time: Duration,
}

impl SchedulingMetrics {
    pub fn calculate(tasks: &[Task], execution_start: Instant, execution_end: Instant) -> Self {
        let completed_tasks: Vec<_> = tasks.iter().filter(|t| t.completion_time.is_some()).collect();
        let total_tasks = tasks.len();
        let completed_count = completed_tasks.len();

        let avg_turnaround = if completed_count > 0 {
            let total_turnaround: Duration = completed_tasks
                .iter()
                .filter_map(|t| t.turnaround_time())
                .sum();
            total_turnaround / completed_count as u32
        } else {
            Duration::ZERO
        };

        let avg_waiting = if completed_count > 0 {
            let total_waiting: Duration = completed_tasks
                .iter()
                .filter_map(|t| t.waiting_time())
                .sum();
            total_waiting / completed_count as u32
        } else {
            Duration::ZERO
        };

        let total_execution_time = execution_end.duration_since(execution_start);
        let throughput = if total_execution_time.as_secs_f64() > 0.0 {
            completed_count as f64 / total_execution_time.as_secs_f64()
        } else {
            0.0
        };

        // Calculate CPU utilization (simplified)
        let total_cpu_time: Duration = completed_tasks
            .iter()
            .map(|t| t.burst_time)
            .sum();
        let cpu_utilization = if total_execution_time.as_secs_f64() > 0.0 {
            (total_cpu_time.as_secs_f64() / total_execution_time.as_secs_f64()).min(1.0)
        } else {
            0.0
        };

        Self {
            total_tasks,
            completed_tasks: completed_count,
            average_turnaround_time: avg_turnaround,
            average_waiting_time: avg_waiting,
            throughput,
            cpu_utilization,
            gpu_utilization: None,
            total_execution_time,
        }
    }

    pub fn print_summary(&self) {
        println!("\n📊 Scheduling Metrics Summary");
        println!("════════════════════════════════");
        println!("Total Tasks: {}", self.total_tasks);
        println!("Completed Tasks: {}", self.completed_tasks);
        println!("Average Turnaround Time: {:.2}s", self.average_turnaround_time.as_secs_f64());
        println!("Average Waiting Time: {:.2}s", self.average_waiting_time.as_secs_f64());
        println!("Throughput: {:.2} tasks/sec", self.throughput);
        println!("CPU Utilization: {:.1}%", self.cpu_utilization * 100.0);
        if let Some(gpu_util) = self.gpu_utilization {
            println!("GPU Utilization: {:.1}%", gpu_util * 100.0);
        }
        println!("Total Execution Time: {:.2}s", self.total_execution_time.as_secs_f64());
    }
}

/// Resource constraints for scheduling
#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    pub cpu_cores: usize,
    pub total_memory: usize, // MB
    pub gpu_cores: Option<usize>,
    pub gpu_memory: Option<usize>, // MB
}

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self {
            cpu_cores: 8,
            total_memory: 16384, // 16GB
            gpu_cores: Some(2048), // Simulated GPU cores
            gpu_memory: Some(8192), // 8GB GPU memory
        }
    }
}
