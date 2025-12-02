use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use chrono::Utc;
use crate::common::{Task, SchedulingMetrics, ResourceConstraints, WorkloadType};
use crate::cpu::CPUScheduler;
use crate::gpu::{GPUScheduler, Kernel, StreamingMultiprocessor};
use colored::*;

/// Hybrid task that can run on CPU or GPU
#[derive(Debug, Clone)]
pub struct HybridTask {
    pub base_task: Task,
    pub cpu_execution_time: Duration,
    pub gpu_execution_time: Duration,
    pub cpu_gpu_affinity: f32,    // 0.0 = CPU preferred, 1.0 = GPU preferred
    pub data_transfer_overhead: Duration,
    pub memory_locality: MemoryLocality,
}

#[derive(Debug, Clone)]
pub enum MemoryLocality {
    CPUMemory,
    GPUMemory,
    Shared,
    Distributed,
}

/// Represents data that needs to be transferred between CPU and GPU
#[derive(Debug, Clone)]
pub struct DataTransfer {
    pub size: usize,           // Bytes
    pub source: ProcessingUnit,
    pub destination: ProcessingUnit,
    pub transfer_time: Duration,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProcessingUnit {
    CPU(usize),  // CPU core ID
    GPU(usize),  // GPU SM ID
}

/// Hybrid CPU-GPU Scheduler
pub struct HybridScheduler {
    resources: ResourceConstraints,
    cpu_cores: usize,
    streaming_multiprocessors: Vec<StreamingMultiprocessor>,
    workload_predictor: WorkloadPredictor,
    load_balancer: LoadBalancer,
    data_transfer_queue: VecDeque<DataTransfer>,
}

/// Predicts optimal placement for tasks based on historical data
#[derive(Debug)]
pub struct WorkloadPredictor {
    execution_history: HashMap<WorkloadType, (Duration, Duration)>, // (avg_cpu_time, avg_gpu_time)
    accuracy_scores: HashMap<WorkloadType, f32>,
}

impl WorkloadPredictor {
    pub fn new() -> Self {
        let mut execution_history = HashMap::new();
        let mut accuracy_scores = HashMap::new();

        execution_history.insert(WorkloadType::AITraining, 
            (Duration::from_secs(10), Duration::from_secs(2)));
        execution_history.insert(WorkloadType::AIInference, 
            (Duration::from_secs(1), Duration::from_millis(100)));
        execution_history.insert(WorkloadType::TensorOperation, 
            (Duration::from_secs(5), Duration::from_millis(500)));
        execution_history.insert(WorkloadType::GeneralCompute, 
            (Duration::from_secs(3), Duration::from_secs(4)));
        execution_history.insert(WorkloadType::MemoryIntensive, 
            (Duration::from_secs(8), Duration::from_secs(6)));
        execution_history.insert(WorkloadType::IOBound, 
            (Duration::from_secs(2), Duration::from_secs(10)));

        for workload_type in [
            WorkloadType::AITraining,
            WorkloadType::AIInference,
            WorkloadType::TensorOperation,
            WorkloadType::GeneralCompute,
            WorkloadType::MemoryIntensive,
            WorkloadType::IOBound,
        ] {
            accuracy_scores.insert(workload_type, 0.8);
        }

        Self {
            execution_history,
            accuracy_scores,
        }
    }

    pub fn predict_optimal_unit(&self, task: &HybridTask) -> ProcessingUnit {
        if let Some((cpu_time, gpu_time)) = self.execution_history.get(&task.base_task.workload_type) {
            let cpu_efficiency = 1.0 / cpu_time.as_secs_f32();
            let gpu_efficiency = 1.0 / (gpu_time.as_secs_f32() + task.data_transfer_overhead.as_secs_f32());
            
            let parallelism_bonus = task.base_task.parallelism_factor * 2.0;
            let adjusted_gpu_efficiency = gpu_efficiency * (1.0 + parallelism_bonus);
            
            if adjusted_gpu_efficiency > cpu_efficiency && task.base_task.gpu_compatibility {
                ProcessingUnit::GPU(0) // Simplified - use first GPU
            } else {
                ProcessingUnit::CPU(0) // Simplified - use first CPU
            }
        } else {
            ProcessingUnit::CPU(0)
        }
    }

    pub fn update_history(&mut self, workload_type: WorkloadType, cpu_time: Duration, gpu_time: Duration) {
        if let Some((avg_cpu, avg_gpu)) = self.execution_history.get_mut(&workload_type) {
            *avg_cpu = Duration::from_secs_f32(avg_cpu.as_secs_f32() * 0.8 + cpu_time.as_secs_f32() * 0.2);
            *avg_gpu = Duration::from_secs_f32(avg_gpu.as_secs_f32() * 0.8 + gpu_time.as_secs_f32() * 0.2);
        }
    }
}

/// Balances load between CPU and GPU based on current utilization
#[derive(Debug)]
pub struct LoadBalancer {
    cpu_utilization: Vec<f32>,
    gpu_utilization: Vec<f32>,
    migration_threshold: f32,
}

impl LoadBalancer {
    pub fn new(cpu_cores: usize, gpu_count: usize) -> Self {
        Self {
            cpu_utilization: vec![0.0; cpu_cores],
            gpu_utilization: vec![0.0; gpu_count],
            migration_threshold: 0.8,
        }
    }

    pub fn should_migrate(&self, current_unit: &ProcessingUnit) -> Option<ProcessingUnit> {
        match current_unit {
            ProcessingUnit::CPU(core_id) => {
                if self.cpu_utilization[*core_id] > self.migration_threshold {
                    let min_gpu_idx = self.gpu_utilization.iter()
                        .enumerate()
                        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(idx, _)| idx)?;
                    
                    if self.gpu_utilization[min_gpu_idx] < self.cpu_utilization[*core_id] - 0.3 {
                        return Some(ProcessingUnit::GPU(min_gpu_idx));
                    }
                }
            },
            ProcessingUnit::GPU(gpu_id) => {
                if self.gpu_utilization[*gpu_id] > self.migration_threshold {
                    let min_cpu_idx = self.cpu_utilization.iter()
                        .enumerate()
                        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(idx, _)| idx)?;
                    
                    if self.cpu_utilization[min_cpu_idx] < self.gpu_utilization[*gpu_id] - 0.3 {
                        return Some(ProcessingUnit::CPU(min_cpu_idx));
                    }
                }
            }
        }
        None
    }

    pub fn update_utilization(&mut self, unit: &ProcessingUnit, utilization: f32) {
        match unit {
            ProcessingUnit::CPU(core_id) => {
                if *core_id < self.cpu_utilization.len() {
                    self.cpu_utilization[*core_id] = utilization.min(1.0);
                }
            },
            ProcessingUnit::GPU(gpu_id) => {
                if *gpu_id < self.gpu_utilization.len() {
                    self.gpu_utilization[*gpu_id] = utilization.min(1.0);
                }
            }
        }
    }

    pub fn get_overall_cpu_utilization(&self) -> f32 {
        self.cpu_utilization.iter().sum::<f32>() / self.cpu_utilization.len() as f32
    }

    pub fn get_overall_gpu_utilization(&self) -> f32 {
        if self.gpu_utilization.is_empty() {
            0.0
        } else {
            self.gpu_utilization.iter().sum::<f32>() / self.gpu_utilization.len() as f32
        }
    }
}

impl HybridScheduler {
    pub fn new(resources: ResourceConstraints) -> Self {
        let cpu_cores = resources.cpu_cores;
        let gpu_count = 1;
        
        let mut sms = Vec::new();
        if resources.gpu_cores.is_some() {
            for i in 0..4 {
                sms.push(StreamingMultiprocessor::new(i as u32));
            }
        }

        Self {
            cpu_cores,
            streaming_multiprocessors: sms,
            workload_predictor: WorkloadPredictor::new(),
            load_balancer: LoadBalancer::new(cpu_cores, gpu_count),
            data_transfer_queue: VecDeque::new(),
            resources,
        }
    }

    pub fn create_hybrid_task(&self, task: Task) -> HybridTask {
        let (cpu_time, gpu_time) = match task.workload_type {
            WorkloadType::AITraining => {
                (task.burst_time * 3, task.burst_time)
            },
            WorkloadType::AIInference => {
                (task.burst_time * 5, task.burst_time)
            },
            WorkloadType::TensorOperation => {
                (task.burst_time * 2, task.burst_time)
            },
            WorkloadType::GeneralCompute => {
                (task.burst_time, task.burst_time * 2)
            },
            WorkloadType::MemoryIntensive => {
                (task.burst_time, task.burst_time * 3)
            },
            WorkloadType::IOBound => {
                (task.burst_time, task.burst_time * 5)
            },
        };

        let affinity = task.parallelism_factor * if task.gpu_compatibility { 1.0 } else { 0.0 };
        let data_transfer = Duration::from_millis((task.memory_requirement as u64 / 1000).max(10));

        HybridTask {
            base_task: task,
            cpu_execution_time: cpu_time,
            gpu_execution_time: gpu_time,
            cpu_gpu_affinity: affinity,
            data_transfer_overhead: data_transfer,
            memory_locality: MemoryLocality::CPUMemory,
        }
    }

    fn execute_on_cpu(&mut self, task: &mut HybridTask, core_id: usize) -> Duration {
        println!("    🖥️  Executing task {} on CPU core {}", task.base_task.id, core_id);
        
        let execution_time = task.cpu_execution_time;
        std::thread::sleep(Duration::from_millis(80));
        
        let new_utilization = (self.load_balancer.cpu_utilization[core_id] + 0.4).min(1.0);
        self.load_balancer.update_utilization(&ProcessingUnit::CPU(core_id), new_utilization);
        
        execution_time
    }

    fn execute_on_gpu(&mut self, task: &mut HybridTask, sm_id: usize) -> Duration {
        println!("    🎮 Executing task {} on GPU SM {}", task.base_task.id, sm_id);
        
        let mut total_time = task.data_transfer_overhead;
        
        total_time += task.gpu_execution_time;
        
        std::thread::sleep(Duration::from_millis(100));
        
        if sm_id < self.streaming_multiprocessors.len() {
            self.streaming_multiprocessors[sm_id].utilization = 
                (self.streaming_multiprocessors[sm_id].utilization + 0.5).min(1.0);
            
            self.load_balancer.update_utilization(&ProcessingUnit::GPU(0), 
                self.streaming_multiprocessors[sm_id].utilization as f32);
        }
        
        total_time
    }

    fn should_migrate_task(&mut self, task: &HybridTask, current_unit: &ProcessingUnit) -> Option<ProcessingUnit> {
        if let Some(better_unit) = self.load_balancer.should_migrate(current_unit) {
            println!("    🔄 Load balancer suggests migration from {:?} to {:?}", current_unit, better_unit);
            return Some(better_unit);
        }

        let predicted_unit = self.workload_predictor.predict_optimal_unit(task);
        if predicted_unit != *current_unit {
            match (&predicted_unit, current_unit) {
                (ProcessingUnit::GPU(_), ProcessingUnit::CPU(_)) => {
                    if task.cpu_gpu_affinity > 0.7 {
                        println!("    🎯 Migrating to GPU based on high parallelism affinity");
                        return Some(predicted_unit);
                    }
                },
                (ProcessingUnit::CPU(_), ProcessingUnit::GPU(_)) => {
                    if task.cpu_gpu_affinity < 0.3 {
                        println!("    🎯 Migrating to CPU based on low parallelism affinity");
                        return Some(predicted_unit);
                    }
                },
                _ => {}
            }
        }

        None
    }
}

/// Main scheduling method for hybrid scheduler
impl HybridScheduler {
    pub fn schedule_hybrid_tasks(&mut self, tasks: Vec<Task>) -> SchedulingMetrics {
        println!("\n🔀 {} Scheduling", "Hybrid CPU-GPU".cyan().bold());
        
        let start_time = Instant::now();
        let mut hybrid_tasks: Vec<HybridTask> = tasks.into_iter()
            .map(|t| self.create_hybrid_task(t))
            .collect();
        
        println!("  📊 Created {} hybrid tasks", hybrid_tasks.len());
        
        let task_placements: Vec<ProcessingUnit> = hybrid_tasks.iter()
            .map(|t| self.workload_predictor.predict_optimal_unit(t))
            .collect();
        
        println!("  🎯 Initial Placement Predictions:");
        for (task, placement) in hybrid_tasks.iter().zip(task_placements.iter()) {
            println!("    Task {} ({:?}): {:?}", 
                task.base_task.id,                 task.base_task.workload_type, placement);
        }
        
        for (i, task) in hybrid_tasks.iter_mut().enumerate() {
            task.base_task.start_time = Some(Utc::now());
            
            let mut current_placement = task_placements[i].clone();
            
            if let Some(better_placement) = self.should_migrate_task(task, &current_placement) {
                current_placement = better_placement;
                println!("  🔄 Migrating task {} to {:?}", task.base_task.id, current_placement);
            }
            
            let execution_time = match current_placement {
                ProcessingUnit::CPU(core_id) => {
                    self.execute_on_cpu(task, core_id)
                },
                ProcessingUnit::GPU(sm_id) => {
                    self.execute_on_gpu(task, sm_id)
                }
            };
            
            task.base_task.completion_time = Some(Utc::now());
            task.base_task.remaining_time = Duration::ZERO;
            
            println!("    ✅ Task {} completed in {:.2}s on {:?}", 
                task.base_task.id, execution_time.as_secs_f64(), current_placement);
            
            self.workload_predictor.update_history(
                task.base_task.workload_type.clone(),
                task.cpu_execution_time,
                task.gpu_execution_time,
            );
        }
        
        let end_time = Instant::now();
        
        println!("\n  📈 Final System Utilization:");
        println!("    CPU: {:.1}%", self.load_balancer.get_overall_cpu_utilization() * 100.0);
        println!("    GPU: {:.1}%", self.load_balancer.get_overall_gpu_utilization() * 100.0);
        
        let completed_tasks: Vec<Task> = hybrid_tasks.into_iter()
            .map(|ht| ht.base_task)
            .collect();
        
        let mut metrics = SchedulingMetrics::calculate(&completed_tasks, start_time, end_time);
        metrics.gpu_utilization = Some(self.load_balancer.get_overall_gpu_utilization() as f64);
        
        metrics
    }
}

/// Cooperative CPU-GPU Scheduler for large-scale AI workloads
pub struct CooperativeScheduler {
    resources: ResourceConstraints,
    cpu_scheduler: Box<dyn CPUScheduler + Send>,
    gpu_scheduler: Box<dyn GPUScheduler + Send>,
    task_splitter: TaskSplitter,
}

/// Splits large tasks into CPU and GPU components
#[derive(Debug)]
pub struct TaskSplitter {
    split_threshold: Duration,
    cpu_gpu_ratio: f32,
}

impl TaskSplitter {
    pub fn new() -> Self {
        Self {
            split_threshold: Duration::from_secs(5),
            cpu_gpu_ratio: 0.3,
        }
    }

    pub fn should_split(&self, task: &Task) -> bool {
        task.burst_time > self.split_threshold && 
        task.gpu_compatibility && 
        matches!(task.workload_type, WorkloadType::AITraining | WorkloadType::TensorOperation)
    }

    pub fn split_task(&self, task: &Task) -> (Task, Kernel) {
        let cpu_component = Task::new(
            task.id * 1000,
            format!("{}_cpu", task.name),
            WorkloadType::GeneralCompute,
            Duration::from_secs_f64(task.burst_time.as_secs_f64() * self.cpu_gpu_ratio as f64),
            task.priority,
            task.memory_requirement / 2,
            0.2,
            false,
        );

        let gpu_component = Kernel {
            id: task.id * 1000 + 1,
            name: format!("{}_gpu", task.name),
            task_id: task.id,
            grid_size: (64, 64, 1),
            block_size: (256, 1, 1),
            shared_memory_size: 48 * 1024,
            registers_per_thread: 32,
            priority: task.priority,
            estimated_execution_time: Duration::from_secs_f64(
                task.burst_time.as_secs_f64() * (1.0 - self.cpu_gpu_ratio as f64)
            ),
            memory_requirements: (task.memory_requirement * 1024 * 1024) / 2,
            workload_type: task.workload_type.clone(),
        };

        (cpu_component, gpu_component)
    }
}
