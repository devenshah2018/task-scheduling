use std::collections::HashMap;
use std::time::{Duration, Instant};
use chrono::Utc;
use crate::common::{Task, SchedulingMetrics, ResourceConstraints, WorkloadType};
use crate::logger::DualLogger;
use colored::*;

/// GPU Warp representation for SIMT execution
#[derive(Debug, Clone)]
pub struct Warp {
    pub id: u32,
    pub threads: Vec<Thread>,
    pub program_counter: usize,
    pub active_mask: u32, // Bit mask for active threads
}

/// GPU Thread representation
#[derive(Debug, Clone)]
pub struct Thread {
    pub id: u32,
    pub warp_id: u32,
    pub active: bool,
    pub registers: HashMap<String, f32>,
}

/// GPU Kernel representation
#[derive(Debug, Clone)]
pub struct Kernel {
    pub id: u32,
    pub name: String,
    pub task_id: u32,
    pub grid_size: (u32, u32, u32),    // Grid dimensions
    pub block_size: (u32, u32, u32),   // Block dimensions
    pub shared_memory_size: usize,      // Bytes
    pub registers_per_thread: u32,
    pub priority: u8,
    pub estimated_execution_time: Duration,
    pub memory_requirements: usize,     // Global memory in bytes
    pub workload_type: WorkloadType,
}

impl Kernel {
    pub fn total_threads(&self) -> u32 {
        self.grid_size.0 * self.grid_size.1 * self.grid_size.2 *
        self.block_size.0 * self.block_size.1 * self.block_size.2
    }

    pub fn total_blocks(&self) -> u32 {
        self.grid_size.0 * self.grid_size.1 * self.grid_size.2
    }
}

/// GPU Streaming Multiprocessor (SM) simulation
#[derive(Debug)]
pub struct StreamingMultiprocessor {
    pub id: u32,
    pub max_warps: usize,
    pub max_blocks: usize,
    pub shared_memory: usize,    // Bytes
    pub register_file: usize,    // Number of registers
    pub active_warps: Vec<Warp>,
    pub active_blocks: Vec<u32>, // Block IDs
    pub utilization: f64,
}

impl StreamingMultiprocessor {
    pub fn new(id: u32) -> Self {
        Self {
            id,
            max_warps: 2048,          // Modern GPUs like A100 can handle ~2048 warps per SM
            max_blocks: 32,           // Typical limit per SM 
            shared_memory: 164 * 1024, // 164KB shared memory (modern GPU)
            register_file: 65536,     // 64K registers
            active_warps: Vec::new(),
            active_blocks: Vec::new(),
            utilization: 0.0,
        }
    }

    pub fn can_schedule_kernel(&self, kernel: &Kernel) -> bool {
        let threads_per_block = kernel.block_size.0 * kernel.block_size.1 * kernel.block_size.2;
        let warps_per_block = (threads_per_block + 31) / 32; // 32 threads per warp
        
        // For large kernels, we only need to check if we can schedule at least one block
        // The scheduler will distribute blocks across multiple SMs
        let shared_mem_per_block = kernel.shared_memory_size;
        let registers_needed = threads_per_block * kernel.registers_per_thread;

        // Check if this SM can handle at least one block of the kernel
        warps_per_block as usize <= self.max_warps &&
        1 <= self.max_blocks && // Can handle at least one block
        shared_mem_per_block <= self.shared_memory &&
        registers_needed as usize <= self.register_file
    }
}

/// GPU Scheduler trait
pub trait GPUScheduler {
    fn schedule_kernels(&mut self, kernels: Vec<Kernel>) -> SchedulingMetrics;
    fn schedule_kernels_with_logger(&mut self, kernels: Vec<Kernel>, logger: Option<&DualLogger>) -> SchedulingMetrics {
        // Default implementation just calls the regular method - schedulers can override for logging
        self.schedule_kernels(kernels)
    }
    fn name(&self) -> &str;
}

/// FIFO GPU Scheduler (Traditional approach)
pub struct FIFOGPUScheduler {
    resources: ResourceConstraints,
    streaming_multiprocessors: Vec<StreamingMultiprocessor>,
}

impl FIFOGPUScheduler {
    pub fn new(resources: ResourceConstraints, sm_count: usize) -> Self {
        let mut sms = Vec::new();
        for i in 0..sm_count {
            sms.push(StreamingMultiprocessor::new(i as u32));
        }
        
        Self {
            resources,
            streaming_multiprocessors: sms,
        }
    }
}

impl GPUScheduler for FIFOGPUScheduler {
    fn name(&self) -> &str {
        "FIFO GPU Scheduler"
    }

    fn schedule_kernels(&mut self, kernels: Vec<Kernel>) -> SchedulingMetrics {
        self.schedule_kernels_with_logger(kernels, None)
    }

    fn schedule_kernels_with_logger(&mut self, kernels: Vec<Kernel>, logger: Option<&DualLogger>) -> SchedulingMetrics {
        if let Some(logger) = logger {
            logger.log_both(&format!("\n🎮 {} Scheduling\n", self.name().green().bold()));
        } else {
            println!("\n🎮 {} Scheduling", self.name().green().bold());
        }
        
        let start_time = Instant::now();
        let mut completed_kernels = Vec::new();
        
        for kernel in kernels {
            let kernel_id = kernel.id; // Store ID for error message
            let launch_msg = format!("  🚀 Launching kernel {} ({:?}) - {} threads in {} blocks", 
                kernel.id, kernel.workload_type, kernel.total_threads(), kernel.total_blocks());
            
            if let Some(logger) = logger {
                logger.log_full_only(&launch_msg);
            } else {
                println!("{}", launch_msg);
            }
            
            // Check if any SM can handle this kernel type
            let mut available_sms = Vec::new();
            for (sm_idx, sm) in self.streaming_multiprocessors.iter().enumerate() {
                if sm.can_schedule_kernel(&kernel) {
                    available_sms.push(sm_idx);
                }
            }
            
            if !available_sms.is_empty() {
                // Distribute blocks across available SMs (simplified distribution)
                let total_blocks = kernel.total_blocks();
                let sms_to_use = available_sms.len().min(total_blocks as usize);
                let dist_msg = format!("    📍 Distributed across {} SMs", sms_to_use);
                
                if let Some(logger) = logger {
                    logger.log_full_only(&dist_msg);
                } else {
                    println!("{}", dist_msg);
                }
                
                // Simulate kernel execution
                let execution_time = match kernel.workload_type {
                    WorkloadType::AITraining => Duration::from_secs_f64(kernel.estimated_execution_time.as_secs_f64() * 1.5), // AI training is intensive
                    WorkloadType::TensorOperation => kernel.estimated_execution_time,
                    WorkloadType::AIInference => Duration::from_secs_f64(kernel.estimated_execution_time.as_secs_f64() * 0.8),
                    _ => Duration::from_secs_f64(kernel.estimated_execution_time.as_secs_f64() * 0.7),
                };
                
                std::thread::sleep(Duration::from_millis(50)); // Demo delay
                
                // Update SM utilizations
                for &sm_idx in available_sms.iter().take(sms_to_use) {
                    let sm = &mut self.streaming_multiprocessors[sm_idx];
                    let blocks_per_sm = total_blocks / sms_to_use as u32;
                    let threads_per_sm = blocks_per_sm * (kernel.block_size.0 * kernel.block_size.1 * kernel.block_size.2);
                    let warps_used = (threads_per_sm + 31) / 32;
                    sm.utilization = (warps_used as f64 / sm.max_warps as f64).min(1.0);
                }
                
                let completion_msg = format!("    ✅ Kernel {} completed in {:.2}s (avg utilization: {:.1}%)", 
                    kernel.id, execution_time.as_secs_f64(), 
                    self.streaming_multiprocessors.iter().map(|sm| sm.utilization).sum::<f64>() / self.streaming_multiprocessors.len() as f64 * 100.0);
                
                if let Some(logger) = logger {
                    logger.log_full_only(&completion_msg);
                } else {
                    println!("{}", completion_msg);
                }
                
                completed_kernels.push(kernel);
            } else {
                let error_msg = format!("    ❌ Kernel {} could not be scheduled (resource constraints)", kernel_id);
                if let Some(logger) = logger {
                    logger.log_both(&error_msg);
                } else {
                    println!("{}", error_msg);
                }
            }
        }
        
        let end_time = Instant::now();
        
        // Convert kernels to tasks for metrics calculation
        let tasks: Vec<Task> = completed_kernels.into_iter().map(|k| {
            let mut task = Task::new(
                k.task_id,
                k.name,
                k.workload_type,
                k.estimated_execution_time,
                k.priority,
                k.memory_requirements / (1024 * 1024), // Convert to MB
                1.0, // GPU kernels are highly parallel
                true,
            );
            task.completion_time = Some(Utc::now());
            task.remaining_time = Duration::ZERO;
            task
        }).collect();
        
        let mut metrics = SchedulingMetrics::calculate(&tasks, start_time, end_time);
        
        // Calculate GPU utilization
        let avg_utilization = self.streaming_multiprocessors.iter()
            .map(|sm| sm.utilization)
            .sum::<f64>() / self.streaming_multiprocessors.len() as f64;
        metrics.gpu_utilization = Some(avg_utilization);
        
        metrics
    }
}

/// Warp Scheduler - Simulates GPU warp-level scheduling
pub struct WarpScheduler {
    resources: ResourceConstraints,
    streaming_multiprocessors: Vec<StreamingMultiprocessor>,
    warp_scheduling_policy: WarpSchedulingPolicy,
}

#[derive(Debug, Clone)]
pub enum WarpSchedulingPolicy {
    GreedyThenOldest,  // GTO - Common in NVIDIA GPUs
    RoundRobin,        // RR - Fair scheduling
    LoosestFirst,      // LRR - Prioritizes warps with fewer ready threads
}

impl WarpScheduler {
    pub fn new(resources: ResourceConstraints, sm_count: usize, policy: WarpSchedulingPolicy) -> Self {
        let mut sms = Vec::new();
        for i in 0..sm_count {
            sms.push(StreamingMultiprocessor::new(i as u32));
        }
        
        Self {
            resources,
            streaming_multiprocessors: sms,
            warp_scheduling_policy: policy,
        }
    }

    fn create_warps_for_kernel(&self, kernel: &Kernel) -> Vec<Warp> {
        let total_threads = kernel.total_threads();
        let mut warps = Vec::new();
        let mut warp_id = 0;
        
        for thread_start in (0..total_threads).step_by(32) {
            let mut threads = Vec::new();
            let thread_end = (thread_start + 32).min(total_threads);
            
            for thread_id in thread_start..thread_end {
                threads.push(Thread {
                    id: thread_id,
                    warp_id,
                    active: true,
                    registers: HashMap::new(),
                });
            }
            
            // Calculate active mask safely to avoid shift overflow
            let num_threads = thread_end - thread_start;
            let active_mask = if num_threads >= 32 {
                u32::MAX
            } else {
                (1u32 << num_threads) - 1
            };
            
            warps.push(Warp {
                id: warp_id,
                threads,
                program_counter: 0,
                active_mask,
            });
            
            warp_id += 1;
        }
        
        warps
    }
}

impl GPUScheduler for WarpScheduler {
    fn name(&self) -> &str {
        match self.warp_scheduling_policy {
            WarpSchedulingPolicy::GreedyThenOldest => "Warp Scheduler (Greedy-Then-Oldest)",
            WarpSchedulingPolicy::RoundRobin => "Warp Scheduler (Round Robin)",
            WarpSchedulingPolicy::LoosestFirst => "Warp Scheduler (Loosest First)",
        }
    }

    fn schedule_kernels(&mut self, kernels: Vec<Kernel>) -> SchedulingMetrics {
        self.schedule_kernels_with_logger(kernels, None)
    }

    fn schedule_kernels_with_logger(&mut self, kernels: Vec<Kernel>, logger: Option<&DualLogger>) -> SchedulingMetrics {
        if let Some(logger) = logger {
            logger.log_both(&format!("\n🎮 {} Scheduling\n", self.name().green().bold()));
        } else {
            println!("\n🎮 {} Scheduling", self.name().green().bold());
        }
        
        let start_time = Instant::now();
        let mut completed_kernels = Vec::new();
        
        for kernel in kernels {
            let launch_msg = format!("  🚀 Launching kernel {} with warp-level scheduling", kernel.id);
            
            if let Some(logger) = logger {
                logger.log_full_only(&launch_msg);
            } else {
                println!("{}", launch_msg);
            }
            
            let warps = self.create_warps_for_kernel(&kernel);
            let warp_msg = format!("    📊 Created {} warps for {} threads", warps.len(), kernel.total_threads());
            
            if let Some(logger) = logger {
                logger.log_full_only(&warp_msg);
            } else {
                println!("{}", warp_msg);
            }
            
            // Find available SM
            for sm in &mut self.streaming_multiprocessors {
                if sm.can_schedule_kernel(&kernel) {
                    sm.active_warps = warps;
                    
                    // Simulate warp execution with different scheduling policies
                    let execution_cycles = match self.warp_scheduling_policy {
                        WarpSchedulingPolicy::GreedyThenOldest => {
                            // Execute warps in order, prioritizing active ones
                            let mut cycles = 0;
                            for warp in &sm.active_warps {
                                if warp.active_mask != 0 {
                                    cycles += 100; // Simulate instruction cycles
                                }
                            }
                            cycles
                        },
                        WarpSchedulingPolicy::RoundRobin => {
                            // Round-robin through warps
                            sm.active_warps.len() * 80 // Slightly better utilization
                        },
                        WarpSchedulingPolicy::LoosestFirst => {
                            // Prioritize warps with better thread utilization
                            let mut cycles = 0;
                            for warp in &sm.active_warps {
                                let active_threads = warp.active_mask.count_ones();
                                cycles += 100 - (active_threads * 2) as usize; // Fewer cycles for fuller warps
                            }
                            cycles
                        }
                    };
                    
                    let _execution_time = Duration::from_millis(execution_cycles as u64);
                    std::thread::sleep(Duration::from_millis(80)); // Demo delay
                    
                    sm.utilization = 0.9; // High utilization for warp scheduling
                    
                    let completion_msg = format!("    ✅ Kernel {} completed with {} execution cycles", 
                        kernel.id, execution_cycles);
                    
                    if let Some(logger) = logger {
                        logger.log_full_only(&completion_msg);
                    } else {
                        println!("{}", completion_msg);
                    }
                    
                    completed_kernels.push(kernel);
                    sm.active_warps.clear();
                    break;
                }
            }
        }
        
        let end_time = Instant::now();
        
        // Convert to tasks for metrics
        let tasks: Vec<Task> = completed_kernels.into_iter().map(|k| {
            let mut task = Task::new(
                k.task_id,
                k.name,
                k.workload_type,
                k.estimated_execution_time,
                k.priority,
                k.memory_requirements / (1024 * 1024),
                1.0,
                true,
            );
            task.completion_time = Some(Utc::now());
            task.remaining_time = Duration::ZERO;
            task
        }).collect();
        
        let mut metrics = SchedulingMetrics::calculate(&tasks, start_time, end_time);
        
        let avg_utilization = self.streaming_multiprocessors.iter()
            .map(|sm| sm.utilization)
            .sum::<f64>() / self.streaming_multiprocessors.len() as f64;
        metrics.gpu_utilization = Some(avg_utilization);
        
        metrics
    }
}

/// Dynamic Priority GPU Scheduler - Adjusts kernel priorities based on workload characteristics
pub struct DynamicPriorityGPUScheduler {
    resources: ResourceConstraints,
    streaming_multiprocessors: Vec<StreamingMultiprocessor>,
    priority_weights: HashMap<WorkloadType, f32>,
}

impl DynamicPriorityGPUScheduler {
    pub fn new(resources: ResourceConstraints, sm_count: usize) -> Self {
        let mut sms = Vec::new();
        for i in 0..sm_count {
            sms.push(StreamingMultiprocessor::new(i as u32));
        }
        
        let mut priority_weights = HashMap::new();
        priority_weights.insert(WorkloadType::AITraining, 1.5);      // High priority for AI training
        priority_weights.insert(WorkloadType::AIInference, 1.8);     // Highest for inference (latency-sensitive)
        priority_weights.insert(WorkloadType::TensorOperation, 1.3); // Medium-high for tensor ops
        priority_weights.insert(WorkloadType::GeneralCompute, 1.0);  // Base priority
        priority_weights.insert(WorkloadType::MemoryIntensive, 0.8); // Lower priority for memory-bound
        priority_weights.insert(WorkloadType::IOBound, 0.5);         // Lowest priority
        
        Self {
            resources,
            streaming_multiprocessors: sms,
            priority_weights,
        }
    }

    fn calculate_dynamic_priority(&self, kernel: &Kernel) -> f32 {
        let base_priority = kernel.priority as f32;
        let workload_weight = self.priority_weights.get(&kernel.workload_type).unwrap_or(&1.0);
        
        // Factor in resource requirements
        let memory_factor = (kernel.memory_requirements as f32 / (1024.0 * 1024.0 * 1024.0)).min(2.0); // GB
        let thread_factor = (kernel.total_threads() as f32 / 10000.0).min(2.0);
        
        base_priority * workload_weight * (1.0 + memory_factor * 0.1 + thread_factor * 0.1)
    }
}

impl GPUScheduler for DynamicPriorityGPUScheduler {
    fn name(&self) -> &str {
        "Dynamic Priority GPU Scheduler"
    }

    fn schedule_kernels(&mut self, kernels: Vec<Kernel>) -> SchedulingMetrics {
        self.schedule_kernels_with_logger(kernels, None)
    }

    fn schedule_kernels_with_logger(&mut self, kernels: Vec<Kernel>, logger: Option<&DualLogger>) -> SchedulingMetrics {
        if let Some(logger) = logger {
            logger.log_both(&format!("\n🎮 {} Scheduling\n", self.name().green().bold()));
        } else {
            println!("\n🎮 {} Scheduling", self.name().green().bold());
        }
        
        let start_time = Instant::now();
        let mut completed_kernels = Vec::new();
        
        // Calculate dynamic priorities and sort
        let mut sorted_kernels: Vec<(f32, Kernel)> = kernels.into_iter()
            .map(|k| (self.calculate_dynamic_priority(&k), k))
            .collect();
        sorted_kernels.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        
        let priority_msg = "  📊 Dynamic Priority Rankings:";
        if let Some(logger) = logger {
            logger.log_full_only(priority_msg);
        } else {
            println!("{}", priority_msg);
        }
        
        for (priority, kernel) in &sorted_kernels {
            let rank_msg = format!("    Kernel {} ({:?}): Priority = {:.2}", 
                kernel.id, kernel.workload_type, priority);
            if let Some(logger) = logger {
                logger.log_full_only(&rank_msg);
            } else {
                println!("{}", rank_msg);
            }
        }
        
        // Schedule kernels in priority order
        for (dynamic_priority, kernel) in sorted_kernels {
            let schedule_msg = format!("  🚀 Scheduling high-priority kernel {} (priority: {:.2})", 
                kernel.id, dynamic_priority);
            
            if let Some(logger) = logger {
                logger.log_full_only(&schedule_msg);
            } else {
                println!("{}", schedule_msg);
            }
            
            // Find best-fit SM based on current utilization
            let mut best_sm_idx = 0;
            let mut lowest_utilization = f64::MAX;
            
            for (idx, sm) in self.streaming_multiprocessors.iter().enumerate() {
                if sm.can_schedule_kernel(&kernel) && sm.utilization < lowest_utilization {
                    lowest_utilization = sm.utilization;
                    best_sm_idx = idx;
                }
            }
            
            if let Some(sm) = self.streaming_multiprocessors.get_mut(best_sm_idx) {
                if sm.can_schedule_kernel(&kernel) {
                    let assign_msg = format!("    📍 Assigned to SM {} (current utilization: {:.1}%)", 
                        sm.id, sm.utilization * 100.0);
                    
                    if let Some(logger) = logger {
                        logger.log_full_only(&assign_msg);
                    } else {
                        println!("{}", assign_msg);
                    }
                    
                    // Adjust execution time based on workload type
                    let execution_time = match kernel.workload_type {
                        WorkloadType::AIInference => kernel.estimated_execution_time / 2, // Optimized inference
                        WorkloadType::AITraining => Duration::from_secs_f64(kernel.estimated_execution_time.as_secs_f64() * 1.5), // Training overhead
                        WorkloadType::TensorOperation => kernel.estimated_execution_time,
                        _ => Duration::from_secs_f64(kernel.estimated_execution_time.as_secs_f64() * 1.2),
                    };
                    
                    std::thread::sleep(Duration::from_millis(90));
                    
                    sm.utilization = (sm.utilization + 0.3).min(1.0);
                    
                    let completion_msg = format!("    ✅ Kernel {} completed in {:.2}s", 
                        kernel.id, execution_time.as_secs_f64());
                    
                    if let Some(logger) = logger {
                        logger.log_full_only(&completion_msg);
                    } else {
                        println!("{}", completion_msg);
                    }
                    
                    completed_kernels.push(kernel);
                }
            }
        }
        
        let end_time = Instant::now();
        
        // Convert to tasks for metrics
        let tasks: Vec<Task> = completed_kernels.into_iter().map(|k| {
            let mut task = Task::new(
                k.task_id,
                k.name,
                k.workload_type,
                k.estimated_execution_time,
                k.priority,
                k.memory_requirements / (1024 * 1024),
                1.0,
                true,
            );
            task.completion_time = Some(Utc::now());
            task.remaining_time = Duration::ZERO;
            task
        }).collect();
        
        let mut metrics = SchedulingMetrics::calculate(&tasks, start_time, end_time);
        
        let avg_utilization = self.streaming_multiprocessors.iter()
            .map(|sm| sm.utilization)
            .sum::<f64>() / self.streaming_multiprocessors.len() as f64;
        metrics.gpu_utilization = Some(avg_utilization);
        
        metrics
    }
}
