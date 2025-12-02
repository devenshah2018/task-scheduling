use std::collections::{VecDeque, BinaryHeap};
use std::cmp::Ordering;
use std::time::{Duration, Instant};
use chrono::Utc;
use crate::common::{Task, SchedulingMetrics, ResourceConstraints};
use crate::logger::DualLogger;
use colored::*;

/// Traditional CPU scheduling algorithms
pub trait CPUScheduler {
    fn schedule(&mut self, tasks: Vec<Task>) -> SchedulingMetrics;
    fn schedule_with_logger(&mut self, tasks: Vec<Task>, logger: Option<&DualLogger>) -> SchedulingMetrics {
        self.schedule(tasks)
    }
    fn name(&self) -> &str;
}

/// First Come First Serve (FCFS) Scheduler
pub struct FCFSScheduler {
    resources: ResourceConstraints,
}

impl FCFSScheduler {
    pub fn new(resources: ResourceConstraints) -> Self {
        Self { resources }
    }
}

impl CPUScheduler for FCFSScheduler {
    fn name(&self) -> &str {
        "First Come First Serve (FCFS)"
    }

    fn schedule(&mut self, tasks: Vec<Task>) -> SchedulingMetrics {
        self.schedule_with_logger(tasks, None)
    }

    fn schedule_with_logger(&mut self, mut tasks: Vec<Task>, logger: Option<&DualLogger>) -> SchedulingMetrics {
        let header = format!("\n🔄 {} Scheduling", self.name().blue().bold());
        
        match logger {
            Some(log) => log.log_both(&header),
            None => println!("{}", header),
        }
        
        let start_time = Instant::now();
        let mut current_time = start_time;
        
        tasks.sort_by_key(|t| t.arrival_time);
        
        for task in &mut tasks {
            task.start_time = Some(Utc::now());
            
            let exec_msg = format!("  ⚡ Executing task {} ({:?}) - Duration: {:.1}s\n", 
                task.id, task.workload_type, task.burst_time.as_secs_f64());
            
            match logger {
                Some(log) => log.log_full_only(&exec_msg),
                None => print!("{}", exec_msg),
            }
            
            std::thread::sleep(Duration::from_millis(100));
            current_time += task.burst_time;
            
            task.completion_time = Some(Utc::now());
            task.remaining_time = Duration::ZERO;
        }
        
        let end_time = Instant::now();
        SchedulingMetrics::calculate(&tasks, start_time, end_time)
    }
}

/// Round Robin Scheduler
pub struct RoundRobinScheduler {
    resources: ResourceConstraints,
    time_quantum: Duration,
}

impl RoundRobinScheduler {
    pub fn new(resources: ResourceConstraints, time_quantum: Duration) -> Self {
        Self { resources, time_quantum }
    }
}

impl CPUScheduler for RoundRobinScheduler {
    fn name(&self) -> &str {
        "Round Robin"
    }

    fn schedule(&mut self, tasks: Vec<Task>) -> SchedulingMetrics {
        self.schedule_with_logger(tasks, None)
    }

    fn schedule_with_logger(&mut self, mut tasks: Vec<Task>, logger: Option<&DualLogger>) -> SchedulingMetrics {
        let header = format!("\n🔄 {} Scheduling (Quantum: {:.1}s)", 
            self.name().blue().bold(), self.time_quantum.as_secs_f64());
        
        match logger {
            Some(log) => log.log_both(&header),
            None => println!("{}", header),
        }
        
        let start_time = Instant::now();
        let mut ready_queue: VecDeque<usize> = (0..tasks.len()).collect();
        let mut current_time = start_time;
        
        while !ready_queue.is_empty() {
            let task_idx = ready_queue.pop_front().unwrap();
            let task = &mut tasks[task_idx];
            
            if task.start_time.is_none() {
                task.start_time = Some(Utc::now());
            }
            
            let execution_time = self.time_quantum.min(task.remaining_time);
            
            let exec_msg = format!("  ⚡ Executing task {} for {:.1}s (remaining: {:.1}s)\n", 
                task.id, execution_time.as_secs_f64(), 
                (task.remaining_time - execution_time).as_secs_f64());
            
            match logger {
                Some(log) => log.log_full_only(&exec_msg),
                None => print!("{}", exec_msg),
            }
            
            std::thread::sleep(Duration::from_millis(50));
            current_time += execution_time;
            task.remaining_time -= execution_time;
            
            if task.remaining_time > Duration::ZERO {
                ready_queue.push_back(task_idx);
            } else {
                task.completion_time = Some(Utc::now());
            }
        }
        
        let end_time = Instant::now();
        SchedulingMetrics::calculate(&tasks, start_time, end_time)
    }
}

/// Shortest Job First (SJF) Scheduler
pub struct SJFScheduler {
    resources: ResourceConstraints,
    preemptive: bool,
}

impl SJFScheduler {
    pub fn new(resources: ResourceConstraints, preemptive: bool) -> Self {
        Self { resources, preemptive }
    }
}

#[derive(Eq, PartialEq)]
struct SJFTask {
    index: usize,
    remaining_time: Duration,
}

impl Ord for SJFTask {
    fn cmp(&self, other: &Self) -> Ordering {
        other.remaining_time.cmp(&self.remaining_time)
    }
}

impl PartialOrd for SJFTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl CPUScheduler for SJFScheduler {
    fn name(&self) -> &str {
        if self.preemptive {
            "Shortest Remaining Time First (SRTF)"
        } else {
            "Shortest Job First (SJF)"
        }
    }

    fn schedule(&mut self, tasks: Vec<Task>) -> SchedulingMetrics {
        self.schedule_with_logger(tasks, None)
    }

    fn schedule_with_logger(&mut self, mut tasks: Vec<Task>, logger: Option<&DualLogger>) -> SchedulingMetrics {
        let header = format!("\n🔄 {} Scheduling", self.name().blue().bold());
        
        match logger {
            Some(log) => log.log_both(&header),
            None => println!("{}", header),
        }
        
        let start_time = Instant::now();
        let mut current_time = start_time;
        
        if !self.preemptive {
            tasks.sort_by_key(|t| t.burst_time);
            
            for task in &mut tasks {
                task.start_time = Some(Utc::now());
                
                let exec_msg = format!("  ⚡ Executing task {} (burst: {:.1}s)\n", 
                    task.id, task.burst_time.as_secs_f64());
                
                match logger {
                    Some(log) => log.log_full_only(&exec_msg),
                    None => print!("{}", exec_msg),
                }
                
                std::thread::sleep(Duration::from_millis(80));
                current_time += task.burst_time;
                
                task.completion_time = Some(Utc::now());
                task.remaining_time = Duration::ZERO;
            }
        } else {
            let mut ready_queue = BinaryHeap::new();
            for (i, _) in tasks.iter().enumerate() {
                ready_queue.push(SJFTask {
                    index: i,
                    remaining_time: tasks[i].remaining_time,
                });
            }
            
            while !ready_queue.is_empty() {
                let sjf_task = ready_queue.pop().unwrap();
                let task = &mut tasks[sjf_task.index];
                
                if task.start_time.is_none() {
                    task.start_time = Some(Utc::now());
                }
                
                let execution_time = Duration::from_millis(100).min(task.remaining_time);
                
                let exec_msg = format!("  ⚡ Executing task {} for {:.1}s (remaining: {:.1}s)\n", 
                    task.id, execution_time.as_secs_f64(), 
                    (task.remaining_time - execution_time).as_secs_f64());
                
                match logger {
                    Some(log) => log.log_full_only(&exec_msg),
                    None => print!("{}", exec_msg),
                }
                
                std::thread::sleep(Duration::from_millis(30));
                current_time += execution_time;
                task.remaining_time -= execution_time;
                
                if task.remaining_time > Duration::ZERO {
                    ready_queue.push(SJFTask {
                        index: sjf_task.index,
                        remaining_time: task.remaining_time,
                    });
                } else {
                    task.completion_time = Some(Utc::now());
                }
            }
        }
        
        let end_time = Instant::now();
        SchedulingMetrics::calculate(&tasks, start_time, end_time)
    }
}

/// Priority Scheduler
pub struct PriorityScheduler {
    resources: ResourceConstraints,
    preemptive: bool,
}

impl PriorityScheduler {
    pub fn new(resources: ResourceConstraints, preemptive: bool) -> Self {
        Self { resources, preemptive }
    }
}

#[derive(Eq, PartialEq)]
struct PriorityTask {
    index: usize,
    priority: u8,
    arrival_order: usize,
}

impl Ord for PriorityTask {
    fn cmp(&self, other: &Self) -> Ordering {
        match other.priority.cmp(&self.priority) {
            Ordering::Equal => self.arrival_order.cmp(&other.arrival_order),
            other => other,
        }
    }
}

impl PartialOrd for PriorityTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl CPUScheduler for PriorityScheduler {
    fn name(&self) -> &str {
        if self.preemptive {
            "Preemptive Priority Scheduling"
        } else {
            "Non-preemptive Priority Scheduling"
        }
    }

    fn schedule(&mut self, tasks: Vec<Task>) -> SchedulingMetrics {
        self.schedule_with_logger(tasks, None)
    }

    fn schedule_with_logger(&mut self, mut tasks: Vec<Task>, logger: Option<&DualLogger>) -> SchedulingMetrics {
        let header = format!("\n🔄 {} Scheduling", self.name().blue().bold());
        
        match logger {
            Some(log) => log.log_both(&header),
            None => println!("{}", header),
        }
        
        let start_time = Instant::now();
        let mut current_time = start_time;
        
        if !self.preemptive {
            tasks.sort_by(|a, b| b.priority.cmp(&a.priority));
            
            for task in &mut tasks {
                task.start_time = Some(Utc::now());
                
                let exec_msg = format!("  ⚡ Executing task {} (priority: {}, burst: {:.1}s)\n", 
                    task.id, task.priority, task.burst_time.as_secs_f64());
                
                match logger {
                    Some(log) => log.log_full_only(&exec_msg),
                    None => print!("{}", exec_msg),
                }
                
                std::thread::sleep(Duration::from_millis(80));
                current_time += task.burst_time;
                
                task.completion_time = Some(Utc::now());
                task.remaining_time = Duration::ZERO;
            }
        } else {
            let mut ready_queue = BinaryHeap::new();
            for (i, task) in tasks.iter().enumerate() {
                ready_queue.push(PriorityTask {
                    index: i,
                    priority: task.priority,
                    arrival_order: i,
                });
            }
            
            while !ready_queue.is_empty() {
                let priority_task = ready_queue.pop().unwrap();
                let task = &mut tasks[priority_task.index];
                
                if task.start_time.is_none() {
                    task.start_time = Some(Utc::now());
                }
                
                let execution_time = Duration::from_millis(100).min(task.remaining_time);
                
                let exec_msg = format!("  ⚡ Executing task {} (priority: {}) for {:.1}s\n", 
                    task.id, task.priority, execution_time.as_secs_f64());
                
                match logger {
                    Some(log) => log.log_full_only(&exec_msg),
                    None => print!("{}", exec_msg),
                }
                
                std::thread::sleep(Duration::from_millis(40));
                current_time += execution_time;
                task.remaining_time -= execution_time;
                
                if task.remaining_time > Duration::ZERO {
                    ready_queue.push(PriorityTask {
                        index: priority_task.index,
                        priority: task.priority,
                        arrival_order: priority_task.arrival_order,
                    });
                } else {
                    task.completion_time = Some(Utc::now());
                }
            }
        }
        
        let end_time = Instant::now();
        SchedulingMetrics::calculate(&tasks, start_time, end_time)
    }
}

/// Multi-level Feedback Queue Scheduler
pub struct MLFQScheduler {
    resources: ResourceConstraints,
    queue_count: usize,
    time_quanta: Vec<Duration>,
}

impl MLFQScheduler {
    pub fn new(resources: ResourceConstraints) -> Self {
        Self {
            resources,
            queue_count: 3,
            time_quanta: vec![
                Duration::from_millis(100),
                Duration::from_millis(200),
                Duration::from_millis(400),
            ],
        }
    }
}

impl CPUScheduler for MLFQScheduler {
    fn name(&self) -> &str {
        "Multi-level Feedback Queue (MLFQ)"
    }

    fn schedule(&mut self, tasks: Vec<Task>) -> SchedulingMetrics {
        self.schedule_with_logger(tasks, None)
    }

    fn schedule_with_logger(&mut self, mut tasks: Vec<Task>, logger: Option<&DualLogger>) -> SchedulingMetrics {
        let header = format!("\n🔄 {} Scheduling", self.name().blue().bold());
        
        match logger {
            Some(log) => log.log_both(&header),
            None => println!("{}", header),
        }
        
        let start_time = Instant::now();
        let mut current_time = start_time;
        
        let mut queues: Vec<VecDeque<usize>> = vec![VecDeque::new(); self.queue_count];
        
        for i in 0..tasks.len() {
            queues[0].push_back(i);
        }
        
        while queues.iter().any(|q| !q.is_empty()) {
            for (queue_level, queue) in queues.iter_mut().enumerate() {
                if let Some(task_idx) = queue.pop_front() {
                    let task = &mut tasks[task_idx];
                    
                    if task.start_time.is_none() {
                        task.start_time = Some(Utc::now());
                    }
                    
                    let quantum = self.time_quanta[queue_level];
                    let execution_time = quantum.min(task.remaining_time);
                    
                    let exec_msg = format!("  ⚡ Queue {} - Executing task {} for {:.1}s\n",
                        queue_level, task.id, execution_time.as_secs_f64());
                    
                    match logger {
                        Some(log) => log.log_full_only(&exec_msg),
                        None => print!("{}", exec_msg),
                    }
                    
                    std::thread::sleep(Duration::from_millis(30));
                    current_time += execution_time;
                    task.remaining_time -= execution_time;
                    
                    if task.remaining_time > Duration::ZERO {
                        let next_queue = (queue_level + 1).min(self.queue_count - 1);
                        queues[next_queue].push_back(task_idx);
                    } else {
                        task.completion_time = Some(Utc::now());
                    }
                    
                    break;
                }
            }
        }
        
        let end_time = Instant::now();
        SchedulingMetrics::calculate(&tasks, start_time, end_time)
    }
}
