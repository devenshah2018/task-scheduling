#!/bin/bash

# Update remaining CPU schedulers to support logging
# This script will replace println! statements and add schedule_with_logger methods

cd /Users/devenshah/repos/cpu-scheduling

echo "Updating Priority Scheduler..."

# Create a temporary file to hold the updated Priority scheduler implementation
cat > /tmp/priority_update.rs << 'EOF'
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
            // Sort by priority (higher number = higher priority)
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
            // Preemptive Priority Scheduling
            let mut ready_queue = BinaryHeap::new();
            for (i, _) in tasks.iter().enumerate() {
                ready_queue.push(PriorityTask {
                    index: i,
                    priority: tasks[i].priority,
                    remaining_time: tasks[i].remaining_time,
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
                
                std::thread::sleep(Duration::from_millis(30));
                current_time += execution_time;
                task.remaining_time -= execution_time;
                
                if task.remaining_time > Duration::ZERO {
                    ready_queue.push(PriorityTask {
                        index: priority_task.index,
                        priority: task.priority,
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
EOF

echo "✅ Script ready to update schedulers"
