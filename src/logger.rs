use std::fs::{self, File, OpenOptions};
use std::io::{self, Write, BufWriter};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use chrono::Local;

pub struct DualLogger {
    full_writer: Arc<Mutex<BufWriter<File>>>,
    simple_writer: Arc<Mutex<BufWriter<File>>>,
    log_dir: PathBuf,
}

impl DualLogger {
    pub fn new(demo_type: &str) -> io::Result<Self> {
        let timestamp = Local::now().format("%Y%m%d_%H%M%S");
        let log_dir = PathBuf::from("logs").join(format!("{}_{}", demo_type, timestamp));
        
        // Create the directory
        fs::create_dir_all(&log_dir)?;
        
        // Create log files
        let full_log_path = log_dir.join("full.log");
        let simple_log_path = log_dir.join("simple.log");
        
        let full_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&full_log_path)?;
        
        let simple_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&simple_log_path)?;
        
        let full_writer = Arc::new(Mutex::new(BufWriter::new(full_file)));
        let simple_writer = Arc::new(Mutex::new(BufWriter::new(simple_file)));
        
        // Write headers
        let timestamp_str = Local::now().format("%Y-%m-%d %H:%M:%S");
        let header = format!("=== {} SCHEDULING DEMONSTRATION LOG ===\n", demo_type.to_uppercase());
        let time_header = format!("Started at: {}\n\n", timestamp_str);
        
        // Write to both files
        if let Ok(mut writer) = full_writer.lock() {
            let _ = writer.write_all(header.as_bytes());
            let _ = writer.write_all(time_header.as_bytes());
            let _ = writer.flush();
        }
        
        if let Ok(mut writer) = simple_writer.lock() {
            let _ = writer.write_all(header.as_bytes());
            let _ = writer.write_all(time_header.as_bytes());
            let _ = writer.flush();
        }
        
        Ok(DualLogger {
            full_writer,
            simple_writer,
            log_dir,
        })
    }
    
    /// Log a message to both full and simple logs
    pub fn log_both(&self, message: &str) {
        let timestamped = format!("[{}] {}\n", Local::now().format("%H:%M:%S"), message);
        let clean_message = Self::strip_color_codes(message);
        let clean_timestamped = format!("[{}] {}\n", Local::now().format("%H:%M:%S"), clean_message);
        
        // Print to console
        print!("{}", message);
        io::stdout().flush().unwrap_or(());
        
        // Write to full log (with colors)
        if let Ok(mut writer) = self.full_writer.lock() {
            let _ = writer.write_all(timestamped.as_bytes());
            let _ = writer.flush();
        }
        
        // Write to simple log (without colors)
        if let Ok(mut writer) = self.simple_writer.lock() {
            let _ = writer.write_all(clean_timestamped.as_bytes());
            let _ = writer.flush();
        }
    }
    
    /// Log a message only to the full log (for verbose output like task execution)
    pub fn log_full_only(&self, message: &str) {
        let timestamped = format!("[{}] {}\n", Local::now().format("%H:%M:%S"), message);
        
        // Print to console
        print!("{}", message);
        io::stdout().flush().unwrap_or(());
        
        // Write only to full log
        if let Ok(mut writer) = self.full_writer.lock() {
            let _ = writer.write_all(timestamped.as_bytes());
            let _ = writer.flush();
        }
    }
    
    /// Log a message only to the simple log (for analysis without console output)
    pub fn log_simple_only(&self, message: &str) {
        let timestamped = format!("[{}] {}\n", Local::now().format("%H:%M:%S"), message);
        
        if let Ok(mut writer) = self.simple_writer.lock() {
            let _ = writer.write_all(timestamped.as_bytes());
            let _ = writer.flush();
        }
    }
    
    /// Log analysis results (formatted for readability in simple log)
    pub fn log_analysis(&self, title: &str, content: &str) {
        let formatted = format!("\n{}\n{}\n{}\n", 
            "=".repeat(title.len() + 4),
            format!("  {}  ", title),
            "=".repeat(title.len() + 4)
        );
        
        // Print to console with colors
        print!("{}", content);
        io::stdout().flush().unwrap_or(());
        
        // Write formatted analysis to both logs
        let analysis_content = format!("{}{}\n", formatted, content);
        let clean_content = Self::strip_color_codes(content);
        let clean_analysis = format!("{}{}\n", formatted, clean_content);
        
        if let Ok(mut writer) = self.full_writer.lock() {
            let timestamped = format!("[{}] {}", Local::now().format("%H:%M:%S"), analysis_content);
            let _ = writer.write_all(timestamped.as_bytes());
            let _ = writer.flush();
        }
        
        // Simple log gets clean version
        if let Ok(mut writer) = self.simple_writer.lock() {
            let _ = writer.write_all(clean_analysis.as_bytes());
            let _ = writer.flush();
        }
    }
    
    /// Strip ANSI color codes from text
    fn strip_color_codes(text: &str) -> String {
        // Simple regex-like approach to remove ANSI escape sequences
        let mut result = String::new();
        let mut chars = text.chars().peekable();
        
        while let Some(ch) = chars.next() {
            if ch == '\x1b' && chars.peek() == Some(&'[') {
                // Skip the escape sequence
                chars.next(); // consume '['
                while let Some(c) = chars.next() {
                    if c.is_ascii_alphabetic() {
                        break; // End of escape sequence
                    }
                }
            } else {
                result.push(ch);
            }
        }
        result
    }
    
    /// Finish logging and return the log directory path
    pub fn finish(&self) -> PathBuf {
        let end_time = Local::now().format("%Y-%m-%d %H:%M:%S");
        let footer = format!("\n=== LOG COMPLETED AT {} ===\n", end_time);
        
        if let Ok(mut writer) = self.full_writer.lock() {
            let _ = writer.write_all(footer.as_bytes());
            let _ = writer.flush();
        }
        
        if let Ok(mut writer) = self.simple_writer.lock() {
            let _ = writer.write_all(footer.as_bytes());
            let _ = writer.flush();
        }
        
        self.log_dir.clone()
    }
}

/// Macro to simplify logging usage
#[macro_export]
macro_rules! log_both {
    ($logger:expr, $($arg:tt)*) => {
        $logger.log_both(&format!($($arg)*))
    };
}

#[macro_export]
macro_rules! log_full {
    ($logger:expr, $($arg:tt)*) => {
        $logger.log_full_only(&format!($($arg)*))
    };
}

#[macro_export]
macro_rules! log_simple {
    ($logger:expr, $($arg:tt)*) => {
        $logger.log_simple_only(&format!($($arg)*))
    };
}
