#!/usr/bin/env python3
"""
PHASE 3a TRAINING PROGRESS MONITOR
Real-time tracking of baseline and optimized training execution
Provides periodic status updates and alerts on completion
"""

import os
import sys
import json
import time
import subprocess
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import psutil
import threading


class Phase3aProgressMonitor:
    """Monitor Phase 3a training progress in real-time."""
    
    def __init__(self, check_interval: int = 10):
        """
        Initialize monitor.
        
        Args:
            check_interval: Seconds between status checks (default 10s)
        """
        self.check_interval = check_interval
        self.ryot_root = Path.cwd()
        self.script_path = self.ryot_root / "train_scaled_model.py"
        self.checkpoints_dir = self.ryot_root / "checkpoints_scaled"
        self.logs_dir = self.ryot_root / "logs_scaled"
        self.monitor_log = self.ryot_root / "logs_scaled" / "monitor.log"
        
        # Create logs directory if needed
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_time = None
        self.process = None
        self.is_complete = False
        self.results = {
            'baseline': None,
            'optimized': None,
            'comparison': None
        }
        self.last_output = ""
        self.epoch_progress = {"baseline": 0, "optimized": 0}
        
        print("\n" + "="*70)
        print("🎯 PHASE 3a TRAINING PROGRESS MONITOR")
        print("="*70)
        print(f"Monitor interval: {check_interval}s")
        print(f"Log directory: {self.logs_dir}")
        print("="*70 + "\n")
    
    def log_progress(self, message: str, level: str = "INFO") -> None:
        """Log progress message to file and console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {level}: {message}"
        
        print(log_msg)
        
        # Write to monitor log
        try:
            with open(self.monitor_log, "a") as f:
                f.write(log_msg + "\n")
        except Exception as e:
            print(f"⚠️  Could not write to log: {e}")
    
    def get_python_process(self) -> Optional[psutil.Process]:
        """Find the running train_scaled_model.py process."""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline and any('train_scaled_model.py' in arg for arg in cmdline):
                        return proc
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            self.log_progress(f"Error scanning processes: {e}", "WARN")
        
        return None
    
    def check_training_status(self) -> Dict:
        """Check current training status."""
        status = {
            'running': False,
            'process_exists': False,
            'has_checkpoints': False,
            'has_logs': False,
            'baseline_complete': False,
            'optimized_complete': False,
            'epoch_progress': self.epoch_progress.copy(),
            'elapsed_time': None,
            'output_files': []
        }
        
        # Check if process exists
        proc = self.get_python_process()
        if proc:
            status['process_exists'] = True
            try:
                status['running'] = proc.is_running()
            except:
                status['running'] = False
        
        # Check output directories
        if self.checkpoints_dir.exists():
            checkpoints = list(self.checkpoints_dir.glob("*.pt"))
            status['has_checkpoints'] = len(checkpoints) > 0
            status['output_files'].extend([str(f) for f in checkpoints])
        
        if self.logs_dir.exists():
            logs = list(self.logs_dir.glob("*.json")) + list(self.logs_dir.glob("*.log"))
            status['has_logs'] = len(logs) > 0
            status['output_files'].extend([str(f) for f in logs if 'monitor' not in str(f)])
        
        # Calculate elapsed time
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            status['elapsed_time'] = str(elapsed).split('.')[0]
        
        return status
    
    def parse_training_output(self, output: str) -> Dict:
        """Parse training output for progress information."""
        progress = {
            'phase': None,
            'epoch': None,
            'step': None,
            'loss': None,
            'throughput': None
        }
        
        # Detect which phase
        if "BASELINE TRAINING" in output:
            progress['phase'] = "baseline"
        elif "OPTIMIZED TRAINING" in output:
            progress['phase'] = "optimized"
        
        # Extract epoch
        epoch_match = re.search(r'Epoch\s+(\d+)/(\d+)', output)
        if epoch_match:
            progress['epoch'] = int(epoch_match.group(1))
        
        # Extract step
        step_match = re.search(r'Step\s+(\d+)/(\d+)', output)
        if step_match:
            progress['step'] = int(step_match.group(1))
        
        # Extract loss
        loss_match = re.search(r'loss=([0-9.]+)', output)
        if loss_match:
            progress['loss'] = float(loss_match.group(1))
        
        # Extract throughput
        throughput_match = re.search(r'throughput=([0-9.]+)\s*tok/s', output)
        if throughput_match:
            progress['throughput'] = float(throughput_match.group(1))
        
        return progress
    
    def display_status(self, status: Dict) -> None:
        """Display formatted status update."""
        print("\n" + "-"*70)
        print("📊 PHASE 3a STATUS UPDATE")
        print("-"*70)
        
        # Process status
        if status['process_exists']:
            print(f"✅ Process: {'RUNNING' if status['running'] else 'COMPLETED'}")
        else:
            print(f"❌ Process: NOT FOUND")
        
        # Timing
        if status['elapsed_time']:
            print(f"⏱️  Elapsed time: {status['elapsed_time']}")
        
        # Output files
        print(f"📁 Checkpoints: {'YES' if status['has_checkpoints'] else 'pending'}")
        print(f"📝 Logs: {'YES' if status['has_logs'] else 'pending'}")
        
        if status['output_files']:
            print(f"   Files created: {len(status['output_files'])}")
            for f in status['output_files'][-3:]:  # Show last 3 files
                print(f"     - {Path(f).name}")
        
        print("-"*70 + "\n")
    
    def check_for_completion(self) -> Tuple[bool, str]:
        """Check if training is complete."""
        # Look for completion indicators
        completion_files = [
            self.logs_dir / "phase3_stage3a_comparison.json",
            self.checkpoints_dir / "scaled_model_epoch_9.pt"
        ]
        
        if all(f.exists() for f in completion_files):
            return True, "Both baseline and optimized training completed"
        
        # Check if process has finished
        proc = self.get_python_process()
        if not proc:
            if self.checkpoints_dir.exists() and any(self.checkpoints_dir.glob("*.pt")):
                return True, "Process completed with checkpoints generated"
        
        return False, ""
    
    def notify_completion(self) -> None:
        """Send completion notification."""
        message = "✅ PHASE 3a TRAINING COMPLETE!\n\nProceeding to results analysis..."
        
        self.log_progress(message, "SUCCESS")
        print("\n" + "🔔 "*20)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("🔔 "*20 + "\n")
        
        # Try to send system notification (Windows)
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0,
                "Phase 3a training is complete!\n\nResults are ready for analysis.",
                "Training Complete",
                0x1000  # MB_ICONINFORMATION
            )
        except:
            pass  # Notification not critical
    
    def load_and_display_results(self) -> None:
        """Load and display final results."""
        result_file = self.logs_dir / "phase3_stage3a_comparison.json"
        
        if result_file.exists():
            try:
                with open(result_file) as f:
                    results = json.load(f)
                
                print("\n" + "="*70)
                print("📈 PHASE 3a RESULTS")
                print("="*70)
                
                if 'baseline' in results:
                    baseline = results['baseline']
                    print(f"\n⏱️  BASELINE (no optimizations):")
                    print(f"   Total time: {baseline.get('total_time', 'N/A')}s")
                    print(f"   Final loss: {baseline.get('final_loss', 'N/A')}")
                
                if 'optimized' in results:
                    optimized = results['optimized']
                    print(f"\n⚡ OPTIMIZED (with Phase 1 stack):")
                    print(f"   Total time: {optimized.get('total_time', 'N/A')}s")
                    print(f"   Final loss: {optimized.get('final_loss', 'N/A')}")
                
                if 'speedup' in results:
                    speedup = results['speedup']
                    print(f"\n🚀 SPEEDUP:")
                    print(f"   Factor: {speedup:.2f}x")
                    
                    target_speedup = 1.25  # 25% improvement = 1.25x
                    status = "✅ EXCEEDS TARGET (25%+)" if speedup >= target_speedup else "⚠️  BELOW TARGET"
                    print(f"   Status: {status}")
                
                print("="*70 + "\n")
            
            except Exception as e:
                self.log_progress(f"Error loading results: {e}", "ERROR")
    
    def run(self, duration_limit: int = 900) -> None:
        """
        Run the monitor.
        
        Args:
            duration_limit: Maximum monitoring time in seconds (default 15 min)
        """
        self.start_time = datetime.now()
        self.log_progress(f"Starting monitor (duration limit: {duration_limit}s)", "START")
        
        check_count = 0
        max_checks = duration_limit // self.check_interval
        
        try:
            while check_count < max_checks:
                check_count += 1
                
                # Check status
                status = self.check_training_status()
                
                # Check for completion
                is_complete, reason = self.check_for_completion()
                
                if is_complete and not self.is_complete:
                    self.is_complete = True
                    self.log_progress(f"Training complete: {reason}", "SUCCESS")
                    self.notify_completion()
                    self.load_and_display_results()
                    break
                
                # Display status every 30 seconds or at key milestones
                if check_count % 3 == 0 or status['has_checkpoints']:
                    self.display_status(status)
                
                # Wait before next check
                time.sleep(self.check_interval)
        
        except KeyboardInterrupt:
            self.log_progress("Monitor stopped by user", "INFO")
            print("\n⏹️  Monitor stopped")
        
        except Exception as e:
            self.log_progress(f"Monitor error: {e}", "ERROR")
            raise
        
        finally:
            # Final status
            status = self.check_training_status()
            print("\n" + "="*70)
            print("📋 FINAL MONITOR STATUS")
            print("="*70)
            self.display_status(status)
            
            self.log_progress("Monitor session ended", "END")


def run_monitor_in_background(interval: int = 10) -> None:
    """Start monitor in background thread."""
    monitor = Phase3aProgressMonitor(check_interval=interval)
    thread = threading.Thread(target=monitor.run, daemon=True)
    thread.start()
    return thread, monitor


def main():
    """Main monitor execution."""
    monitor = Phase3aProgressMonitor(check_interval=10)
    
    # Monitor for up to 20 minutes (training expected to take 12-15 minutes)
    monitor.run(duration_limit=1200)


if __name__ == "__main__":
    main()
