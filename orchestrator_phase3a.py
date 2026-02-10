#!/usr/bin/env python3
"""
PHASE 3a TRAINING ORCHESTRATOR
Complete training monitoring, status checking, and alert system
Launches all components and provides unified control
"""

import os
import sys
import subprocess
import threading
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional


class TrainingOrchestrator:
    """Orchestrate all Phase 3a training monitoring components."""
    
    def __init__(self):
        self.ryot_root = Path.cwd()
        self.logs_dir = self.ryot_root / "logs_scaled"
        self.monitor_script = self.ryot_root / "monitor_phase3a_training.py"
        self.checker_script = self.ryot_root / "status_checker_phase3a.py"
        self.alert_script = self.ryot_root / "alert_service_phase3a.py"
        self.orchestrator_log = self.logs_dir / "orchestrator.log"
        
        self.processes = {}
        self.running = False
    
    def log(self, message: str) -> None:
        """Log message to console and file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        
        print(log_msg)
        
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            with open(self.orchestrator_log, "a") as f:
                f.write(log_msg + "\n")
        except:
            pass
    
    def start_monitor(self) -> Optional[subprocess.Popen]:
        """Start the main progress monitor."""
        try:
            self.log("Starting progress monitor...")
            proc = subprocess.Popen(
                [sys.executable, str(self.monitor_script)],
                cwd=self.ryot_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            self.processes['monitor'] = proc
            self.log(f"✅ Monitor started (PID: {proc.pid})")
            
            return proc
        
        except Exception as e:
            self.log(f"❌ Failed to start monitor: {e}")
            return None
    
    def start_periodic_checker(self) -> Optional[subprocess.Popen]:
        """Start the periodic status checker."""
        try:
            self.log("Starting periodic status checker...")
            proc = subprocess.Popen(
                [sys.executable, str(self.checker_script), "--periodic", "30"],
                cwd=self.ryot_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            self.processes['checker'] = proc
            self.log(f"✅ Status checker started (PID: {proc.pid})")
            
            return proc
        
        except Exception as e:
            self.log(f"❌ Failed to start status checker: {e}")
            return None
    
    def monitor_subprocess_output(self, name: str, proc: subprocess.Popen) -> None:
        """Monitor subprocess output in background thread."""
        try:
            while proc.poll() is None:
                line = proc.stdout.readline()
                if line:
                    # Print subprocess output
                    print(f"  [{name}] {line.rstrip()}")
                time.sleep(0.01)
        except Exception as e:
            self.log(f"Error monitoring {name}: {e}")
    
    def start_output_monitors(self) -> None:
        """Start threads to monitor subprocess output."""
        for name, proc in self.processes.items():
            if proc:
                thread = threading.Thread(
                    target=self.monitor_subprocess_output,
                    args=(name, proc),
                    daemon=True
                )
                thread.start()
    
    def wait_for_completion(self, timeout: int = 1200) -> bool:
        """
        Wait for training completion.
        
        Args:
            timeout: Maximum wait time in seconds
        
        Returns:
            True if completed, False if timeout
        """
        result_file = self.logs_dir / "phase3_stage3a_comparison.json"
        start_time = time.time()
        
        self.log(f"Waiting for training completion (timeout: {timeout}s)...")
        
        while time.time() - start_time < timeout:
            if result_file.exists():
                self.log("✅ Training complete! Results file detected.")
                return True
            
            elapsed = int(time.time() - start_time)
            if elapsed % 60 == 0:
                self.log(f"⏳ Still waiting... ({elapsed}s elapsed)")
            
            time.sleep(1)
        
        self.log(f"⚠️  Timeout after {timeout}s waiting for completion")
        return False
    
    def trigger_alerts(self) -> None:
        """Trigger completion alerts."""
        try:
            self.log("Triggering completion alerts...")
            
            proc = subprocess.Popen(
                [sys.executable, str(self.alert_script), "--detailed"],
                cwd=self.ryot_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            
            output, _ = proc.communicate(timeout=10)
            self.log("✅ Alerts triggered successfully")
        
        except Exception as e:
            self.log(f"Error triggering alerts: {e}")
    
    def stop_all(self) -> None:
        """Stop all running processes."""
        self.log("Stopping all processes...")
        
        for name, proc in self.processes.items():
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                    self.log(f"✅ {name} stopped")
                except:
                    try:
                        proc.kill()
                        self.log(f"⚠️  {name} killed")
                    except Exception as e:
                        self.log(f"❌ Failed to stop {name}: {e}")
        
        self.running = False
    
    def run_full_orchestration(self, check_interval: int = 30) -> None:
        """
        Run complete orchestration workflow.
        
        Args:
            check_interval: Seconds for status checker interval
        """
        print("\n" + "="*80)
        print("🚀 PHASE 3a TRAINING ORCHESTRATOR")
        print("="*80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
        
        self.running = True
        
        try:
            # Start components
            monitor_proc = self.start_monitor()
            time.sleep(2)  # Let monitor initialize
            
            checker_proc = self.start_periodic_checker()
            time.sleep(1)
            
            # Monitor subprocess output
            self.start_output_monitors()
            
            # Wait for completion
            completed = self.wait_for_completion(timeout=1200)
            
            if completed:
                self.log("\n✅ Training completed!")
                time.sleep(2)  # Let processes catch up
                
                # Trigger alerts
                self.trigger_alerts()
            else:
                self.log("\n⚠️  Training did not complete within timeout")
        
        except KeyboardInterrupt:
            self.log("\n⏹️  Orchestration stopped by user")
        
        except Exception as e:
            self.log(f"\n❌ Orchestration error: {e}")
        
        finally:
            # Cleanup
            self.stop_all()
            
            print("\n" + "="*80)
            print("📋 ORCHESTRATION COMPLETE")
            print("="*80)
            print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Log file: {self.orchestrator_log}")
            print("="*80 + "\n")
    
    def run_with_wait_mode(self) -> None:
        """Run in wait-only mode (monitor background training)."""
        print("\n" + "="*80)
        print("⏳ PHASE 3a TRAINING WAIT MODE")
        print("="*80)
        print("Monitoring background training for completion...")
        print("Press Ctrl+C to cancel\n")
        
        result_file = self.logs_dir / "phase3_stage3a_comparison.json"
        
        try:
            completed = self.wait_for_completion(timeout=1200)
            
            if completed:
                self.log("✅ Training detected as complete!")
                self.trigger_alerts()
                
                # Display results
                if result_file.exists():
                    try:
                        with open(result_file) as f:
                            results = json.load(f)
                        
                        print("\n" + "-"*80)
                        print("📊 RESULTS:")
                        print("-"*80)
                        print(json.dumps(results, indent=2))
                    except Exception as e:
                        print(f"Error loading results: {e}")
            else:
                print("\n⚠️  Timeout waiting for completion")
        
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped")
        
        finally:
            print("\n" + "="*80)


def main():
    """Main orchestrator entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Phase 3a Training Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full orchestration (starts monitor, checker, waits for completion)
  python orchestrator_phase3a.py --full
  
  # Wait mode (monitor background training only)
  python orchestrator_phase3a.py --wait
  
  # Start just the monitor
  python orchestrator_phase3a.py --monitor
  
  # Start just the periodic checker
  python orchestrator_phase3a.py --checker
  
  # Trigger alerts immediately
  python orchestrator_phase3a.py --alert
        """
    )
    
    parser.add_argument('--full', action='store_true', help='Run full orchestration')
    parser.add_argument('--wait', action='store_true', help='Wait for completion mode')
    parser.add_argument('--monitor', action='store_true', help='Start monitor only')
    parser.add_argument('--checker', action='store_true', help='Start checker only')
    parser.add_argument('--alert', action='store_true', help='Trigger alerts')
    
    args = parser.parse_args()
    
    orchestrator = TrainingOrchestrator()
    
    try:
        if args.full:
            orchestrator.run_full_orchestration()
        
        elif args.wait:
            orchestrator.run_with_wait_mode()
        
        elif args.monitor:
            monitor = orchestrator.start_monitor()
            if monitor:
                monitor.wait()
        
        elif args.checker:
            checker = orchestrator.start_periodic_checker()
            if checker:
                checker.wait()
        
        elif args.alert:
            orchestrator.trigger_alerts()
        
        else:
            # Default: full orchestration
            orchestrator.run_full_orchestration()
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
