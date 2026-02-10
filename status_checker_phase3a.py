#!/usr/bin/env python3
"""
PERIODIC STATUS CHECKER FOR PHASE 3a TRAINING
Lightweight status checker with configurable intervals
Designed to run in parallel with training
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import psutil


class StatusChecker:
    """Lightweight status checker for Phase 3a training."""
    
    def __init__(self):
        self.ryot_root = Path.cwd()
        self.checkpoints_dir = self.ryot_root / "checkpoints_scaled"
        self.logs_dir = self.ryot_root / "logs_scaled"
        self.status_file = self.ryot_root / "phase3a_status.json"
    
    def get_process_info(self) -> Dict:
        """Get information about running training process."""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and any('train_scaled_model.py' in arg for arg in cmdline):
                    return {
                        'pid': proc.pid,
                        'running': proc.is_running(),
                        'memory_mb': proc.memory_info().rss / 1024 / 1024,
                        'cpu_percent': proc.cpu_percent(interval=0.1)
                    }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return {'pid': None, 'running': False, 'memory_mb': 0, 'cpu_percent': 0}
    
    def get_file_stats(self) -> Dict:
        """Get file system statistics."""
        stats = {
            'checkpoint_files': 0,
            'latest_checkpoint': None,
            'checkpoint_size_mb': 0,
            'log_files': 0,
            'hasComparison': False
        }
        
        if self.checkpoints_dir.exists():
            checkpoints = list(self.checkpoints_dir.glob("*.pt"))
            stats['checkpoint_files'] = len(checkpoints)
            
            if checkpoints:
                latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                stats['latest_checkpoint'] = latest.name
                stats['checkpoint_size_mb'] = latest.stat().st_size / 1024 / 1024
        
        if self.logs_dir.exists():
            logs = list(self.logs_dir.glob("*.log")) + list(self.logs_dir.glob("*.json"))
            stats['log_files'] = len([f for f in logs if 'monitor' not in f.name])
            stats['hasComparison'] = (self.logs_dir / "phase3_stage3a_comparison.json").exists()
        
        return stats
    
    def get_quickstatus(self) -> Dict:
        """Quick status snapshot."""
        process_info = self.get_process_info()
        file_stats = self.get_file_stats()
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'process': process_info,
            'files': file_stats,
            'complete': file_stats['hasComparison']
        }
        
        return status
    
    def print_status_compact(self, status: Dict) -> None:
        """Print compact status for console output."""
        proc = status['process']
        files = status['files']
        
        # Process status
        proc_status = "🟢 RUNNING" if proc['running'] else "🔴 STOPPED"
        
        # File status
        ckpt_status = f"📦 {files['checkpoint_files']} checkpoints"
        log_status = f"📝 {files['log_files']} logs"
        
        # Completion
        complete_status = "✅ COMPLETE" if status['complete'] else "⏳ IN PROGRESS"
        
        print(f"{proc_status} | {ckpt_status} | {log_status} | {complete_status}")
        
        if proc['running']:
            print(f"   Memory: {proc['memory_mb']:.0f}MB | CPU: {proc['cpu_percent']:.1f}%")
    
    def save_status(self, status: Dict) -> None:
        """Save status to JSON file for external monitoring."""
        try:
            with open(self.status_file, 'w') as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            print(f"Error saving status: {e}")
    
    def check_once(self, verbose: bool = False) -> Dict:
        """Perform single status check."""
        status = self.get_quickstatus()
        
        if verbose:
            self.print_status_compact(status)
        
        self.save_status(status)
        
        return status
    
    def check_periodic(self, interval: int = 30, max_iterations: int = 0) -> None:
        """
        Perform periodic status checks.
        
        Args:
            interval: Seconds between checks
            max_iterations: Max checks (0 = infinite)
        """
        print(f"Starting periodic status checks every {interval}s...")
        print("Press Ctrl+C to stop\n")
        
        iteration = 0
        while max_iterations == 0 or iteration < max_iterations:
            iteration += 1
            
            status = self.check_once(verbose=True)
            
            if status['complete']:
                print("\n✅ Training complete! Analysis ready.")
                break
            
            time.sleep(interval)
    
    def wait_for_completion(self, timeout: int = 1200) -> bool:
        """
        Wait for training completion.
        
        Args:
            timeout: Maximum wait time in seconds
        
        Returns:
            True if completed, False if timeout
        """
        start = time.time()
        
        while time.time() - start < timeout:
            status = self.check_once()
            
            if status['complete']:
                print("\n✅ Training completed successfully!")
                return True
            
            # Print progress dot every 30 seconds
            if int(time.time() - start) % 30 == 0:
                elapsed = int(time.time() - start)
                print(f"⏳ Still running ({elapsed}s elapsed)...", end='\r')
            
            time.sleep(5)
        
        print(f"\n⚠️  Timeout after {timeout}s")
        return False


def main():
    """Main CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 3a training status checker")
    parser.add_argument('--once', action='store_true', help='Check status once and exit')
    parser.add_argument('--periodic', type=int, default=30, help='Periodic check interval (seconds)')
    parser.add_argument('--wait', action='store_true', help='Wait for completion')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    checker = StatusChecker()
    
    if args.once:
        print("📊 PHASE 3a TRAINING STATUS")
        print("="*50)
        status = checker.check_once(verbose=True)
        
    elif args.wait:
        print("⏳ Waiting for training completion...")
        print("   (Timeout: 20 minutes)")
        completed = checker.wait_for_completion(timeout=1200)
        
        if completed:
            # Load and show results
            result_file = checker.logs_dir / "phase3_stage3a_comparison.json"
            try:
                with open(result_file) as f:
                    results = json.load(f)
                
                print("\n" + "="*50)
                print("📈 TRAINING RESULTS")
                print("="*50)
                print(json.dumps(results, indent=2))
            except:
                pass
    
    else:
        print("📊 PHASE 3a TRAINING STATUS MONITOR")
        print("="*50)
        checker.check_periodic(interval=args.periodic)


if __name__ == "__main__":
    main()
