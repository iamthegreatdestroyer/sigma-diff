#!/usr/bin/env python3
"""
Phase 3a Environment Verification Script
Checks all prerequisites before launching monitoring tools
"""

import os
import sys
import json
from pathlib import Path
from typing import Tuple, Dict, List

class EnvironmentChecker:
    def __init__(self):
        self.ryot_root = Path.cwd()
        self.checks: Dict[str, Tuple[bool, str]] = {}
        self.critical_failures = []
        
    def check_python_version(self) -> bool:
        """Verify Python 3.8+"""
        version = sys.version_info
        required = (3, 8)
        passed = version >= required
        
        status = f"Python {version.major}.{version.minor}.{version.micro}"
        if passed:
            self.checks["Python Version"] = (True, f"✅ {status} (≥3.8 required)")
        else:
            self.checks["Python Version"] = (False, f"❌ {status} (requires ≥3.8)")
            self.critical_failures.append("Python version too old")
        
        return passed
    
    def check_module(self, module_name: str) -> bool:
        """Verify required module is installed"""
        try:
            __import__(module_name)
            self.checks[f"Module: {module_name}"] = (True, f"✅ Installed")
            return True
        except ImportError:
            self.checks[f"Module: {module_name}"] = (False, f"❌ Not found (run: pip install {module_name})")
            if module_name == "psutil":
                self.critical_failures.append(f"Missing module: {module_name}")
            return False
    
    def check_script_exists(self, script_name: str) -> bool:
        """Verify monitoring script exists"""
        path = self.ryot_root / script_name
        if path.exists():
            size_kb = path.stat().st_size / 1024
            self.checks[f"Script: {script_name}"] = (True, f"✅ Found ({size_kb:.1f} KB)")
            return True
        else:
            self.checks[f"Script: {script_name}"] = (False, f"❌ Not found at {path}")
            self.critical_failures.append(f"Missing script: {script_name}")
            return False
    
    def check_training_script(self) -> bool:
        """Verify training script exists"""
        path = self.ryot_root / "train_scaled_model.py"
        if path.exists():
            self.checks["Training Script"] = (True, "✅ train_scaled_model.py found")
            return True
        else:
            self.checks["Training Script"] = (False, f"⚠️  Not found (may not be needed if training active)")
            return True  # Not critical
    
    def check_logs_directory(self) -> bool:
        """Verify logs directory exists/can be created"""
        log_dir = self.ryot_root / "logs_scaled"
        try:
            log_dir.mkdir(exist_ok=True)
            self.checks["Logs Directory"] = (True, f"✅ logs_scaled/ ready")
            return True
        except Exception as e:
            self.checks["Logs Directory"] = (False, f"❌ Cannot create: {e}")
            self.critical_failures.append(f"Cannot create logs directory: {e}")
            return False
    
    def check_checkpoints_directory(self) -> bool:
        """Verify checkpoints directory exists"""
        ckpt_dir = self.ryot_root / "checkpoints_scaled"
        if ckpt_dir.exists():
            count = len(list(ckpt_dir.glob("*.pt")))
            self.checks["Checkpoints Directory"] = (True, f"✅ Found ({count} checkpoints)")
            return True
        else:
            self.checks["Checkpoints Directory"] = (True, f"⚠️  Not created yet (will be created during training)")
            return True  # Not critical
    
    def check_disk_space(self) -> bool:
        """Check available disk space"""
        try:
            stat = os.statvfs(self.ryot_root)
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            
            if free_gb > 1:
                self.checks["Disk Space"] = (True, f"✅ {free_gb:.1f} GB available")
                return True
            else:
                self.checks["Disk Space"] = (False, f"⚠️  Low: {free_gb:.1f} GB (need >1 GB)")
                return True  # Warning only
        except Exception as e:
            self.checks["Disk Space"] = (False, f"⚠️  Cannot check: {e}")
            return True
    
    def check_training_status(self) -> str:
        """Check if training is currently running"""
        try:
            import psutil
            
            training_proc = None
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.name() == 'python.exe' or proc.name() == 'python':
                        cmdline = ' '.join(proc.cmdline())
                        if 'train_scaled_model.py' in cmdline:
                            training_proc = proc
                            break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if training_proc:
                self.checks["Training Status"] = (True, f"✅ Running (PID {training_proc.pid})")
                return "running"
            else:
                self.checks["Training Status"] = (True, f"⚠️  Not currently detected")
                return "unknown"
        except Exception as e:
            self.checks["Training Status"] = (True, f"⚠️  Cannot detect: {e}")
            return "unknown"
    
    def check_launch_scripts(self) -> bool:
        """Check if launcher scripts exist"""
        on_windows = sys.platform == 'win32'
        
        if on_windows:
            bat_exists = (self.ryot_root / "launch_orchestrator.bat").exists()
            ps1_exists = (self.ryot_root / "launch_orchestrator.ps1").exists()
            
            if bat_exists and ps1_exists:
                self.checks["Launcher Scripts"] = (True, f"✅ Both .bat and .ps1 available")
                return True
            elif bat_exists or ps1_exists:
                what = "bat" if bat_exists else "ps1"
                self.checks["Launcher Scripts"] = (True, f"✅ {what} launcher available")
                return True
            else:
                self.checks["Launcher Scripts"] = (False, f"❌ No launcher scripts found")
                return False
        else:
            ps1_exists = (self.ryot_root / "launch_orchestrator.ps1").exists()
            if ps1_exists:
                self.checks["Launcher Scripts"] = (True, f"✅ PowerShell launcher available (Linux/Mac)")
                return True
            else:
                self.checks["Launcher Scripts"] = (True, f"⚠️  No launcher (can run tools directly)")
                return True
    
    def run_all_checks(self) -> bool:
        """Run all environment checks"""
        print("\n" + "="*80)
        print("🔍 PHASE 3a ENVIRONMENT VERIFICATION".center(80))
        print("="*80 + "\n")
        
        # Python version
        self.check_python_version()
        
        # Required modules
        self.check_module("psutil")
        self.check_module("numpy")
        
        # Scripts
        print("Checking monitoring scripts...")
        scripts = [
            "orchestrator_phase3a.py",
            "monitor_phase3a_training.py",
            "status_checker_phase3a.py",
            "alert_service_phase3a.py",
        ]
        for script in scripts:
            self.check_script_exists(script)
        
        # Training script
        self.check_training_script()
        
        # Directories
        print("Checking directories...")
        self.check_logs_directory()
        self.check_checkpoints_directory()
        
        # Resources
        print("Checking resources...")
        self.check_disk_space()
        
        # Launchers
        self.check_launch_scripts()
        
        # Training status
        print("Checking training status...")
        self.check_training_status()
        
        return True
    
    def print_results(self):
        """Print verification results"""
        print("\n" + "="*80)
        print("VERIFICATION RESULTS".center(80))
        print("="*80 + "\n")
        
        for check_name, (passed, message) in self.checks.items():
            print(f"{check_name:.<40} {message}")
        
        print("\n" + "="*80)
        
        if self.critical_failures:
            print("❌ CRITICAL FAILURES DETECTED:\n")
            for i, failure in enumerate(self.critical_failures, 1):
                print(f"  {i}. {failure}")
            print("\n⚠️  Please fix the above issues before launching monitoring tools.\n")
            return False
        else:
            print("✅ ALL CHECKS PASSED - READY TO LAUNCH MONITORING TOOLS\n")
            return True
    
    def print_recommendations(self):
        """Print recommendations based on checks"""
        print("📋 NEXT STEPS:\n")
        
        if "Training Status" in self.checks:
            status_msg = self.checks["Training Status"][1]
            if "✅ Running" in status_msg:
                print("  1. ✅ Training is running - perfect!")
                print("  2. Choose a launch method from QUICK_START_PHASE3A.md")
                print("  3. Recommended: python orchestrator_phase3a.py --full")
            else:
                print("  ⚠️  Training not detected as running")
                print("  1. Start training: python train_scaled_model.py")
                print("  2. Then launch monitoring: python orchestrator_phase3a.py --full")
        
        print("\n📖 FOR DETAILED INFORMATION:")
        print("  • Read: QUICK_START_PHASE3A.md")
        print("  • Read: MONITORING_GUIDE.md")
        print("  • Use launcher: launch_orchestrator.bat (Windows)")
        print("                 ./launch_orchestrator.ps1 (PowerShell)")
        print("")

def main():
    checker = EnvironmentChecker()
    checker.run_all_checks()
    
    success = checker.print_results()
    checker.print_recommendations()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
