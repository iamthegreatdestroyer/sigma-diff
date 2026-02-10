#!/usr/bin/env python3
"""
COMPLETION ALERT SERVICE FOR PHASE 3a TRAINING
Multi-channel notification system for training completion
Supports console, system notifications, and optional email
"""

import os
import sys
import json
import time
import subprocess
import smtplib
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class AlertService:
    """Multi-channel alert service for training completion."""
    
    def __init__(self):
        self.ryot_root = Path.cwd()
        self.logs_dir = self.ryot_root / "logs_scaled"
        self.result_file = self.logs_dir / "phase3_stage3a_comparison.json"
        self.checkpoints_dir = self.ryot_root / "checkpoints_scaled"
        self.alert_log = self.logs_dir / "alerts.log"
    
    def log_alert(self, message: str, level: str = "INFO") -> None:
        """Log alert to file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {level}: {message}"
        
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            with open(self.alert_log, "a") as f:
                f.write(log_msg + "\n")
        except Exception as e:
            print(f"Error logging alert: {e}")
    
    def console_alert_simple(self) -> None:
        """Send simple console alert."""
        print("\n" + "🔔 "*40)
        print("✅ PHASE 3a TRAINING COMPLETED!")
        print("🔔 "*40)
        
        self.log_alert("Console alert sent", "ALERT")
    
    def console_alert_detailed(self, results: Dict) -> None:
        """Send detailed console alert with results."""
        self.console_alert_simple()
        
        print("\n" + "="*80)
        print("📊 TRAINING SUMMARY")
        print("="*80)
        
        if 'baseline' in results:
            print("\n⏱️  BASELINE (No Optimizations):")
            baseline = results['baseline']
            print(f"   Time: {baseline.get('total_time', 'N/A')} seconds")
            print(f"   Loss: {baseline.get('final_loss', 'N/A')}")
            print(f"   Throughput: {baseline.get('throughput', 'N/A')} tokens/sec")
        
        if 'optimized' in results:
            print("\n⚡ OPTIMIZED (With Phase 1 Stack):")
            optimized = results['optimized']
            print(f"   Time: {optimized.get('total_time', 'N/A')} seconds")
            print(f"   Loss: {optimized.get('final_loss', 'N/A')}")
            print(f"   Throughput: {optimized.get('throughput', 'N/A')} tokens/sec")
        
        if 'speedup' in results:
            speedup = results['speedup']
            target = 1.25  # 25% improvement
            target_met = "✅ YES (Exceeds 25% target)" if speedup >= target else "⚠️  NO (Below target)"
            
            print(f"\n🚀 SPEEDUP ACHIEVED:")
            print(f"   Factor: {speedup:.2f}x")
            print(f"   Percentage: {(speedup - 1) * 100:.1f}% faster")
            print(f"   Target Met: {target_met}")
        
        if 'checkpoint_info' in results:
            print(f"\n💾 CHECKPOINTS:")
            for ckpt in results.get('checkpoint_info', []):
                print(f"   - {ckpt}")
        
        print("="*80 + "\n")
        
        self.log_alert("Detailed console alert sent with results", "ALERT")
    
    def windows_notification(self, title: str = "Phase 3a Training", 
                            message: str = "Training completed successfully!") -> bool:
        """Send Windows system notification."""
        try:
            import ctypes
            
            # Windows message box
            MB_ICONINFORMATION = 0x1000
            result = ctypes.windll.user32.MessageBoxW(
                0,
                message,
                title,
                MB_ICONINFORMATION
            )
            
            self.log_alert("Windows notification sent", "ALERT")
            return True
        
        except Exception as e:
            self.log_alert(f"Windows notification failed: {e}", "WARN")
            return False
    
    def powershell_notification(self) -> bool:
        """Send PowerShell toast notification (Windows 10+)."""
        try:
            ps_script = """
$title = 'Phase 3a Training'
$message = 'Training completed successfully!'
$Icon = 'C:\\Windows\\System32\\Shell32.dll'

[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.UI.Notifications.ToastNotification, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, System.Xml.XmlDocument, ContentType = WindowsRuntime] | Out-Null

$APP_ID = 'RYZEN-LLM'

$template = @"
<toast>
    <visual>
        <binding template="ToastText02">
            <text id="1">$title</text>
            <text id="2">$message</text>
        </binding>
    </visual>
</toast>
"@

$xml = New-Object Windows.Data.Xml.Dom.XmlDocument
$xml.LoadXml($template)
$toast = New-Object Windows.UI.Notifications.ToastNotification $xml
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($APP_ID).Show($toast)
"""
            subprocess.run(
                ["powershell", "-Command", ps_script],
                capture_output=True,
                timeout=5
            )
            
            self.log_alert("PowerShell notification sent", "ALERT")
            return True
        
        except Exception as e:
            self.log_alert(f"PowerShell notification failed: {e}", "WARN")
            return False
    
    def stdout_beep(self) -> None:
        """Send audible alert via system beep."""
        try:
            for _ in range(3):
                print("\a", end="", flush=True)
                time.sleep(0.3)
            
            self.log_alert("Audible alert sent", "ALERT")
        except:
            pass
    
    def email_alert(self, recipient: str, smtp_config: Dict) -> bool:
        """
        Send email alert.
        
        Args:
            recipient: Email address to send to
            smtp_config: Dict with 'server', 'port', 'username', 'password'
        """
        try:
            # Load results
            if not self.result_file.exists():
                self.log_alert("Results file not found for email", "WARN")
                return False
            
            with open(self.result_file) as f:
                results = json.load(f)
            
            # Prepare email
            sender = smtp_config.get('username', 'ryzen-llm@example.com')
            subject = "✅ Phase 3a Training Completed Successfully"
            
            body = self._format_email_body(results)
            
            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = recipient
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_config['server'], smtp_config['port']) as server:
                server.starttls()
                server.login(smtp_config['username'], smtp_config['password'])
                server.send_message(msg)
            
            self.log_alert(f"Email alert sent to {recipient}", "ALERT")
            return True
        
        except Exception as e:
            self.log_alert(f"Email alert failed: {e}", "WARN")
            return False
    
    def _format_email_body(self, results: Dict) -> str:
        """Format email body with results."""
        body = "Phase 3a Training Completion Report\n"
        body += "=" * 60 + "\n\n"
        
        body += "STATUS: ✅ COMPLETED SUCCESSFULLY\n"
        body += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        if 'baseline' in results:
            baseline = results['baseline']
            body += "BASELINE TRAINING (No Optimizations):\n"
            body += f"  - Time: {baseline.get('total_time', 'N/A')}s\n"
            body += f"  - Final Loss: {baseline.get('final_loss', 'N/A')}\n"
            body += f"  - Throughput: {baseline.get('throughput', 'N/A')} tok/s\n\n"
        
        if 'optimized' in results:
            optimized = results['optimized']
            body += "OPTIMIZED TRAINING (Phase 1 Stack):\n"
            body += f"  - Time: {optimized.get('total_time', 'N/A')}s\n"
            body += f"  - Final Loss: {optimized.get('final_loss', 'N/A')}\n"
            body += f"  - Throughput: {optimized.get('throughput', 'N/A')} tok/s\n\n"
        
        if 'speedup' in results:
            speedup = results['speedup']
            body += "SPEEDUP ANALYSIS:\n"
            body += f"  - Factor: {speedup:.2f}x\n"
            body += f"  - Improvement: {(speedup - 1) * 100:.1f}%\n"
            target_met = speedup >= 1.25
            body += f"  - Target (25%) Met: {'YES' if target_met else 'NO'}\n\n"
        
        body += "-" * 60 + "\n"
        body += "Next Steps: Proceed to Phase 3b (Production Server)\n"
        
        return body
    
    def trigger_all_alerts(self, detailed: bool = True, attemp_notifications: bool = True) -> Dict:
        """
        Trigger all available alert channels.
        
        Args:
            detailed: Show detailed results
            attemp_notifications: Try system notifications
        
        Returns:
            Dict with results of each alert channel
        """
        alerts_sent = {
            'console': False,
            'windows_notification': False,
            'powershell_notification': False,
            'audible': False,
            'email': False
        }
        
        try:
            # Load results
            if self.result_file.exists():
                with open(self.result_file) as f:
                    results = json.load(f)
                
                if detailed:
                    self.console_alert_detailed(results)
                else:
                    self.console_alert_simple()
            else:
                self.console_alert_simple()
            
            alerts_sent['console'] = True
        
        except Exception as e:
            self.log_alert(f"Console alert error: {e}", "ERROR")
        
        # Try system notifications if requested
        if attemp_notifications:
            try:
                if self.windows_notification():
                    alerts_sent['windows_notification'] = True
            except:
                pass
            
            try:
                if self.powershell_notification():
                    alerts_sent['powershell_notification'] = True
            except:
                pass
        
        # Audible alert
        try:
            self.stdout_beep()
            alerts_sent['audible'] = True
        except:
            pass
        
        # Log summary
        successful = sum(1 for v in alerts_sent.values() if v)
        self.log_alert(f"Alerts triggered: {successful}/{len(alerts_sent)}", "ALERT")
        
        return alerts_sent


def main():
    """Main alert service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 3a completion alert service")
    parser.add_argument('--detailed', action='store_true', help='Show detailed results')
    parser.add_argument('--quiet', action='store_true', help='No notifications')
    parser.add_argument('--email', type=str, help='Send email to address')
    parser.add_argument('--no-system', action='store_true', help='Skip system notify')
    
    args = parser.parse_args()
    
    service = AlertService()
    
    # Trigger alerts
    alerts = service.trigger_all_alerts(
        detailed=args.detailed,
        attemp_notifications=not args.no_system and not args.quiet
    )
    
    print("\n" + "="*60)
    print("Alert Service Summary:")
    for channel, sent in alerts.items():
        status = "✅" if sent else "❌"
        print(f"  {status} {channel.replace('_', ' ').title()}")
    print("="*60)


if __name__ == "__main__":
    main()
