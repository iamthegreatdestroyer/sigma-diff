#!/usr/bin/env python3
"""
Training Dashboard Generator
Generates comprehensive training progress reports from metrics JSON.
"""

import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingDashboard:
    """Generate and display training progress reports."""
    
    def __init__(self, metrics_file: str):
        """Initialize dashboard with metrics file."""
        self.metrics_file = Path(metrics_file)
        self.metrics: Dict[str, Any] = {}
        self._load_metrics()
    
    def _load_metrics(self):
        """Load metrics from JSON file."""
        try:
            with open(self.metrics_file, 'r') as f:
                self.metrics = json.load(f)
            logger.info(f"✅ Loaded metrics from {self.metrics_file}")
        except FileNotFoundError:
            logger.error(f"❌ Metrics file not found: {self.metrics_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in metrics file: {e}")
            raise
    
    def generate_text_report(self) -> str:
        """Generate text-based training report."""
        report_lines = []
        
        # Header
        report_lines.append("=" * 70)
        report_lines.append("📊 TRAINING PROGRESS REPORT")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # Training Summary
        report_lines.append("📈 TRAINING SUMMARY")
        report_lines.append("-" * 70)
        report_lines.append(f"  Current Epoch:       {self.metrics.get('current_epoch', 'N/A')} / {self.metrics.get('total_epochs', 'N/A')}")
        report_lines.append(f"  Total Steps:         {self.metrics.get('total_steps', 'N/A')}")
        report_lines.append(f"  Max Steps:           {self.metrics.get('max_steps', 'N/A')}")
        report_lines.append("")
        
        # Loss Metrics
        report_lines.append("📉 LOSS METRICS")
        report_lines.append("-" * 70)
        report_lines.append(f"  Training Loss:       {self.metrics.get('loss', 'N/A'):.4f}")
        report_lines.append(f"  Final Loss:          {self.metrics.get('final_loss', 'N/A'):.4f}")
        
        if 'eval_loss' in self.metrics:
            report_lines.append(f"  Validation Loss:     {self.metrics.get('eval_loss', 'N/A'):.4f}")
        
        if 'loss_min' in self.metrics:
            report_lines.append(f"  Min Loss:            {self.metrics.get('loss_min', 'N/A'):.4f}")
        
        if 'loss_max' in self.metrics:
            report_lines.append(f"  Max Loss:            {self.metrics.get('loss_max', 'N/A'):.4f}")
        
        report_lines.append("")
        
        # Learning Rate
        report_lines.append("🎯 OPTIMIZATION")
        report_lines.append("-" * 70)
        report_lines.append(f"  Learning Rate:       {self.metrics.get('lr', 'N/A'):.2e}")
        report_lines.append(f"  Batch Size:          {self.metrics.get('batch_size', 'N/A')}")
        report_lines.append(f"  Gradient Accum:      {self.metrics.get('gradient_accumulation_steps', 'N/A')}")
        report_lines.append(f"  Effective Batch:     {self.metrics.get('effective_batch_size', 'N/A')}")
        report_lines.append("")
        
        # GPU Usage
        report_lines.append("💾 GPU RESOURCE USAGE")
        report_lines.append("-" * 70)
        report_lines.append(f"  GPU Memory Used:     {self.metrics.get('gpu_memory_gb', 'N/A'):.1f} GB")
        report_lines.append(f"  GPU Memory Reserved: {self.metrics.get('gpu_memory_reserved_gb', 'N/A'):.1f} GB")
        
        if 'gpu_utilization_percent' in self.metrics:
            report_lines.append(f"  GPU Utilization:     {self.metrics.get('gpu_utilization_percent', 'N/A'):.1f}%")
        
        report_lines.append("")
        
        # Performance
        report_lines.append("⚡ PERFORMANCE METRICS")
        report_lines.append("-" * 70)
        report_lines.append(f"  Samples/Second:      {self.metrics.get('samples_per_sec', 'N/A'):.1f}")
        
        if 'tokens_per_sec' in self.metrics:
            report_lines.append(f"  Tokens/Second:       {self.metrics.get('tokens_per_sec', 'N/A'):.1f}")
        
        if 'avg_tokens_per_sec' in self.metrics:
            report_lines.append(f"  Avg Tokens/Second:   {self.metrics.get('avg_tokens_per_sec', 'N/A'):.1f}")
        
        report_lines.append(f"  Epoch Duration:      {self.metrics.get('epoch_duration_minutes', 'N/A'):.1f} min")
        report_lines.append(f"  Total Duration:      {self.metrics.get('total_duration_hours', 'N/A'):.1f} hours")
        report_lines.append("")
        
        # Detailed Breakdown (if available)
        if 'detailed_metrics' in self.metrics:
            report_lines.append("📋 DETAILED METRICS")
            report_lines.append("-" * 70)
            for key, value in self.metrics['detailed_metrics'].items():
                if isinstance(value, (int, float)):
                    report_lines.append(f"  {key:<30} {value:.4f}")
                else:
                    report_lines.append(f"  {key:<30} {value}")
            report_lines.append("")
        
        # Training Status
        report_lines.append("✅ STATUS")
        report_lines.append("-" * 70)
        status = self.metrics.get('status', 'Unknown')
        report_lines.append(f"  Training Status:     {status}")
        
        if 'completion_percent' in self.metrics:
            report_lines.append(f"  Completion:          {self.metrics.get('completion_percent', 'N/A'):.1f}%")
        
        report_lines.append("")
        
        # Metadata
        report_lines.append("📝 METADATA")
        report_lines.append("-" * 70)
        report_lines.append(f"  Timestamp:           {self.metrics.get('timestamp', datetime.now().isoformat())}")
        report_lines.append(f"  Git SHA:             {self.metrics.get('git_sha', 'N/A')}")
        report_lines.append(f"  Run Number:          {self.metrics.get('run_number', 'N/A')}")
        report_lines.append("")
        
        # Footer
        report_lines.append("=" * 70)
        report_lines.append(f"Generated: {datetime.now().isoformat()}")
        report_lines.append("=" * 70)
        
        return "\n".join(report_lines)
    
    def generate_json_report(self) -> Dict[str, Any]:
        """Generate JSON-formatted report."""
        return {
            'generated_at': datetime.now().isoformat(),
            'metrics': self.metrics,
            'summary': {
                'total_epochs': self.metrics.get('total_epochs', 'N/A'),
                'current_epoch': self.metrics.get('current_epoch', 'N/A'),
                'final_loss': self.metrics.get('final_loss', 'N/A'),
                'duration_hours': self.metrics.get('total_duration_hours', 'N/A'),
                'status': self.metrics.get('status', 'Unknown'),
            }
        }
    
    def print_report(self):
        """Print formatted report to console."""
        print(self.generate_text_report())
    
    def save_report(self, output_file: str):
        """Save report to file."""
        try:
            with open(output_file, 'w') as f:
                f.write(self.generate_text_report())
            logger.info(f"✅ Report saved to {output_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save report: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description='Generate training progress dashboard',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--metrics-file',
        default='reports/training_metrics.json',
        help='Path to metrics JSON file'
    )
    parser.add_argument(
        '--output-file',
        default='reports/training_report.txt',
        help='Output file for text report'
    )
    parser.add_argument(
        '--json-output',
        help='Output file for JSON report'
    )
    parser.add_argument(
        '--print',
        action='store_true',
        default=True,
        help='Print report to console'
    )
    
    args = parser.parse_args()
    
    try:
        # Create dashboard
        dashboard = TrainingDashboard(args.metrics_file)
        
        # Print to console
        if args.print:
            dashboard.print_report()
        
        # Save text report
        dashboard.save_report(args.output_file)
        
        # Save JSON report if requested
        if args.json_output:
            with open(args.json_output, 'w') as f:
                json.dump(dashboard.generate_json_report(), f, indent=2)
            logger.info(f"✅ JSON report saved to {args.json_output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Dashboard generation failed: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
