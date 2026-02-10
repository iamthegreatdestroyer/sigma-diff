"""
Alerting System for LLM Inference Monitoring.

This module provides alerting capabilities with configurable rules,
severity levels, and notification handlers.

Classes:
    AlertSeverity: Enum of alert severity levels
    AlertRule: Definition of an alert condition
    Alert: An active alert instance
    AlertManager: Manages alert rules and notifications
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import logging
import json

from .metrics import MetricRegistry

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Severity levels for alerts."""
    INFO = "info"           # Informational, no action needed
    WARNING = "warning"     # May need attention soon
    CRITICAL = "critical"   # Requires immediate attention
    EMERGENCY = "emergency" # System-wide emergency


class AlertState(Enum):
    """State of an alert."""
    PENDING = "pending"     # Condition met, waiting for duration
    FIRING = "firing"       # Alert is active
    RESOLVED = "resolved"   # Alert has been resolved


@dataclass
class AlertRule:
    """
    Definition of an alert condition.
    
    Alerts fire when the condition is met for the specified duration.
    """
    name: str
    description: str
    metric_name: str
    condition: str  # 'gt', 'lt', 'eq', 'ne', 'gte', 'lte'
    threshold: float
    severity: AlertSeverity
    labels: Dict[str, str] = field(default_factory=dict)
    for_duration_seconds: float = 0.0  # How long condition must be true
    annotations: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    
    def __post_init__(self):
        """Validate the rule."""
        valid_conditions = {'gt', 'lt', 'eq', 'ne', 'gte', 'lte'}
        if self.condition not in valid_conditions:
            raise ValueError(f"Invalid condition: {self.condition}. Must be one of {valid_conditions}")
    
    def evaluate(self, value: float) -> bool:
        """
        Evaluate if the condition is met.
        
        Args:
            value: The metric value to check
            
        Returns:
            True if condition is met
        """
        if self.condition == 'gt':
            return value > self.threshold
        elif self.condition == 'lt':
            return value < self.threshold
        elif self.condition == 'eq':
            return value == self.threshold
        elif self.condition == 'ne':
            return value != self.threshold
        elif self.condition == 'gte':
            return value >= self.threshold
        elif self.condition == 'lte':
            return value <= self.threshold
        return False


@dataclass
class Alert:
    """An active or recently resolved alert."""
    rule: AlertRule
    state: AlertState
    value: float
    started_at: float
    fired_at: Optional[float] = None
    resolved_at: Optional[float] = None
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        """Get how long the alert has been active."""
        if self.resolved_at:
            return self.resolved_at - self.started_at
        return time.time() - self.started_at
    
    @property
    def fingerprint(self) -> str:
        """Generate unique fingerprint for this alert."""
        label_str = json.dumps(self.labels, sort_keys=True)
        return f"{self.rule.name}:{label_str}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "name": self.rule.name,
            "description": self.rule.description,
            "severity": self.rule.severity.value,
            "state": self.state.value,
            "value": self.value,
            "threshold": self.rule.threshold,
            "condition": self.rule.condition,
            "labels": self.labels,
            "annotations": self.annotations,
            "started_at": datetime.fromtimestamp(self.started_at).isoformat(),
            "fired_at": datetime.fromtimestamp(self.fired_at).isoformat() if self.fired_at else None,
            "resolved_at": datetime.fromtimestamp(self.resolved_at).isoformat() if self.resolved_at else None,
            "duration_seconds": self.duration_seconds
        }


class AlertManager:
    """
    Manages alert rules and notifications.
    
    Evaluates rules against metrics, tracks alert state,
    and dispatches notifications.
    
    Example:
        manager = AlertManager(registry)
        
        # Add alert rules
        manager.add_rule(AlertRule(
            name="high_latency",
            description="Inference latency is too high",
            metric_name="time_to_first_token_seconds",
            condition="gt",
            threshold=1.0,
            severity=AlertSeverity.WARNING,
            for_duration_seconds=60.0
        ))
        
        # Add notification handler
        manager.add_handler(lambda alert: print(f"ALERT: {alert.rule.name}"))
        
        # Start evaluation loop
        manager.start(interval_seconds=15.0)
    """
    
    def __init__(self, registry: Optional[MetricRegistry] = None):
        """
        Initialize the alert manager.
        
        Args:
            registry: The metric registry to evaluate (optional, creates default if not provided)
        """
        from .metrics import MetricRegistry  # Avoid circular import at module level
        self._registry = registry if registry is not None else MetricRegistry()
        self._rules: Dict[str, AlertRule] = {}
        self._pending_alerts: Dict[str, Alert] = {}  # fingerprint -> Alert
        self._active_alerts: Dict[str, Alert] = {}   # fingerprint -> Alert
        self._resolved_alerts: List[Alert] = []
        self._handlers: List[Callable[[Alert], None]] = []
        self._lock = threading.RLock()
        self._eval_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Configuration
        self._max_resolved_alerts = 100
        self._resolve_after_seconds = 300.0  # Remove from resolved after 5 min
    
    @property
    def rule_count(self) -> int:
        """Get number of registered rules."""
        return len(self._rules)
    
    @property
    def active_alert_count(self) -> int:
        """Get number of active alerts."""
        return len(self._active_alerts)
    
    def add_rule(self, rule: AlertRule) -> None:
        """
        Add an alert rule.
        
        Args:
            rule: The rule to add
        """
        with self._lock:
            self._rules[rule.name] = rule
            logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, name: str) -> None:
        """Remove an alert rule by name."""
        with self._lock:
            self._rules.pop(name, None)
            # Clean up any alerts for this rule
            to_remove = [fp for fp, a in self._pending_alerts.items() if a.rule.name == name]
            for fp in to_remove:
                del self._pending_alerts[fp]
            to_remove = [fp for fp, a in self._active_alerts.items() if a.rule.name == name]
            for fp in to_remove:
                del self._active_alerts[fp]
    
    def get_rule(self, name: str) -> Optional[AlertRule]:
        """Get a rule by name."""
        return self._rules.get(name)
    
    def get_rules(self) -> List[AlertRule]:
        """Get all rules."""
        return list(self._rules.values())
    
    def add_handler(self, handler: Callable[[Alert], None]) -> None:
        """
        Add a notification handler.
        
        Args:
            handler: Function called when an alert fires or resolves
        """
        self._handlers.append(handler)
    
    def remove_handler(self, handler: Callable[[Alert], None]) -> None:
        """Remove a notification handler."""
        if handler in self._handlers:
            self._handlers.remove(handler)
    
    def _notify(self, alert: Alert) -> None:
        """Notify all handlers of an alert."""
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    def _get_metric_value(self, rule: AlertRule) -> Optional[float]:
        """Get the current value of a metric for a rule."""
        # Try to get the metric value
        full_name = f"{self._registry.namespace}_{rule.metric_name}"
        
        # Check in gauges first (most common for alerting)
        if full_name in self._registry._gauges:
            values = self._registry._gauges[full_name]
            if values:
                # Use labels if specified, otherwise take first value
                if rule.labels:
                    key = tuple(rule.labels.get(l, "") for l in 
                               self._registry._definitions.get(full_name, 
                                   type('', (), {'labels': []})()).labels)
                    return values.get(key)
                else:
                    # Return average of all label combinations
                    return sum(values.values()) / len(values) if values else None
        
        # Check counters
        if full_name in self._registry._counters:
            values = self._registry._counters[full_name]
            if values:
                if rule.labels:
                    key = tuple(rule.labels.get(l, "") for l in 
                               self._registry._definitions.get(full_name,
                                   type('', (), {'labels': []})()).labels)
                    return values.get(key)
                else:
                    return sum(values.values())
        
        return None
    
    def evaluate_rule(self, rule: AlertRule) -> None:
        """
        Evaluate a single rule.
        
        Args:
            rule: The rule to evaluate
        """
        if not rule.enabled:
            return
        
        value = self._get_metric_value(rule)
        if value is None:
            return
        
        condition_met = rule.evaluate(value)
        fingerprint = f"{rule.name}:{json.dumps(rule.labels, sort_keys=True)}"
        now = time.time()
        
        with self._lock:
            if condition_met:
                # Condition is met
                if fingerprint in self._active_alerts:
                    # Already firing - update value
                    self._active_alerts[fingerprint].value = value
                elif fingerprint in self._pending_alerts:
                    # Check if duration has been met
                    pending = self._pending_alerts[fingerprint]
                    pending.value = value
                    
                    if now - pending.started_at >= rule.for_duration_seconds:
                        # Promote to firing
                        pending.state = AlertState.FIRING
                        pending.fired_at = now
                        self._active_alerts[fingerprint] = pending
                        del self._pending_alerts[fingerprint]
                        
                        logger.warning(f"Alert firing: {rule.name} (value={value})")
                        self._notify(pending)
                else:
                    # New alert - start pending
                    alert = Alert(
                        rule=rule,
                        state=AlertState.PENDING if rule.for_duration_seconds > 0 else AlertState.FIRING,
                        value=value,
                        started_at=now,
                        labels=rule.labels.copy(),
                        annotations=rule.annotations.copy()
                    )
                    
                    if rule.for_duration_seconds > 0:
                        self._pending_alerts[fingerprint] = alert
                    else:
                        alert.fired_at = now
                        self._active_alerts[fingerprint] = alert
                        logger.warning(f"Alert firing: {rule.name} (value={value})")
                        self._notify(alert)
            else:
                # Condition not met
                if fingerprint in self._pending_alerts:
                    # Clear pending
                    del self._pending_alerts[fingerprint]
                elif fingerprint in self._active_alerts:
                    # Resolve active alert
                    alert = self._active_alerts[fingerprint]
                    alert.state = AlertState.RESOLVED
                    alert.resolved_at = now
                    
                    del self._active_alerts[fingerprint]
                    self._resolved_alerts.append(alert)
                    
                    # Trim resolved list
                    if len(self._resolved_alerts) > self._max_resolved_alerts:
                        self._resolved_alerts = self._resolved_alerts[-self._max_resolved_alerts:]
                    
                    logger.info(f"Alert resolved: {rule.name}")
                    self._notify(alert)
    
    def evaluate_all(self) -> None:
        """Evaluate all rules."""
        for rule in list(self._rules.values()):
            try:
                self.evaluate_rule(rule)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (firing) alerts."""
        with self._lock:
            return list(self._active_alerts.values())
    
    def get_pending_alerts(self) -> List[Alert]:
        """Get all pending alerts."""
        with self._lock:
            return list(self._pending_alerts.values())
    
    def get_resolved_alerts(self) -> List[Alert]:
        """Get recently resolved alerts."""
        with self._lock:
            return list(self._resolved_alerts)
    
    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get active alerts filtered by severity."""
        with self._lock:
            return [a for a in self._active_alerts.values() if a.rule.severity == severity]
    
    def start(self, interval_seconds: float = 15.0) -> None:
        """
        Start background alert evaluation.
        
        Args:
            interval_seconds: Evaluation interval
        """
        if self._eval_thread is not None and self._eval_thread.is_alive():
            return
        
        self._stop_event.clear()
        
        def eval_loop():
            while not self._stop_event.is_set():
                self.evaluate_all()
                self._stop_event.wait(interval_seconds)
        
        self._eval_thread = threading.Thread(
            target=eval_loop,
            daemon=True,
            name="alert-evaluator"
        )
        self._eval_thread.start()
        logger.info(f"Started alert evaluation with {interval_seconds}s interval")
    
    def stop(self) -> None:
        """Stop background alert evaluation."""
        self._stop_event.set()
        if self._eval_thread is not None:
            self._eval_thread.join(timeout=5.0)
            self._eval_thread = None
        logger.info("Stopped alert evaluation")
    
    def get_status(self) -> Dict[str, Any]:
        """Get alert manager status."""
        with self._lock:
            return {
                "rules_count": len(self._rules),
                "active_alerts": len(self._active_alerts),
                "pending_alerts": len(self._pending_alerts),
                "resolved_recent": len(self._resolved_alerts),
                "handlers_count": len(self._handlers),
                "is_running": self._eval_thread is not None and self._eval_thread.is_alive(),
                "alerts_by_severity": {
                    severity.value: len(self.get_alerts_by_severity(severity))
                    for severity in AlertSeverity
                }
            }


# Pre-defined alert rules for LLM inference
def get_default_inference_rules() -> List[AlertRule]:
    """Get default alert rules for LLM inference systems."""
    return [
        AlertRule(
            name="high_latency_p99",
            description="P99 inference latency exceeds 5 seconds",
            metric_name="generation_duration_seconds",
            condition="gt",
            threshold=5.0,
            severity=AlertSeverity.WARNING,
            for_duration_seconds=60.0,
            annotations={"runbook": "Check GPU utilization and batch size"}
        ),
        AlertRule(
            name="very_high_latency",
            description="Inference latency exceeds 10 seconds",
            metric_name="generation_duration_seconds",
            condition="gt",
            threshold=10.0,
            severity=AlertSeverity.CRITICAL,
            for_duration_seconds=30.0,
            annotations={"runbook": "Scale up resources or reduce load"}
        ),
        AlertRule(
            name="low_throughput",
            description="Token generation rate below threshold",
            metric_name="tokens_per_second",
            condition="lt",
            threshold=10.0,
            severity=AlertSeverity.WARNING,
            for_duration_seconds=120.0,
            annotations={"runbook": "Check for resource constraints"}
        ),
        AlertRule(
            name="high_gpu_memory",
            description="GPU memory utilization above 90%",
            metric_name="gpu_memory_utilization_ratio",
            condition="gt",
            threshold=0.9,
            severity=AlertSeverity.WARNING,
            for_duration_seconds=60.0,
            annotations={"runbook": "Consider reducing batch size or sequence length"}
        ),
        AlertRule(
            name="critical_gpu_memory",
            description="GPU memory utilization above 95%",
            metric_name="gpu_memory_utilization_ratio",
            condition="gt",
            threshold=0.95,
            severity=AlertSeverity.CRITICAL,
            for_duration_seconds=30.0,
            annotations={"runbook": "Immediate action required - OOM risk"}
        ),
        AlertRule(
            name="high_error_rate",
            description="Request error rate above 1%",
            metric_name="request_errors_total",
            condition="gt",
            threshold=10.0,  # Absolute count
            severity=AlertSeverity.WARNING,
            for_duration_seconds=120.0,
            annotations={"runbook": "Check error logs for root cause"}
        ),
        AlertRule(
            name="low_cache_hit_rate",
            description="Cache hit rate below 50%",
            metric_name="cache_hit_ratio",
            condition="lt",
            threshold=0.5,
            severity=AlertSeverity.INFO,
            for_duration_seconds=300.0,
            annotations={"runbook": "Consider increasing cache size"}
        ),
        AlertRule(
            name="cluster_degraded",
            description="Less than 80% of nodes healthy",
            metric_name="cluster_nodes_total",
            condition="lt",
            threshold=0.8,  # Would need custom logic for ratio
            severity=AlertSeverity.WARNING,
            labels={"status": "healthy"},
            for_duration_seconds=60.0,
            annotations={"runbook": "Check unhealthy nodes"}
        ),
        AlertRule(
            name="high_queue_depth",
            description="Request queue depth above threshold",
            metric_name="request_queue_depth",
            condition="gt",
            threshold=100.0,
            severity=AlertSeverity.WARNING,
            for_duration_seconds=60.0,
            annotations={"runbook": "Scale up or throttle incoming requests"}
        ),
        AlertRule(
            name="gpu_temperature_high",
            description="GPU temperature above 80°C",
            metric_name="gpu_temperature_celsius",
            condition="gt",
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            for_duration_seconds=120.0,
            annotations={"runbook": "Check cooling system"}
        )
    ]
