"""
Alert management system for pipeline monitoring and notifications.

Monitors metrics and triggers alerts based on configurable thresholds,
with support for multiple notification channels.
"""

import logging
import smtplib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
try:
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
except ImportError:
    # Fallback for environments where email modules are not available
    MimeText = None
    MimeMultipart = None
import threading
import time

from .metrics import MetricsCollector

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status states."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"


@dataclass
class AlertRule:
    """Configuration for an alert rule."""
    name: str
    description: str
    metric_name: str
    condition: str  # 'gt', 'lt', 'eq', 'gte', 'lte'
    threshold: Union[int, float]
    severity: AlertSeverity
    window_minutes: int = 5
    min_occurrences: int = 1
    enabled: bool = True
    notification_channels: List[str] = field(default_factory=list)
    cooldown_minutes: int = 30


@dataclass
class Alert:
    """Individual alert instance."""
    id: str
    rule_name: str
    message: str
    severity: AlertSeverity
    status: AlertStatus
    created_at: datetime
    updated_at: datetime
    metric_value: Union[int, float]
    threshold: Union[int, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None


@dataclass
class NotificationChannel:
    """Configuration for notification channels."""
    name: str
    type: str  # 'email', 'webhook', 'log', 'console'
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


class AlertManager:
    """
    Manages alert rules, monitoring, and notifications for pipeline metrics.
    
    Monitors metrics against configurable thresholds and sends notifications
    through various channels when alerts are triggered.
    """
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        """
        Initialize alert manager.
        
        Args:
            metrics_collector: Metrics collector to monitor
        """
        self.metrics_collector = metrics_collector
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_channels: Dict[str, NotificationChannel] = {}
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._alert_counter = 0
        self._lock = threading.Lock()
        
        # Default alert rules
        self._setup_default_rules()
        self._setup_default_channels()
    
    def _setup_default_rules(self):
        """Setup default alert rules for common scenarios."""
        default_rules = [
            AlertRule(
                name="high_error_rate",
                description="Error rate exceeds 10%",
                metric_name="error_rate_percent",
                condition="gt",
                threshold=10.0,
                severity=AlertSeverity.HIGH,
                window_minutes=5,
                notification_channels=["console", "log"]
            ),
            AlertRule(
                name="low_quality_score",
                description="Average quality score below 0.5",
                metric_name="avg_quality_score",
                condition="lt",
                threshold=0.5,
                severity=AlertSeverity.MEDIUM,
                window_minutes=10,
                notification_channels=["console", "log"]
            ),
            AlertRule(
                name="high_memory_usage",
                description="Memory usage exceeds 800MB",
                metric_name="memory_usage_mb",
                condition="gt",
                threshold=800.0,
                severity=AlertSeverity.MEDIUM,
                window_minutes=5,
                notification_channels=["console", "log"]
            ),
            AlertRule(
                name="processing_stalled",
                description="No documents processed in 30 minutes",
                metric_name="throughput_docs_per_sec",
                condition="eq",
                threshold=0.0,
                severity=AlertSeverity.HIGH,
                window_minutes=30,
                notification_channels=["console", "log"]
            ),
            AlertRule(
                name="queue_overflow",
                description="Processing queue size exceeds 1000",
                metric_name="queue_size",
                condition="gt",
                threshold=1000,
                severity=AlertSeverity.CRITICAL,
                window_minutes=1,
                notification_channels=["console", "log"]
            )
        ]
        
        for rule in default_rules:
            self.add_alert_rule(rule)
    
    def _setup_default_channels(self):
        """Setup default notification channels."""
        default_channels = [
            NotificationChannel(
                name="console",
                type="console",
                config={},
                enabled=True
            ),
            NotificationChannel(
                name="log",
                type="log",
                config={"logger_name": "qudata.alerts"},
                enabled=True
            )
        ]
        
        for channel in default_channels:
            self.add_notification_channel(channel)
    
    def add_alert_rule(self, rule: AlertRule):
        """
        Add or update an alert rule.
        
        Args:
            rule: Alert rule configuration
        """
        with self._lock:
            self.alert_rules[rule.name] = rule
            logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """
        Remove an alert rule.
        
        Args:
            rule_name: Name of rule to remove
        """
        with self._lock:
            if rule_name in self.alert_rules:
                del self.alert_rules[rule_name]
                logger.info(f"Removed alert rule: {rule_name}")
    
    def add_notification_channel(self, channel: NotificationChannel):
        """
        Add or update a notification channel.
        
        Args:
            channel: Notification channel configuration
        """
        with self._lock:
            self.notification_channels[channel.name] = channel
            logger.info(f"Added notification channel: {channel.name}")
    
    def start_monitoring(self, check_interval_seconds: int = 60):
        """
        Start background monitoring thread.
        
        Args:
            check_interval_seconds: How often to check metrics
        """
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            logger.warning("Monitoring already started")
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(check_interval_seconds,),
            daemon=True
        )
        self._monitoring_thread.start()
        logger.info("Alert monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring thread."""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("Alert monitoring stopped")
    
    def _monitoring_loop(self, check_interval: int):
        """Main monitoring loop that runs in background thread."""
        while not self._stop_monitoring.is_set():
            try:
                self._check_all_rules()
                time.sleep(check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(check_interval)
    
    def _check_all_rules(self):
        """Check all alert rules against current metrics."""
        if not self.metrics_collector:
            return
        
        current_metrics = self.metrics_collector.get_current_metrics()
        
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            try:
                self._check_rule(rule, current_metrics)
            except Exception as e:
                logger.error(f"Error checking rule {rule_name}: {e}")
    
    def _check_rule(self, rule: AlertRule, current_metrics: Dict[str, Any]):
        """
        Check a specific rule against current metrics.
        
        Args:
            rule: Alert rule to check
            current_metrics: Current metric values
        """
        # Extract metric value from nested structure
        metric_value = self._extract_metric_value(rule.metric_name, current_metrics)
        
        if metric_value is None:
            return
        
        # Check condition
        condition_met = self._evaluate_condition(
            metric_value, rule.condition, rule.threshold
        )
        
        alert_id = f"{rule.name}_{rule.metric_name}"
        
        if condition_met:
            # Check if alert already exists and is in cooldown
            if alert_id in self.active_alerts:
                existing_alert = self.active_alerts[alert_id]
                cooldown_end = existing_alert.created_at + timedelta(minutes=rule.cooldown_minutes)
                if datetime.now() < cooldown_end:
                    return  # Still in cooldown
            
            # Create new alert
            alert = Alert(
                id=alert_id,
                rule_name=rule.name,
                message=f"{rule.description} (value: {metric_value}, threshold: {rule.threshold})",
                severity=rule.severity,
                status=AlertStatus.ACTIVE,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                metric_value=metric_value,
                threshold=rule.threshold,
                metadata={"rule": rule.__dict__}
            )
            
            self._trigger_alert(alert, rule)
            
        else:
            # Check if we should resolve existing alert
            if alert_id in self.active_alerts:
                self._resolve_alert(alert_id)
    
    def _extract_metric_value(self, metric_name: str, metrics: Dict[str, Any]) -> Optional[Union[int, float]]:
        """
        Extract metric value from nested metrics structure.
        
        Args:
            metric_name: Name of metric to extract
            metrics: Nested metrics dictionary
            
        Returns:
            Metric value or None if not found
        """
        # Check direct metrics first
        if metric_name in metrics:
            return metrics[metric_name]
        
        # Check nested structures
        for category in ['processing', 'quality', 'errors']:
            if category in metrics:
                category_data = metrics[category]
                if hasattr(category_data, metric_name):
                    return getattr(category_data, metric_name)
                elif isinstance(category_data, dict) and metric_name in category_data:
                    return category_data[metric_name]
        
        return None
    
    def _evaluate_condition(self, value: Union[int, float], condition: str, 
                          threshold: Union[int, float]) -> bool:
        """
        Evaluate alert condition.
        
        Args:
            value: Current metric value
            condition: Condition type ('gt', 'lt', 'eq', 'gte', 'lte')
            threshold: Threshold value
            
        Returns:
            True if condition is met
        """
        if condition == 'gt':
            return value > threshold
        elif condition == 'lt':
            return value < threshold
        elif condition == 'eq':
            return value == threshold
        elif condition == 'gte':
            return value >= threshold
        elif condition == 'lte':
            return value <= threshold
        else:
            logger.warning(f"Unknown condition: {condition}")
            return False
    
    def _trigger_alert(self, alert: Alert, rule: AlertRule):
        """
        Trigger an alert and send notifications.
        
        Args:
            alert: Alert to trigger
            rule: Associated alert rule
        """
        with self._lock:
            self.active_alerts[alert.id] = alert
            self.alert_history.append(alert)
            self._alert_counter += 1
        
        logger.warning(f"Alert triggered: {alert.message}")
        
        # Send notifications
        for channel_name in rule.notification_channels:
            if channel_name in self.notification_channels:
                channel = self.notification_channels[channel_name]
                if channel.enabled:
                    self._send_notification(alert, channel)
    
    def _resolve_alert(self, alert_id: str):
        """
        Resolve an active alert.
        
        Args:
            alert_id: ID of alert to resolve
        """
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()
                alert.updated_at = datetime.now()
                
                del self.active_alerts[alert_id]
                logger.info(f"Alert resolved: {alert.message}")
    
    def _send_notification(self, alert: Alert, channel: NotificationChannel):
        """
        Send notification through specified channel.
        
        Args:
            alert: Alert to send
            channel: Notification channel
        """
        try:
            if channel.type == 'console':
                self._send_console_notification(alert)
            elif channel.type == 'log':
                self._send_log_notification(alert, channel.config)
            elif channel.type == 'email':
                self._send_email_notification(alert, channel.config)
            elif channel.type == 'webhook':
                self._send_webhook_notification(alert, channel.config)
            else:
                logger.warning(f"Unknown notification channel type: {channel.type}")
                
        except Exception as e:
            logger.error(f"Failed to send notification via {channel.name}: {e}")
    
    def _send_console_notification(self, alert: Alert):
        """Send notification to console."""
        severity_colors = {
            AlertSeverity.LOW: '\033[94m',      # Blue
            AlertSeverity.MEDIUM: '\033[93m',   # Yellow
            AlertSeverity.HIGH: '\033[91m',     # Red
            AlertSeverity.CRITICAL: '\033[95m'  # Magenta
        }
        reset_color = '\033[0m'
        
        color = severity_colors.get(alert.severity, '')
        print(f"{color}[ALERT {alert.severity.value.upper()}] {alert.message}{reset_color}")
    
    def _send_log_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send notification to log."""
        logger_name = config.get('logger_name', 'qudata.alerts')
        alert_logger = logging.getLogger(logger_name)
        
        if alert.severity == AlertSeverity.CRITICAL:
            alert_logger.critical(f"ALERT: {alert.message}")
        elif alert.severity == AlertSeverity.HIGH:
            alert_logger.error(f"ALERT: {alert.message}")
        elif alert.severity == AlertSeverity.MEDIUM:
            alert_logger.warning(f"ALERT: {alert.message}")
        else:
            alert_logger.info(f"ALERT: {alert.message}")
    
    def _send_email_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send email notification."""
        if MimeText is None or MimeMultipart is None:
            logger.error("Email modules not available")
            return
            
        smtp_server = config.get('smtp_server')
        smtp_port = config.get('smtp_port', 587)
        username = config.get('username')
        password = config.get('password')
        to_emails = config.get('to_emails', [])
        
        if not all([smtp_server, username, password, to_emails]):
            logger.error("Incomplete email configuration")
            return
        
        msg = MimeMultipart()
        msg['From'] = username
        msg['To'] = ', '.join(to_emails)
        msg['Subject'] = f"QuData Alert: {alert.rule_name}"
        
        body = f"""
        Alert: {alert.message}
        Severity: {alert.severity.value}
        Time: {alert.created_at}
        Metric Value: {alert.metric_value}
        Threshold: {alert.threshold}
        """
        
        msg.attach(MimeText(body, 'plain'))
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
    
    def _send_webhook_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send webhook notification."""
        import requests
        
        url = config.get('url')
        headers = config.get('headers', {})
        
        if not url:
            logger.error("No webhook URL configured")
            return
        
        payload = {
            'alert_id': alert.id,
            'rule_name': alert.rule_name,
            'message': alert.message,
            'severity': alert.severity.value,
            'status': alert.status.value,
            'created_at': alert.created_at.isoformat(),
            'metric_value': alert.metric_value,
            'threshold': alert.threshold
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """
        Acknowledge an active alert.
        
        Args:
            alert_id: ID of alert to acknowledge
            acknowledged_by: User who acknowledged the alert
        """
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = acknowledged_by
                alert.updated_at = datetime.now()
                logger.info(f"Alert acknowledged by {acknowledged_by}: {alert.message}")
    
    def suppress_alert(self, alert_id: str, duration_minutes: int = 60):
        """
        Suppress an alert for a specified duration.
        
        Args:
            alert_id: ID of alert to suppress
            duration_minutes: How long to suppress the alert
        """
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.SUPPRESSED
                alert.updated_at = datetime.now()
                alert.metadata['suppressed_until'] = (
                    datetime.now() + timedelta(minutes=duration_minutes)
                ).isoformat()
                logger.info(f"Alert suppressed for {duration_minutes} minutes: {alert.message}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get list of currently active alerts."""
        with self._lock:
            return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """
        Get alert history for specified time period.
        
        Args:
            hours: Number of hours of history to return
            
        Returns:
            List of alerts from the specified time period
        """
        since = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.created_at >= since]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert statistics."""
        with self._lock:
            active_count = len(self.active_alerts)
            total_count = len(self.alert_history)
            
            severity_counts = {}
            for alert in self.alert_history:
                severity = alert.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            return {
                'active_alerts': active_count,
                'total_alerts': total_count,
                'severity_distribution': severity_counts,
                'alert_rules_count': len(self.alert_rules),
                'notification_channels_count': len(self.notification_channels)
            }
    
    def export_alerts(self, format: str = 'json') -> str:
        """
        Export alert data in specified format.
        
        Args:
            format: Export format ('json', 'csv')
            
        Returns:
            Formatted alert data
        """
        if format == 'json':
            data = {
                'active_alerts': [alert.__dict__ for alert in self.active_alerts.values()],
                'alert_history': [alert.__dict__ for alert in self.alert_history],
                'summary': self.get_alert_summary()
            }
            return json.dumps(data, default=str, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")