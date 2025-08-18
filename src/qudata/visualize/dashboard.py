"""
Dashboard server using Streamlit for interactive data visualization.

Provides a web-based dashboard for monitoring pipeline metrics,
quality scores, and system performance in real-time.
"""

import streamlit as st
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import logging
import threading
import json

from .metrics import MetricsCollector, get_metrics_collector
from .charts import ChartGenerator
from .alerts import AlertManager, AlertSeverity
from ..analyze.analysis_engine import AnalysisEngine

logger = logging.getLogger(__name__)


class DashboardServer:
    """
    Streamlit-based dashboard server for pipeline monitoring and visualization.
    
    Provides interactive web interface for viewing metrics, charts, alerts,
    and analysis results in real-time.
    """
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None,
                 alert_manager: Optional[AlertManager] = None,
                 analysis_engine: Optional[AnalysisEngine] = None):
        """
        Initialize dashboard server.
        
        Args:
            metrics_collector: Metrics collector for data source
            alert_manager: Alert manager for notifications
            analysis_engine: Analysis engine for data insights
        """
        self.metrics_collector = metrics_collector or get_metrics_collector()
        self.alert_manager = alert_manager or AlertManager(self.metrics_collector)
        self.analysis_engine = analysis_engine
        self.chart_generator = ChartGenerator(self.metrics_collector)
        
        # Dashboard configuration
        self.refresh_interval = 30  # seconds
        self.auto_refresh = True
        self.theme = "light"
        
        # Page configuration
        self.pages = {
            "Overview": self._render_overview_page,
            "Processing Metrics": self._render_processing_page,
            "Quality Analysis": self._render_quality_page,
            "System Monitoring": self._render_system_page,
            "Alerts": self._render_alerts_page,
            "Data Analysis": self._render_analysis_page,
            "Settings": self._render_settings_page
        }
    
    def run(self, host: str = "localhost", port: int = 8501, debug: bool = False):
        """
        Run the dashboard server.
        
        Args:
            host: Host address to bind to
            port: Port number to use
            debug: Enable debug mode
        """
        # Configure Streamlit
        st.set_page_config(
            page_title="QuData Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Start alert monitoring if not already running
        if not self.alert_manager._monitoring_thread or not self.alert_manager._monitoring_thread.is_alive():
            self.alert_manager.start_monitoring()
        
        # Render main dashboard
        self._render_dashboard()
    
    def _render_dashboard(self):
        """Render the main dashboard interface."""
        # Sidebar navigation
        st.sidebar.title("QuData Dashboard")
        
        # Page selection
        selected_page = st.sidebar.selectbox(
            "Navigate to:",
            list(self.pages.keys()),
            index=0
        )
        
        # Auto-refresh toggle
        self.auto_refresh = st.sidebar.checkbox("Auto-refresh", value=self.auto_refresh)
        
        if self.auto_refresh:
            self.refresh_interval = st.sidebar.slider(
                "Refresh interval (seconds)",
                min_value=5,
                max_value=300,
                value=self.refresh_interval
            )
        
        # Alert summary in sidebar
        self._render_sidebar_alerts()
        
        # Main content area
        if selected_page in self.pages:
            self.pages[selected_page]()
        
        # Auto-refresh logic
        if self.auto_refresh:
            time.sleep(self.refresh_interval)
            st.rerun()
    
    def _render_sidebar_alerts(self):
        """Render alert summary in sidebar."""
        st.sidebar.markdown("---")
        st.sidebar.subheader("🚨 Alerts")
        
        active_alerts = self.alert_manager.get_active_alerts()
        
        if not active_alerts:
            st.sidebar.success("No active alerts")
        else:
            # Group by severity
            severity_counts = {}
            for alert in active_alerts:
                severity = alert.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            for severity, count in severity_counts.items():
                if severity == "critical":
                    st.sidebar.error(f"Critical: {count}")
                elif severity == "high":
                    st.sidebar.warning(f"High: {count}")
                elif severity == "medium":
                    st.sidebar.info(f"Medium: {count}")
                else:
                    st.sidebar.info(f"Low: {count}")
    
    def _render_overview_page(self):
        """Render overview dashboard page."""
        st.title("📊 QuData Pipeline Overview")
        
        # Get current metrics
        current_metrics = self.metrics_collector.get_current_metrics()
        processing = current_metrics.get('processing', {})
        quality = current_metrics.get('quality', {})
        errors = current_metrics.get('errors', {})
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            docs_processed = getattr(processing, 'documents_processed', 0)
            st.metric(
                label="Documents Processed",
                value=docs_processed,
                delta=None
            )
        
        with col2:
            throughput = getattr(processing, 'throughput_docs_per_sec', 0)
            st.metric(
                label="Throughput (docs/sec)",
                value=f"{throughput:.2f}",
                delta=None
            )
        
        with col3:
            avg_quality = getattr(quality, 'avg_quality_score', 0)
            st.metric(
                label="Avg Quality Score",
                value=f"{avg_quality:.3f}",
                delta=None
            )
        
        with col4:
            error_rate = getattr(errors, 'error_rate_percent', 0)
            st.metric(
                label="Error Rate (%)",
                value=f"{error_rate:.1f}",
                delta=None
            )
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Processing Timeline")
            timeline_chart = self.chart_generator.create_processing_timeline(hours=24)
            st.plotly_chart(timeline_chart, use_container_width=True)
        
        with col2:
            st.subheader("System Metrics")
            system_chart = self.chart_generator.create_system_metrics_dashboard()
            st.plotly_chart(system_chart, use_container_width=True)
        
        # Recent alerts
        st.subheader("Recent Alerts")
        recent_alerts = self.alert_manager.get_alert_history(hours=24)
        
        if recent_alerts:
            alert_data = []
            for alert in recent_alerts[-10:]:  # Show last 10 alerts
                alert_data.append({
                    'Time': alert.created_at.strftime('%H:%M:%S'),
                    'Rule': alert.rule_name,
                    'Severity': alert.severity.value,
                    'Message': alert.message,
                    'Status': alert.status.value
                })
            
            df = pd.DataFrame(alert_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No recent alerts")
    
    def _render_processing_page(self):
        """Render processing metrics page."""
        st.title("⚙️ Processing Metrics")
        
        # Processing statistics
        current_metrics = self.metrics_collector.get_current_metrics()
        processing = current_metrics.get('processing', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Processing Stats")
            st.write(f"**Documents Processed:** {getattr(processing, 'documents_processed', 0)}")
            st.write(f"**Documents Failed:** {getattr(processing, 'documents_failed', 0)}")
            st.write(f"**Average Processing Time:** {getattr(processing, 'processing_time_avg', 0):.3f}s")
            st.write(f"**Total Processing Time:** {getattr(processing, 'processing_time_total', 0):.1f}s")
        
        with col2:
            st.subheader("System Resources")
            st.write(f"**Memory Usage:** {getattr(processing, 'memory_usage_mb', 0):.1f} MB")
            st.write(f"**CPU Usage:** {getattr(processing, 'cpu_usage_percent', 0):.1f}%")
            st.write(f"**Queue Size:** {getattr(processing, 'queue_size', 0)}")
        
        with col3:
            st.subheader("Throughput")
            throughput_chart = self.chart_generator.create_throughput_gauge()
            st.plotly_chart(throughput_chart, use_container_width=True)
        
        # Processing timeline
        st.subheader("Processing Timeline (24 hours)")
        timeline_chart = self.chart_generator.create_processing_timeline(hours=24)
        st.plotly_chart(timeline_chart, use_container_width=True)
        
        # Error analysis
        st.subheader("Error Analysis")
        error_chart = self.chart_generator.create_error_heatmap(days=7)
        st.plotly_chart(error_chart, use_container_width=True)
    
    def _render_quality_page(self):
        """Render quality analysis page."""
        st.title("🎯 Quality Analysis")
        
        current_metrics = self.metrics_collector.get_current_metrics()
        quality = current_metrics.get('quality', {})
        
        # Quality statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Quality Statistics")
            st.write(f"**Average Quality Score:** {getattr(quality, 'avg_quality_score', 0):.3f}")
            st.write(f"**Minimum Quality Score:** {getattr(quality, 'min_quality_score', 0):.3f}")
            st.write(f"**Maximum Quality Score:** {getattr(quality, 'max_quality_score', 0):.3f}")
            st.write(f"**Documents Below Threshold:** {getattr(quality, 'documents_below_threshold', 0)}")
            st.write(f"**Duplicate Documents:** {getattr(quality, 'duplicate_documents', 0)}")
            st.write(f"**Average Content Length:** {getattr(quality, 'content_length_avg', 0):.0f} chars")
        
        with col2:
            st.subheader("Quality Score Distribution")
            quality_chart = self.chart_generator.create_quality_distribution()
            st.plotly_chart(quality_chart, use_container_width=True)
        
        # Language distribution
        st.subheader("Language Distribution")
        language_chart = self.chart_generator.create_language_pie_chart()
        st.plotly_chart(language_chart, use_container_width=True)
        
        # Quality trends (if analysis engine is available)
        if self.analysis_engine:
            st.subheader("Quality Trends")
            st.info("Quality trend analysis requires processed dataset")
    
    def _render_system_page(self):
        """Render system monitoring page."""
        st.title("🖥️ System Monitoring")
        
        # System metrics dashboard
        system_chart = self.chart_generator.create_system_metrics_dashboard()
        st.plotly_chart(system_chart, use_container_width=True)
        
        # Resource usage over time
        st.subheader("Resource Usage Timeline")
        
        # Get historical metrics for system resources
        memory_metrics = self.metrics_collector.get_metrics_history('memory_usage_mb')
        cpu_metrics = self.metrics_collector.get_metrics_history('cpu_usage_percent')
        
        if memory_metrics or cpu_metrics:
            # Create timeline chart for resources
            col1, col2 = st.columns(2)
            
            with col1:
                if memory_metrics:
                    memory_data = pd.DataFrame([
                        {'timestamp': m.timestamp, 'value': m.value}
                        for m in memory_metrics[-100:]  # Last 100 points
                    ])
                    st.line_chart(memory_data.set_index('timestamp')['value'])
                    st.caption("Memory Usage (MB)")
            
            with col2:
                if cpu_metrics:
                    cpu_data = pd.DataFrame([
                        {'timestamp': m.timestamp, 'value': m.value}
                        for m in cpu_metrics[-100:]  # Last 100 points
                    ])
                    st.line_chart(cpu_data.set_index('timestamp')['value'])
                    st.caption("CPU Usage (%)")
        else:
            st.info("No historical system metrics available")
        
        # System information
        st.subheader("System Information")
        uptime = current_metrics.get('uptime_seconds', 0)
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        st.write(f"**Uptime:** {hours}h {minutes}m")
        st.write(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _render_alerts_page(self):
        """Render alerts management page."""
        st.title("🚨 Alerts Management")
        
        # Alert summary
        alert_summary = self.alert_manager.get_alert_summary()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Active Alerts", alert_summary['active_alerts'])
        
        with col2:
            st.metric("Total Alerts", alert_summary['total_alerts'])
        
        with col3:
            st.metric("Alert Rules", alert_summary['alert_rules_count'])
        
        # Active alerts
        st.subheader("Active Alerts")
        active_alerts = self.alert_manager.get_active_alerts()
        
        if active_alerts:
            alert_data = []
            for alert in active_alerts:
                alert_data.append({
                    'ID': alert.id,
                    'Rule': alert.rule_name,
                    'Severity': alert.severity.value,
                    'Message': alert.message,
                    'Created': alert.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                    'Status': alert.status.value,
                    'Value': alert.metric_value,
                    'Threshold': alert.threshold
                })
            
            df = pd.DataFrame(alert_data)
            st.dataframe(df, use_container_width=True)
            
            # Alert actions
            st.subheader("Alert Actions")
            selected_alert = st.selectbox("Select Alert", [a['ID'] for a in alert_data])
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Acknowledge Alert"):
                    self.alert_manager.acknowledge_alert(selected_alert, "dashboard_user")
                    st.success(f"Alert {selected_alert} acknowledged")
                    st.rerun()
            
            with col2:
                suppress_duration = st.number_input("Suppress Duration (minutes)", min_value=1, value=60)
                if st.button("Suppress Alert"):
                    self.alert_manager.suppress_alert(selected_alert, suppress_duration)
                    st.success(f"Alert {selected_alert} suppressed for {suppress_duration} minutes")
                    st.rerun()
        else:
            st.success("No active alerts")
        
        # Alert history
        st.subheader("Alert History (24 hours)")
        alert_history = self.alert_manager.get_alert_history(hours=24)
        
        if alert_history:
            history_data = []
            for alert in alert_history[-20:]:  # Show last 20 alerts
                history_data.append({
                    'Time': alert.created_at.strftime('%H:%M:%S'),
                    'Rule': alert.rule_name,
                    'Severity': alert.severity.value,
                    'Message': alert.message,
                    'Status': alert.status.value,
                    'Resolved': alert.resolved_at.strftime('%H:%M:%S') if alert.resolved_at else 'N/A'
                })
            
            df = pd.DataFrame(history_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No alert history available")
    
    def _render_analysis_page(self):
        """Render data analysis page."""
        st.title("📈 Data Analysis")
        
        if not self.analysis_engine:
            st.warning("Analysis engine not available. Please configure analysis engine to view insights.")
            return
        
        st.subheader("Dataset Analysis")
        st.info("Analysis features require a processed dataset. Upload or process data to see analysis results.")
        
        # Placeholder for analysis results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Topic Analysis")
            st.info("Topic modeling results will appear here after processing documents")
        
        with col2:
            st.subheader("Sentiment Analysis")
            st.info("Sentiment analysis results will appear here after processing documents")
        
        # Token length analysis
        st.subheader("Token Length Analysis")
        st.info("Token length distribution will appear here after processing documents")
    
    def _render_settings_page(self):
        """Render settings and configuration page."""
        st.title("⚙️ Settings")
        
        # Dashboard settings
        st.subheader("Dashboard Settings")
        
        new_refresh_interval = st.slider(
            "Auto-refresh interval (seconds)",
            min_value=5,
            max_value=300,
            value=self.refresh_interval
        )
        
        if new_refresh_interval != self.refresh_interval:
            self.refresh_interval = new_refresh_interval
            st.success("Refresh interval updated")
        
        # Theme selection
        new_theme = st.selectbox("Theme", ["light", "dark"], index=0 if self.theme == "light" else 1)
        if new_theme != self.theme:
            self.theme = new_theme
            st.success("Theme updated (restart required)")
        
        # Metrics settings
        st.subheader("Metrics Settings")
        
        if st.button("Reset Metrics"):
            self.metrics_collector.reset_metrics()
            st.success("Metrics reset successfully")
            st.rerun()
        
        if st.button("Export Metrics"):
            metrics_json = self.metrics_collector.export_metrics('json')
            st.download_button(
                label="Download Metrics JSON",
                data=metrics_json,
                file_name=f"qudata_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Alert settings
        st.subheader("Alert Settings")
        
        alert_summary = self.alert_manager.get_alert_summary()
        st.write(f"**Active Alert Rules:** {alert_summary['alert_rules_count']}")
        st.write(f"**Notification Channels:** {alert_summary['notification_channels_count']}")
        
        if st.button("Export Alerts"):
            alerts_json = self.alert_manager.export_alerts('json')
            st.download_button(
                label="Download Alerts JSON",
                data=alerts_json,
                file_name=f"qudata_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # System information
        st.subheader("System Information")
        current_metrics = self.metrics_collector.get_current_metrics()
        
        st.write(f"**Dashboard Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"**Metrics Collector:** {'Active' if self.metrics_collector else 'Inactive'}")
        st.write(f"**Alert Manager:** {'Active' if self.alert_manager else 'Inactive'}")
        st.write(f"**Analysis Engine:** {'Active' if self.analysis_engine else 'Inactive'}")


def create_dashboard_app(metrics_collector: Optional[MetricsCollector] = None,
                        alert_manager: Optional[AlertManager] = None,
                        analysis_engine: Optional[AnalysisEngine] = None) -> DashboardServer:
    """
    Create and configure dashboard application.
    
    Args:
        metrics_collector: Optional metrics collector instance
        alert_manager: Optional alert manager instance
        analysis_engine: Optional analysis engine instance
        
    Returns:
        Configured dashboard server
    """
    return DashboardServer(
        metrics_collector=metrics_collector,
        alert_manager=alert_manager,
        analysis_engine=analysis_engine
    )


def run_dashboard(host: str = "localhost", port: int = 8501, debug: bool = False):
    """
    Run the dashboard server with default configuration.
    
    Args:
        host: Host address to bind to
        port: Port number to use
        debug: Enable debug mode
    """
    dashboard = create_dashboard_app()
    dashboard.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_dashboard()