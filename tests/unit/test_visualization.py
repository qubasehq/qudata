"""Unit tests for visualization and monitoring dashboard components.

Tests the dashboard server, chart generator, metrics collector,
alert manager, and report renderer functionality.
"""

import pytest
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from src.qudata.visualize.dashboard import DashboardServer, create_dashboard_app
from src.qudata.visualize.charts import ChartGenerator
from src.qudata.visualize.metrics import (
    MetricsCollector, Metric, ProcessingMetrics, QualityMetrics, ErrorMetrics,
    get_metrics_collector, set_metrics_collector
)
from src.qudata.visualize.alerts import (
    AlertManager, Alert, AlertRule, AlertSeverity, AlertStatus, NotificationChannel
)
from src.qudata.visualize.reports import ReportRenderer, ReportConfig, ReportSection


class TestMetricsCollector:
    """Test metrics collection functionality."""
    
    def test_metrics_collector_initialization(self):
        """Test metrics collector initialization."""
        collector = MetricsCollector(max_history_size=100)
        
        assert collector.max_history_size == 100
        assert len(collector._metrics_history) == 0
        assert isinstance(collector.processing_metrics, ProcessingMetrics)
        assert isinstance(collector.quality_metrics, QualityMetrics)
        assert isinstance(collector.error_metrics, ErrorMetrics)
    
    def test_record_metric(self):
        """Test recording individual metrics."""
        collector = MetricsCollector()
        
        collector.record_metric("test_metric", 42.5, {"tag": "value"}, "units")
        
        assert len(collector._metrics_history) == 1
        metric = collector._metrics_history[0]
        assert metric.name == "test_metric"
        assert metric.value == 42.5
        assert metric.tags == {"tag": "value"}
        assert metric.unit == "units"
        assert isinstance(metric.timestamp, datetime)
    
    def test_increment_counter(self):
        """Test counter increment functionality."""
        collector = MetricsCollector()
        
        collector.increment_counter("test_counter", 5)
        collector.increment_counter("test_counter", 3)
        
        assert collector._counters["test_counter"] == 8
        assert len(collector._metrics_history) == 2
    
    def test_set_gauge(self):
        """Test gauge setting functionality."""
        collector = MetricsCollector()
        
        collector.set_gauge("test_gauge", 75.5)
        
        assert collector._gauges["test_gauge"] == 75.5
        assert len(collector._metrics_history) == 1
    
    def test_record_timer(self):
        """Test timer recording functionality."""
        collector = MetricsCollector()
        
        collector.record_timer("test_timer", 1.5)
        collector.record_timer("test_timer", 2.5)
        
        assert len(collector._timers["test_timer"]) == 2
        assert collector._timers["test_timer"] == [1.5, 2.5]
        # Should record both latest and average
        assert len(collector._metrics_history) == 4  # 2 latest + 2 avg
    
    def test_record_document_processed(self):
        """Test document processing metrics recording."""
        collector = MetricsCollector()
        
        collector.record_document_processed(
            processing_time=1.2,
            quality_score=0.85,
            language="en",
            content_length=1000,
            success=True
        )
        
        assert collector.processing_metrics.documents_processed == 1
        assert collector.processing_metrics.processing_time_avg == 1.2
        assert collector.quality_metrics.avg_quality_score == 0.85
        assert collector.quality_metrics.language_distribution["en"] == 1
        assert collector.quality_metrics.content_length_avg == 1000
    
    def test_record_document_failed(self):
        """Test failed document processing metrics."""
        collector = MetricsCollector()
        
        collector.record_document_processed(
            processing_time=0.5,
            quality_score=0.0,
            language="en",
            content_length=100,
            success=False
        )
        
        assert collector.processing_metrics.documents_failed == 1
        assert collector.processing_metrics.documents_processed == 0
    
    def test_record_error(self):
        """Test error recording functionality."""
        collector = MetricsCollector()
        
        collector.record_error("FileNotFound", "ingestion", "File not found")
        
        assert collector.error_metrics.total_errors == 1
        assert collector.error_metrics.errors_by_type["FileNotFound"] == 1
        assert collector.error_metrics.errors_by_stage["ingestion"] == 1
        assert collector.error_metrics.last_error_time is not None
    
    def test_update_system_metrics(self):
        """Test system metrics update."""
        collector = MetricsCollector()
        
        collector.update_system_metrics(512.0, 75.5, 10)
        
        assert collector.processing_metrics.memory_usage_mb == 512.0
        assert collector.processing_metrics.cpu_usage_percent == 75.5
        assert collector.processing_metrics.queue_size == 10
    
    def test_get_current_metrics(self):
        """Test current metrics retrieval."""
        collector = MetricsCollector()
        collector.record_document_processed(1.0, 0.8, "en", 500, True)
        
        metrics = collector.get_current_metrics()
        
        assert 'processing' in metrics
        assert 'quality' in metrics
        assert 'errors' in metrics
        assert 'timestamp' in metrics
        assert 'uptime_seconds' in metrics
    
    def test_get_metrics_history(self):
        """Test metrics history retrieval."""
        collector = MetricsCollector()
        
        collector.record_metric("test1", 1)
        collector.record_metric("test2", 2)
        collector.record_metric("test1", 3)
        
        # Get all metrics
        all_metrics = collector.get_metrics_history()
        assert len(all_metrics) == 3
        
        # Filter by name
        test1_metrics = collector.get_metrics_history("test1")
        assert len(test1_metrics) == 2
        assert all(m.name == "test1" for m in test1_metrics)
        
        # Filter by time
        since = datetime.now() - timedelta(seconds=1)
        recent_metrics = collector.get_metrics_history(since=since)
        assert len(recent_metrics) <= 3
    
    def test_get_metric_summary(self):
        """Test metric summary statistics."""
        collector = MetricsCollector()
        
        collector.record_metric("test_metric", 10)
        collector.record_metric("test_metric", 20)
        collector.record_metric("test_metric", 30)
        
        summary = collector.get_metric_summary("test_metric", window_minutes=60)
        
        assert summary['min'] == 10
        assert summary['max'] == 30
        assert summary['avg'] == 20
        assert summary['count'] == 3
    
    def test_export_metrics(self):
        """Test metrics export functionality."""
        collector = MetricsCollector()
        collector.record_document_processed(1.0, 0.8, "en", 500, True)
        
        # Test JSON export
        json_export = collector.export_metrics('json')
        data = json.loads(json_export)
        assert 'processing' in data
        assert 'quality' in data
        
        # Test CSV export
        csv_export = collector.export_metrics('csv')
        assert 'metric,value,timestamp' in csv_export
        
        # Test invalid format
        with pytest.raises(ValueError):
            collector.export_metrics('invalid')
    
    def test_reset_metrics(self):
        """Test metrics reset functionality."""
        collector = MetricsCollector()
        collector.record_metric("test", 1)
        collector.record_document_processed(1.0, 0.8, "en", 500, True)
        
        collector.reset_metrics()
        
        assert len(collector._metrics_history) == 0
        assert collector.processing_metrics.documents_processed == 0
        assert collector.quality_metrics.avg_quality_score == 0.0
    
    def test_global_collector(self):
        """Test global metrics collector functionality."""
        # Test getting default collector
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        assert collector1 is collector2
        
        # Test setting custom collector
        custom_collector = MetricsCollector()
        set_metrics_collector(custom_collector)
        collector3 = get_metrics_collector()
        assert collector3 is custom_collector


class TestChartGenerator:
    """Test chart generation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metrics_collector = MetricsCollector()
        self.chart_generator = ChartGenerator(self.metrics_collector)
        
        # Add some test data
        self.metrics_collector.record_document_processed(1.0, 0.8, "en", 500, True)
        self.metrics_collector.record_document_processed(1.5, 0.9, "es", 600, True)
        self.metrics_collector.record_error("ParseError", "extraction", "Parse failed")
    
    def test_chart_generator_initialization(self):
        """Test chart generator initialization."""
        generator = ChartGenerator()
        assert generator.metrics_collector is None
        
        generator = ChartGenerator(self.metrics_collector)
        assert generator.metrics_collector is self.metrics_collector
    
    def test_create_processing_timeline(self):
        """Test processing timeline chart creation."""
        chart = self.chart_generator.create_processing_timeline(hours=24)
        
        assert chart is not None
        assert hasattr(chart, 'data')
        assert hasattr(chart, 'layout')
    
    def test_create_quality_distribution(self):
        """Test quality distribution chart creation."""
        chart = self.chart_generator.create_quality_distribution()
        
        assert chart is not None
        assert hasattr(chart, 'data')
    
    def test_create_language_pie_chart(self):
        """Test language distribution pie chart creation."""
        chart = self.chart_generator.create_language_pie_chart()
        
        assert chart is not None
        assert hasattr(chart, 'data')
    
    def test_create_error_heatmap(self):
        """Test error heatmap creation."""
        chart = self.chart_generator.create_error_heatmap(days=7)
        
        assert chart is not None
        assert hasattr(chart, 'data')
    
    def test_create_throughput_gauge(self):
        """Test throughput gauge creation."""
        chart = self.chart_generator.create_throughput_gauge()
        
        assert chart is not None
        assert hasattr(chart, 'data')
    
    def test_create_system_metrics_dashboard(self):
        """Test system metrics dashboard creation."""
        chart = self.chart_generator.create_system_metrics_dashboard()
        
        assert chart is not None
        assert hasattr(chart, 'data')
    
    def test_create_token_length_distribution(self):
        """Test token length distribution chart."""
        data = [
            {'token_count': 100},
            {'token_count': 200},
            {'token_count': 150}
        ]
        
        chart = self.chart_generator.create_token_length_distribution(data)
        
        assert chart is not None
        assert hasattr(chart, 'data')
    
    def test_create_sentiment_trends(self):
        """Test sentiment trends chart."""
        data = [
            {'sentiment': 'positive'},
            {'sentiment': 'negative'},
            {'sentiment': 'positive'}
        ]
        
        chart = self.chart_generator.create_sentiment_trends(data)
        
        assert chart is not None
        assert hasattr(chart, 'data')
    
    def test_empty_chart_creation(self):
        """Test empty chart creation when no data available."""
        empty_generator = ChartGenerator()
        
        chart = empty_generator.create_processing_timeline()
        assert chart is not None
        
        chart = empty_generator.create_quality_distribution()
        assert chart is not None
    
    @patch('src.qudata.visualize.charts.go.Figure.write_html')
    def test_export_chart(self, mock_write_html):
        """Test chart export functionality."""
        chart = self.chart_generator.create_processing_timeline()
        
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
            result = self.chart_generator.export_chart(chart, tmp.name, 'html')
            assert result == tmp.name
            mock_write_html.assert_called_once()


class TestAlertManager:
    """Test alert management functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
    
    def test_alert_manager_initialization(self):
        """Test alert manager initialization."""
        manager = AlertManager()
        assert manager.metrics_collector is None
        assert len(manager.alert_rules) > 0  # Should have default rules
        assert len(manager.notification_channels) > 0  # Should have default channels
    
    def test_add_alert_rule(self):
        """Test adding alert rules."""
        rule = AlertRule(
            name="test_rule",
            description="Test rule",
            metric_name="test_metric",
            condition="gt",
            threshold=10.0,
            severity=AlertSeverity.HIGH
        )
        
        self.alert_manager.add_alert_rule(rule)
        
        assert "test_rule" in self.alert_manager.alert_rules
        assert self.alert_manager.alert_rules["test_rule"] == rule
    
    def test_remove_alert_rule(self):
        """Test removing alert rules."""
        rule = AlertRule(
            name="test_rule",
            description="Test rule",
            metric_name="test_metric",
            condition="gt",
            threshold=10.0,
            severity=AlertSeverity.HIGH
        )
        
        self.alert_manager.add_alert_rule(rule)
        self.alert_manager.remove_alert_rule("test_rule")
        
        assert "test_rule" not in self.alert_manager.alert_rules
    
    def test_add_notification_channel(self):
        """Test adding notification channels."""
        channel = NotificationChannel(
            name="test_channel",
            type="email",
            config={"smtp_server": "localhost"}
        )
        
        self.alert_manager.add_notification_channel(channel)
        
        assert "test_channel" in self.alert_manager.notification_channels
        assert self.alert_manager.notification_channels["test_channel"] == channel
    
    def test_evaluate_condition(self):
        """Test condition evaluation."""
        manager = self.alert_manager
        
        assert manager._evaluate_condition(15, "gt", 10) == True
        assert manager._evaluate_condition(5, "gt", 10) == False
        assert manager._evaluate_condition(5, "lt", 10) == True
        assert manager._evaluate_condition(10, "eq", 10) == True
        assert manager._evaluate_condition(10, "gte", 10) == True
        assert manager._evaluate_condition(10, "lte", 10) == True
    
    def test_extract_metric_value(self):
        """Test metric value extraction from nested structure."""
        metrics = {
            'processing': Mock(documents_processed=100),
            'quality': {'avg_quality_score': 0.85},
            'direct_metric': 42
        }
        
        manager = self.alert_manager
        
        assert manager._extract_metric_value('documents_processed', metrics) == 100
        assert manager._extract_metric_value('avg_quality_score', metrics) == 0.85
        assert manager._extract_metric_value('direct_metric', metrics) == 42
        assert manager._extract_metric_value('nonexistent', metrics) is None
    
    def test_trigger_alert(self):
        """Test alert triggering."""
        rule = AlertRule(
            name="test_rule",
            description="Test alert",
            metric_name="test_metric",
            condition="gt",
            threshold=10.0,
            severity=AlertSeverity.HIGH,
            notification_channels=["console"]
        )
        
        alert = Alert(
            id="test_alert",
            rule_name="test_rule",
            message="Test alert message",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metric_value=15.0,
            threshold=10.0
        )
        
        self.alert_manager._trigger_alert(alert, rule)
        
        assert "test_alert" in self.alert_manager.active_alerts
        assert len(self.alert_manager.alert_history) == 1
    
    def test_resolve_alert(self):
        """Test alert resolution."""
        alert = Alert(
            id="test_alert",
            rule_name="test_rule",
            message="Test alert",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metric_value=15.0,
            threshold=10.0
        )
        
        self.alert_manager.active_alerts["test_alert"] = alert
        self.alert_manager._resolve_alert("test_alert")
        
        assert "test_alert" not in self.alert_manager.active_alerts
        assert alert.status == AlertStatus.RESOLVED
        assert alert.resolved_at is not None
    
    def test_acknowledge_alert(self):
        """Test alert acknowledgment."""
        alert = Alert(
            id="test_alert",
            rule_name="test_rule",
            message="Test alert",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metric_value=15.0,
            threshold=10.0
        )
        
        self.alert_manager.active_alerts["test_alert"] = alert
        self.alert_manager.acknowledge_alert("test_alert", "test_user")
        
        assert alert.status == AlertStatus.ACKNOWLEDGED
        assert alert.acknowledged_by == "test_user"
    
    def test_suppress_alert(self):
        """Test alert suppression."""
        alert = Alert(
            id="test_alert",
            rule_name="test_rule",
            message="Test alert",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metric_value=15.0,
            threshold=10.0
        )
        
        self.alert_manager.active_alerts["test_alert"] = alert
        self.alert_manager.suppress_alert("test_alert", 60)
        
        assert alert.status == AlertStatus.SUPPRESSED
        assert 'suppressed_until' in alert.metadata
    
    def test_get_alert_summary(self):
        """Test alert summary generation."""
        summary = self.alert_manager.get_alert_summary()
        
        assert 'active_alerts' in summary
        assert 'total_alerts' in summary
        assert 'severity_distribution' in summary
        assert 'alert_rules_count' in summary
        assert 'notification_channels_count' in summary
    
    def test_export_alerts(self):
        """Test alert export functionality."""
        export_data = self.alert_manager.export_alerts('json')
        data = json.loads(export_data)
        
        assert 'active_alerts' in data
        assert 'alert_history' in data
        assert 'summary' in data
        
        # Test invalid format
        with pytest.raises(ValueError):
            self.alert_manager.export_alerts('invalid')
    
    @patch('src.qudata.visualize.alerts.print')
    def test_console_notification(self, mock_print):
        """Test console notification sending."""
        alert = Alert(
            id="test_alert",
            rule_name="test_rule",
            message="Test alert",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metric_value=15.0,
            threshold=10.0
        )
        
        self.alert_manager._send_console_notification(alert)
        mock_print.assert_called_once()


class TestReportRenderer:
    """Test report rendering functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metrics_collector = MetricsCollector()
        self.chart_generator = ChartGenerator(self.metrics_collector)
        self.alert_manager = AlertManager(self.metrics_collector)
        self.report_renderer = ReportRenderer(
            self.metrics_collector,
            self.chart_generator,
            self.alert_manager
        )
        
        # Add some test data
        self.metrics_collector.record_document_processed(1.0, 0.8, "en", 500, True)
    
    def test_report_renderer_initialization(self):
        """Test report renderer initialization."""
        renderer = ReportRenderer()
        assert renderer.metrics_collector is None
        assert renderer.chart_generator is None
        assert renderer.alert_manager is None
        assert renderer.output_dir.exists()
        assert renderer.chart_dir.exists()
    
    def test_collect_report_data(self):
        """Test report data collection."""
        config = ReportConfig(time_range_hours=24)
        data = self.report_renderer._collect_report_data(config)
        
        assert 'processing_metrics' in data
        assert 'quality_metrics' in data
        assert 'error_metrics' in data
        assert 'alert_summary' in data
        assert 'active_alerts' in data
        assert 'alert_history' in data
    
    def test_generate_summary_report(self):
        """Test summary report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.report_renderer.output_ = Path(tmpdir)
            
            report_path = self.report_renderer.generate_summary_report("html")
            
            assert Path(report_path).exists()
            assert report_path.endswith('.html')
    
    def test_generate_detailed_report(self):
        """Test detailed report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.report_renderer.output_dir = Path(tmpdir)
            
            report_path = self.report_renderer.generate_detailed_report("html", 168)
            
            assert Path(report_path).exists()
            assert report_path.endswith('.html')
    
    @patch('src.qudata.visualize.reports.JINJA2_AVAILABLE', True)
    def test_generate_html_report(self):
        """Test HTML report generation."""
        cig = ReportConfig(output_format="html")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            self.report_renderer.output_dir = Path(tmpdir)
            output_path = Path(tmpdir) / "test_report.html"
            
            result_path = self.report_renderer._generate_html_report(config, str(output_path))
            
            assert Path(result_path).exists()
            assert result_path == str(output_path)
    
    @patch('src.qudata.visualize.reports.REPORTLAB_AVAILABLE', True)
    @patch('src.qudata.visualize.reports.SimpleDocTemplate')
    def test_generate_pdf_report(self, mock_doc):
        """Test PDF report generation."""
        config = ReportConfig(output_format="pdf")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            self.report_renderer.output_dir = Path(tmpdir)
            output_path = Path(tmpdir) / "test_report.pdf"
            
            # Mock the PDF document
            mock_doc_instance = Mock()
            mock_doc.return_value = mock_doc_instance
            
            result_path = self.report_renderer._generate_pdf_report(config, str(output_path))
            
            assert result_path == str(output_path)
            mock_doc_instance.build.assert_called_once()
    
    def test_export_report_data(self):
        """Test report data export."""
        config = ReportConfig()
        
        # Test JSON export
        json_data = self.report_renderer.export_report_data(config, "json")
        data = json.loads(json_data)
        assert isinstance(data, dict)
        
        # Test CSV export
        csv_data = self.report_renderer.export_report_data(config, "csv")
        assert 'metric,value,timestamp' in csv_data
        
        # Test invalid format
        with pytest.raises(ValueError):
            self.report_renderer.export_report_data(config, "invalid")
    
    def test_unsupported_format_error(self):
        """Test error handling for unsupported formats."""
        config = ReportConfig(output_format="unsupported")
        
        with pytest.raises(ValueError):
            self.report_renderer.generate_report(config)


class TestDashboardServer:
    """Test dashboard server functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
        self.dashboard = DashboardServer(
            self.metrics_collector,
            self.alert_manager
        )
    
    def test_dashboard_initialization(self):
        """Test dashboard server initialization."""
        dashboard = DashboardServer()
        assert dashboard.metrics_collector is not None
        assert dashboard.alert_manager is not None
        assert dashboard.chart_generator is not None
        assert dashboard.refresh_interval == 30
        assert dashboard.auto_refresh == True
    
    def test_create_dashboard_app(self):
        """Test dashboard app creation."""
        app = create_dashboard_app()
        assert isinstance(app, DashboardServer)
        
        app = create_dashboard_app(
            metrics_collector=self.metrics_collector,
            alert_manager=self.alert_manager
        )
        assert app.metrics_collector is self.metrics_collector
        assert app.alert_manager is self.alert_manager
    
    @patch('streamlit.set_page_config')
    @patch('streamlit.sidebar')
    @patch('streamlit.title')
    def test_dashboard_rendering(self, mock_title, mock_sidebar, mock_config):
        """Test dashboard rendering components."""
        # Mock streamlit components
        mock_sidebar.title.return_value = None
        mock_sidebar.selectbox.return_value = "Overview"
        mock_sidebar.checkbox.return_value = True
        mock_sidebar.slider.return_value = 30
        
        # Test that dashboard can be initialized without errors
        dashboard = DashboardServer(self.metrics_collector, self.alert_manager)
        assert dashboard is not None


class TestIntegration:
    """Test integration between visualization components."""
    
    def test_full_pipeline_integration(self):
        """Test full visualization pipeline integration."""
        # Create components
        metrics_collector = MetricsCollector()
        chart_generator = ChartGenerator(metrics_collector)
        alert_manager = AlertManager(metrics_collector)
        report_renderer = ReportRenderer(metrics_collector, chart_generator, alert_manager)
        dashboard = DashboardServer(metrics_collector, alert_manager)
        
        # Add test data
        metrics_collector.record_document_processed(1.0, 0.8, "en", 500, True)
        metrics_collector.record_document_processed(1.5, 0.9, "es", 600, True)
        metrics_collector.record_error("ParseError", "extraction", "Parse failed")
        
        # Test metrics collection
        current_metrics = metrics_collector.get_current_metrics()
        assert current_metrics['processing'].documents_processed == 2
        
        # Test chart generation
        timeline_chart = chart_generator.create_processing_timeline()
        assert timeline_chart is not None
        
        # Test alert management
        alert_summary = alert_manager.get_alert_summary()
        assert 'active_alerts' in alert_summary
        
        # Test report generation
        with tempfile.TemporaryDirectory() as tmpdir:
            report_renderer.output_dir = Path(tmpdir)
            report_path = report_renderer.generate_summary_report()
            assert Path(report_path).ists()
        
        # Test dashboard components
        assert dashboard.metrics_collector is metrics_collector
        assert dashboard.alert_manager is alert_manager
    
    def test_error_handling_integration(self):
        """Test error handling across components."""
        # Test with None components
        dashboard = DashboardServer(None, None, None)
        assert dashboard.metrics_collector is not None  # Should use default
        
        # Test chart generation with no data
        chart_generator = ChartGenerator(None)
        chart = chart_generator.create_processing_timeline()
        assert chart is not None  # Should return empty chart
        
        # Test report generation with no data
        report_renderer = ReportRenderer(None, None, None)
        config = ReportConfig()
        data = report_renderer._collect_report_data(config)
        assert isinstance(data, dict)  # Should return empty dict


if __name__ == "__main__":
    pytest.main([__file__])