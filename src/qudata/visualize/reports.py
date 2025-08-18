"""
Report generation for PDF and HTML output.

Creates comprehensive reports with charts, metrics, and analysis results
for pipeline monitoring and data quality assessment.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import logging
from pathlib import Path

# PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    from reportlab.graphics.charts.piecharts import Pie
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# HTML template engine
try:
    from jinja2 import Template, Environment, FileSystemLoader
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

from .metrics import MetricsCollector
from .charts import ChartGenerator
from .alerts import AlertManager

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    title: str = "QuData Pipeline Report"
    subtitle: str = "Data Processing and Quality Analysis"
    author: str = "QuData System"
    include_charts: bool = True
    include_metrics: bool = True
    include_alerts: bool = True
    include_analysis: bool = True
    time_range_hours: int = 24
    output_format: str = "html"  # 'html', 'pdf'
    template_path: Optional[str] = None
    logo_path: Optional[str] = None


@dataclass
class ReportSection:
    """Individual report section."""
    title: str
    content: str
    charts: List[str] = None
    tables: List[Dict[str, Any]] = None
    order: int = 0


class ReportRenderer:
    """
    Generates comprehensive reports in PDF and HTML formats.
    
    Creates detailed reports with metrics, charts, alerts, and analysis
    results for pipeline monitoring and quality assessment.
    """
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None,
                 chart_generator: Optional[ChartGenerator] = None,
                 alert_manager: Optional[AlertManager] = None):
        """
        Initialize report renderer.
        
        Args:
            metrics_collector: Metrics collector for data source
            chart_generator: Chart generator for visualizations
            alert_manager: Alert manager for alert data
        """
        self.metrics_collector = metrics_collector
        self.chart_generator = chart_generator
        self.alert_manager = alert_manager
        
        # Report templates
        self.html_template = self._get_default_html_template()
        self.css_styles = self._get_default_css_styles()
        
        # Output directory
        self.output_dir = Path("reports")
        self.output_dir.mkdir(exist_ok=True)
        
        # Chart export directory
        self.chart_dir = self.output_dir / "charts"
        self.chart_dir.mkdir(exist_ok=True)
    
    def generate_report(self, config: ReportConfig, output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive report.
        
        Args:
            config: Report configuration
            output_path: Optional custom output path
            
        Returns:
            Path to generated report file
        """
        if config.output_format == "html":
            return self._generate_html_report(config, output_path)
        elif config.output_format == "pdf":
            return self._generate_pdf_report(config, output_path)
        else:
            raise ValueError(f"Unsupported output format: {config.output_format}")
    
    def _generate_html_report(self, config: ReportConfig, output_path: Optional[str] = None) -> str:
        """Generate HTML report."""
        if not JINJA2_AVAILABLE:
            raise ImportError("Jinja2 is required for HTML report generation")
        
        # Collect report data
        report_data = self._collect_report_data(config)
        
        # Generate charts if requested
        chart_files = []
        if config.include_charts and self.chart_generator:
            chart_files = self._generate_report_charts()
        
        # Prepare template data
        template_data = {
            'config': config,
            'report_data': report_data,
            'chart_files': chart_files,
            'generation_time': datetime.now(),
            'css_styles': self.css_styles
        }
        
        # Render template
        template = Template(self.html_template)
        html_content = template.render(**template_data)
        
        # Save to file
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"qudata_report_{timestamp}.html"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {output_path}")
        return str(output_path)
    
    def _generate_pdf_report(self, config: ReportConfig, output_path: Optional[str] = None) -> str:
        """Generate PDF report."""
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab is required for PDF report generation")
        
        # Collect report data
        report_data = self._collect_report_data(config)
        
        # Setup PDF document
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"qudata_report_{timestamp}.pdf"
        
        doc = SimpleDocTemplate(str(output_path), pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title page
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        
        story.append(Paragraph(config.title, title_style))
        story.append(Paragraph(config.subtitle, styles['Heading2']))
        story.append(Spacer(1, 20))
        
        # Report metadata
        metadata_data = [
            ['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Author:', config.author],
            ['Time Range:', f'{config.time_range_hours} hours'],
            ['Report Type:', 'Pipeline Analysis']
        ]
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 3*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        
        story.append(metadata_table)
        story.append(PageBreak())
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['Heading1']))
        
        processing_metrics = report_data.get('processing_metrics', {})
        quality_metrics = report_data.get('quality_metrics', {})
        error_metrics = report_data.get('error_metrics', {})
        
        summary_text = f"""
        This report provides a comprehensive analysis of the QuData pipeline performance 
        over the last {config.time_range_hours} hours.
        
        Key Findings:
        • {processing_metrics.get('documents_processed', 0)} documents processed successfully
        • Average quality score: {quality_metrics.get('avg_quality_score', 0):.3f}
        • Error rate: {error_metrics.get('error_rate_percent', 0):.1f}%
        • Processing throughput: {processing_metrics.get('throughput_docs_per_sec', 0):.2f} docs/sec
        """
        
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Processing Metrics Section
        if config.include_metrics:
            story.append(Paragraph("Processing Metrics", styles['Heading1']))
            
            metrics_data = [
                ['Metric', 'Value'],
                ['Documents Processed', str(processing_metrics.get('documents_processed', 0))],
                ['Documents Failed', str(processing_metrics.get('documents_failed', 0))],
                ['Average Processing Time', f"{processing_metrics.get('processing_time_avg', 0):.3f}s"],
                ['Total Processing Time', f"{processing_metrics.get('processing_time_total', 0):.1f}s"],
                ['Throughput', f"{processing_metrics.get('throughput_docs_per_sec', 0):.2f} docs/sec"],
                ['Memory Usage', f"{processing_metrics.get('memory_usage_mb', 0):.1f} MB"],
                ['CPU Usage', f"{processing_metrics.get('cpu_usage_percent', 0):.1f}%"],
                ['Queue Size', str(processing_metrics.get('queue_size', 0))]
            ]
            
            metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(metrics_table)
            story.append(Spacer(1, 20))
        
        # Quality Metrics Section
        story.append(Paragraph("Quality Analysis", styles['Heading1']))
        
        quality_data = [
            ['Quality Metric', 'Value'],
            ['Average Quality Score', f"{quality_metrics.get('avg_quality_score', 0):.3f}"],
            ['Minimum Quality Score', f"{quality_metrics.get('min_quality_score', 0):.3f}"],
            ['Maximum Quality Score', f"{quality_metrics.get('max_quality_score', 0):.3f}"],
            ['Documents Below Threshold', str(quality_metrics.get('documents_below_threshold', 0))],
            ['Duplicate Documents', str(quality_metrics.get('duplicate_documents', 0))],
            ['Average Content Length', f"{quality_metrics.get('content_length_avg', 0):.0f} chars"]
        ]
        
        quality_table = Table(quality_data, colWidths=[3*inch, 2*inch])
        quality_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(quality_table)
        story.append(Spacer(1, 20))
        
        # Alerts Section
        if config.include_alerts and self.alert_manager:
            story.append(Paragraph("Alerts Summary", styles['Heading1']))
            
            alert_summary = report_data.get('alert_summary', {})
            active_alerts = report_data.get('active_alerts', [])
            
            alert_text = f"""
            Alert Status:
            • Active Alerts: {alert_summary.get('active_alerts', 0)}
            • Total Alerts (24h): {len(report_data.get('alert_history', []))}
            • Alert Rules: {alert_summary.get('alert_rules_count', 0)}
            """
            
            story.append(Paragraph(alert_text, styles['Normal']))
            
            if active_alerts:
                story.append(Paragraph("Active Alerts", styles['Heading2']))
                
                alert_data = [['Rule', 'Severity', 'Message', 'Created']]
                for alert in active_alerts[:10]:  # Limit to 10 alerts
                    alert_data.append([
                        alert.rule_name,
                        alert.severity.value,
                        alert.message[:50] + "..." if len(alert.message) > 50 else alert.message,
                        alert.created_at.strftime('%H:%M:%S')
                    ])
                
                alert_table = Table(alert_data, colWidths=[1.5*inch, 1*inch, 2.5*inch, 1*inch])
                alert_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(alert_table)
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"PDF report generated: {output_path}")
        return str(output_path)
    
    def _collect_report_data(self, config: ReportConfig) -> Dict[str, Any]:
        """Collect all data needed for the report."""
        report_data = {}
        
        if self.metrics_collector:
            current_metrics = self.metrics_collector.get_current_metrics()
            report_data['processing_metrics'] = current_metrics.get('processing', {})
            report_data['quality_metrics'] = current_metrics.get('quality', {})
            report_data['error_metrics'] = current_metrics.get('errors', {})
            report_data['uptime_seconds'] = current_metrics.get('uptime_seconds', 0)
            
            # Historical metrics
            since = datetime.now() - timedelta(hours=config.time_range_hours)
            report_data['metrics_history'] = self.metrics_collector.get_metrics_history(since=since)
        
        if self.alert_manager:
            report_data['alert_summary'] = self.alert_manager.get_alert_summary()
            report_data['active_alerts'] = self.alert_manager.get_active_alerts()
            report_data['alert_history'] = self.alert_manager.get_alert_history(hours=config.time_range_hours)
        
        return report_data
    
    def _generate_report_charts(self) -> List[str]:
        """Generate charts for the report and return file paths."""
        chart_files = []
        
        if not self.chart_generator:
            return chart_files
        
        try:
            # Processing timeline
            timeline_chart = self.chart_generator.create_processing_timeline(hours=24)
            timeline_path = self.chart_dir / "processing_timeline.html"
            timeline_chart.write_html(str(timeline_path))
            chart_files.append(str(timeline_path))
            
            # Quality distribution
            quality_chart = self.chart_generator.create_quality_distribution()
            quality_path = self.chart_dir / "quality_distribution.html"
            quality_chart.write_html(str(quality_path))
            chart_files.append(str(quality_path))
            
            # Language distribution
            language_chart = self.chart_generator.create_language_pie_chart()
            language_path = self.chart_dir / "language_distribution.html"
            language_chart.write_html(str(language_path))
            chart_files.append(str(language_path))
            
            # System metrics
            system_chart = self.chart_generator.create_system_metrics_dashboard()
            system_path = self.chart_dir / "system_metrics.html"
            system_chart.write_html(str(system_path))
            chart_files.append(str(system_path))
            
            # Error heatmap
            error_chart = self.chart_generator.create_error_heatmap(days=7)
            error_path = self.chart_dir / "error_heatmap.html"
            error_chart.write_html(str(error_path))
            chart_files.append(str(error_path))
            
        except Exception as e:
            logger.error(f"Error generating charts for report: {e}")
        
        return chart_files
    
    def generate_summary_report(self, output_format: str = "html") -> str:
        """
        Generate a quick summary report with default settings.
        
        Args:
            output_format: Output format ('html' or 'pdf')
            
        Returns:
            Path to generated report
        """
        config = ReportConfig(
            title="QuData Pipeline Summary",
            subtitle="Quick Status Report",
            output_format=output_format,
            time_range_hours=24
        )
        
        return self.generate_report(config)
    
    def generate_detailed_report(self, output_format: str = "html", time_range_hours: int = 168) -> str:
        """
        Generate a detailed report with extended time range.
        
        Args:
            output_format: Output format ('html' or 'pdf')
            time_range_hours: Time range in hours (default: 1 week)
            
        Returns:
            Path to generated report
        """
        config = ReportConfig(
            title="QuData Pipeline Detailed Analysis",
            subtitle="Comprehensive Performance Report",
            output_format=output_format,
            time_range_hours=time_range_hours,
            include_charts=True,
            include_metrics=True,
            include_alerts=True,
            include_analysis=True
        )
        
        return self.generate_report(config)
    
    def _get_default_html_template(self) -> str:
        """Get default HTML template for reports."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ config.title }}</title>
    <style>{{ css_styles }}</style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{{ config.title }}</h1>
            <h2>{{ config.subtitle }}</h2>
            <div class="metadata">
                <p><strong>Generated:</strong> {{ generation_time.strftime('%Y-%m-%d %H:%M:%S') }}</p>
                <p><strong>Author:</strong> {{ config.author }}</p>
                <p><strong>Time Range:</strong> {{ config.time_range_hours }} hours</p>
            </div>
        </header>
        
        <section class="executive-summary">
            <h2>Executive Summary</h2>
            <div class="summary-grid">
                <div class="metric-card">
                    <h3>Documents Processed</h3>
                    <div class="metric-value">{{ report_data.processing_metrics.documents_processed or 0 }}</div>
                </div>
                <div class="metric-card">
                    <h3>Average Quality Score</h3>
                    <div class="metric-value">{{ "%.3f"|format(report_data.quality_metrics.avg_quality_score or 0) }}</div>
                </div>
                <div class="metric-card">
                    <h3>Error Rate</h3>
                    <div class="metric-value">{{ "%.1f"|format(report_data.error_metrics.error_rate_percent or 0) }}%</div>
                </div>
                <div class="metric-card">
                    <h3>Throughput</h3>
                    <div class="metric-value">{{ "%.2f"|format(report_data.processing_metrics.throughput_docs_per_sec or 0) }} docs/sec</div>
                </div>
            </div>
        </section>
        
        {% if config.include_metrics %}
        <section class="processing-metrics">
            <h2>Processing Metrics</h2>
            <table class="metrics-table">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Documents Processed</td><td>{{ report_data.processing_metrics.documents_processed or 0 }}</td></tr>
                <tr><td>Documents Failed</td><td>{{ report_data.processing_metrics.documents_failed or 0 }}</td></tr>
                <tr><td>Average Processing Time</td><td>{{ "%.3f"|format(report_data.processing_metrics.processing_time_avg or 0) }}s</td></tr>
                <tr><td>Total Processing Time</td><td>{{ "%.1f"|format(report_data.processing_metrics.processing_time_total or 0) }}s</td></tr>
                <tr><td>Memory Usage</td><td>{{ "%.1f"|format(report_data.processing_metrics.memory_usage_mb or 0) }} MB</td></tr>
                <tr><td>CPU Usage</td><td>{{ "%.1f"|format(report_data.processing_metrics.cpu_usage_percent or 0) }}%</td></tr>
                <tr><td>Queue Size</td><td>{{ report_data.processing_metrics.queue_size or 0 }}</td></tr>
            </table>
        </section>
        {% endif %}
        
        <section class="quality-analysis">
            <h2>Quality Analysis</h2>
            <table class="metrics-table">
                <tr><th>Quality Metric</th><th>Value</th></tr>
                <tr><td>Average Quality Score</td><td>{{ "%.3f"|format(report_data.quality_metrics.avg_quality_score or 0) }}</td></tr>
                <tr><td>Minimum Quality Score</td><td>{{ "%.3f"|format(report_data.quality_metrics.min_quality_score or 0) }}</td></tr>
                <tr><td>Maximum Quality Score</td><td>{{ "%.3f"|format(report_data.quality_metrics.max_quality_score or 0) }}</td></tr>
                <tr><td>Documents Below Threshold</td><td>{{ report_data.quality_metrics.documents_below_threshold or 0 }}</td></tr>
                <tr><td>Duplicate Documents</td><td>{{ report_data.quality_metrics.duplicate_documents or 0 }}</td></tr>
                <tr><td>Average Content Length</td><td>{{ "%.0f"|format(report_data.quality_metrics.content_length_avg or 0) }} chars</td></tr>
            </table>
        </section>
        
        {% if config.include_alerts and report_data.alert_summary %}
        <section class="alerts-summary">
            <h2>Alerts Summary</h2>
            <div class="alert-stats">
                <p><strong>Active Alerts:</strong> {{ report_data.alert_summary.active_alerts }}</p>
                <p><strong>Total Alerts (24h):</strong> {{ report_data.alert_history|length }}</p>
                <p><strong>Alert Rules:</strong> {{ report_data.alert_summary.alert_rules_count }}</p>
            </div>
            
            {% if report_data.active_alerts %}
            <h3>Active Alerts</h3>
            <table class="alerts-table">
                <tr><th>Rule</th><th>Severity</th><th>Message</th><th>Created</th></tr>
                {% for alert in report_data.active_alerts[:10] %}
                <tr class="alert-{{ alert.severity.value }}">
                    <td>{{ alert.rule_name }}</td>
                    <td>{{ alert.severity.value }}</td>
                    <td>{{ alert.message[:100] }}{% if alert.message|length > 100 %}...{% endif %}</td>
                    <td>{{ alert.created_at.strftime('%H:%M:%S') }}</td>
                </tr>
                {% endfor %}
            </table>
            {% endif %}
        </section>
        {% endif %}
        
        {% if config.include_charts and chart_files %}
        <section class="charts">
            <h2>Visualizations</h2>
            <div class="charts-grid">
                {% for chart_file in chart_files %}
                <div class="chart-container">
                    <iframe src="{{ chart_file }}" width="100%" height="400"></iframe>
                </div>
                {% endfor %}
            </div>
        </section>
        {% endif %}
        
        <footer>
            <p>Report generated by QuData Pipeline Monitoring System</p>
            <p>{{ generation_time.strftime('%Y-%m-%d %H:%M:%S') }}</p>
        </footer>
    </div>
</body>
</html>
        """
    
    def _get_default_css_styles(self) -> str:
        """Get default CSS styles for HTML reports."""
        return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        
        header {
            text-align: center;
            border-bottom: 2px solid #3498db;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        
        h1 {
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        h2 {
            color: #34495e;
            border-bottom: 1px solid #ecf0f1;
            padding-bottom: 10px;
        }
        
        .metadata {
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .metric-card {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #3498db;
        }
        
        .metric-card h3 {
            margin: 0 0 10px 0;
            color: #2c3e50;
            font-size: 14px;
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }
        
        .metrics-table, .alerts-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        .metrics-table th, .alerts-table th {
            background-color: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
        }
        
        .metrics-table td, .alerts-table td {
            padding: 10px;
            border-bottom: 1px solid #ecf0f1;
        }
        
        .metrics-table tr:nth-child(even), .alerts-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        
        .alert-critical {
            background-color: #ffebee !important;
        }
        
        .alert-high {
            background-color: #fff3e0 !important;
        }
        
        .alert-medium {
            background-color: #e8f5e8 !important;
        }
        
        .alert-stats {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .chart-container {
            border: 1px solid #ecf0f1;
            border-radius: 5px;
            overflow: hidden;
        }
        
        footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
            color: #7f8c8d;
        }
        
        section {
            margin-bottom: 30px;
        }
        """
    
    def export_report_data(self, config: ReportConfig, format: str = "json") -> str:
        """
        Export raw report data in specified format.
        
        Args:
            config: Report configuration
            format: Export format ('json', 'csv')
            
        Returns:
            Formatted report data
        """
        report_data = self._collect_report_data(config)
        
        if format == "json":
            return json.dumps(report_data, default=str, indent=2)
        elif format == "csv":
            # Simple CSV export of key metrics
            lines = ['metric,value,timestamp']
            timestamp = datetime.now().isoformat()
            
            # Processing metrics
            processing = report_data.get('processing_metrics', {})
            for key, value in processing.__dict__.items() if hasattr(processing, '__dict__') else {}:
                lines.append(f"processing_{key},{value},{timestamp}")
            
            # Quality metrics
            quality = report_data.get('quality_metrics', {})
            for key, value in quality.__dict__.items() if hasattr(quality, '__dict__') else {}:
                lines.append(f"quality_{key},{value},{timestamp}")
            
            return '\n'.join(lines)
        else:
            raise ValueError(f"Unsupported export format: {format}")