"""
Chart generation using Plotly for data visualization.

Creates interactive charts and visualizations for pipeline metrics,
quality scores, and analysis results.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
import logging

from .metrics import MetricsCollector, Metric

logger = logging.getLogger(__name__)


class ChartGenerator:
    """
    Generates interactive charts using Plotly for dashboard visualization.
    
    Creates various chart types for metrics visualization including
    time series, distributions, heatmaps, and statistical summaries.
    """
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        """
        Initialize chart generator.
        
        Args:
            metrics_collector: Metrics collector instance for data source
        """
        self.metrics_collector = metrics_collector
        self.default_colors = px.colors.qualitative.Set3
        self.theme = {
            'background_color': '#f8f9fa',
            'paper_color': 'white',
            'text_color': '#2c3e50',
            'grid_color': '#ecf0f1'
        }
    
    def create_processing_timeline(self, hours: int = 24) -> go.Figure:
        """
        Create timeline chart of processing metrics.
        
        Args:
            hours: Number of hours of history to show
            
        Returns:
            Plotly figure with processing timeline
        """
        if not self.metrics_collector:
            return self._create_empty_chart("No metrics collector available")
        
        since = datetime.now() - timedelta(hours=hours)
        metrics = self.metrics_collector.get_metrics_history(since=since)
        
        if not metrics:
            return self._create_empty_chart("No metrics data available")
        
        # Convert to DataFrame for easier manipulation
        df_data = []
        for metric in metrics:
            if metric.name in ['documents_total', 'document_processing_time_avg', 'quality_score']:
                df_data.append({
                    'timestamp': metric.timestamp,
                    'metric': metric.name,
                    'value': float(metric.value) if isinstance(metric.value, (int, float)) else 0
                })
        
        if not df_data:
            return self._create_empty_chart("No processing metrics available")
        
        df = pd.DataFrame(df_data)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Documents Processed', 'Processing Time (avg)', 'Quality Score'],
            vertical_spacing=0.08
        )
        
        # Documents processed
        docs_data = df[df['metric'] == 'documents_total']
        if not docs_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=docs_data['timestamp'],
                    y=docs_data['value'],
                    mode='lines+markers',
                    name='Documents',
                    line=dict(color='#3498db', width=2)
                ),
                row=1, col=1
            )
        
        # Processing time
        time_data = df[df['metric'] == 'document_processing_time_avg']
        if not time_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=time_data['timestamp'],
                    y=time_data['value'],
                    mode='lines+markers',
                    name='Avg Time (s)',
                    line=dict(color='#e74c3c', width=2)
                ),
                row=2, col=1
            )
        
        # Quality score
        quality_data = df[df['metric'] == 'quality_score']
        if not quality_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=quality_data['timestamp'],
                    y=quality_data['value'],
                    mode='lines+markers',
                    name='Quality Score',
                    line=dict(color='#2ecc71', width=2)
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            title='Processing Pipeline Timeline',
            height=600,
            showlegend=False,
            **self._get_layout_theme()
        )
        
        return fig
    
    def create_quality_distribution(self) -> go.Figure:
        """
        Create distribution chart of quality scores.
        
        Returns:
            Plotly figure with quality score distribution
        """
        if not self.metrics_collector:
            return self._create_empty_chart("No metrics collector available")
        
        quality_metrics = self.metrics_collector.get_metrics_history('quality_score')
        
        if not quality_metrics:
            return self._create_empty_chart("No quality metrics available")
        
        scores = [float(m.value) for m in quality_metrics if isinstance(m.value, (int, float))]
        
        if not scores:
            return self._create_empty_chart("No valid quality scores")
        
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=scores,
            nbinsx=20,
            name='Quality Score Distribution',
            marker_color='#3498db',
            opacity=0.7
        ))
        
        # Add mean line
        mean_score = sum(scores) / len(scores)
        fig.add_vline(
            x=mean_score,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_score:.2f}"
        )
        
        fig.update_layout(
            title='Quality Score Distribution',
            xaxis_title='Quality Score',
            yaxis_title='Count',
            **self._get_layout_theme()
        )
        
        return fig
    
    def create_language_pie_chart(self) -> go.Figure:
        """
        Create pie chart of language distribution.
        
        Returns:
            Plotly figure with language distribution
        """
        if not self.metrics_collector:
            return self._create_empty_chart("No metrics collector available")
        
        current_metrics = self.metrics_collector.get_current_metrics()
        language_dist = current_metrics.get('quality', {})
        
        if hasattr(language_dist, 'language_distribution'):
            lang_data = language_dist.language_distribution
        else:
            lang_data = {}
        
        if not lang_data:
            return self._create_empty_chart("No language data available")
        
        languages = list(lang_data.keys())
        counts = list(lang_data.values())
        
        fig = go.Figure(data=[go.Pie(
            labels=languages,
            values=counts,
            hole=0.3,
            marker_colors=self.default_colors[:len(languages)]
        )])
        
        fig.update_layout(
            title='Language Distribution',
            **self._get_layout_theme()
        )
        
        return fig
    
    def create_error_heatmap(self, days: int = 7) -> go.Figure:
        """
        Create heatmap of errors by type and time.
        
        Args:
            days: Number of days to include
            
        Returns:
            Plotly figure with error heatmap
        """
        if not self.metrics_collector:
            return self._create_empty_chart("No metrics collector available")
        
        since = datetime.now() - timedelta(days=days)
        error_metrics = self.metrics_collector.get_metrics_history(since=since)
        
        # Filter error metrics
        error_data = [m for m in error_metrics if m.name.startswith('errors_')]
        
        if not error_data:
            return self._create_empty_chart("No error data available")
        
        # Group by hour and error type
        hourly_errors = {}
        for metric in error_data:
            hour = metric.timestamp.strftime('%Y-%m-%d %H:00')
            error_type = metric.name.replace('errors_', '')
            
            if hour not in hourly_errors:
                hourly_errors[hour] = {}
            
            hourly_errors[hour][error_type] = hourly_errors[hour].get(error_type, 0) + 1
        
        if not hourly_errors:
            return self._create_empty_chart("No error patterns found")
        
        # Convert to matrix format
        hours = sorted(hourly_errors.keys())
        error_types = set()
        for hour_data in hourly_errors.values():
            error_types.update(hour_data.keys())
        error_types = sorted(error_types)
        
        z_matrix = []
        for error_type in error_types:
            row = [hourly_errors.get(hour, {}).get(error_type, 0) for hour in hours]
            z_matrix.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=z_matrix,
            x=hours,
            y=error_types,
            colorscale='Reds',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Error Heatmap (by Hour)',
            xaxis_title='Time',
            yaxis_title='Error Type',
            **self._get_layout_theme()
        )
        
        return fig
    
    def create_throughput_gauge(self) -> go.Figure:
        """
        Create gauge chart for processing throughput.
        
        Returns:
            Plotly figure with throughput gauge
        """
        if not self.metrics_collector:
            return self._create_empty_chart("No metrics collector available")
        
        current_metrics = self.metrics_collector.get_current_metrics()
        processing = current_metrics.get('processing', {})
        
        if hasattr(processing, 'throughput_docs_per_sec'):
            throughput = processing.throughput_docs_per_sec
        else:
            throughput = 0
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=throughput,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Throughput (docs/sec)"},
            delta={'reference': 1.0},
            gauge={
                'axis': {'range': [None, 10]},
                'bar': {'color': "#3498db"},
                'steps': [
                    {'range': [0, 2], 'color': "#ecf0f1"},
                    {'range': [2, 5], 'color': "#bdc3c7"},
                    {'range': [5, 10], 'color': "#95a5a6"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 8
                }
            }
        ))
        
        fig.update_layout(**self._get_layout_theme())
        
        return fig
    
    def create_system_metrics_dashboard(self) -> go.Figure:
        """
        Create comprehensive system metrics dashboard.
        
        Returns:
            Plotly figure with system metrics
        """
        if not self.metrics_collector:
            return self._create_empty_chart("No metrics collector available")
        
        current_metrics = self.metrics_collector.get_current_metrics()
        processing = current_metrics.get('processing', {})
        quality = current_metrics.get('quality', {})
        errors = current_metrics.get('errors', {})
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Memory Usage', 'CPU Usage', 'Queue Size', 'Error Rate'],
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # Memory usage
        memory_mb = getattr(processing, 'memory_usage_mb', 0)
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=memory_mb,
            title={'text': "Memory (MB)"},
            gauge={'axis': {'range': [0, 1000]}, 'bar': {'color': "#e74c3c"}},
        ), row=1, col=1)
        
        # CPU usage
        cpu_percent = getattr(processing, 'cpu_usage_percent', 0)
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=cpu_percent,
            title={'text': "CPU (%)"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#f39c12"}},
        ), row=1, col=2)
        
        # Queue size
        queue_size = getattr(processing, 'queue_size', 0)
        fig.add_trace(go.Indicator(
            mode="number",
            value=queue_size,
            title={'text': "Queue Size"},
        ), row=2, col=1)
        
        # Error rate
        error_rate = getattr(errors, 'error_rate_percent', 0)
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=error_rate,
            title={'text': "Error Rate (%)"},
            gauge={'axis': {'range': [0, 20]}, 'bar': {'color': "#c0392b"}},
        ), row=2, col=2)
        
        fig.update_layout(
            title='System Metrics Dashboard',
            height=500,
            **self._get_layout_theme()
        )
        
        return fig
    
    def create_token_length_distribution(self, data: List[Dict[str, Any]]) -> go.Figure:
        """
        Create token length distribution chart.
        
        Args:
            data: List of documents with token counts
            
        Returns:
            Plotly figure with token distribution
        """
        if not data:
            return self._create_empty_chart("No token data available")
        
        token_counts = [doc.get('token_count', 0) for doc in data if 'token_count' in doc]
        
        if not token_counts:
            return self._create_empty_chart("No valid token counts")
        
        fig = go.Figure()
        
        # Box plot
        fig.add_trace(go.Box(
            y=token_counts,
            name='Token Count Distribution',
            marker_color='#9b59b6'
        ))
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=token_counts,
            nbinsx=30,
            name='Token Count Histogram',
            opacity=0.7,
            marker_color='#3498db',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Token Length Distribution',
            xaxis_title='Token Count',
            yaxis_title='Distribution',
            yaxis2=dict(overlaying='y', side='right', title='Frequency'),
            **self._get_layout_theme()
        )
        
        return fig
    
    def create_sentiment_trends(self, sentiment_data: List[Dict[str, Any]]) -> go.Figure:
        """
        Create sentiment analysis trends chart.
        
        Args:
            sentiment_data: List of sentiment analysis results
            
        Returns:
            Plotly figure with sentiment trends
        """
        if not sentiment_data:
            return self._create_empty_chart("No sentiment data available")
        
        df = pd.DataFrame(sentiment_data)
        
        if df.empty or 'sentiment' not in df.columns:
            return self._create_empty_chart("Invalid sentiment data format")
        
        # Count sentiments
        sentiment_counts = df['sentiment'].value_counts()
        
        fig = go.Figure(data=[
            go.Bar(
                x=sentiment_counts.index,
                y=sentiment_counts.values,
                marker_color=['#e74c3c' if s == 'negative' else 
                             '#f39c12' if s == 'neutral' else '#2ecc71' 
                             for s in sentiment_counts.index]
            )
        ])
        
        fig.update_layout(
            title='Sentiment Distribution',
            xaxis_title='Sentiment',
            yaxis_title='Count',
            **self._get_layout_theme()
        )
        
        return fig
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create an empty chart with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=self.theme['text_color'])
        )
        fig.update_layout(**self._get_layout_theme())
        return fig
    
    def _get_layout_theme(self) -> Dict[str, Any]:
        """Get consistent layout theme for all charts."""
        return {
            'plot_bgcolor': self.theme['background_color'],
            'paper_bgcolor': self.theme['paper_color'],
            'font': {'color': self.theme['text_color']},
            'xaxis': {'gridcolor': self.theme['grid_color']},
            'yaxis': {'gridcolor': self.theme['grid_color']}
        }
    
    def export_chart(self, fig: go.Figure, filename: str, format: str = 'html') -> str:
        """
        Export chart to file.
        
        Args:
            fig: Plotly figure to export
            filename: Output filename
            format: Export format ('html', 'png', 'pdf', 'svg')
            
        Returns:
            Path to exported file
        """
        try:
            if format == 'html':
                fig.write_html(filename)
            elif format == 'png':
                fig.write_image(filename, format='png')
            elif format == 'pdf':
                fig.write_image(filename, format='pdf')
            elif format == 'svg':
                fig.write_image(filename, format='svg')
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Chart exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to export chart: {e}")
            raise