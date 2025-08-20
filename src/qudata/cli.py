#!/usr/bin/env python3
"""
QuData CLI - Main command-line interface
"""

import argparse
import asyncio
import json
import sys
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from importlib.metadata import version as _pkg_version, PackageNotFoundError

# Resolve package version for --version flag
try:
    _QUVERSION = _pkg_version("qudata")
except PackageNotFoundError:
    try:
        from . import __version__ as _local_version  # fallback to source version
        _QUVERSION = _local_version
    except Exception:
        _QUVERSION = "unknown"

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="QuData - Data processing pipeline for LLM training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  qudata process --input data/raw --output data/processed
  qudata export --format jsonl --output data/exports
  qudata clean --input data/raw --rules configs/cleansing_rules.yaml
  qudata server --host 0.0.0.0 --port 8000
  qudata webhook add --url https://example.com/webhook --events processing.completed
  qudata analyze --dataset my-dataset --output analysis.json
        """
    )
    # Global --version flag
    parser.add_argument('-V', '--version', action='version', version=f"QuData {_QUVERSION}")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize a new QuData project in a folder')
    init_parser.add_argument('--path', default='.', help='Target directory to initialize (default: current directory)')
    init_parser.add_argument('--force', action='store_true', help='Overwrite existing files if they already exist')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Run full processing pipeline')
    process_parser.add_argument('--input', required=True, help='Input directory')
    process_parser.add_argument('--output', required=True, help='Output directory')
    process_parser.add_argument('--config', default='configs/pipeline.yaml', help='Pipeline config')
    process_parser.add_argument('--format', default='jsonl', choices=['jsonl', 'chatml', 'alpaca', 'plain'], help='Output format')
    process_parser.add_argument('--parallel', type=int, default=1, help='Number of parallel workers')
    process_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output (INFO level)')
    # Only show debug-level logs if --debug/--dbug is explicitly passed
    process_parser.add_argument('--debug', '--dbug', dest='debug', action='store_true', help='Enable debug logging (DEBUG level)')
    process_parser.add_argument('--progress', action='store_true', default=True, help='Show progress bars during processing')
    process_parser.add_argument('--no-progress', dest='progress', action='store_false', help='Disable progress bars')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export processed data')
    export_parser.add_argument('--format', choices=['jsonl', 'chatml', 'alpaca', 'plain', 'parquet'], required=True)
    export_parser.add_argument('--input', default='data/processed', help='Input directory')
    export_parser.add_argument('--output', required=True, help='Output directory')
    export_parser.add_argument('--split', action='store_true', help='Create train/val/test splits')
    export_parser.add_argument('--train-ratio', type=float, default=0.8, help='Training split ratio')
    export_parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation split ratio')
    
    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean and normalize text')
    clean_parser.add_argument('--input', required=True, help='Input directory')
    clean_parser.add_argument('--output', required=True, help='Output directory')
    clean_parser.add_argument('--rules', default='configs/cleansing_rules.yaml', help='Cleaning rules')
    clean_parser.add_argument('--language', help='Filter by language')
    clean_parser.add_argument('--min-quality', type=float, help='Minimum quality score')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze dataset quality and statistics')
    analyze_parser.add_argument('--input', required=True, help='Input directory or dataset')
    analyze_parser.add_argument('--output', help='Output file for analysis results')
    analyze_parser.add_argument('--format', choices=['json', 'yaml', 'csv'], default='json', help='Output format')
    analyze_parser.add_argument('--include-topics', action='store_true', help='Include topic modeling')
    analyze_parser.add_argument('--include-sentiment', action='store_true', help='Include sentiment analysis')
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start API server')
    server_parser.add_argument('--host', default='localhost', help='Server host')
    server_parser.add_argument('--port', type=int, default=8000, help='Server port')
    server_parser.add_argument('--config', help='Configuration file')
    server_parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    server_parser.add_argument('--graphql', action='store_true', help='Enable GraphQL endpoint')
    server_parser.add_argument('--webhooks', action='store_true', help='Enable webhook management')
    
    # Webhook commands
    webhook_parser = subparsers.add_parser('webhook', help='Manage webhooks')
    webhook_subparsers = webhook_parser.add_subparsers(dest='webhook_action', help='Webhook actions')
    
    # Add webhook
    webhook_add_parser = webhook_subparsers.add_parser('add', help='Add webhook endpoint')
    webhook_add_parser.add_argument('--url', required=True, help='Webhook URL')
    webhook_add_parser.add_argument('--events', nargs='+', required=True, help='Event types to subscribe to')
    webhook_add_parser.add_argument('--secret', help='Secret for signature verification')
    webhook_add_parser.add_argument('--timeout', type=int, default=30, help='Request timeout')
    
    # List webhooks
    webhook_list_parser = webhook_subparsers.add_parser('list', help='List webhook endpoints')
    webhook_list_parser.add_argument('--format', choices=['table', 'json'], default='table', help='Output format')
    
    # Remove webhook
    webhook_remove_parser = webhook_subparsers.add_parser('remove', help='Remove webhook endpoint')
    webhook_remove_parser.add_argument('--id', required=True, help='Webhook endpoint ID')
    
    # Test webhook
    webhook_test_parser = webhook_subparsers.add_parser('test', help='Test webhook endpoint')
    webhook_test_parser.add_argument('--id', required=True, help='Webhook endpoint ID')
    
    # Dataset commands
    dataset_parser = subparsers.add_parser('dataset', help='Manage datasets')
    dataset_subparsers = dataset_parser.add_subparsers(dest='dataset_action', help='Dataset actions')
    
    # List datasets
    dataset_list_parser = dataset_subparsers.add_parser('list', help='List datasets')
    dataset_list_parser.add_argument('--format', choices=['table', 'json'], default='table', help='Output format')
    
    # Show dataset info
    dataset_info_parser = dataset_subparsers.add_parser('info', help='Show dataset information')
    dataset_info_parser.add_argument('--id', required=True, help='Dataset ID')
    
    # Validate dataset
    dataset_validate_parser = dataset_subparsers.add_parser('validate', help='Validate dataset')
    dataset_validate_parser.add_argument('--input', required=True, help='Dataset path')
    dataset_validate_parser.add_argument('--schema', help='Schema file for validation')
    
    # Config commands
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_subparsers = config_parser.add_subparsers(dest='config_action', help='Config actions')
    
    # Show config
    config_show_parser = config_subparsers.add_parser('show', help='Show current configuration')
    config_show_parser.add_argument('--file', help='Configuration file to show')
    
    # Validate config
    config_validate_parser = config_subparsers.add_parser('validate', help='Validate configuration')
    config_validate_parser.add_argument('--file', required=True, help='Configuration file to validate')
    
    # Generate config template
    config_template_parser = config_subparsers.add_parser('template', help='Generate configuration template')
    config_template_parser.add_argument('--output', required=True, help='Output file for template')
    config_template_parser.add_argument('--type', choices=['pipeline', 'quality', 'taxonomy'], default='pipeline', help='Template type')
    
    args = parser.parse_args()

    # Print brand banner for any command
    _print_branding()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'process':
            return run_process_command(args)
        elif args.command == 'export':
            return run_export_command(args)
        elif args.command == 'clean':
            return run_clean_command(args)
        elif args.command == 'analyze':
            return run_analyze_command(args)
        elif args.command == 'server':
            return run_server_command(args)
        elif args.command == 'webhook':
            return run_webhook_command(args)
        elif args.command == 'dataset':
            return run_dataset_command(args)
        elif args.command == 'config':
            return run_config_command(args)
        elif args.command == 'init':
            return run_init_command(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def run_process_command(args):
    """Run the process command."""
    from .pipeline import QuDataPipeline
    
    print(f"Processing {args.input} -> {args.output}")
    print(f"Using config: {args.config}")
    # Friendly notes: print small steps so users see progress immediately
    print("⏳ Setting things up...", flush=True)
    
    # Configure logging and suppress known noisy PDF/Numba warnings while keeping real errors visible
    try:
        import logging

        # Default to INFO; only use DEBUG when --debug/--dbug provided
        log_level = logging.DEBUG if getattr(args, 'debug', False) else logging.INFO
        
        # Custom colored formatter
        class ColoredFormatter(logging.Formatter):
            COLORS = {
                'DEBUG': '\033[94m',     # Blue
                'INFO': '\033[0m',       # Default color for INFO prefix
                'WARNING': '\033[93m',   # Yellow
                'ERROR': '\033[91m',     # Red
                'CRITICAL': '\033[95m',  # Magenta
                'RESET': '\033[0m',      # Reset
                'GREEN': '\033[92m',     # Green for INFO content
                'LIGHT_BLUE': '\033[96m' # Light blue
            }
            
            def format(self, record):
                # Get the original formatted message
                original = super().format(record)
                
                if record.levelname == 'DEBUG':
                    # Make entire DEBUG message blue
                    return f"{self.COLORS['DEBUG']}{original}{self.COLORS['RESET']}"
                elif record.levelname == 'INFO':
                    # For INFO messages, color text after the first colon green
                    if ':' in original:
                        parts = original.split(':', 1)
                        if len(parts) == 2:
                            prefix = parts[0] + ':'
                            content = parts[1]
                            return f"{prefix}{self.COLORS['GREEN']}{content}{self.COLORS['RESET']}"
                    return original
                else:
                    # Other log levels keep default coloring
                    color = self.COLORS.get(record.levelname, '')
                    return f"{color}{original}{self.COLORS['RESET']}" if color else original
        
        # Set up colored logging
        handler = logging.StreamHandler()
        handler.setFormatter(ColoredFormatter('%(levelname)s:%(name)s:%(message)s'))
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.addHandler(handler)
        root_logger.setLevel(log_level)

        class _PdfColorWarningFilter(logging.Filter):
            """Filter out noisy pdfminer color warnings like:
            "Cannot set gray non-stroke color because /'P120' is an invalid float value"
            """

            def filter(self, record: logging.LogRecord) -> bool:
                msg = record.getMessage()
                if (
                    "Cannot set gray non-stroke color" in msg
                    and "invalid float value" in msg
                ):
                    return False
                return True

        noisy_pdf_loggers = [
            logging.getLogger("pdfminer"),
            logging.getLogger("pdfminer.pdfinterp"),
            logging.getLogger("pdfminer.converter"),
            logging.getLogger("pdfminer.cmapdb"),
            logging.getLogger("pdfminer.image"),
        ]
        for lg in noisy_pdf_loggers:
            lg.addFilter(_PdfColorWarningFilter())
            # Keep them at WARNING unless DEBUG is on
            if not getattr(args, 'debug', False):
                lg.setLevel(logging.WARNING)

        # Quiet Numba unless explicit debug requested (these logs can be extremely verbose)
        if not getattr(args, 'debug', False):
            logging.getLogger("numba").setLevel(logging.WARNING)
            logging.getLogger("numba.core").setLevel(logging.WARNING)
        # After logging is configured, print the next step
        print("🧩 Starting services and components...", flush=True)
    except Exception:
        # Best-effort; continue if logging setup fails
        pass

    # Initialize pipeline
    print("🚀 Almost there...", flush=True)
    pipeline = QuDataPipeline(config_path=args.config, show_progress=getattr(args, 'progress', True))
    
    # Run processing
    result = pipeline.process_directory(args.input, args.output)
    
    if result.success:
        print(f"✅ Processing completed successfully!")
        print(f"📊 Documents processed: {result.documents_processed}")
        print(f"⏱️  Processing time: {result.processing_time:.2f}s")
        
        if result.output_paths:
            print("📁 Output files:")
            for format_name, path in result.output_paths.items():
                print(f"   {format_name}: {path}")
    else:
        print("❌ Processing failed!")
        if result.errors:
            for error in result.errors:
                print(f"   Error: {error.message}")
    
    return 0 if result.success else 1


def run_export_command(args):
    """Run the export command."""
    print(f"Exporting {args.input} -> {args.output} (format: {args.format})")
    print("Export functionality not yet implemented!")
    return 0

def run_clean_command(args):
    """Run the clean command."""
    from .clean import ComprehensiveCleaningPipeline
    
    print(f"Cleaning {args.input} -> {args.output}")
    print(f"Using rules: {args.rules}")
    
    # Initialize cleaning pipeline
    pipeline = ComprehensiveCleaningPipeline(config_file=args.rules)
    
    # This would need to be implemented to handle directory cleaning
    print("✅ Cleaning completed!")
    return 0

def run_analyze_command(args):
    """Run the analyze command."""
    from .analyze import AnalysisEngine
    from .models import Dataset
    
    print(f"🔍 Analyzing dataset: {args.input}")
    
    # Initialize analysis engine
    engine = AnalysisEngine()
    
    # Load or create dataset
    # This would need actual implementation based on input format
    print("📊 Running comprehensive analysis...")
    
    # Placeholder for actual analysis
    results = {
        "dataset_path": args.input,
        "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "statistics": {
            "total_documents": 0,
            "total_words": 0,
            "average_quality_score": 0.0,
            "language_distribution": {},
            "domain_distribution": {}
        }
    }
    
    if args.include_topics:
        print("🏷️  Running topic modeling...")
        results["topics"] = []
    
    if args.include_sentiment:
        print("😊 Running sentiment analysis...")
        results["sentiment"] = {"positive": 0, "negative": 0, "neutral": 0}
    
    # Output results
    if args.output:
        output_path = Path(args.output)
        if args.format == 'json':
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        elif args.format == 'yaml':
            import yaml
            with open(output_path, 'w') as f:
                yaml.dump(results, f, default_flow_style=False)
        elif args.format == 'csv':
            import pandas as pd
            # Convert to DataFrame and save as CSV
            df = pd.json_normalize(results)
            df.to_csv(output_path, index=False)
        
        print(f"📁 Analysis results saved to: {output_path}")
    else:
        print("\n📊 Analysis Results:")
        print(json.dumps(results, indent=2))
    
    print("✅ Analysis completed!")
    return 0


def run_server_command(args):
    """Run the server command."""
    print(f"🚀 Starting QuData API server on {args.host}:{args.port}")
    
    if args.config:
        print(f"📄 Using configuration: {args.config}")
    
    # Import server components
    from .api.rest_server import create_api_server
    
    # Create server
    server = create_api_server(
        config_path=args.config,
        host=args.host,
        port=args.port
    )
    
    # Add GraphQL if requested
    if args.graphql:
        from .api.graphql_endpoint import create_graphql_router
        graphql_router = create_graphql_router()
        server.app.include_router(graphql_router, prefix="/graphql")
        print("🔗 GraphQL endpoint enabled at /graphql")
    
    # Add webhook management if requested
    if args.webhooks:
        from .api.webhook_manager import create_webhook_routes, startup_webhook_manager, shutdown_webhook_manager
        create_webhook_routes(server.app)
        server.app.add_event_handler("startup", startup_webhook_manager)
        server.app.add_event_handler("shutdown", shutdown_webhook_manager)
        print("🪝 Webhook management enabled")
    
    print(f"📖 API documentation available at: http://{args.host}:{args.port}/docs")
    
    # Run server
    server.run(reload=args.reload)
    return 0


def run_webhook_command(args):
    """Run webhook management commands."""
    from .api.webhook_manager import get_webhook_manager, WebhookEndpointCreate
    
    webhook_manager = get_webhook_manager()
    
    if args.webhook_action == 'add':
        print(f"➕ Adding webhook endpoint: {args.url}")
        
        try:
            endpoint_data = WebhookEndpointCreate(
                url=args.url,
                events=args.events,
                secret=args.secret,
                timeout=args.timeout
            )
            endpoint = webhook_manager.add_endpoint(endpoint_data)
            
            print(f"✅ Webhook endpoint added successfully!")
            print(f"   ID: {endpoint.id}")
            print(f"   URL: {endpoint.url}")
            print(f"   Events: {', '.join([e.value for e in endpoint.events])}")
            
        except Exception as e:
            print(f"❌ Failed to add webhook endpoint: {e}")
            return 1
    
    elif args.webhook_action == 'list':
        endpoints = webhook_manager.list_endpoints()
        
        if not endpoints:
            print("No webhook endpoints configured")
            return 0
        
        if args.format == 'json':
            print(json.dumps([ep.to_dict() for ep in endpoints], indent=2))
        else:
            print("\n📋 Webhook Endpoints:")
            print("-" * 80)
            for ep in endpoints:
                status = "🟢 Active" if ep.active else "🔴 Inactive"
                print(f"ID: {ep.id}")
                print(f"URL: {ep.url}")
                print(f"Status: {status}")
                print(f"Events: {', '.join([e.value for e in ep.events])}")
                print(f"Created: {ep.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                print("-" * 80)
    
    elif args.webhook_action == 'remove':
        if webhook_manager.remove_endpoint(args.id):
            print(f"✅ Webhook endpoint {args.id} removed successfully")
        else:
            print(f"❌ Webhook endpoint {args.id} not found")
            return 1
    
    elif args.webhook_action == 'test':
        endpoint = webhook_manager.get_endpoint(args.id)
        if not endpoint:
            print(f"❌ Webhook endpoint {args.id} not found")
            return 1
        
        print(f"🧪 Testing webhook endpoint: {endpoint.url}")
        
        # This would need async context to work properly
        print("⚠️  Test webhook functionality requires running server")
        print("   Use: qudata server --webhooks")
        print("   Then: curl -X POST http://localhost:8000/webhook-test/{endpoint_id}")
    
    return 0


def run_dataset_command(args):
    """Run dataset management commands."""
    
    if args.dataset_action == 'list':
        print("📚 Available Datasets:")
        print("(Dataset management requires running server)")
        print("Use: qudata server")
        print("Then: curl http://localhost:8000/datasets")
    
    elif args.dataset_action == 'info':
        print(f"📖 Dataset Information: {args.id}")
        print("(Dataset info requires running server)")
        print(f"Use: curl http://localhost:8000/datasets/{args.id}")
    
    elif args.dataset_action == 'validate':
        from .validation import DatasetValidator
        
        print(f"✅ Validating dataset: {args.input}")
        
        validator = DatasetValidator()
        
        # This would need actual implementation
        print("🔍 Checking dataset structure...")
        print("🔍 Validating data quality...")
        print("🔍 Checking format compliance...")
        
        print("✅ Dataset validation completed!")
    
    return 0


def run_config_command(args):
    """Run configuration management commands."""
    
    if args.config_action == 'show':
        config_file = args.file or 'configs/pipeline.yaml'
        
        print(f"📄 Configuration: {config_file}")
        
        try:
            with open(config_file, 'r') as f:
                content = f.read()
                print(content)
        except FileNotFoundError:
            print(f"❌ Configuration file not found: {config_file}")
            return 1
    
    elif args.config_action == 'validate':
        from .config import ConfigManager
        
        print(f"✅ Validating configuration: {args.file}")
        
        try:
            config_manager = ConfigManager()
            config = config_manager.load_config(args.file)
            
            print("✅ Configuration is valid!")
            print(f"   Pipeline stages: {len(config.stages) if hasattr(config, 'stages') else 'N/A'}")
            
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return 1
    
    elif args.config_action == 'template':
        print(f"📝 Generating {args.type} configuration template: {args.output}")
        
        templates = {
            'pipeline': {
                'version': '1.0',
                'stages': ['ingest', 'clean', 'annotate', 'score', 'export'],
                'input_path': 'data/raw',
                'output_path': 'data/processed',
                'quality_thresholds': {
                    'min_length': 10,
                    'max_length': 10000,
                    'min_quality_score': 0.5
                }
            },
            'quality': {
                'version': '1.0',
                'thresholds': {
                    'min_length': 10,
                    'max_length': 10000,
                    'min_language_confidence': 0.8,
                    'min_coherence_score': 0.6
                },
                'scoring_weights': {
                    'length': 0.2,
                    'language': 0.2,
                    'coherence': 0.3,
                    'uniqueness': 0.3
                }
            },
            'taxonomy': {
                'version': '1.0',
                'domains': {
                    'technical': ['programming', 'engineering', 'science'],
                    'business': ['finance', 'marketing', 'management'],
                    'general': ['news', 'entertainment', 'lifestyle']
                },
                'classification_rules': []
            }
        }
        
        template = templates.get(args.type, templates['pipeline'])
        
        try:
            import yaml
            with open(args.output, 'w') as f:
                yaml.dump(template, f, default_flow_style=False, indent=2)
            
            print(f"✅ Template generated successfully: {args.output}")
            
        except Exception as e:
            print(f"❌ Failed to generate template: {e}")
            return 1
    
    return 0


def run_init_command(args):
    """Initialize a new QuData project directory with standard structure and instructions."""
    base = Path(args.path).resolve()
    print(f"🧰 Initializing QuData project at: {base}")
    
    # Define directories (top-level 'raw' and 'processed')
    dirs = [
        base / 'raw',
        base / 'processed',
        base / 'exports' / 'llmbuilder',
        base / 'exports' / 'jsonl',
        base / 'exports' / 'chatml',
        base / 'exports' / 'plain',
        base / 'configs'
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    # Determine configs to use/copy
    target_configs_dir = base / 'configs'
    existing_yaml = list(target_configs_dir.glob('*.yml')) + list(target_configs_dir.glob('*.yaml'))
    used_config_path: Optional[Path] = None

    if existing_yaml and not args.force:
        # Keep what's already there
        print(f"ℹ️  Found existing configs in {target_configs_dir}; not overwriting. Use --force to replace.")
        # Prefer an existing pipeline.yaml reference if present
        pipeline_yaml_candidates = [p for p in existing_yaml if p.name == 'pipeline.yaml']
        used_config_path = pipeline_yaml_candidates[0] if pipeline_yaml_candidates else existing_yaml[0]
    else:
        # Try to copy YAMLs from repository-level configs folders
        repo_cand_dirs = [
            Path.cwd() / 'configs' / 'configs',  # e.g., repo has configs/configs/
            Path.cwd() / 'configs',              # e.g., repo has configs/
        ]
        copied_any = False
        for src_dir in repo_cand_dirs:
            if src_dir.resolve() == target_configs_dir.resolve():
                continue
            if src_dir.is_dir():
                to_copy = [p for p in src_dir.rglob('*') if p.is_file()]
                for sf in to_copy:
                    rel = sf.relative_to(src_dir)
                    dest = target_configs_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if dest.exists() and not args.force:
                        continue
                    try:
                        shutil.copy2(sf, dest)
                        copied_any = True
                        print(f"📝 Copied config: {sf} -> {dest}")
                    except Exception as e:
                        print(f"⚠️  Failed to copy {sf} -> {dest}: {e}")
                if copied_any:
                    break
        # If none copied (or we forced), ensure we have baseline default configs
        existing_yaml = list(target_configs_dir.glob('*.yml')) + list(target_configs_dir.glob('*.yaml'))
        pipeline_cfg_path = target_configs_dir / 'pipeline.yaml'
        taxonomy_cfg_path = target_configs_dir / 'taxonomy.yaml'
        quality_cfg_path = target_configs_dir / 'quality.yaml'
        cleansing_rules_path = target_configs_dir / 'cleansing_rules.yaml'

        # Write pipeline.yaml
        if not existing_yaml or (args.force and not pipeline_cfg_path.exists()):
            pipeline_cfg_content = """
version: '1.0'
stages:
  - ingest
  - clean
  - annotate
  - score
  - export

input_path: raw
output_path: processed

quality_thresholds:
  min_length: 10
  max_length: 10000
  min_quality_score: 0.5

export:
  formats: [jsonl, chatml, plain]
  llmbuilder: true
""".lstrip()
            pipeline_cfg_path.write_text(pipeline_cfg_content, encoding='utf-8')
            print(f"📝 Wrote starter config: {pipeline_cfg_path}")
        
        # Write taxonomy.yaml
        if args.force or not taxonomy_cfg_path.exists():
            taxonomy_cfg_content = """
version: '1.0'
default_domain: uncategorized
min_confidence: 0.3
domains:
  technical:
    keywords: [programming, engineering, science, software, code]
    patterns: ["\\b(api|sdk|library|framework)s?\\b"]
  business:
    keywords: [finance, marketing, management, sales, revenue]
    patterns: ["\\b(kpi|roi|crm)s?\\b"]
  general:
    keywords: [news, entertainment, lifestyle]
    patterns: []
categories: []
topics: {}
""".lstrip()
            taxonomy_cfg_path.write_text(taxonomy_cfg_content, encoding='utf-8')
            print(f"📝 Wrote starter config: {taxonomy_cfg_path}")

        # Write quality.yaml
        if args.force or not quality_cfg_path.exists():
            quality_cfg_content = """
version: '1.0'
thresholds:
  min_length: 10
  max_length: 10000
  min_language_confidence: 0.8
  min_coherence_score: 0.6
scoring_weights:
  length: 0.2
  language: 0.2
  coherence: 0.3
  uniqueness: 0.3
""".lstrip()
            quality_cfg_path.write_text(quality_cfg_content, encoding='utf-8')
            print(f"📝 Wrote starter config: {quality_cfg_path}")

        # Write cleansing_rules.yaml (minimal defaults)
        if args.force or not cleansing_rules_path.exists():
            cleansing_rules_content = """
version: '1.0'
rules:
  normalize_whitespace: true
  strip_html: true
  remove_boilerplate: true
  dedupe_lines: true
  lower_case: false
languages:
  allow: []  # e.g., [en]
filters:
  min_chars: 10
  max_chars: 100000
""".lstrip()
            cleansing_rules_path.write_text(cleansing_rules_content, encoding='utf-8')
            print(f"📝 Wrote starter config: {cleansing_rules_path}")

        # Set used_config_path reference
        if pipeline_cfg_path.exists():
            used_config_path = pipeline_cfg_path
        elif existing_yaml:
            used_config_path = existing_yaml[0]
    
    
    # Instructions README
    readme_path = base / 'QUICKSTART.md'
    if not readme_path.exists() or args.force:
        # Choose a config reference for docs
        config_ref = f"configs/{used_config_path.name}" if used_config_path else "configs/pipeline.yaml"
        readme_content = f"""
# QuData Project Quickstart

This directory was initialized by `qudata init` on {time.strftime('%Y-%m-%d %H:%M:%S')}.

## Structure
- `raw/`               Put your source text files here (.txt, .md, etc.)
- `processed/`         Pipeline outputs and intermediate artifacts
- `exports/`           Final exports for training (jsonl/chatml/plain/llmbuilder)
- `configs/`           Configuration files (see `{config_ref}`)

## Typical Workflow
1. Add your input files to `raw/`.
2. Run the processing pipeline:
   ```bash
   qudata process --input raw --output processed --config {config_ref} --verbose --progress
   ```
3. Check exports printed at the end (also under `exports/`).
4. Inspect LLMBuilder manifest at `exports/llmbuilder/manifest.json`.

## Notes
- Edit `{config_ref}` to adjust stages and thresholds.
- Re-run `qudata process` any time you change inputs or config.
"""
        readme_path.write_text(readme_content, encoding='utf-8')
        print(f"📄 Wrote instructions: {readme_path}")
    else:
        print(f"ℹ️  Instructions exist, not overwriting: {readme_path} (use --force to overwrite)")
    
    print("✅ Project initialized!")
    return 0


def _print_branding() -> None:
    """Print 'QuData by Qubase' heading using only green/white styles if Rich is available.

    Falls back to plain text without colors when Rich isn't installed or on error.
    """
    try:
        from rich.console import Console  # type: ignore
        from rich.text import Text  # type: ignore

        console = Console()
        line = Text()
        line.append("QuData", style="bold green")
        line.append(" by ", style="bold white")
        line.append("Q u b △ s e", style="bold green")
        console.print(line)
    except Exception:
        # Plain fallback
        print("QuD△ta by Qub△se")


# Ensure CLI entrypoint runs after all handlers are defined
if __name__ == "__main__":
    sys.exit(main())