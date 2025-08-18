#!/usr/bin/env python3
"""
QuData Command Line Tutorial

This script teaches users how to use QuData from the command line
with step-by-step examples and explanations.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

def print_header(title: str):
    """Print a tutorial section header."""
    print("\n" + "📚" + "=" * 60)
    print(f"TUTORIAL: {title}")
    print("=" * 61)

def print_command(description: str, command: str, explanation: str = ""):
    """Print a command with explanation."""
    print(f"\n🔸 {description}")
    print(f"💻 Command: {command}")
    if explanation:
        print(f"📝 Explanation: {explanation}")

def print_tip(tip: str):
    """Print a helpful tip."""
    print(f"💡 TIP: {tip}")

def print_example_output(output: str):
    """Print example command output."""
    print("📤 Example Output:")
    print("   " + output.replace("\n", "\n   "))

def create_sample_data():
    """Create sample data for the tutorial."""
    print_header("Setting Up Sample Data")
    
    # Create directory structure
    directories = ["tutorial_data/raw", "tutorial_data/processed", "tutorial_data/exports"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Create sample files
    sample_files = {
        "tutorial_data/raw/article1.txt": """
        The Rise of Artificial Intelligence in Healthcare
        
        Artificial Intelligence (AI) is transforming healthcare by enabling 
        more accurate diagnoses, personalized treatments, and efficient 
        drug discovery processes.
        
        Key applications include:
        - Medical imaging analysis
        - Drug discovery acceleration
        - Personalized treatment plans
        - Predictive analytics for patient outcomes
        
        The future of healthcare will be increasingly AI-driven, with 
        human doctors and AI systems working together to provide 
        better patient care.
        """,
        
        "tutorial_data/raw/article2.txt": """
        Climate Change and Renewable Energy Solutions
        
        Climate change represents one of the most pressing challenges 
        of our time. However, renewable energy technologies offer 
        promising solutions.
        
        Renewable energy sources include:
        - Solar power systems
        - Wind energy generation
        - Hydroelectric power
        - Geothermal energy
        
        The transition to renewable energy is not just an environmental 
        necessity but also an economic opportunity for sustainable growth.
        """,
        
        "tutorial_data/raw/short_note.txt": """
        This is a very short document that might not meet quality thresholds.
        """,
        
        "tutorial_data/raw/duplicate_content.txt": """
        The Rise of Artificial Intelligence in Healthcare
        
        Artificial Intelligence (AI) is transforming healthcare by enabling 
        more accurate diagnoses, personalized treatments, and efficient 
        drug discovery processes.
        
        This is essentially the same content as article1.txt to demonstrate 
        deduplication features.
        """
    }
    
    for file_path, content in sample_files.items():
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
    
    print("✅ Sample data created in tutorial_data/ directory")
    print(f"   Created {len(sample_files)} sample files")
    print("   Directory structure:")
    print("   tutorial_data/")
    print("   ├── raw/           (input files)")
    print("   ├── processed/     (cleaned files)")
    print("   └── exports/       (final datasets)")

def tutorial_basic_commands():
    """Tutorial for basic QuData commands."""
    print_header("Basic QuData Commands")
    
    print_command(
        "Check if QuData is installed",
        "qudata --version",
        "This shows the installed version of QuData"
    )
    print_example_output("QuData version 1.0.0")
    
    print_command(
        "Get help for any command",
        "qudata --help",
        "Shows all available commands and options"
    )
    print_example_output("""
Usage: qudata [OPTIONS] COMMAND [ARGS]...

Commands:
  process   Process documents
  export    Export processed data
  analyze   Analyze dataset quality
  server    Start API server
  config    Configuration management
    """)
    
    print_command(
        "Get help for specific commands",
        "qudata process --help",
        "Shows detailed help for the process command"
    )
    
    print_tip("Use --help with any command to see all available options")

def tutorial_processing():
    """Tutorial for document processing."""
    print_header("Document Processing")
    
    print_command(
        "Process all files in a directory (basic)",
        "qudata process --input tutorial_data/raw --output tutorial_data/processed",
        "Processes all supported files from input directory to output directory"
    )
    print_example_output("""
Processing files from tutorial_data/raw...
✓ Processing article1.txt... (Quality: 0.85)
✓ Processing article2.txt... (Quality: 0.82)
⚠ Processing short_note.txt... (Quality: 0.45 - below threshold)
✓ Processing duplicate_content.txt... (Duplicate detected - skipped)
Processed 2 documents successfully, 2 skipped
    """)
    
    print_command(
        "Process with verbose output",
        "qudata process --input tutorial_data/raw --output tutorial_data/processed --verbose",
        "Shows detailed processing information for debugging"
    )
    
    print_command(
        "Process specific file types only",
        "qudata process --input tutorial_data/raw --output tutorial_data/processed --format txt",
        "Only processes .txt files, ignoring other formats"
    )
    
    print_command(
        "Process with custom configuration",
        "qudata process --input tutorial_data/raw --output tutorial_data/processed --config my_config.yaml",
        "Uses custom settings from configuration file"
    )
    
    print_command(
        "Process with parallel processing",
        "qudata process --input tutorial_data/raw --output tutorial_data/processed --parallel 4",
        "Uses 4 CPU cores for faster processing of large datasets"
    )
    
    print_tip("Start with basic processing, then add options as needed")

def tutorial_export():
    """Tutorial for data export."""
    print_header("Data Export")
    
    print_command(
        "Export to JSONL format (most common)",
        "qudata export --format jsonl --input tutorial_data/processed --output tutorial_data/exports/training.jsonl",
        "Creates a JSONL file suitable for AI training"
    )
    print_example_output("""
Exporting 2 documents to JSONL format...
✓ Export completed: tutorial_data/exports/training.jsonl
File size: 1.2 MB
    """)
    
    print_command(
        "Export with train/validation/test splits",
        "qudata export --format jsonl --input tutorial_data/processed --output tutorial_data/exports --split",
        "Creates separate files for training, validation, and testing"
    )
    print_example_output("""
Creating dataset splits...
✓ train.jsonl: 8 documents (80%)
✓ validation.jsonl: 1 document (10%)
✓ test.jsonl: 1 document (10%)
    """)
    
    print_command(
        "Export to multiple formats",
        "qudata export --format csv --input tutorial_data/processed --output tutorial_data/exports/analysis.csv",
        "Exports metadata and statistics in CSV format for analysis"
    )
    
    print_command(
        "Export to ChatML format for conversational AI",
        "qudata export --format chatml --input tutorial_data/processed --output tutorial_data/exports/chat_training.json",
        "Creates conversational training data format"
    )
    
    print_tip("JSONL is the most versatile format for AI training")

def tutorial_analysis():
    """Tutorial for data analysis."""
    print_header("Data Analysis")
    
    print_command(
        "Basic quality analysis",
        "qudata analyze --input tutorial_data/processed --output analysis.json",
        "Generates quality metrics and statistics"
    )
    print_example_output("""
Analyzing 2 documents...
✓ Analysis completed: analysis.json
Average quality score: 0.84
Language distribution: en (100%)
Total words: 1,247
    """)
    
    print_command(
        "Comprehensive analysis with topics",
        "qudata analyze --input tutorial_data/processed --output analysis.json --include-topics",
        "Includes topic modeling in the analysis"
    )
    
    print_command(
        "Analysis with sentiment scoring",
        "qudata analyze --input tutorial_data/processed --output analysis.json --include-sentiment",
        "Adds sentiment analysis to the report"
    )
    
    print_command(
        "Export analysis as CSV",
        "qudata analyze --input tutorial_data/processed --output analysis.csv --format csv",
        "Creates analysis report in spreadsheet format"
    )
    
    print_tip("Use analysis results to optimize your processing configuration")

def tutorial_configuration():
    """Tutorial for configuration management."""
    print_header("Configuration Management")
    
    print_command(
        "Generate configuration template",
        "qudata config template --output my_config.yaml",
        "Creates a template configuration file you can customize"
    )
    print_example_output("""
✓ Configuration template created: my_config.yaml
Edit this file to customize processing behavior
    """)
    
    print_command(
        "Validate configuration file",
        "qudata config validate --file my_config.yaml",
        "Checks if your configuration file is valid"
    )
    print_example_output("""
Validating configuration...
✓ Configuration is valid
No errors found
    """)
    
    print_command(
        "Show current configuration",
        "qudata config show --file my_config.yaml",
        "Displays the current configuration settings"
    )
    
    print_tip("Always validate configuration files before using them in production")

def tutorial_server():
    """Tutorial for API server."""
    print_header("API Server")
    
    print_command(
        "Start basic API server",
        "qudata server --host 0.0.0.0 --port 8000",
        "Starts REST API server for programmatic access"
    )
    print_example_output("""
Starting QuData API server...
✓ Server running at http://0.0.0.0:8000
✓ API documentation at http://0.0.0.0:8000/docs
Press Ctrl+C to stop
    """)
    
    print_command(
        "Start server with all features",
        "qudata server --host 0.0.0.0 --port 8000 --graphql --webhooks",
        "Enables GraphQL endpoint and webhook support"
    )
    
    print_command(
        "Start server with custom configuration",
        "qudata server --config api_config.yaml --port 8000",
        "Uses custom API configuration settings"
    )
    
    print_tip("The API server allows integration with other systems and web interfaces")

def tutorial_advanced():
    """Tutorial for advanced features."""
    print_header("Advanced Features")
    
    print_command(
        "Process with quality filtering",
        "qudata process --input tutorial_data/raw --output tutorial_data/processed --min-quality 0.7",
        "Only keeps documents with quality score >= 0.7"
    )
    
    print_command(
        "Process with language filtering",
        "qudata process --input tutorial_data/raw --output tutorial_data/processed --language en",
        "Only processes English documents"
    )
    
    print_command(
        "Process with custom batch size",
        "qudata process --input tutorial_data/raw --output tutorial_data/processed --batch-size 50",
        "Processes 50 documents at a time (useful for memory management)"
    )
    
    print_command(
        "Enable streaming mode for large files",
        "qudata process --input tutorial_data/raw --output tutorial_data/processed --streaming",
        "Uses streaming processing to handle very large files"
    )
    
    print_command(
        "Process with checkpointing",
        "qudata process --input tutorial_data/raw --output tutorial_data/processed --checkpoint-every 100",
        "Saves progress every 100 documents for recovery"
    )
    
    print_tip("Advanced options help optimize processing for specific use cases")

def tutorial_troubleshooting():
    """Tutorial for troubleshooting."""
    print_header("Troubleshooting")
    
    print("🔧 Common Issues and Solutions:")
    
    print("\n❌ Problem: Command not found")
    print("💡 Solution: Make sure QuData is installed: pip install -e .")
    print("💻 Test: qudata --version")
    
    print("\n❌ Problem: Out of memory errors")
    print("💡 Solution: Reduce batch size and enable streaming")
    print("💻 Command: qudata process --input data --output processed --batch-size 10 --streaming")
    
    print("\n❌ Problem: Processing is too slow")
    print("💡 Solution: Enable parallel processing")
    print("💻 Command: qudata process --input data --output processed --parallel 4")
    
    print("\n❌ Problem: Low quality scores")
    print("💡 Solution: Lower the quality threshold")
    print("💻 Command: qudata process --input data --output processed --min-quality 0.4")
    
    print("\n❌ Problem: Files not being processed")
    print("💡 Solution: Check supported formats and enable verbose logging")
    print("💻 Command: qudata process --input data --output processed --verbose")
    
    print("\n❌ Problem: Configuration errors")
    print("💡 Solution: Validate your configuration file")
    print("💻 Command: qudata config validate --file my_config.yaml")
    
    print_tip("Always start with verbose logging when troubleshooting: --verbose")

def tutorial_workflows():
    """Tutorial for common workflows."""
    print_header("Common Workflows")
    
    print("🔄 Workflow 1: Basic Document Processing")
    print("1. qudata process --input raw_docs --output processed_docs")
    print("2. qudata analyze --input processed_docs --output analysis.json")
    print("3. qudata export --format jsonl --input processed_docs --output training.jsonl")
    
    print("\n🔄 Workflow 2: Academic Paper Processing")
    print("1. qudata config template --output academic_config.yaml")
    print("2. # Edit academic_config.yaml for academic papers")
    print("3. qudata process --input papers --output processed --config academic_config.yaml")
    print("4. qudata export --format jsonl --input processed --output academic_training.jsonl --split")
    
    print("\n🔄 Workflow 3: Web Content Processing")
    print("1. # Download web content to web_content/ directory")
    print("2. qudata process --input web_content --output processed --format html")
    print("3. qudata analyze --input processed --output web_analysis.json --include-topics")
    print("4. qudata export --format chatml --input processed --output chat_data.json")
    
    print("\n🔄 Workflow 4: Quality Control Pipeline")
    print("1. qudata process --input docs --output processed --min-quality 0.8")
    print("2. qudata analyze --input processed --output quality_report.json")
    print("3. # Review quality_report.json")
    print("4. qudata export --format jsonl --input processed --output high_quality.jsonl")
    
    print_tip("Create shell scripts to automate your common workflows")

def create_cheat_sheet():
    """Create a command-line cheat sheet."""
    print_header("Command Line Cheat Sheet")
    
    cheat_sheet = """
# QuData Command Line Cheat Sheet

## Essential Commands
qudata --version                    # Check version
qudata --help                       # Get help
qudata COMMAND --help               # Get help for specific command

## Processing
qudata process --input DIR --output DIR                    # Basic processing
qudata process --input DIR --output DIR --verbose         # With detailed output
qudata process --input DIR --output DIR --parallel 4      # Parallel processing
qudata process --input DIR --output DIR --config FILE     # Custom configuration

## Export
qudata export --format jsonl --input DIR --output FILE    # Export to JSONL
qudata export --format jsonl --input DIR --output DIR --split  # With splits
qudata export --format csv --input DIR --output FILE      # Export to CSV
qudata export --format chatml --input DIR --output FILE   # Export to ChatML

## Analysis
qudata analyze --input DIR --output FILE                  # Basic analysis
qudata analyze --input DIR --output FILE --include-topics # With topics
qudata analyze --input DIR --output FILE --format csv     # CSV format

## Configuration
qudata config template --output FILE                      # Create template
qudata config validate --file FILE                        # Validate config
qudata config show --file FILE                           # Show config

## Server
qudata server --host 0.0.0.0 --port 8000                # Start API server
qudata server --port 8000 --graphql --webhooks          # Full features

## Advanced Options
--min-quality 0.7          # Quality threshold
--language en              # Language filter
--batch-size 50           # Batch size
--streaming               # Streaming mode
--checkpoint-every 100    # Checkpointing
--format pdf              # File type filter
--parallel 4              # Parallel workers
--verbose                 # Detailed output
--config FILE             # Custom configuration
"""
    
    # Save cheat sheet to file
    cheat_sheet_path = Path("tutorial_data/qudata_cheat_sheet.txt")
    with open(cheat_sheet_path, 'w', encoding='utf-8') as f:
        f.write(cheat_sheet.strip())
    
    print(cheat_sheet)
    print(f"\n✅ Cheat sheet saved to: {cheat_sheet_path}")
    print("💡 Keep this file handy for quick reference!")

def main():
    """Run the complete command line tutorial."""
    print("📚 QuData Command Line Tutorial")
    print("Learn how to use QuData from the command line with practical examples")
    print("=" * 70)
    
    try:
        # Create sample data first
        create_sample_data()
        
        # Run tutorial sections
        tutorial_basic_commands()
        tutorial_processing()
        tutorial_export()
        tutorial_analysis()
        tutorial_configuration()
        tutorial_server()
        tutorial_advanced()
        tutorial_troubleshooting()
        tutorial_workflows()
        create_cheat_sheet()
        
        print("\n" + "🎉" + "=" * 69)
        print("TUTORIAL COMPLETE!")
        print("=" * 70)
        
        print("\n📋 What You've Learned:")
        print("✅ Basic QuData commands and options")
        print("✅ Document processing workflows")
        print("✅ Data export in multiple formats")
        print("✅ Quality analysis and reporting")
        print("✅ Configuration management")
        print("✅ API server setup")
        print("✅ Advanced processing options")
        print("✅ Troubleshooting common issues")
        print("✅ Real-world workflow examples")
        
        print("\n🚀 Next Steps:")
        print("• Try the commands with your own data")
        print("• Experiment with different configurations")
        print("• Set up automated workflows with shell scripts")
        print("• Explore the API server for integration projects")
        
        print("\n📁 Files Created:")
        print("• tutorial_data/: Sample data and processed results")
        print("• tutorial_data/qudata_cheat_sheet.txt: Quick reference")
        
        print("\n💡 Pro Tips:")
        print("• Always start with --verbose when learning")
        print("• Use --help to explore command options")
        print("• Test with small datasets first")
        print("• Keep the cheat sheet handy for reference")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tutorial interrupted by user")
        print("You can run this script again anytime to continue learning!")
    except Exception as e:
        print(f"\n❌ Tutorial failed: {e}")
        print("This might be due to missing dependencies or system issues")

if __name__ == "__main__":
    main()