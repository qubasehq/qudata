Skip to content
Navigation Menu
qubasehq
llmbuilder-package

Type / to search
Code
Issues
Pull requests
Actions
Projects
Security
Insights
Settings
Owner avatar
llmbuilder-package
Public
qubasehq/llmbuilder-package
Go to file
t
Name		
loyality7
loyality7
updated enhanced fetures and everything working fine V:0.4.6
e9096b6
 · 
2 days ago
.github
updated enhanced fetures and everything working fine V:0.4.6
2 days ago
docs
updated enhanced fetures and everything working fine V:0.4.6
2 days ago
llmbuilder
updated enhanced fetures and everything working fine V:0.4.6
2 days ago
scripts
updated enhanced fetures and everything working fine V:0.4.6
2 days ago
site
updated enhanced fetures and everything working fine V:0.4.6
2 days ago
.gitignore
updated enhanced fetures and everything working fine V:0.4.6
2 days ago
CLI.md
updated enhanced fetures and everything working fine V:0.4.6
2 days ago
CONTRIBUTING.md
updated enhanced fetures and everything working fine V:0.4.6
2 days ago
MANIFEST.in
updated enhanced fetures and everything working fine V:0.4.6
2 days ago
MIGRATION.md
updated enhanced fetures and everything working fine V:0.4.6
2 days ago
NON_TECH_README.md
updated enhanced fetures and everything working fine V:0.4.6
2 days ago
README.md
updated enhanced fetures and everything working fine V:0.4.6
2 days ago
bandit-report.json
updated enhanced fetures and everything working fine V:0.4.6
2 days ago
custom_config.json
updated enhanced fetures and everything working fine V:0.4.6
2 days ago
mkdocs.yml
updated enhanced fetures and everything working fine V:0.4.6
2 days ago
pyproject.toml
updated enhanced fetures and everything working fine V:0.4.6
2 days ago
setup.py
updated enhanced fetures and everything working fine V:0.4.6
2 days ago
Repository files navigation
README
Contributing
🤖 LLMBuilder
Documentation Python 3.8+ License: MIT

A comprehensive toolkit for building, training, fine-tuning, and deploying GPT-style language models with advanced data processing capabilities and CPU-friendly defaults.

About LLMBuilder Framework
LLMBuilder is a production-ready framework for training and fine-tuning Large Language Models (LLMs) — not a model itself. Designed for developers, researchers, and AI engineers, LLMBuilder provides a full pipeline to go from raw text data to deployable, optimized LLMs, all running locally on CPUs or GPUs.

Complete Framework Structure
The full LLMBuilder framework includes:

LLMBuilder/
├── data/                   # Data directories
│   ├── raw/               # Raw input files (.txt, .pdf, .docx)
│   ├── cleaned/           # Processed text files
│   └── tokens/            # Tokenized datasets
│   ├── download_data.py   # Script to download datasets
│   └── SOURCES.md         # Data sources documentation
│
├── debug_scripts/         # Debugging utilities
│   ├── debug_loader.py    # Data loading debugger
│   ├── debug_training.py  # Training process debugger
│   └── debug_timestamps.py # Timing analysis
│
├── eval/                  # Model evaluation
│   └── eval.py           # Evaluation scripts
│
├── exports/               # Output directories
│   ├── checkpoints/      # Training checkpoints
│   ├── gguf/             # GGUF model exports
│   ├── onnx/             # ONNX model exports
│   └── tokenizer/        # Saved tokenizer files
│
├── finetune/             # Fine-tuning scripts
│   ├── finetune.py      # Fine-tuning implementation
│   └── __init__.py      # Package initialization
│
├── logs/                 # Training and evaluation logs
│
├── model/                # Model architecture
│   └── gpt_model.py     # GPT model implementation
│
├── tools/                # Utility scripts
│   ├── analyze_data.ps1  # PowerShell data analysis
│   ├── analyze_data.sh   # Bash data analysis
│   ├── download_hf_model.py # HuggingFace model downloader
│   └── export_gguf.py    # GGUF export utility
│
├── training/             # Training pipeline
│   ├── dataset.py       # Dataset handling
│   ├── preprocess.py    # Data preprocessing
│   ├── quantization.py  # Model quantization
│   ├── train.py         # Main training script
│   ├── train_tokenizer.py # Tokenizer training
│   └── utils.py         # Training utilities
│
├── .gitignore           # Git ignore rules
├── config.json          # Main configuration
├── config_cpu_small.json # Small CPU config
├── config_gpu.json      # GPU configuration
├── inference.py         # Inference script
├── quantize_model.py    # Model quantization
├── README.md           # Documentation
├── requirements.txt    # Python dependencies
├── run.ps1            # PowerShell runner
└── run.sh             # Bash runner
🔗 Full Framework Repository: https://github.com/Qubasehq/llmbuilder

Note

This is a separate framework - The complete LLMBuilder framework shown above is not related to this package. It's a standalone, comprehensive framework available at the GitHub repository. This package (llmbuilder_package) provides the core modular toolkit, while the complete framework offers additional utilities, debugging tools, and production-ready scripts for comprehensive LLM development workflows.

✨ Key Features
🔄 Advanced Data Processing
Multi-Format Ingestion: Process HTML, Markdown, EPUB, PDF, and text files with intelligent extraction
OCR Integration: Automatic OCR fallback for scanned PDFs using Tesseract
Smart Deduplication: Both exact and semantic duplicate detection with configurable similarity thresholds
Batch Processing: Parallel processing with configurable worker threads and batch sizes
🔤 Flexible Tokenization
Multiple Algorithms: BPE, SentencePiece, Unigram, and WordPiece tokenizers
Custom Training: Train tokenizers on your specific datasets with advanced configuration options
Validation Tools: Built-in tokenizer testing and benchmarking utilities
⚡ Model Conversion & Optimization
GGUF Pipeline: Convert trained models to GGUF format for llama.cpp compatibility
Quantization Options: Support for F32, F16, Q8_0, Q5_1, Q5_0, Q4_1, Q4_0 quantization levels
Batch Conversion: Convert multiple models with different quantization levels simultaneously
Validation: Automatic output validation and integrity checking
⚙️ Configuration Management
Template System: Pre-configured templates for different use cases (CPU, GPU, inference, etc.)
Validation: Comprehensive configuration validation with detailed error reporting
Override Support: Easy configuration customization with dot-notation overrides
CLI Integration: Full command-line configuration management tools
🖥️ Production-Ready CLI
Complete Interface: Full command-line interface for all operations
Interactive Modes: Guided setup and configuration for new users
Progress Tracking: Real-time progress reporting with detailed logging
Batch Operations: Support for processing multiple files and models
🧪 Quality Assurance
Extensive Testing: 200+ unit and integration tests covering all functionality
Performance Tests: Memory usage monitoring and performance benchmarking
CI/CD Pipeline: Automated testing and validation on multiple platforms
Documentation: Comprehensive documentation with examples and troubleshooting guides
🚀 Quick Start
Installation
pip install llmbuilder
Optional Dependencies
LLMBuilder works out of the box, but you can install additional dependencies for advanced features:

# Complete installation with all features
pip install llmbuilder[all]

# Or install specific feature sets:
pip install llmbuilder[pdf]        # PDF processing with OCR
pip install llmbuilder[epub]       # EPUB document processing
pip install llmbuilder[html]       # Advanced HTML processing
pip install llmbuilder[semantic]   # Semantic deduplication
pip install llmbuilder[conversion] # GGUF model conversion

# Manual installation of optional dependencies:
pip install pymupdf pytesseract    # PDF processing with OCR
pip install ebooklib               # EPUB processing
pip install beautifulsoup4 lxml    # HTML processing
pip install sentence-transformers  # Semantic deduplication
System Dependencies
For PDF OCR Processing:

# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
# macOS:
brew install tesseract

# Ubuntu/Debian:
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# Verify installation:
tesseract --version
For GGUF Model Conversion:

# Option 1: Install llama.cpp (recommended)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make

# Option 2: Python package (alternative)
pip install llama-cpp-python

# Verify installation:
llmbuilder convert gguf --help
5-Minute Complete Pipeline
import llmbuilder as lb

# 1. Create configuration from template
from llmbuilder.config.manager import create_config_from_template
config = create_config_from_template("basic_config", {
    "model": {"vocab_size": 16000},
    "data": {
        "ingestion": {"enable_ocr": True, "supported_formats": ["pdf", "html", "txt"]},
        "deduplication": {"similarity_threshold": 0.85}
    }
})

# 2. Process multi-format documents with advanced features
from llmbuilder.data.ingest import IngestionPipeline
pipeline = IngestionPipeline(config.data.ingestion)
pipeline.process_directory("./raw_data", "./processed_data")
print(f"Processed {pipeline.get_stats()['files_processed']} files")

# 3. Advanced deduplication (exact + semantic)
from llmbuilder.data.dedup import DeduplicationPipeline
dedup = DeduplicationPipeline(config.data.deduplication)
stats = dedup.process_file("./processed_data/combined.txt", "./clean_data/deduplicated.txt")
print(f"Removed {stats['duplicates_removed']} duplicates")

# 4. Train custom tokenizer with validation
from llmbuilder.tokenizer import TokenizerTrainer
trainer = TokenizerTrainer(config.tokenizer_training)
results = trainer.train("./clean_data/deduplicated.txt", "./tokenizers")
print(f"Trained tokenizer with {results['vocab_size']} tokens")

# 5. Build and train model
model = lb.build_model(config.model)
from llmbuilder.data import TextDataset
dataset = TextDataset("./clean_data/deduplicated.txt", block_size=config.model.max_seq_length)
training_results = lb.train_model(model, dataset, config.training)
print(f"Training completed. Final loss: {training_results['final_loss']:.4f}")

# 6. Convert to multiple GGUF formats
from llmbuilder.tools.convert_to_gguf import GGUFConverter
converter = GGUFConverter()
quantization_levels = ["Q8_0", "Q4_0", "Q4_1"]
for quant in quantization_levels:
    result = converter.convert_model(
        "./checkpoints/model.pt",
        f"./exports/model_{quant.lower()}.gguf",
        quant
    )
    if result.success:
        print(f"✅ {quant}: {result.file_size_bytes / (1024*1024):.1f} MB")

# 7. Generate text with different sampling strategies
text_creative = lb.generate_text(
    model_path="./checkpoints/model.pt",
    tokenizer_path="./tokenizers",
    prompt="The future of AI is",
    max_new_tokens=100,
    temperature=0.8,  # More creative
    top_p=0.9
)

text_focused = lb.generate_text(
    model_path="./checkpoints/model.pt",
    tokenizer_path="./tokenizers",
    prompt="The future of AI is",
    max_new_tokens=100,
    temperature=0.3,  # More focused
    top_k=40
)

print("Creative:", text_creative)
print("Focused:", text_focused)
📚 Documentation
Complete documentation is available at: https://qubasehq.github.io/llmbuilder-package/

The documentation includes:

📖 Getting Started Guide - From installation to your first model
🎯 User Guides - Comprehensive guides for all features
🖥️ CLI Reference - Complete command-line interface documentation
🐍 Python API - Full API reference with examples
📋 Examples - Working code examples for common tasks
❓ FAQ - Answers to frequently asked questions
CLI Quickstart
Getting Started
# Show help and available commands
llmbuilder --help

# Interactive welcome guide for new users
llmbuilder welcome

# Show package information and credits
llmbuilder info
Data Processing Pipeline
# Multi-format document ingestion with OCR
llmbuilder data load -i ./documents -o ./processed.txt --format all --clean --enable-ocr

# Advanced deduplication (exact + semantic)
llmbuilder data deduplicate -i ./processed.txt -o ./clean.txt --method both --threshold 0.85

# Train custom tokenizer with validation
llmbuilder data tokenizer -i ./clean.txt -o ./tokenizers --algorithm sentencepiece --vocab-size 16000
Configuration Management
# List available configuration templates
llmbuilder config templates

# Create configuration from template with overrides
llmbuilder config from-template advanced_processing_config -o my_config.json \
  --override model.vocab_size=24000 \
  --override training.batch_size=32

# Validate configuration with detailed reporting
llmbuilder config validate my_config.json --detailed

# Show comprehensive configuration summary
llmbuilder config summary my_config.json
Model Training & Operations
# Train model with configuration file
llmbuilder train model --config my_config.json --data ./clean.txt --tokenizer ./tokenizers --output ./checkpoints

# Interactive text generation setup
llmbuilder generate text --setup

# Generate text with custom parameters
llmbuilder generate text -m ./checkpoints/model.pt -t ./tokenizers -p "Hello world" --temperature 0.8 --max-tokens 100
GGUF Model Conversion
# Convert single model with validation
llmbuilder convert gguf ./checkpoints/model.pt -o ./exports/model.gguf -q Q8_0 --validate

# Convert with all quantization levels
llmbuilder convert gguf ./checkpoints/model.pt -o ./exports/model.gguf --all-quantizations

# Batch convert multiple models
llmbuilder convert batch -i ./models -o ./exports -q Q8_0 Q4_0 Q4_1 --pattern "*.pt"
Advanced Operations
# Process large datasets with custom settings
llmbuilder data load -i ./large_docs -o ./processed.txt --batch-size 200 --workers 8 --format pdf,html

# Semantic deduplication with GPU acceleration
llmbuilder data deduplicate -i ./dataset.txt -o ./clean.txt --method semantic --use-gpu --batch-size 2000

# Train tokenizer with advanced options
llmbuilder data tokenizer -i ./corpus.txt -o ./tokenizers \
  --algorithm sentencepiece \
  --vocab-size 32000 \
  --character-coverage 0.9998 \
  --special-tokens "<pad>,<unk>,<s>,</s>,<mask>"
Python API Quickstart
import llmbuilder as lb

# Load a preset config and build a model
cfg = lb.load_config(preset="cpu_small")
model = lb.build_model(cfg.model)

# Train (example; see examples/train_tiny.py for a runnable script)
from llmbuilder.data import TextDataset
dataset = TextDataset("./data/clean.txt", block_size=cfg.model.max_seq_length)
results = lb.train_model(model, dataset, cfg.training)

# Generate text
text = lb.generate_text(
    model_path="./checkpoints/model.pt",
    tokenizer_path="./tokenizers",
    prompt="Hello world",
    max_new_tokens=50,
)
print(text)
Full Example Script: docs/train_model.py
"""
Example: Train a small GPT model on cybersecurity text files using LLMBuilder.

Usage:
  python docs/train_model.py --data_dir ./Model_Test --output_dir ./Model_Test/output \
      --prompt "Cybersecurity is important because" --epochs 5

If --data_dir is omitted, it defaults to the directory containing this script.
If --output_dir is omitted, it defaults to <data_dir>/output.

This script uses small-friendly settings (block_size=64, batch_size=1) so it
works on tiny datasets. It trains, saves checkpoints, and performs a sample
text generation from the latest/best checkpoint.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import llmbuilder


def main():
    parser = argparse.ArgumentParser(description="Train and generate with LLMBuilder on small text datasets.")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory with .txt files (default: folder of this script)")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to save outputs (default: <data_dir>/output)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size (small data friendly)")
    parser.add_argument("--block_size", type=int, default=64, help="Context window size for training")
    parser.add_argument("--embed_dim", type=int, default=256, help="Model embedding dimension")
    parser.add_argument("--layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--prompt", type=str, default="Cybersecurity is important because", help="Prompt for sample generation")
    parser.add_argument("--max_new_tokens", type=int, default=80, help="Tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling top_p")
    args = parser.parse_args()

    # Resolve paths
    if args.data_dir is None:
        data_dir = Path(__file__).parent
    else:
        data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else (data_dir / "output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Configs mapped to llmbuilder expected keys
    config = {
        # tokenizer/dataset convenience
        "vocab_size": 8000,
        "block_size": int(args.block_size),
        # training config -> llmbuilder.config.TrainingConfig
        "training": {
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.lr),
            "num_epochs": int(args.epochs),
            "max_grad_norm": 1.0,
            "save_every": 1,
            "log_every": 10,
        },
        # model config -> llmbuilder.config.ModelConfig
        "model": {
            "embedding_dim": int(args.embed_dim),
            "num_layers": int(args.layers),
            "num_heads": int(args.heads),
            "max_seq_length": int(args.block_size),
            "dropout": 0.1,
        },
    }

    print("Starting LLMBuilder training pipeline...")
    pipeline = llmbuilder.train(
        data_path=str(data_dir),
        output_dir=str(output_dir),
        config=config,
        clean=False,
    )

    # Generation
    best_ckpt = output_dir / "checkpoints" / "best_checkpoint.pt"
    latest_ckpt = output_dir / "checkpoints" / "latest_checkpoint.pt"
    model_ckpt = best_ckpt if best_ckpt.exists() else latest_ckpt
    tokenizer_dir = output_dir / "tokenizer"

    if model_ckpt.exists() and tokenizer_dir.exists():
        print("\nGenerating sample text with trained model...")
        text = llmbuilder.generate_text(
            model_path=str(model_ckpt),
            tokenizer_path=str(tokenizer_dir),
            prompt=args.prompt,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
        )
        print("\nSample generation:\n" + text)
    else:
        print("\nSkipping generation because artifacts were not found.")


if __name__ == "__main__":
    main()
More
Examples: see the examples/ folder
examples/generate_text.py
examples/train_tiny.py
Migration from older scripts: see MIGRATION.md
For Developers and Advanced Users
Python API quickstart:

import llmbuilder as lb
cfg = lb.load_config(preset="cpu_small")
model = lb.build_model(cfg.model)
from llmbuilder.data import TextDataset
dataset = TextDataset("./data/clean.txt", block_size=cfg.model.max_seq_length)
results = lb.train_model(model, dataset, cfg.training)
text = lb.generate_text(
    model_path="./checkpoints/model.pt",
    tokenizer_path="./tokenizers",
    prompt="Hello",
    max_new_tokens=64,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
)
print(text)
Config presets and legacy keys:

Use lb.load_config(preset="cpu_small") or path="config.yaml".
Legacy flat keys like n_layer, n_head, n_embd are accepted and mapped internally.
Useful CLI flags:

Training: --epochs, --batch-size, --lr, --eval-interval, --save-interval (see llmbuilder train model --help).
Generation: --max-new-tokens, --temperature, --top-k, --top-p, --device (see llmbuilder generate text --help).
Environment knobs:

Enable slow tests: set RUN_SLOW=1
Enable performance tests: set RUN_PERF=1
Performance tips:

Prefer CPU wheels for broad compatibility; use smaller seq length and batch size.
Checkpoints are saved under checkpoints/; consider periodic eval to monitor perplexity.
🔧 Troubleshooting
Installation Issues
Missing Optional Dependencies

# Check what's installed
python -c "import llmbuilder; print('✅ LLMBuilder installed')"

# Install missing dependencies
pip install pymupdf pytesseract ebooklib beautifulsoup4 lxml sentence-transformers

# Verify specific features
python -c "import pytesseract; print('✅ OCR available')"
python -c "import sentence_transformers; print('✅ Semantic deduplication available')"
System Dependencies

# Tesseract OCR (for PDF processing)
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
# macOS: brew install tesseract
# Ubuntu: sudo apt-get install tesseract-ocr tesseract-ocr-eng

# Verify Tesseract installation
tesseract --version
python -c "import pytesseract; pytesseract.get_tesseract_version()"
Processing Issues
PDF Processing Problems

# Enable debug logging
export LLMBUILDER_LOG_LEVEL=DEBUG

# Common fixes:
# 1. Install language packs: sudo apt-get install tesseract-ocr-eng
# 2. Set Tesseract path: export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
# 3. Lower OCR threshold: --ocr-threshold 0.3
Memory Issues with Large Datasets

# Use configuration to optimize memory usage
llmbuilder config from-template cpu_optimized_config -o memory_config.json \
  --override data.ingestion.batch_size=50 \
  --override data.deduplication.batch_size=500 \
  --override data.deduplication.use_gpu_for_embeddings=false

# Process in smaller chunks
llmbuilder data load -i large_dataset/ -o processed.txt --batch-size 25 --workers 2
Semantic Deduplication Performance

# GPU issues - disable GPU acceleration
llmbuilder data deduplicate -i dataset.txt -o clean.txt --method semantic --no-gpu

# Slow processing - increase batch size
llmbuilder data deduplicate -i dataset.txt -o clean.txt --method semantic --batch-size 2000

# Memory issues - reduce embedding cache
llmbuilder config from-template basic_config -o config.json \
  --override data.deduplication.embedding_cache_size=5000
GGUF Conversion Issues
Missing llama.cpp

# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make

# Add to PATH or specify location
export PATH=$PATH:/path/to/llama.cpp

# Alternative: Use Python package
pip install llama-cpp-python

# Test conversion
llmbuilder convert gguf --help
Conversion Failures

# Check available conversion scripts
llmbuilder convert gguf model.pt -o test.gguf --verbose

# Try different quantization levels
llmbuilder convert gguf model.pt -o test.gguf -q F16  # Less compression
llmbuilder convert gguf model.pt -o test.gguf -q Q8_0 # Balanced

# Increase timeout for large models
llmbuilder config from-template basic_config -o config.json \
  --override gguf_conversion.conversion_timeout=7200
Configuration Issues
Validation Errors

# Validate configuration with detailed output
llmbuilder config validate my_config.json --detailed

# Common fixes:
# 1. Vocab size mismatch - ensure model.vocab_size matches tokenizer_training.vocab_size
# 2. Sequence length issues - ensure data.max_length <= model.max_seq_length
# 3. Invalid quantization level - use: F32, F16, Q8_0, Q5_1, Q5_0, Q4_1, Q4_0
Template Issues

# List available templates
llmbuilder config templates

# Create from working template
llmbuilder config from-template basic_config -o working_config.json

# Validate before use
llmbuilder config validate working_config.json
Performance Optimization
Speed Up Processing

# Use more workers for I/O bound tasks
llmbuilder data load -i docs/ -o processed.txt --workers 8

# Enable GPU for semantic operations
llmbuilder data deduplicate -i dataset.txt -o clean.txt --use-gpu --batch-size 2000

# Use faster HTML parser
llmbuilder config from-template basic_config -o config.json \
  --override data.ingestion.html_parser=lxml
Reduce Memory Usage

# Smaller batch sizes
llmbuilder data load -i docs/ -o processed.txt --batch-size 25

# Disable semantic deduplication for large datasets
llmbuilder data deduplicate -i dataset.txt -o clean.txt --method exact

# Use CPU-optimized configuration
llmbuilder config from-template cpu_optimized_config -o config.json
Debug Mode
Enable Verbose Logging

# Set environment variable
export LLMBUILDER_LOG_LEVEL=DEBUG

# Or use CLI flag
llmbuilder data load -i docs/ -o processed.txt --verbose

# Check processing statistics
llmbuilder data load -i docs/ -o processed.txt --stats
Getting Help
📖 Documentation: https://qubasehq.github.io/llmbuilder-package/
🐛 Issues: GitHub Issues
💬 Discussions: GitHub Discussions
Testing (developers)
Fast tests: python -m pytest -q tests
Slow tests: set RUN_SLOW=1 && python -m pytest -q tests
Performance tests: set RUN_PERF=1 && python -m pytest -q tests\performance
License
Apache-2.0 (or project license).

About
A comprehensive toolkit for building, training, and deploying language models

qubasehq.github.io/llmbuilder-package/
Resources
 Readme
Contributing
 Contributing
 Activity
 Custom properties
Stars
 0 stars
Watchers
 0 watching
Forks
 0 forks
Report repository
Releases
 1 tags
Create a new release
Packages
No packages published
Publish your first package
Deployments
4
 github-pages 2 days ago
+ 3 deployments
Languages
HTML
79.7%
 
Python
19.5%
 
Ot