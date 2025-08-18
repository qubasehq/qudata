#!/usr/bin/env python3
"""
Complete Beginner's Guide to QuData

This comprehensive example demonstrates every major feature of QuData
in a step-by-step manner that even non-technical users can understand.

Run this script to see QuData in action with detailed explanations.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def print_section(title: str, description: str = ""):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"📚 {title}")
    if description:
        print(f"   {description}")
    print("=" * 60)

def print_step(step: str, description: str = ""):
    """Print a formatted step."""
    print(f"\n🔸 {step}")
    if description:
        print(f"   {description}")

def print_success(message: str):
    """Print a success message."""
    print(f"✅ {message}")

def print_info(message: str):
    """Print an info message."""
    print(f"ℹ️  {message}")

def print_warning(message: str):
    """Print a warning message."""
    print(f"⚠️  {message}")

def print_error(message: str):
    """Print an error message."""
    print(f"❌ {message}")

def create_sample_documents() -> List[str]:
    """Create sample documents for demonstration."""
    print_step("Creating sample documents for demonstration")
    
    # Ensure sample directory exists
    sample_dir = Path("sample_documents")
    sample_dir.mkdir(exist_ok=True)
    
    # Sample documents with different content types
    samples = {
        "technology_article.txt": """
        The Future of Artificial Intelligence in Healthcare
        
        Artificial Intelligence (AI) is revolutionizing healthcare by enabling 
        more accurate diagnoses, personalized treatments, and efficient drug discovery.
        
        Key Applications:
        - Medical imaging analysis for cancer detection
        - Drug discovery and development acceleration
        - Personalized treatment recommendations
        - Predictive analytics for patient outcomes
        
        Companies like Google DeepMind, IBM Watson Health, and numerous startups
        are developing AI solutions that could save millions of lives.
        
        The integration of AI in healthcare represents one of the most promising
        applications of artificial intelligence technology.
        """,
        
        "business_report.txt": """
        Q3 2024 Market Analysis Report
        
        Executive Summary:
        The technology sector showed strong growth in Q3 2024, with AI companies
        leading the charge. Cloud computing and cybersecurity also demonstrated
        robust performance.
        
        Key Findings:
        - AI sector grew 45% year-over-year
        - Cloud services increased 28% from Q2
        - Cybersecurity investments up 35%
        - Remote work tools maintained steady demand
        
        Market Outlook:
        We expect continued growth in AI and cloud technologies through 2025,
        driven by enterprise digital transformation initiatives.
        
        Companies to watch: Microsoft, Google, Amazon, NVIDIA, and emerging
        AI startups in the healthcare and finance sectors.
        """,
        
        "scientific_paper.txt": """
        Climate Change Impact on Ocean Ecosystems
        
        Abstract:
        This study examines the effects of rising ocean temperatures on marine
        biodiversity. Our research indicates significant changes in species
        distribution and ecosystem dynamics.
        
        Introduction:
        Climate change is causing unprecedented changes in ocean temperatures,
        affecting marine life at all levels of the food chain.
        
        Methodology:
        We analyzed temperature data from 500 monitoring stations worldwide
        over a 20-year period, correlating changes with species population data.
        
        Results:
        - Average ocean temperature increased 0.6°C over 20 years
        - 23% of species showed significant habitat shifts
        - Coral reef ecosystems most severely affected
        - Arctic marine life experiencing rapid changes
        
        Conclusion:
        Urgent action is needed to mitigate climate change impacts on ocean
        ecosystems. International cooperation is essential for effective
        conservation strategies.
        """,
        
        "tutorial_content.txt": """
        Python Programming Tutorial: Getting Started with Machine Learning
        
        Welcome to this beginner-friendly tutorial on machine learning with Python!
        
        What You'll Learn:
        1. Setting up your Python environment
        2. Understanding basic ML concepts
        3. Building your first ML model
        4. Evaluating model performance
        
        Prerequisites:
        - Basic Python knowledge
        - High school level mathematics
        - Curiosity about AI and data science
        
        Step 1: Install Required Libraries
        ```python
        pip install pandas numpy scikit-learn matplotlib
        ```
        
        Step 2: Load Your Data
        ```python
        import pandas as pd
        data = pd.read_csv('your_data.csv')
        print(data.head())
        ```
        
        Step 3: Build a Simple Model
        ```python
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train, y_train)
        ```
        
        This tutorial will guide you through each step with practical examples
        and clear explanations suitable for beginners.
        """,
        
        "news_article.txt": """
        Breaking: Major Breakthrough in Quantum Computing Announced
        
        Published: March 15, 2024
        By: Sarah Johnson, Technology Reporter
        
        Scientists at MIT have announced a significant breakthrough in quantum
        computing that could accelerate the development of practical quantum
        applications.
        
        The research team, led by Dr. Michael Chen, has developed a new method
        for maintaining quantum coherence that could make quantum computers
        more stable and reliable.
        
        "This breakthrough brings us closer to quantum computers that can solve
        real-world problems," said Dr. Chen in a press conference yesterday.
        
        Key Achievements:
        - Quantum coherence maintained for 10x longer than previous methods
        - Error rates reduced by 90%
        - Scalable to larger quantum systems
        - Compatible with existing quantum hardware
        
        Industry experts believe this development could accelerate quantum
        computing adoption in fields like cryptography, drug discovery, and
        financial modeling.
        
        The research was published in Nature Quantum Information and funded
        by the National Science Foundation.
        """,
        
        "short_note.txt": """
        Quick reminder: Team meeting tomorrow at 2 PM in Conference Room A.
        Please bring your project updates and quarterly reports.
        """,
        
        "multilingual_content.txt": """
        English: Welcome to our international conference on artificial intelligence.
        
        Español: Bienvenidos a nuestra conferencia internacional sobre inteligencia artificial.
        
        Français: Bienvenue à notre conférence internationale sur l'intelligence artificielle.
        
        Deutsch: Willkommen zu unserer internationalen Konferenz über künstliche Intelligenz.
        
        This document demonstrates QuData's ability to handle multilingual content
        and detect different languages within the same document.
        """
    }
    
    created_files = []
    for filename, content in samples.items():
        file_path = sample_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        created_files.append(str(file_path))
        print_info(f"Created: {filename}")
    
    print_success(f"Created {len(created_files)} sample documents")
    return created_files

def demonstrate_basic_processing():
    """Demonstrate basic document processing."""
    print_section("Basic Document Processing", 
                  "Learn how to process documents with QuData")
    
    try:
        from qudata import QuDataPipeline
        from qudata.models import Document, DocumentMetadata
        
        print_step("1. Initialize QuData Pipeline")
        pipeline = QuDataPipeline()
        print_success("Pipeline initialized successfully")
        
        print_step("2. Create sample documents")
        sample_files = create_sample_documents()
        
        print_step("3. Process documents one by one")
        results = []
        
        for i, file_path in enumerate(sample_files, 1):
            print(f"\n   Processing file {i}/{len(sample_files)}: {Path(file_path).name}")
            
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Create document object
                metadata = DocumentMetadata(
                    title=Path(file_path).stem.replace('_', ' ').title(),
                    file_type="txt",
                    language="en"
                )
                
                document = Document(
                    id=f"doc_{i}",
                    source_path=file_path,
                    content=content,
                    metadata=metadata
                )
                
                # Process document
                result = pipeline.process_document(document)
                results.append(result)
                
                if result.success:
                    doc = result.document
                    print_success(f"Processed successfully!")
                    print(f"      Quality Score: {doc.quality_score:.2f}")
                    print(f"      Language: {doc.metadata.language}")
                    print(f"      Content Length: {len(doc.content)} characters")
                    print(f"      Processing Time: {result.processing_time:.2f}s")
                    
                    if doc.metadata.topics:
                        print(f"      Topics: {', '.join(doc.metadata.topics)}")
                else:
                    print_error("Processing failed")
                    for error in result.errors:
                        print(f"      Error: {error.message}")
                        
            except Exception as e:
                print_error(f"Failed to process {file_path}: {e}")
        
        print_step("4. Summary of Results")
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        print_success(f"Successfully processed: {len(successful)} documents")
        if failed:
            print_warning(f"Failed to process: {len(failed)} documents")
        
        if successful:
            avg_quality = sum(r.document.quality_score for r in successful) / len(successful)
            print_info(f"Average quality score: {avg_quality:.2f}")
            
            # Language distribution
            languages = [r.document.metadata.language for r in successful]
            from collections import Counter
            lang_counts = Counter(languages)
            print_info("Language distribution:")
            for lang, count in lang_counts.items():
                print(f"      {lang}: {count} documents")
        
        return successful
        
    except ImportError as e:
        print_error(f"QuData not available: {e}")
        print_info("Please install QuData first: pip install -e .")
        return []

def demonstrate_batch_processing():
    """Demonstrate batch processing of multiple documents."""
    print_section("Batch Processing", 
                  "Process multiple documents at once")
    
    try:
        from qudata import QuDataPipeline
        
        print_step("1. Initialize pipeline for batch processing")
        pipeline = QuDataPipeline()
        
        print_step("2. Process entire directory")
        input_dir = "sample_documents"
        output_dir = "processed_documents"
        
        # Ensure output directory exists
        Path(output_dir).mkdir(exist_ok=True)
        
        print_info(f"Processing all files in: {input_dir}")
        print_info(f"Output will be saved to: {output_dir}")
        
        # Process directory
        start_time = time.time()
        results = pipeline.process_directory(input_dir, output_dir)
        processing_time = time.time() - start_time
        
        print_step("3. Batch Processing Results")
        print_success(f"Batch processing completed in {processing_time:.2f} seconds")
        print_info(f"Total documents processed: {len(results)}")
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        print_success(f"Successful: {len(successful)}")
        if failed:
            print_warning(f"Failed: {len(failed)}")
        
        if successful:
            # Calculate statistics
            quality_scores = [r.document.quality_score for r in successful]
            content_lengths = [len(r.document.content) for r in successful]
            processing_times = [r.processing_time for r in successful]
            
            print_info("Processing Statistics:")
            print(f"      Average quality: {sum(quality_scores) / len(quality_scores):.2f}")
            print(f"      Quality range: {min(quality_scores):.2f} - {max(quality_scores):.2f}")
            print(f"      Average content length: {sum(content_lengths) / len(content_lengths):.0f} chars")
            print(f"      Average processing time: {sum(processing_times) / len(processing_times):.2f}s")
            print(f"      Total processing throughput: {len(successful) / processing_time:.1f} docs/second")
        
        return successful
        
    except ImportError as e:
        print_error(f"QuData not available: {e}")
        return []

def demonstrate_export_formats():
    """Demonstrate exporting data in different formats."""
    print_section("Export Formats", 
                  "Export processed data for AI training and analysis")
    
    try:
        from qudata import QuDataPipeline
        from qudata.pack import JSONLFormatter, ChatMLFormatter
        
        print_step("1. Process sample documents")
        pipeline = QuDataPipeline()
        results = pipeline.process_directory("sample_documents", "processed_documents")
        successful_docs = [r.document for r in results if r.success]
        
        if not successful_docs:
            print_warning("No documents to export. Processing some samples first...")
            successful_docs = demonstrate_basic_processing()
            successful_docs = [r.document for r in successful_docs if r.success]
        
        if not successful_docs:
            print_error("No successful documents to export")
            return
        
        # Ensure output directory exists
        output_dir = Path("exports")
        output_dir.mkdir(exist_ok=True)
        
        print_step("2. Export to JSONL format (most common for AI training)")
        jsonl_formatter = JSONLFormatter()
        jsonl_path = output_dir / "training_data.jsonl"
        
        jsonl_formatter.export_to_file(
            documents=successful_docs,
            output_path=str(jsonl_path),
            fields=["text", "metadata", "quality_score"]
        )
        print_success(f"JSONL export completed: {jsonl_path}")
        print_info(f"Exported {len(successful_docs)} documents")
        
        # Show sample of JSONL content
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            sample_data = json.loads(first_line)
            print_info("Sample JSONL entry:")
            print(f"      Text preview: {sample_data['text'][:100]}...")
            print(f"      Quality score: {sample_data['quality_score']}")
            print(f"      Language: {sample_data['metadata']['language']}")
        
        print_step("3. Export to ChatML format (for conversational AI)")
        chatml_formatter = ChatMLFormatter()
        chatml_data = chatml_formatter.format_documents(
            successful_docs,
            system_message="You are a helpful assistant with knowledge from various documents."
        )
        
        chatml_path = output_dir / "chat_training.json"
        with open(chatml_path, 'w', encoding='utf-8') as f:
            json.dump(chatml_data, f, indent=2, ensure_ascii=False)
        
        print_success(f"ChatML export completed: {chatml_path}")
        print_info(f"Created {len(chatml_data)} conversation examples")
        
        print_step("4. Export to CSV format (for spreadsheet analysis)")
        csv_path = output_dir / "document_analysis.csv"
        
        # Create CSV data
        csv_data = []
        for doc in successful_docs:
            csv_data.append({
                'title': doc.metadata.title or 'Untitled',
                'language': doc.metadata.language,
                'quality_score': doc.quality_score,
                'content_length': len(doc.content),
                'word_count': len(doc.content.split()),
                'domain': doc.metadata.domain or 'general',
                'source': doc.source_path
            })
        
        import csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            if csv_data:
                writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                writer.writeheader()
                writer.writerows(csv_data)
        
        print_success(f"CSV export completed: {csv_path}")
        print_info("CSV contains document metadata and statistics")
        
        print_step("5. Create train/validation/test splits")
        # Split documents for machine learning
        import random
        random.seed(42)  # For reproducible splits
        
        shuffled_docs = successful_docs.copy()
        random.shuffle(shuffled_docs)
        
        total = len(shuffled_docs)
        train_size = int(0.8 * total)
        val_size = int(0.1 * total)
        
        train_docs = shuffled_docs[:train_size]
        val_docs = shuffled_docs[train_size:train_size + val_size]
        test_docs = shuffled_docs[train_size + val_size:]
        
        # Export splits
        splits = {
            'train': train_docs,
            'validation': val_docs,
            'test': test_docs
        }
        
        for split_name, docs in splits.items():
            if docs:
                split_path = output_dir / f"{split_name}.jsonl"
                jsonl_formatter.export_to_file(
                    documents=docs,
                    output_path=str(split_path),
                    fields=["text", "metadata", "quality_score"]
                )
                print_success(f"{split_name.title()} split: {len(docs)} documents -> {split_path}")
        
        print_step("6. Export Summary")
        print_info("All exports completed successfully!")
        print_info("Files created:")
        for file_path in output_dir.glob("*"):
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"      {file_path.name}: {size_mb:.2f} MB")
        
    except ImportError as e:
        print_error(f"Export functionality not available: {e}")

def demonstrate_quality_analysis():
    """Demonstrate quality analysis and reporting."""
    print_section("Quality Analysis", 
                  "Analyze and understand your data quality")
    
    try:
        from qudata import QuDataPipeline
        from qudata.analyze import AnalysisEngine
        
        print_step("1. Process documents for analysis")
        pipeline = QuDataPipeline()
        results = pipeline.process_directory("sample_documents", "processed_documents")
        successful_docs = [r.document for r in results if r.success]
        
        if not successful_docs:
            print_warning("No documents available for analysis")
            return
        
        print_step("2. Initialize analysis engine")
        analyzer = AnalysisEngine()
        
        print_step("3. Perform comprehensive analysis")
        analysis_result = analyzer.analyze_documents(
            successful_docs,
            include_topics=True,
            include_sentiment=True,
            include_entities=True
        )
        
        print_step("4. Text Statistics")
        stats = analysis_result.text_statistics
        print_info("Document Statistics:")
        print(f"      Total documents: {stats.total_documents}")
        print(f"      Total words: {stats.total_words:,}")
        print(f"      Unique words: {stats.unique_words:,}")
        print(f"      Average document length: {stats.avg_document_length:.0f} words")
        print(f"      Vocabulary richness: {stats.unique_words / stats.total_words:.3f}")
        
        print_info("Top keywords:")
        for keyword, count in stats.top_keywords[:10]:
            print(f"      {keyword}: {count} occurrences")
        
        print_step("5. Quality Analysis")
        quality_scores = [doc.quality_score for doc in successful_docs]
        print_info("Quality Distribution:")
        print(f"      Average quality: {sum(quality_scores) / len(quality_scores):.2f}")
        print(f"      Minimum quality: {min(quality_scores):.2f}")
        print(f"      Maximum quality: {max(quality_scores):.2f}")
        
        # Quality categories
        high_quality = [s for s in quality_scores if s >= 0.8]
        medium_quality = [s for s in quality_scores if 0.6 <= s < 0.8]
        low_quality = [s for s in quality_scores if s < 0.6]
        
        print(f"      High quality (≥0.8): {len(high_quality)} documents")
        print(f"      Medium quality (0.6-0.8): {len(medium_quality)} documents")
        print(f"      Low quality (<0.6): {len(low_quality)} documents")
        
        print_step("6. Language Analysis")
        languages = [doc.metadata.language for doc in successful_docs]
        from collections import Counter
        lang_counts = Counter(languages)
        
        print_info("Language Distribution:")
        for lang, count in lang_counts.items():
            percentage = (count / len(successful_docs)) * 100
            print(f"      {lang}: {count} documents ({percentage:.1f}%)")
        
        print_step("7. Content Analysis")
        content_lengths = [len(doc.content) for doc in successful_docs]
        word_counts = [len(doc.content.split()) for doc in successful_docs]
        
        print_info("Content Length Analysis:")
        print(f"      Average characters: {sum(content_lengths) / len(content_lengths):.0f}")
        print(f"      Average words: {sum(word_counts) / len(word_counts):.0f}")
        print(f"      Shortest document: {min(word_counts)} words")
        print(f"      Longest document: {max(word_counts)} words")
        
        print_step("8. Topic Analysis")
        if hasattr(analysis_result, 'topics') and analysis_result.topics:
            print_info("Discovered Topics:")
            for i, topic in enumerate(analysis_result.topics[:5], 1):
                print(f"      Topic {i}: {', '.join(topic.keywords[:5])}")
        else:
            print_info("Topic analysis not available (requires ML dependencies)")
        
        print_step("9. Generate Analysis Report")
        report_path = Path("exports") / "quality_analysis.json"
        report_path.parent.mkdir(exist_ok=True)
        
        report_data = {
            "summary": {
                "total_documents": len(successful_docs),
                "average_quality": sum(quality_scores) / len(quality_scores),
                "language_distribution": dict(lang_counts),
                "quality_distribution": {
                    "high": len(high_quality),
                    "medium": len(medium_quality),
                    "low": len(low_quality)
                }
            },
            "statistics": {
                "total_words": sum(word_counts),
                "unique_words": len(set(" ".join(doc.content for doc in successful_docs).split())),
                "average_length": sum(word_counts) / len(word_counts),
                "content_range": {
                    "min_words": min(word_counts),
                    "max_words": max(word_counts)
                }
            }
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print_success(f"Analysis report saved: {report_path}")
        
    except ImportError as e:
        print_error(f"Analysis functionality not available: {e}")

def demonstrate_configuration():
    """Demonstrate configuration customization."""
    print_section("Configuration Customization", 
                  "Learn how to customize QuData for your needs")
    
    print_step("1. Understanding Configuration Files")
    print_info("QuData uses YAML files to control processing behavior")
    print_info("Configuration files are like 'recipes' that tell QuData what to do")
    
    print_step("2. Create a Basic Configuration")
    config_dir = Path("custom_configs")
    config_dir.mkdir(exist_ok=True)
    
    basic_config = {
        'pipeline': {
            'name': 'my_first_pipeline',
            'input_directory': 'sample_documents',
            'output_directory': 'processed_documents',
            'parallel_processing': True,
            'max_workers': 2
        },
        'ingest': {
            'file_types': ['txt', 'pdf', 'docx', 'html'],
            'max_file_size': '50MB',
            'extract_metadata': True
        },
        'clean': {
            'normalize_text': True,
            'remove_duplicates': True,
            'similarity_threshold': 0.85,
            'min_length': 50,
            'max_length': 10000
        },
        'quality': {
            'min_score': 0.5,
            'auto_filter': True,
            'dimensions': {
                'content': 0.4,
                'language': 0.3,
                'structure': 0.3
            }
        },
        'export': {
            'formats': ['jsonl', 'csv'],
            'include_metadata': True,
            'split_data': True,
            'split_ratios': {
                'train': 0.8,
                'validation': 0.1,
                'test': 0.1
            }
        }
    }
    
    basic_config_path = config_dir / "basic_config.yaml"
    
    import yaml
    with open(basic_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(basic_config, f, default_flow_style=False, indent=2)
    
    print_success(f"Basic configuration created: {basic_config_path}")
    
    print_step("3. Create Specialized Configurations")
    
    # Academic papers configuration
    academic_config = basic_config.copy()
    academic_config['pipeline']['name'] = 'academic_papers'
    academic_config['clean']['min_length'] = 1000  # Longer minimum for papers
    academic_config['quality']['min_score'] = 0.7  # Higher quality threshold
    academic_config['ingest']['file_types'] = ['pdf']  # Only PDFs
    
    academic_config_path = config_dir / "academic_papers.yaml"
    with open(academic_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(academic_config, f, default_flow_style=False, indent=2)
    
    print_success(f"Academic papers config: {academic_config_path}")
    
    # Web content configuration
    web_config = basic_config.copy()
    web_config['pipeline']['name'] = 'web_content'
    web_config['clean']['remove_html_tags'] = True
    web_config['clean']['remove_boilerplate'] = True
    web_config['quality']['min_score'] = 0.4  # Lower threshold for web content
    web_config['ingest']['file_types'] = ['html', 'txt']
    
    web_config_path = config_dir / "web_content.yaml"
    with open(web_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(web_config, f, default_flow_style=False, indent=2)
    
    print_success(f"Web content config: {web_config_path}")
    
    print_step("4. Configuration Explanation")
    print_info("Configuration sections explained:")
    print("      pipeline: Basic settings like input/output directories")
    print("      ingest: What file types to process and how")
    print("      clean: How to clean and normalize text")
    print("      quality: Quality thresholds and scoring")
    print("      export: Output formats and data splitting")
    
    print_step("5. Using Custom Configurations")
    print_info("To use a custom configuration:")
    print("      Command line: qudata process --config custom_configs/basic_config.yaml")
    print("      Python: pipeline = QuDataPipeline(config_path='custom_configs/basic_config.yaml')")
    
    return [basic_config_path, academic_config_path, web_config_path]

def demonstrate_monitoring():
    """Demonstrate monitoring and progress tracking."""
    print_section("Monitoring and Progress Tracking", 
                  "Keep track of your processing jobs")
    
    print_step("1. Processing with Progress Callbacks")
    
    def progress_callback(stage: str, progress: float, message: str):
        """Custom progress callback function."""
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f"\r   [{stage}] |{bar}| {progress:.1%} - {message}", end='', flush=True)
    
    try:
        from qudata import QuDataPipeline
        
        pipeline = QuDataPipeline()
        
        print_info("Processing with progress tracking...")
        results = pipeline.process_directory(
            "sample_documents", 
            "processed_documents",
            progress_callback=progress_callback
        )
        print()  # New line after progress bar
        
        print_success("Processing completed with progress tracking")
        
    except Exception as e:
        print_error(f"Progress tracking demo failed: {e}")
    
    print_step("2. Performance Monitoring")
    
    # Simulate performance monitoring
    import time
    import psutil
    
    print_info("System Resource Usage:")
    print(f"      CPU Usage: {psutil.cpu_percent()}%")
    print(f"      Memory Usage: {psutil.virtual_memory().percent}%")
    print(f"      Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    
    print_step("3. Processing Statistics")
    
    if 'results' in locals():
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        if successful:
            processing_times = [r.processing_time for r in successful]
            quality_scores = [r.document.quality_score for r in successful]
            
            print_info("Processing Performance:")
            print(f"      Total documents: {len(results)}")
            print(f"      Success rate: {len(successful) / len(results) * 100:.1f}%")
            print(f"      Average processing time: {sum(processing_times) / len(processing_times):.2f}s")
            print(f"      Fastest processing: {min(processing_times):.2f}s")
            print(f"      Slowest processing: {max(processing_times):.2f}s")
            print(f"      Average quality: {sum(quality_scores) / len(quality_scores):.2f}")

def demonstrate_troubleshooting():
    """Demonstrate common troubleshooting scenarios."""
    print_section("Troubleshooting Common Issues", 
                  "Learn how to solve common problems")
    
    print_step("1. File Processing Issues")
    print_info("Common file issues and solutions:")
    print("      ❌ File not found → Check file path and permissions")
    print("      ❌ Unsupported format → Check supported file types")
    print("      ❌ File too large → Increase max_file_size in config")
    print("      ❌ Corrupted file → Enable skip_corrupted in config")
    
    print_step("2. Memory Issues")
    print_info("Memory management solutions:")
    print("      ❌ Out of memory → Reduce batch_size in config")
    print("      ❌ Slow processing → Enable streaming_mode")
    print("      ❌ System freeze → Set max_memory_usage limit")
    
    print_step("3. Quality Issues")
    print_info("Quality improvement strategies:")
    print("      ❌ Low quality scores → Lower min_score threshold")
    print("      ❌ Too many filtered docs → Disable auto_filter")
    print("      ❌ Poor text extraction → Check file format support")
    
    print_step("4. Configuration Issues")
    print_info("Configuration troubleshooting:")
    print("      ❌ Config not loading → Check YAML syntax")
    print("      ❌ Settings ignored → Verify config file path")
    print("      ❌ Unexpected behavior → Review default settings")
    
    print_step("5. Performance Issues")
    print_info("Performance optimization tips:")
    print("      🚀 Enable parallel processing")
    print("      🚀 Increase max_workers (but not more than CPU cores)")
    print("      🚀 Use SSD storage for better I/O")
    print("      🚀 Enable caching for repeated operations")
    
    print_step("6. Debugging Tips")
    print_info("When things go wrong:")
    print("      🔍 Enable verbose logging: --verbose")
    print("      🔍 Check log files for detailed errors")
    print("      🔍 Test with small sample first")
    print("      🔍 Verify input data format and quality")
    print("      🔍 Check system resources (CPU, memory, disk)")

def main():
    """Run the complete beginner's guide demonstration."""
    print("🎯 QuData Complete Beginner's Guide")
    print("This comprehensive demo will show you everything QuData can do!")
    print("Follow along to learn step by step.")
    
    # Create necessary directories
    for directory in ["sample_documents", "processed_documents", "exports", "custom_configs"]:
        Path(directory).mkdir(exist_ok=True)
    
    try:
        # Run all demonstrations
        demonstrate_basic_processing()
        demonstrate_batch_processing()
        demonstrate_export_formats()
        demonstrate_quality_analysis()
        demonstrate_configuration()
        demonstrate_monitoring()
        demonstrate_troubleshooting()
        
        print_section("🎉 Congratulations!", 
                      "You've completed the QuData beginner's guide!")
        
        print_info("What you've learned:")
        print("   ✅ How to process individual documents")
        print("   ✅ How to batch process multiple documents")
        print("   ✅ How to export data in different formats")
        print("   ✅ How to analyze data quality")
        print("   ✅ How to customize configurations")
        print("   ✅ How to monitor processing progress")
        print("   ✅ How to troubleshoot common issues")
        
        print_info("Next steps:")
        print("   🚀 Try processing your own documents")
        print("   🚀 Experiment with different configurations")
        print("   🚀 Explore advanced features like web scraping")
        print("   🚀 Integrate QuData with your AI training pipeline")
        
        print_info("Files created during this demo:")
        for directory in ["sample_documents", "processed_documents", "exports", "custom_configs"]:
            if Path(directory).exists():
                files = list(Path(directory).glob("*"))
                if files:
                    print(f"   📁 {directory}/: {len(files)} files")
        
        print("\n" + "=" * 60)
        print("Thank you for trying QuData! 🙏")
        print("For more help, visit: https://github.com/qubasehq/qudata")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        print("You can run this script again anytime to continue learning!")
    except Exception as e:
        print_error(f"Demo failed with error: {e}")
        print_info("This might be due to missing dependencies or system issues")
        print_info("Please check the installation and try again")

if __name__ == "__main__":
    main()