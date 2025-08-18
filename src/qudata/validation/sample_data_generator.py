"""
Sample data generator for creating test datasets and validation data.

This module provides comprehensive capabilities for generating synthetic test data
to validate processing pipeline functionality across different scenarios.
"""

import json
import random
import string
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class DataType(Enum):
    """Types of sample data to generate."""
    TEXT = "text"
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    PDF = "pdf"
    DOCX = "docx"
    XML = "xml"
    MARKDOWN = "markdown"


class ContentCategory(Enum):
    """Categories of content to generate."""
    TECHNICAL = "technical"
    ACADEMIC = "academic"
    NEWS = "news"
    LEGAL = "legal"
    MEDICAL = "medical"
    FINANCIAL = "financial"
    GENERAL = "general"
    CORRUPTED = "corrupted"


class QualityLevel(Enum):
    """Quality levels for generated content."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CORRUPTED = "corrupted"


@dataclass
class GenerationConfig:
    """Configuration for data generation."""
    data_type: DataType
    content_category: ContentCategory
    quality_level: QualityLevel
    size_range: Tuple[int, int] = (100, 1000)  # Size in characters
    language: str = "en"
    include_metadata: bool = True
    corruption_rate: float = 0.0  # 0.0 to 1.0
    duplicate_rate: float = 0.0  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "data_type": self.data_type.value,
            "content_category": self.content_category.value,
            "quality_level": self.quality_level.value,
            "size_range": self.size_range,
            "language": self.language,
            "include_metadata": self.include_metadata,
            "corruption_rate": self.corruption_rate,
            "duplicate_rate": self.duplicate_rate
        }


@dataclass
class SampleDocument:
    """Generated sample document."""
    content: str
    metadata: Dict[str, Any]
    file_path: Optional[str] = None
    expected_quality_score: Optional[float] = None
    expected_language: Optional[str] = None
    expected_domain: Optional[str] = None
    generation_config: Optional[GenerationConfig] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert sample document to dictionary."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "file_path": self.file_path,
            "expected_quality_score": self.expected_quality_score,
            "expected_language": self.expected_language,
            "expected_domain": self.expected_domain,
            "generation_config": self.generation_config.to_dict() if self.generation_config else None
        }


class ContentGenerator(ABC):
    """Abstract base class for content generators."""
    
    @abstractmethod
    def generate_content(self, config: GenerationConfig) -> str:
        """Generate content based on configuration."""
        pass
    
    @abstractmethod
    def get_expected_metadata(self, config: GenerationConfig) -> Dict[str, Any]:
        """Get expected metadata for generated content."""
        pass


class TextContentGenerator(ContentGenerator):
    """Generator for plain text content."""
    
    def __init__(self):
        """Initialize text content generator."""
        self.word_lists = {
            ContentCategory.TECHNICAL: [
                "algorithm", "database", "framework", "implementation", "optimization",
                "architecture", "scalability", "performance", "security", "integration",
                "deployment", "monitoring", "testing", "debugging", "refactoring"
            ],
            ContentCategory.ACADEMIC: [
                "research", "methodology", "hypothesis", "analysis", "conclusion",
                "literature", "experiment", "observation", "theory", "evidence",
                "publication", "peer-review", "citation", "abstract", "dissertation"
            ],
            ContentCategory.NEWS: [
                "breaking", "report", "investigation", "interview", "statement",
                "announcement", "development", "situation", "response", "impact",
                "community", "government", "policy", "economy", "society"
            ],
            ContentCategory.LEGAL: [
                "contract", "agreement", "clause", "provision", "liability",
                "jurisdiction", "compliance", "regulation", "statute", "precedent",
                "plaintiff", "defendant", "court", "judgment", "appeal"
            ],
            ContentCategory.MEDICAL: [
                "patient", "diagnosis", "treatment", "symptoms", "medication",
                "therapy", "procedure", "clinical", "medical", "health",
                "disease", "condition", "recovery", "prevention", "care"
            ],
            ContentCategory.FINANCIAL: [
                "investment", "portfolio", "revenue", "profit", "loss",
                "market", "trading", "analysis", "forecast", "budget",
                "capital", "assets", "liability", "equity", "dividend"
            ],
            ContentCategory.GENERAL: [
                "information", "content", "document", "text", "data",
                "example", "sample", "test", "demonstration", "illustration",
                "description", "explanation", "overview", "summary", "details"
            ]
        }
    
    def generate_content(self, config: GenerationConfig) -> str:
        """Generate text content."""
        if config.quality_level == QualityLevel.CORRUPTED:
            return self._generate_corrupted_content(config)
        
        # Get word list for category
        words = self.word_lists.get(config.content_category, self.word_lists[ContentCategory.GENERAL])
        
        # Generate content
        target_size = random.randint(*config.size_range)
        content_parts = []
        current_size = 0
        
        while current_size < target_size:
            if config.quality_level == QualityLevel.HIGH:
                sentence = self._generate_high_quality_sentence(words)
            elif config.quality_level == QualityLevel.MEDIUM:
                sentence = self._generate_medium_quality_sentence(words)
            else:  # LOW quality
                sentence = self._generate_low_quality_sentence(words)
            
            content_parts.append(sentence)
            current_size += len(sentence)
        
        content = " ".join(content_parts)
        
        # Apply corruption if specified
        if config.corruption_rate > 0:
            content = self._apply_corruption(content, config.corruption_rate)
        
        return content[:target_size]  # Trim to exact size
    
    def _generate_high_quality_sentence(self, words: List[str]) -> str:
        """Generate a high-quality sentence."""
        sentence_templates = [
            "The {noun} demonstrates {adjective} {noun} through {verb} {noun}.",
            "In order to {verb} the {noun}, we must {verb} the {adjective} {noun}.",
            "This {noun} provides {adjective} {noun} for {verb} {noun}.",
            "The {adjective} {noun} enables {verb} of {noun} and {noun}."
        ]
        
        template = random.choice(sentence_templates)
        
        # Fill template with words
        filled = template.format(
            noun=random.choice(words),
            adjective=random.choice(["comprehensive", "detailed", "effective", "efficient", "robust"]),
            verb=random.choice(["analyze", "implement", "optimize", "evaluate", "develop"])
        )
        
        return filled
    
    def _generate_medium_quality_sentence(self, words: List[str]) -> str:
        """Generate a medium-quality sentence."""
        sentence_length = random.randint(5, 15)
        sentence_words = random.choices(words, k=sentence_length)
        
        # Add some structure
        if len(sentence_words) > 3:
            sentence_words[0] = sentence_words[0].capitalize()
            sentence_words.append("and")
            sentence_words.extend(random.choices(words, k=2))
        
        return " ".join(sentence_words) + "."
    
    def _generate_low_quality_sentence(self, words: List[str]) -> str:
        """Generate a low-quality sentence."""
        sentence_length = random.randint(2, 8)
        sentence_words = random.choices(words, k=sentence_length)
        
        # Add some randomness and poor structure
        if random.random() < 0.3:
            sentence_words.insert(random.randint(0, len(sentence_words)), "...")
        
        if random.random() < 0.2:
            sentence_words.append(random.choice(["!!!", "???", "..."]))
        
        return " ".join(sentence_words)
    
    def _generate_corrupted_content(self, config: GenerationConfig) -> str:
        """Generate corrupted content."""
        base_content = "corrupted data " * 10
        
        # Add encoding artifacts
        corrupted_chars = ["�", "\ufffd", "\x00", "\x01", "\x02"]
        corruption_points = random.randint(5, 20)
        
        content_list = list(base_content)
        for _ in range(corruption_points):
            pos = random.randint(0, len(content_list) - 1)
            content_list[pos] = random.choice(corrupted_chars)
        
        return "".join(content_list)
    
    def _apply_corruption(self, content: str, corruption_rate: float) -> str:
        """Apply corruption to content."""
        content_list = list(content)
        corruption_count = int(len(content_list) * corruption_rate)
        
        for _ in range(corruption_count):
            pos = random.randint(0, len(content_list) - 1)
            content_list[pos] = random.choice(["�", "?", "#", "@"])
        
        return "".join(content_list)
    
    def get_expected_metadata(self, config: GenerationConfig) -> Dict[str, Any]:
        """Get expected metadata for text content."""
        quality_scores = {
            QualityLevel.HIGH: (0.8, 1.0),
            QualityLevel.MEDIUM: (0.5, 0.8),
            QualityLevel.LOW: (0.2, 0.5),
            QualityLevel.CORRUPTED: (0.0, 0.2)
        }
        
        domain_mapping = {
            ContentCategory.TECHNICAL: "technology",
            ContentCategory.ACADEMIC: "education",
            ContentCategory.NEWS: "news",
            ContentCategory.LEGAL: "legal",
            ContentCategory.MEDICAL: "healthcare",
            ContentCategory.FINANCIAL: "finance",
            ContentCategory.GENERAL: "general",
            ContentCategory.CORRUPTED: "unknown"
        }
        
        quality_range = quality_scores[config.quality_level]
        
        return {
            "expected_quality_score": random.uniform(*quality_range),
            "expected_language": config.language,
            "expected_domain": domain_mapping[config.content_category],
            "expected_file_type": "txt"
        }


class JSONContentGenerator(ContentGenerator):
    """Generator for JSON content."""
    
    def generate_content(self, config: GenerationConfig) -> str:
        """Generate JSON content."""
        if config.quality_level == QualityLevel.CORRUPTED:
            return self._generate_corrupted_json()
        
        # Generate structured data based on category
        if config.content_category == ContentCategory.TECHNICAL:
            data = self._generate_technical_json()
        elif config.content_category == ContentCategory.FINANCIAL:
            data = self._generate_financial_json()
        else:
            data = self._generate_general_json()
        
        # Convert to JSON string
        if config.quality_level == QualityLevel.HIGH:
            return json.dumps(data, indent=2, sort_keys=True)
        elif config.quality_level == QualityLevel.MEDIUM:
            return json.dumps(data, indent=1)
        else:  # LOW quality
            return json.dumps(data, separators=(',', ':'))
    
    def _generate_technical_json(self) -> Dict[str, Any]:
        """Generate technical JSON data."""
        return {
            "api_version": "v1.0",
            "endpoints": [
                {"path": "/users", "method": "GET", "auth_required": True},
                {"path": "/data", "method": "POST", "auth_required": False}
            ],
            "configuration": {
                "timeout": 30,
                "retry_count": 3,
                "cache_enabled": True
            },
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "version": "1.2.3"
            }
        }
    
    def _generate_financial_json(self) -> Dict[str, Any]:
        """Generate financial JSON data."""
        return {
            "account_id": f"ACC{random.randint(100000, 999999)}",
            "balance": round(random.uniform(1000, 50000), 2),
            "transactions": [
                {
                    "id": f"TXN{i}",
                    "amount": round(random.uniform(-500, 1000), 2),
                    "date": (datetime.now() - timedelta(days=i)).isoformat(),
                    "description": f"Transaction {i}"
                }
                for i in range(5)
            ],
            "currency": "USD"
        }
    
    def _generate_general_json(self) -> Dict[str, Any]:
        """Generate general JSON data."""
        return {
            "id": random.randint(1, 1000),
            "name": f"Sample Item {random.randint(1, 100)}",
            "description": "This is a sample JSON document for testing purposes.",
            "tags": ["sample", "test", "data"],
            "active": random.choice([True, False]),
            "created_date": datetime.now().isoformat()
        }
    
    def _generate_corrupted_json(self) -> str:
        """Generate corrupted JSON."""
        # Invalid JSON with syntax errors
        corrupted_samples = [
            '{"key": "value",}',  # Trailing comma
            '{"key": value}',     # Unquoted value
            '{"key": "value"',    # Missing closing brace
            '{key: "value"}',     # Unquoted key
            '{"key": "value" "key2": "value2"}'  # Missing comma
        ]
        return random.choice(corrupted_samples)
    
    def get_expected_metadata(self, config: GenerationConfig) -> Dict[str, Any]:
        """Get expected metadata for JSON content."""
        quality_scores = {
            QualityLevel.HIGH: (0.9, 1.0),
            QualityLevel.MEDIUM: (0.7, 0.9),
            QualityLevel.LOW: (0.4, 0.7),
            QualityLevel.CORRUPTED: (0.0, 0.3)
        }
        
        quality_range = quality_scores[config.quality_level]
        
        return {
            "expected_quality_score": random.uniform(*quality_range),
            "expected_language": "en",  # JSON is typically English
            "expected_domain": "data",
            "expected_file_type": "json"
        }


class CSVContentGenerator(ContentGenerator):
    """Generator for CSV content."""
    
    def generate_content(self, config: GenerationConfig) -> str:
        """Generate CSV content."""
        if config.quality_level == QualityLevel.CORRUPTED:
            return self._generate_corrupted_csv()
        
        # Generate headers and data based on category
        if config.content_category == ContentCategory.FINANCIAL:
            headers, rows = self._generate_financial_csv_data()
        elif config.content_category == ContentCategory.TECHNICAL:
            headers, rows = self._generate_technical_csv_data()
        else:
            headers, rows = self._generate_general_csv_data()
        
        # Format CSV
        lines = [",".join(headers)]
        lines.extend([",".join(map(str, row)) for row in rows])
        
        return "\n".join(lines)
    
    def _generate_financial_csv_data(self) -> Tuple[List[str], List[List[Any]]]:
        """Generate financial CSV data."""
        headers = ["Date", "Amount", "Description", "Category", "Balance"]
        rows = []
        
        balance = 10000.0
        for i in range(10):
            amount = round(random.uniform(-500, 1000), 2)
            balance += amount
            
            rows.append([
                (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"),
                amount,
                f"Transaction {i+1}",
                random.choice(["Food", "Transport", "Entertainment", "Utilities"]),
                round(balance, 2)
            ])
        
        return headers, rows
    
    def _generate_technical_csv_data(self) -> Tuple[List[str], List[List[Any]]]:
        """Generate technical CSV data."""
        headers = ["Timestamp", "CPU_Usage", "Memory_Usage", "Disk_IO", "Network_IO"]
        rows = []
        
        for i in range(10):
            rows.append([
                (datetime.now() - timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S"),
                round(random.uniform(10, 90), 1),
                round(random.uniform(20, 80), 1),
                random.randint(100, 1000),
                random.randint(50, 500)
            ])
        
        return headers, rows
    
    def _generate_general_csv_data(self) -> Tuple[List[str], List[List[Any]]]:
        """Generate general CSV data."""
        headers = ["ID", "Name", "Value", "Category", "Status"]
        rows = []
        
        for i in range(10):
            rows.append([
                i + 1,
                f"Item {i + 1}",
                round(random.uniform(1, 100), 2),
                random.choice(["A", "B", "C"]),
                random.choice(["Active", "Inactive"])
            ])
        
        return headers, rows
    
    def _generate_corrupted_csv(self) -> str:
        """Generate corrupted CSV."""
        # CSV with inconsistent columns, missing quotes, etc.
        corrupted_samples = [
            "col1,col2,col3\nval1,val2\nval1,val2,val3,val4",  # Inconsistent columns
            'col1,col2\n"val1,val2\nval3,"val4',  # Unmatched quotes
            "col1;col2\nval1,val2\nval3;val4",  # Mixed separators
        ]
        return random.choice(corrupted_samples)
    
    def get_expected_metadata(self, config: GenerationConfig) -> Dict[str, Any]:
        """Get expected metadata for CSV content."""
        quality_scores = {
            QualityLevel.HIGH: (0.8, 1.0),
            QualityLevel.MEDIUM: (0.6, 0.8),
            QualityLevel.LOW: (0.3, 0.6),
            QualityLevel.CORRUPTED: (0.0, 0.3)
        }
        
        quality_range = quality_scores[config.quality_level]
        
        return {
            "expected_quality_score": random.uniform(*quality_range),
            "expected_language": "en",
            "expected_domain": "data",
            "expected_file_type": "csv"
        }


class HTMLContentGenerator(ContentGenerator):
    """Generator for HTML content."""
    
    def generate_content(self, config: GenerationConfig) -> str:
        """Generate HTML content."""
        if config.quality_level == QualityLevel.CORRUPTED:
            return self._generate_corrupted_html()
        
        # Generate HTML based on category
        if config.content_category == ContentCategory.NEWS:
            return self._generate_news_html()
        elif config.content_category == ContentCategory.ACADEMIC:
            return self._generate_academic_html()
        else:
            return self._generate_general_html()
    
    def _generate_news_html(self) -> str:
        """Generate news article HTML."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Breaking News: Sample Article</title>
        </head>
        <body>
            <h1>Breaking News: Sample Article</h1>
            <p class="byline">By Reporter Name | Published: 2024-01-01</p>
            <p>This is a sample news article generated for testing purposes. 
            The article contains multiple paragraphs with relevant content.</p>
            <p>In a recent development, sources confirm that this is indeed 
            a test article designed to validate HTML processing capabilities.</p>
            <p>More details will be provided as the story develops.</p>
        </body>
        </html>
        """
    
    def _generate_academic_html(self) -> str:
        """Generate academic paper HTML."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Research Paper: Sample Study</title>
        </head>
        <body>
            <h1>Research Paper: Sample Study</h1>
            <h2>Abstract</h2>
            <p>This paper presents a comprehensive analysis of sample data 
            processing techniques in academic research contexts.</p>
            <h2>Introduction</h2>
            <p>The field of data processing has evolved significantly in recent years.
            This study examines the implications of these developments.</p>
            <h2>Methodology</h2>
            <p>We employed a mixed-methods approach to analyze the data.</p>
            <h2>Conclusion</h2>
            <p>The results demonstrate the effectiveness of the proposed approach.</p>
        </body>
        </html>
        """
    
    def _generate_general_html(self) -> str:
        """Generate general HTML content."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sample HTML Document</title>
        </head>
        <body>
            <h1>Sample HTML Document</h1>
            <p>This is a sample HTML document for testing purposes.</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
                <li>Item 3</li>
            </ul>
            <p>The document contains various HTML elements to test parsing.</p>
        </body>
        </html>
        """
    
    def _generate_corrupted_html(self) -> str:
        """Generate corrupted HTML."""
        return """
        <html>
        <head>
            <title>Corrupted HTML
        </head>
        <body>
            <h1>Missing closing tag
            <p>Unclosed paragraph
            <div>Nested without closing</div>
            <img src="missing.jpg"
        </body>
        """
    
    def get_expected_metadata(self, config: GenerationConfig) -> Dict[str, Any]:
        """Get expected metadata for HTML content."""
        quality_scores = {
            QualityLevel.HIGH: (0.7, 0.9),
            QualityLevel.MEDIUM: (0.5, 0.7),
            QualityLevel.LOW: (0.3, 0.5),
            QualityLevel.CORRUPTED: (0.0, 0.3)
        }
        
        quality_range = quality_scores[config.quality_level]
        
        return {
            "expected_quality_score": random.uniform(*quality_range),
            "expected_language": config.language,
            "expected_domain": "web",
            "expected_file_type": "html"
        }


class SampleDataGenerator:
    """Main class for generating sample test data."""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize sample data generator.
        
        Args:
            output_dir: Directory to save generated files
        """
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir()) / "sample_data"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize content generators
        self.generators = {
            DataType.TEXT: TextContentGenerator(),
            DataType.JSON: JSONContentGenerator(),
            DataType.CSV: CSVContentGenerator(),
            DataType.HTML: HTMLContentGenerator()
        }
        
        self.generated_files: List[str] = []
    
    def generate_sample_document(self, config: GenerationConfig) -> SampleDocument:
        """
        Generate a single sample document.
        
        Args:
            config: Generation configuration
            
        Returns:
            SampleDocument with generated content
        """
        if config.data_type not in self.generators:
            raise ValueError(f"Unsupported data type: {config.data_type}")
        
        generator = self.generators[config.data_type]
        
        # Generate content
        content = generator.generate_content(config)
        
        # Get expected metadata
        expected_metadata = generator.get_expected_metadata(config)
        
        # Create sample document
        document = SampleDocument(
            content=content,
            metadata={
                "size_bytes": len(content.encode('utf-8')),
                "generation_timestamp": datetime.now().isoformat(),
                "config": config.to_dict()
            },
            expected_quality_score=expected_metadata.get("expected_quality_score"),
            expected_language=expected_metadata.get("expected_language"),
            expected_domain=expected_metadata.get("expected_domain"),
            generation_config=config
        )
        
        return document
    
    def generate_sample_dataset(self, configs: List[GenerationConfig], 
                              save_files: bool = True) -> List[SampleDocument]:
        """
        Generate a dataset of sample documents.
        
        Args:
            configs: List of generation configurations
            save_files: Whether to save documents to files
            
        Returns:
            List of generated sample documents
        """
        documents = []
        
        for i, config in enumerate(configs):
            document = self.generate_sample_document(config)
            
            if save_files:
                # Save to file
                file_extension = config.data_type.value
                filename = f"sample_{i:03d}.{file_extension}"
                file_path = self.output_dir / filename
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(document.content)
                
                document.file_path = str(file_path)
                self.generated_files.append(str(file_path))
            
            documents.append(document)
        
        return documents
    
    def generate_test_suite_data(self, suite_name: str = "default") -> Dict[str, List[SampleDocument]]:
        """
        Generate a comprehensive test suite with various data types and quality levels.
        
        Args:
            suite_name: Name of the test suite
            
        Returns:
            Dictionary of test categories and their sample documents
        """
        test_suite = {}
        
        # High-quality samples
        high_quality_configs = [
            GenerationConfig(DataType.TEXT, ContentCategory.TECHNICAL, QualityLevel.HIGH),
            GenerationConfig(DataType.JSON, ContentCategory.TECHNICAL, QualityLevel.HIGH),
            GenerationConfig(DataType.CSV, ContentCategory.FINANCIAL, QualityLevel.HIGH),
            GenerationConfig(DataType.HTML, ContentCategory.NEWS, QualityLevel.HIGH)
        ]
        test_suite["high_quality"] = self.generate_sample_dataset(high_quality_configs)
        
        # Medium-quality samples
        medium_quality_configs = [
            GenerationConfig(DataType.TEXT, ContentCategory.GENERAL, QualityLevel.MEDIUM),
            GenerationConfig(DataType.JSON, ContentCategory.GENERAL, QualityLevel.MEDIUM),
            GenerationConfig(DataType.CSV, ContentCategory.GENERAL, QualityLevel.MEDIUM)
        ]
        test_suite["medium_quality"] = self.generate_sample_dataset(medium_quality_configs)
        
        # Low-quality samples
        low_quality_configs = [
            GenerationConfig(DataType.TEXT, ContentCategory.GENERAL, QualityLevel.LOW),
            GenerationConfig(DataType.JSON, ContentCategory.GENERAL, QualityLevel.LOW)
        ]
        test_suite["low_quality"] = self.generate_sample_dataset(low_quality_configs)
        
        # Corrupted samples
        corrupted_configs = [
            GenerationConfig(DataType.TEXT, ContentCategory.CORRUPTED, QualityLevel.CORRUPTED),
            GenerationConfig(DataType.JSON, ContentCategory.CORRUPTED, QualityLevel.CORRUPTED),
            GenerationConfig(DataType.HTML, ContentCategory.CORRUPTED, QualityLevel.CORRUPTED)
        ]
        test_suite["corrupted"] = self.generate_sample_dataset(corrupted_configs)
        
        # Large files for performance testing
        large_file_configs = [
            GenerationConfig(
                DataType.TEXT, 
                ContentCategory.TECHNICAL, 
                QualityLevel.HIGH,
                size_range=(50000, 100000)  # 50-100KB
            )
        ]
        test_suite["large_files"] = self.generate_sample_dataset(large_file_configs)
        
        # Multi-language samples
        multilang_configs = [
            GenerationConfig(DataType.TEXT, ContentCategory.GENERAL, QualityLevel.HIGH, language="es"),
            GenerationConfig(DataType.TEXT, ContentCategory.GENERAL, QualityLevel.HIGH, language="fr"),
            GenerationConfig(DataType.TEXT, ContentCategory.GENERAL, QualityLevel.HIGH, language="de")
        ]
        test_suite["multilingual"] = self.generate_sample_dataset(multilang_configs)
        
        return test_suite
    
    def cleanup_generated_files(self) -> None:
        """Clean up all generated files."""
        for file_path in self.generated_files:
            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception:
                pass  # Ignore cleanup errors
        self.generated_files.clear()
    
    def save_test_suite_metadata(self, test_suite: Dict[str, List[SampleDocument]], 
                                output_path: str) -> None:
        """
        Save test suite metadata to file.
        
        Args:
            test_suite: Test suite data
            output_path: Path to save metadata
        """
        metadata = {
            "generation_timestamp": datetime.now().isoformat(),
            "categories": {},
            "total_documents": 0
        }
        
        for category, documents in test_suite.items():
            metadata["categories"][category] = {
                "document_count": len(documents),
                "documents": [doc.to_dict() for doc in documents]
            }
            metadata["total_documents"] += len(documents)
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_test_suite_metadata(self, metadata_path: str) -> Dict[str, Any]:
        """
        Load test suite metadata from file.
        
        Args:
            metadata_path: Path to metadata file
            
        Returns:
            Test suite metadata
        """
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def generate_regression_test_data(self) -> List[SampleDocument]:
        """
        Generate test data specifically for regression testing.
        
        Returns:
            List of sample documents with known characteristics for regression testing
        """
        regression_configs = []
        
        # Create configs with different quality levels and categories
        for quality in [QualityLevel.HIGH, QualityLevel.MEDIUM, QualityLevel.LOW]:
            for category in [ContentCategory.TECHNICAL, ContentCategory.GENERAL, ContentCategory.ACADEMIC]:
                for data_type in [DataType.TEXT, DataType.JSON, DataType.CSV]:
                    config = GenerationConfig(
                        data_type=data_type,
                        content_category=category,
                        quality_level=quality,
                        size_range=(200, 500)
                    )
                    regression_configs.append(config)
        
        return self.generate_sample_dataset(regression_configs, save_files=True)
    
    def save_test_suite_metadata(self, test_suite: Dict[str, List[SampleDocument]], 
                                filepath: str) -> None:
        """
        Save test suite metadata to file.
        
        Args:
            test_suite: Test suite data
            filepath: Path to save metadata
        """
        metadata = {}
        
        for category, documents in test_suite.items():
            metadata[category] = []
            for doc in documents:
                doc_metadata = {
                    "content_length": len(doc.content),
                    "expected_quality_score": doc.expected_quality_score,
                    "expected_language": doc.expected_language,
                    "expected_domain": doc.expected_domain,
                    "file_path": doc.file_path,
                    "generation_config": doc.generation_config.to_dict() if doc.generation_config else None,
                    "metadata": doc.metadata
                }
                metadata[category].append(doc_metadata)
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_test_suite_metadata(self, filepath: str) -> Dict[str, Any]:
        """
        Load test suite metadata from file.
        
        Args:
            filepath: Path to metadata file
            
        Returns:
            Loaded metadata dictionary
        """
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def cleanup_generated_files(self) -> None:
        """Clean up all generated files."""
        for file_path in self.generated_files:
            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception:
                pass  # Ignore cleanup errors
        
        self.generated_files.clear()
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about generated data.
        
        Returns:
            Dictionary with generation statistics
        """
        return {
            "total_files_generated": len(self.generated_files),
            "output_directory": str(self.output_dir),
            "available_generators": list(self.generators.keys()),
            "supported_data_types": [dt.value for dt in DataType],
            "supported_categories": [cc.value for cc in ContentCategory],
            "supported_quality_levels": [ql.value for ql in QualityLevel]
        }
        
        return test_suite
    
    def generate_regression_test_data(self) -> List[SampleDocument]:
        """Generate data specifically for regression testing."""
        configs = [
            # Baseline configurations that should always work
            GenerationConfig(DataType.TEXT, ContentCategory.GENERAL, QualityLevel.HIGH),
            GenerationConfig(DataType.JSON, ContentCategory.GENERAL, QualityLevel.HIGH),
            GenerationConfig(DataType.CSV, ContentCategory.GENERAL, QualityLevel.HIGH),
            
            # Edge cases
            GenerationConfig(DataType.TEXT, ContentCategory.GENERAL, QualityLevel.HIGH, size_range=(1, 10)),  # Very small
            GenerationConfig(DataType.TEXT, ContentCategory.GENERAL, QualityLevel.HIGH, size_range=(10000, 20000)),  # Large
            
            # Quality variations
            GenerationConfig(DataType.TEXT, ContentCategory.GENERAL, QualityLevel.MEDIUM),
            GenerationConfig(DataType.TEXT, ContentCategory.GENERAL, QualityLevel.LOW),
        ]
        
        return self.generate_sample_dataset(configs)
    
    def cleanup_generated_files(self) -> None:
        """Clean up all generated files."""
        for file_path in self.generated_files:
            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception:
                pass  # Ignore cleanup errors
        
        self.generated_files.clear()
        
        # Remove output directory if empty
        try:
            if self.output_dir.exists() and not any(self.output_dir.iterdir()):
                self.output_dir.rmdir()
        except Exception:
            pass  # Ignore cleanup errors
    
    def save_test_suite_metadata(self, test_suite: Dict[str, List[SampleDocument]], 
                                filename: str = "test_suite_metadata.json") -> str:
        """
        Save test suite metadata to a JSON file.
        
        Args:
            test_suite: Test suite data
            filename: Output filename
            
        Returns:
            Path to saved metadata file
        """
        metadata = {}
        
        for category, documents in test_suite.items():
            metadata[category] = [doc.to_dict() for doc in documents]
        
        metadata_path = self.output_dir / filename
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
    
    def generate_regression_test_data(self) -> List[SampleDocument]:
        """
        Generate test data specifically for regression testing.
        
        Returns:
            List of sample documents with known characteristics for regression testing
        """
        regression_configs = [
            # Known high-quality technical document
            GenerationConfig(
                DataType.TEXT, 
                ContentCategory.TECHNICAL, 
                QualityLevel.HIGH,
                size_range=(500, 1000)
            ),
            
            # Known medium-quality general document
            GenerationConfig(
                DataType.TEXT, 
                ContentCategory.GENERAL, 
                QualityLevel.MEDIUM,
                size_range=(200, 400)
            ),
            
            # Known structured data
            GenerationConfig(
                DataType.JSON, 
                ContentCategory.TECHNICAL, 
                QualityLevel.HIGH
            ),
            
            # Known CSV data
            GenerationConfig(
                DataType.CSV, 
                ContentCategory.FINANCIAL, 
                QualityLevel.HIGH
            )
        ]
        
        return self.generate_sample_dataset(regression_configs, save_files=True)
    
    def generate_stress_test_data(self, document_count: int = 100) -> List[SampleDocument]:
        """
        Generate large amounts of test data for stress testing.
        
        Args:
            document_count: Number of documents to generate
            
        Returns:
            List of sample documents for stress testing
        """
        stress_configs = []
        
        # Generate varied configurations
        data_types = [DataType.TEXT, DataType.JSON, DataType.CSV, DataType.HTML]
        categories = [ContentCategory.TECHNICAL, ContentCategory.GENERAL, ContentCategory.NEWS]
        qualities = [QualityLevel.HIGH, QualityLevel.MEDIUM, QualityLevel.LOW]
        
        for i in range(document_count):
            config = GenerationConfig(
                data_type=random.choice(data_types),
                content_category=random.choice(categories),
                quality_level=random.choice(qualities),
                size_range=(100, 2000)
            )
            stress_configs.append(config)
        
        return self.generate_sample_dataset(stress_configs, save_files=True)
    
    def get_generation_statistics(self, documents: List[SampleDocument]) -> Dict[str, Any]:
        """
        Get statistics about generated documents.
        
        Args:
            documents: List of generated documents
            
        Returns:
            Statistics about the generated documents
        """
        if not documents:
            return {"total_documents": 0}
        
        # Count by data type
        data_type_counts = {}
        category_counts = {}
        quality_counts = {}
        language_counts = {}
        
        total_size = 0
        quality_scores = []
        
        for doc in documents:
            if doc.generation_config:
                # Data type counts
                dt = doc.generation_config.data_type.value
                data_type_counts[dt] = data_type_counts.get(dt, 0) + 1
                
                # Category counts
                cat = doc.generation_config.content_category.value
                category_counts[cat] = category_counts.get(cat, 0) + 1
                
                # Quality counts
                qual = doc.generation_config.quality_level.value
                quality_counts[qual] = quality_counts.get(qual, 0) + 1
                
                # Language counts
                lang = doc.generation_config.language
                language_counts[lang] = language_counts.get(lang, 0) + 1
            
            # Size and quality
            total_size += len(doc.content)
            if doc.expected_quality_score:
                quality_scores.append(doc.expected_quality_score)
        
        return {
            "total_documents": len(documents),
            "total_size_chars": total_size,
            "average_size_chars": total_size / len(documents),
            "data_type_distribution": data_type_counts,
            "category_distribution": category_counts,
            "quality_distribution": quality_counts,
            "language_distribution": language_counts,
            "average_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "quality_score_range": (min(quality_scores), max(quality_scores)) if quality_scores else (0, 0)
        }