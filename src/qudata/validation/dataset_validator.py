"""
Dataset validation for schema compliance and data integrity.

This module provides comprehensive validation of datasets to ensure they meet
quality standards and schema requirements before being used for training.
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..models import Dataset, Document, DocumentMetadata, QualityMetrics


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(Enum):
    """Categories of validation checks."""
    SCHEMA = "schema"
    CONTENT = "content"
    QUALITY = "quality"
    CONSISTENCY = "consistency"
    COMPLETENESS = "completeness"


@dataclass
class ValidationIssue:
    """Represents a validation issue found in a dataset."""
    category: ValidationCategory
    severity: ValidationSeverity
    message: str
    document_id: Optional[str] = None
    field_name: Optional[str] = None
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    suggestion: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation issue to dictionary."""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "document_id": self.document_id,
            "field_name": self.field_name,
            "expected_value": self.expected_value,
            "actual_value": self.actual_value,
            "suggestion": self.suggestion
        }


@dataclass
class ValidationResult:
    """Result of dataset validation."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    validation_timestamp: datetime = field(default_factory=datetime.now)
    validation_duration: float = 0.0
    documents_validated: int = 0
    rules_applied: List[str] = field(default_factory=list)
    
    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue."""
        self.issues.append(issue)
        if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.is_valid = False
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get all issues of a specific severity."""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def get_issues_by_category(self, category: ValidationCategory) -> List[ValidationIssue]:
        """Get all issues of a specific category."""
        return [issue for issue in self.issues if issue.category == category]
    
    def get_critical_issues(self) -> List[ValidationIssue]:
        """Get all critical issues."""
        return self.get_issues_by_severity(ValidationSeverity.CRITICAL)
    
    def get_error_issues(self) -> List[ValidationIssue]:
        """Get all error issues."""
        return self.get_issues_by_severity(ValidationSeverity.ERROR)
    
    def has_critical_issues(self) -> bool:
        """Check if there are any critical issues."""
        return len(self.get_critical_issues()) > 0
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.get_error_issues()) > 0
    
    def get_summary(self) -> Dict[str, int]:
        """Get summary of issues by severity."""
        summary = {severity.value: 0 for severity in ValidationSeverity}
        for issue in self.issues:
            summary[issue.severity.value] += 1
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary."""
        return {
            "is_valid": self.is_valid,
            "issues": [issue.to_dict() for issue in self.issues],
            "validation_timestamp": self.validation_timestamp.isoformat(),
            "validation_duration": self.validation_duration,
            "documents_validated": self.documents_validated,
            "rules_applied": self.rules_applied,
            "summary": self.get_summary()
        }


class ValidationRule(ABC):
    """Abstract base class for validation rules."""
    
    def __init__(self, name: str, description: str, category: ValidationCategory):
        """
        Initialize validation rule.
        
        Args:
            name: Name of the validation rule
            description: Description of what the rule validates
            category: Category of validation this rule belongs to
        """
        self.name = name
        self.description = description
        self.category = category
    
    @abstractmethod
    def validate(self, dataset: Dataset) -> List[ValidationIssue]:
        """
        Validate a dataset against this rule.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            List of validation issues found
        """
        pass
    
    def create_issue(self, severity: ValidationSeverity, message: str,
                    document_id: str = None, field_name: str = None,
                    expected_value: Any = None, actual_value: Any = None,
                    suggestion: str = None) -> ValidationIssue:
        """Helper method to create validation issues."""
        return ValidationIssue(
            category=self.category,
            severity=severity,
            message=message,
            document_id=document_id,
            field_name=field_name,
            expected_value=expected_value,
            actual_value=actual_value,
            suggestion=suggestion
        )


class SchemaValidationRule(ValidationRule):
    """Validates dataset schema compliance."""
    
    def __init__(self):
        super().__init__(
            name="schema_validation",
            description="Validates dataset schema compliance",
            category=ValidationCategory.SCHEMA
        )
    
    def validate(self, dataset: Dataset) -> List[ValidationIssue]:
        """Validate dataset schema."""
        issues = []
        
        # Validate dataset-level fields
        if not dataset.id:
            issues.append(self.create_issue(
                ValidationSeverity.CRITICAL,
                "Dataset ID is required",
                field_name="id",
                suggestion="Provide a unique dataset identifier"
            ))
        
        if not dataset.name:
            issues.append(self.create_issue(
                ValidationSeverity.ERROR,
                "Dataset name is required",
                field_name="name",
                suggestion="Provide a descriptive dataset name"
            ))
        
        if not dataset.version:
            issues.append(self.create_issue(
                ValidationSeverity.ERROR,
                "Dataset version is required",
                field_name="version",
                suggestion="Provide a version string (e.g., '1.0')"
            ))
        
        # Validate documents
        for doc in dataset.documents:
            doc_issues = self._validate_document_schema(doc)
            issues.extend(doc_issues)
        
        return issues
    
    def _validate_document_schema(self, document: Document) -> List[ValidationIssue]:
        """Validate individual document schema."""
        issues = []
        
        # Required fields
        if not document.id:
            issues.append(self.create_issue(
                ValidationSeverity.CRITICAL,
                "Document ID is required",
                document_id=document.id,
                field_name="id"
            ))
        
        if not document.source_path:
            issues.append(self.create_issue(
                ValidationSeverity.ERROR,
                "Document source path is required",
                document_id=document.id,
                field_name="source_path"
            ))
        
        if not document.content:
            issues.append(self.create_issue(
                ValidationSeverity.ERROR,
                "Document content is empty",
                document_id=document.id,
                field_name="content",
                suggestion="Ensure document has extractable content"
            ))
        
        # Validate metadata
        if document.metadata:
            metadata_issues = self._validate_metadata_schema(document.metadata, document.id)
            issues.extend(metadata_issues)
        else:
            issues.append(self.create_issue(
                ValidationSeverity.ERROR,
                "Document metadata is missing",
                document_id=document.id,
                field_name="metadata"
            ))
        
        return issues
    
    def _validate_metadata_schema(self, metadata: DocumentMetadata, document_id: str) -> List[ValidationIssue]:
        """Validate document metadata schema."""
        issues = []
        
        if not metadata.file_type:
            issues.append(self.create_issue(
                ValidationSeverity.ERROR,
                "File type is required in metadata",
                document_id=document_id,
                field_name="metadata.file_type"
            ))
        
        if metadata.size_bytes <= 0:
            issues.append(self.create_issue(
                ValidationSeverity.WARNING,
                "File size should be positive",
                document_id=document_id,
                field_name="metadata.size_bytes",
                actual_value=metadata.size_bytes
            ))
        
        if not metadata.language:
            issues.append(self.create_issue(
                ValidationSeverity.WARNING,
                "Language not detected",
                document_id=document_id,
                field_name="metadata.language",
                suggestion="Run language detection on content"
            ))
        
        return issues


class ContentValidationRule(ValidationRule):
    """Validates document content quality and structure."""
    
    def __init__(self, min_content_length: int = 10, max_content_length: int = 1000000):
        super().__init__(
            name="content_validation",
            description="Validates document content quality and structure",
            category=ValidationCategory.CONTENT
        )
        self.min_content_length = min_content_length
        self.max_content_length = max_content_length
    
    def validate(self, dataset: Dataset) -> List[ValidationIssue]:
        """Validate document content."""
        issues = []
        
        for doc in dataset.documents:
            # Content length validation
            content_length = len(doc.content)
            
            if content_length < self.min_content_length:
                issues.append(self.create_issue(
                    ValidationSeverity.WARNING,
                    f"Content too short ({content_length} characters)",
                    document_id=doc.id,
                    field_name="content",
                    expected_value=f">= {self.min_content_length}",
                    actual_value=content_length,
                    suggestion="Consider removing very short documents"
                ))
            
            if content_length > self.max_content_length:
                issues.append(self.create_issue(
                    ValidationSeverity.WARNING,
                    f"Content very long ({content_length} characters)",
                    document_id=doc.id,
                    field_name="content",
                    expected_value=f"<= {self.max_content_length}",
                    actual_value=content_length,
                    suggestion="Consider splitting long documents"
                ))
            
            # Content quality checks
            if self._is_mostly_whitespace(doc.content):
                issues.append(self.create_issue(
                    ValidationSeverity.ERROR,
                    "Content is mostly whitespace",
                    document_id=doc.id,
                    field_name="content",
                    suggestion="Clean whitespace or remove document"
                ))
            
            if self._has_encoding_issues(doc.content):
                issues.append(self.create_issue(
                    ValidationSeverity.WARNING,
                    "Content may have encoding issues",
                    document_id=doc.id,
                    field_name="content",
                    suggestion="Check and fix text encoding"
                ))
            
            if self._is_likely_corrupted(doc.content):
                issues.append(self.create_issue(
                    ValidationSeverity.ERROR,
                    "Content appears corrupted",
                    document_id=doc.id,
                    field_name="content",
                    suggestion="Re-extract content from source"
                ))
        
        return issues
    
    def _is_mostly_whitespace(self, content: str) -> bool:
        """Check if content is mostly whitespace."""
        if not content:
            return True
        non_whitespace = len(content.strip())
        return non_whitespace / len(content) < 0.1
    
    def _has_encoding_issues(self, content: str) -> bool:
        """Check for common encoding issues."""
        # Look for common encoding artifacts
        encoding_artifacts = ['�', '\ufffd', '\x00', '\x01', '\x02']
        return any(artifact in content for artifact in encoding_artifacts)
    
    def _is_likely_corrupted(self, content: str) -> bool:
        """Check if content appears corrupted."""
        if not content:
            return True
        
        # Check for excessive repetition of characters
        char_counts = {}
        for char in content:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # If any single character makes up more than 50% of content, likely corrupted
        max_char_ratio = max(char_counts.values()) / len(content)
        return max_char_ratio > 0.5


class QualityValidationRule(ValidationRule):
    """Validates dataset quality metrics."""
    
    def __init__(self, min_quality_score: float = 0.3):
        super().__init__(
            name="quality_validation",
            description="Validates dataset quality metrics",
            category=ValidationCategory.QUALITY
        )
        self.min_quality_score = min_quality_score
    
    def validate(self, dataset: Dataset) -> List[ValidationIssue]:
        """Validate quality metrics."""
        issues = []
        
        # Validate dataset-level quality metrics
        if dataset.quality_metrics.overall_score < self.min_quality_score:
            issues.append(self.create_issue(
                ValidationSeverity.WARNING,
                f"Dataset quality score below threshold ({dataset.quality_metrics.overall_score:.2f})",
                field_name="quality_metrics.overall_score",
                expected_value=f">= {self.min_quality_score}",
                actual_value=dataset.quality_metrics.overall_score,
                suggestion="Review and improve data quality"
            ))
        
        # Validate document-level quality
        for doc in dataset.documents:
            if doc.metadata.quality_score < self.min_quality_score:
                issues.append(self.create_issue(
                    ValidationSeverity.INFO,
                    f"Document quality score below threshold ({doc.metadata.quality_score:.2f})",
                    document_id=doc.id,
                    field_name="metadata.quality_score",
                    expected_value=f">= {self.min_quality_score}",
                    actual_value=doc.metadata.quality_score,
                    suggestion="Consider removing low-quality documents"
                ))
        
        return issues


class ConsistencyValidationRule(ValidationRule):
    """Validates data consistency across the dataset."""
    
    def __init__(self):
        super().__init__(
            name="consistency_validation",
            description="Validates data consistency across the dataset",
            category=ValidationCategory.CONSISTENCY
        )
    
    def validate(self, dataset: Dataset) -> List[ValidationIssue]:
        """Validate data consistency."""
        issues = []
        
        # Check for duplicate document IDs
        doc_ids = [doc.id for doc in dataset.documents]
        duplicate_ids = self._find_duplicates(doc_ids)
        
        for duplicate_id in duplicate_ids:
            issues.append(self.create_issue(
                ValidationSeverity.CRITICAL,
                f"Duplicate document ID found: {duplicate_id}",
                document_id=duplicate_id,
                field_name="id",
                suggestion="Ensure all document IDs are unique"
            ))
        
        # Check for inconsistent metadata
        file_types = set()
        languages = set()
        domains = set()
        
        for doc in dataset.documents:
            if doc.metadata.file_type:
                file_types.add(doc.metadata.file_type)
            if doc.metadata.language:
                languages.add(doc.metadata.language)
            if doc.metadata.domain:
                domains.add(doc.metadata.domain)
        
        # Warn about excessive diversity (might indicate inconsistent processing)
        if len(file_types) > 10:
            issues.append(self.create_issue(
                ValidationSeverity.INFO,
                f"Many different file types ({len(file_types)}) in dataset",
                field_name="metadata.file_type",
                actual_value=len(file_types),
                suggestion="Verify file type detection is working correctly"
            ))
        
        if len(languages) > 20:
            issues.append(self.create_issue(
                ValidationSeverity.WARNING,
                f"Many different languages ({len(languages)}) detected",
                field_name="metadata.language",
                actual_value=len(languages),
                suggestion="Review language detection accuracy"
            ))
        
        return issues
    
    def _find_duplicates(self, items: List[str]) -> Set[str]:
        """Find duplicate items in a list."""
        seen = set()
        duplicates = set()
        
        for item in items:
            if item in seen:
                duplicates.add(item)
            else:
                seen.add(item)
        
        return duplicates


class CompletenessValidationRule(ValidationRule):
    """Validates dataset completeness."""
    
    def __init__(self):
        super().__init__(
            name="completeness_validation",
            description="Validates dataset completeness",
            category=ValidationCategory.COMPLETENESS
        )
    
    def validate(self, dataset: Dataset) -> List[ValidationIssue]:
        """Validate dataset completeness."""
        issues = []
        
        # Check if dataset is empty
        if not dataset.documents:
            issues.append(self.create_issue(
                ValidationSeverity.CRITICAL,
                "Dataset contains no documents",
                field_name="documents",
                suggestion="Add documents to the dataset"
            ))
            return issues
        
        # Check for missing metadata fields
        missing_authors = 0
        missing_dates = 0
        missing_domains = 0
        missing_languages = 0
        
        for doc in dataset.documents:
            if not doc.metadata.author:
                missing_authors += 1
            if not doc.metadata.creation_date:
                missing_dates += 1
            if not doc.metadata.domain or doc.metadata.domain == "uncategorized":
                missing_domains += 1
            if not doc.metadata.language or doc.metadata.language == "unknown":
                missing_languages += 1
        
        total_docs = len(dataset.documents)
        
        # Report missing metadata as warnings if significant portion is missing
        if missing_authors / total_docs > 0.5:
            issues.append(self.create_issue(
                ValidationSeverity.INFO,
                f"Author information missing for {missing_authors}/{total_docs} documents",
                field_name="metadata.author",
                suggestion="Extract author information where available"
            ))
        
        if missing_dates / total_docs > 0.5:
            issues.append(self.create_issue(
                ValidationSeverity.INFO,
                f"Creation date missing for {missing_dates}/{total_docs} documents",
                field_name="metadata.creation_date",
                suggestion="Extract creation dates where available"
            ))
        
        if missing_domains / total_docs > 0.3:
            issues.append(self.create_issue(
                ValidationSeverity.WARNING,
                f"Domain classification missing for {missing_domains}/{total_docs} documents",
                field_name="metadata.domain",
                suggestion="Run domain classification on documents"
            ))
        
        if missing_languages / total_docs > 0.1:
            issues.append(self.create_issue(
                ValidationSeverity.WARNING,
                f"Language detection missing for {missing_languages}/{total_docs} documents",
                field_name="metadata.language",
                suggestion="Run language detection on documents"
            ))
        
        return issues


class DatasetValidator:
    """Main dataset validator that applies multiple validation rules."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize dataset validator.
        
        Args:
            config: Configuration for validation rules and thresholds
        """
        self.config = config or {}
        self.rules: List[ValidationRule] = []
        self._setup_default_rules()
    
    def _setup_default_rules(self) -> None:
        """Setup default validation rules."""
        # Schema validation
        self.rules.append(SchemaValidationRule())
        
        # Content validation
        min_length = self.config.get("min_content_length", 10)
        max_length = self.config.get("max_content_length", 1000000)
        self.rules.append(ContentValidationRule(min_length, max_length))
        
        # Quality validation
        min_quality = self.config.get("min_quality_score", 0.3)
        self.rules.append(QualityValidationRule(min_quality))
        
        # Consistency validation
        self.rules.append(ConsistencyValidationRule())
        
        # Completeness validation
        self.rules.append(CompletenessValidationRule())
    
    def add_rule(self, rule: ValidationRule) -> None:
        """Add a custom validation rule."""
        self.rules.append(rule)
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a validation rule by name."""
        original_count = len(self.rules)
        self.rules = [rule for rule in self.rules if rule.name != rule_name]
        return len(self.rules) < original_count
    
    def validate(self, dataset: Dataset) -> ValidationResult:
        """
        Validate a dataset against all configured rules.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            ValidationResult with all issues found
        """
        start_time = datetime.now()
        result = ValidationResult(is_valid=True)
        
        # Apply each validation rule
        for rule in self.rules:
            try:
                issues = rule.validate(dataset)
                for issue in issues:
                    result.add_issue(issue)
                result.rules_applied.append(rule.name)
            except Exception as e:
                # If a rule fails, add it as a critical issue
                result.add_issue(ValidationIssue(
                    category=ValidationCategory.SCHEMA,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Validation rule '{rule.name}' failed: {str(e)}",
                    suggestion="Check validation rule implementation"
                ))
        
        # Set result metadata
        result.documents_validated = len(dataset.documents)
        result.validation_duration = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def validate_json_schema(self, dataset_dict: Dict[str, Any], 
                           schema_path: str = None) -> ValidationResult:
        """
        Validate dataset against JSON schema.
        
        Args:
            dataset_dict: Dataset as dictionary
            schema_path: Path to JSON schema file
            
        Returns:
            ValidationResult with schema validation issues
        """
        result = ValidationResult(is_valid=True)
        
        try:
            import jsonschema
            
            # Load schema
            if schema_path:
                with open(schema_path, 'r') as f:
                    schema = json.load(f)
            else:
                schema = self._get_default_schema()
            
            # Validate against schema
            validator = jsonschema.Draft7Validator(schema)
            errors = list(validator.iter_errors(dataset_dict))
            
            for error in errors:
                result.add_issue(ValidationIssue(
                    category=ValidationCategory.SCHEMA,
                    severity=ValidationSeverity.ERROR,
                    message=f"Schema validation error: {error.message}",
                    field_name=".".join(str(p) for p in error.absolute_path),
                    suggestion="Fix schema compliance issue"
                ))
        
        except ImportError:
            result.add_issue(ValidationIssue(
                category=ValidationCategory.SCHEMA,
                severity=ValidationSeverity.WARNING,
                message="jsonschema library not available for schema validation",
                suggestion="Install jsonschema library for schema validation"
            ))
        except Exception as e:
            result.add_issue(ValidationIssue(
                category=ValidationCategory.SCHEMA,
                severity=ValidationSeverity.ERROR,
                message=f"Schema validation failed: {str(e)}",
                suggestion="Check schema file and dataset format"
            ))
        
        return result
    
    def _get_default_schema(self) -> Dict[str, Any]:
        """Get default JSON schema for dataset validation."""
        return {
            "type": "object",
            "required": ["id", "name", "version", "documents"],
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "name": {"type": "string", "minLength": 1},
                "version": {"type": "string", "minLength": 1},
                "documents": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["id", "source_path", "content", "metadata"],
                        "properties": {
                            "id": {"type": "string", "minLength": 1},
                            "source_path": {"type": "string", "minLength": 1},
                            "content": {"type": "string", "minLength": 1},
                            "metadata": {
                                "type": "object",
                                "required": ["file_type", "size_bytes", "language"],
                                "properties": {
                                    "file_type": {"type": "string", "minLength": 1},
                                    "size_bytes": {"type": "integer", "minimum": 0},
                                    "language": {"type": "string", "minLength": 1}
                                }
                            }
                        }
                    }
                }
            }
        }
    
    def generate_report(self, result: ValidationResult, format: str = "text") -> str:
        """
        Generate a validation report.
        
        Args:
            result: Validation result to report on
            format: Report format ("text", "json", "html")
            
        Returns:
            Formatted validation report
        """
        if format == "json":
            return json.dumps(result.to_dict(), indent=2)
        elif format == "html":
            return self._generate_html_report(result)
        else:
            return self._generate_text_report(result)
    
    def _generate_text_report(self, result: ValidationResult) -> str:
        """Generate text format validation report."""
        lines = []
        lines.append("Dataset Validation Report")
        lines.append("=" * 50)
        lines.append(f"Validation Status: {'PASSED' if result.is_valid else 'FAILED'}")
        lines.append(f"Documents Validated: {result.documents_validated}")
        lines.append(f"Validation Duration: {result.validation_duration:.2f}s")
        lines.append(f"Rules Applied: {', '.join(result.rules_applied)}")
        lines.append("")
        
        # Summary
        summary = result.get_summary()
        lines.append("Issue Summary:")
        for severity, count in summary.items():
            if count > 0:
                lines.append(f"  {severity.upper()}: {count}")
        lines.append("")
        
        # Issues by category
        for category in ValidationCategory:
            category_issues = result.get_issues_by_category(category)
            if category_issues:
                lines.append(f"{category.value.upper()} Issues:")
                for issue in category_issues:
                    lines.append(f"  [{issue.severity.value.upper()}] {issue.message}")
                    if issue.document_id:
                        lines.append(f"    Document: {issue.document_id}")
                    if issue.suggestion:
                        lines.append(f"    Suggestion: {issue.suggestion}")
                lines.append("")
        
        return "\n".join(lines)
    
    def _generate_html_report(self, result: ValidationResult) -> str:
        """Generate HTML format validation report."""
        status_color = "green" if result.is_valid else "red"
        status_text = "PASSED" if result.is_valid else "FAILED"
        
        html = f"""
        <html>
        <head>
            <title>Dataset Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
                .status {{ color: {status_color}; font-weight: bold; }}
                .issue {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }}
                .critical {{ border-left-color: #ff0000; }}
                .error {{ border-left-color: #ff6600; }}
                .warning {{ border-left-color: #ffcc00; }}
                .info {{ border-left-color: #0066ff; }}
                .suggestion {{ font-style: italic; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Dataset Validation Report</h1>
                <p><strong>Status:</strong> <span class="status">{status_text}</span></p>
                <p><strong>Documents Validated:</strong> {result.documents_validated}</p>
                <p><strong>Duration:</strong> {result.validation_duration:.2f}s</p>
                <p><strong>Rules Applied:</strong> {', '.join(result.rules_applied)}</p>
            </div>
        """
        
        # Add issues
        for category in ValidationCategory:
            category_issues = result.get_issues_by_category(category)
            if category_issues:
                html += f"<h2>{category.value.title()} Issues</h2>"
                for issue in category_issues:
                    html += f'<div class="issue {issue.severity.value}">'
                    html += f'<strong>[{issue.severity.value.upper()}]</strong> {issue.message}'
                    if issue.document_id:
                        html += f'<br><strong>Document:</strong> {issue.document_id}'
                    if issue.suggestion:
                        html += f'<br><span class="suggestion">Suggestion: {issue.suggestion}</span>'
                    html += '</div>'
        
        html += "</body></html>"
        return html