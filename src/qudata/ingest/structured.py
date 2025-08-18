"""
Structured data extraction module for QuData.

This module provides extraction capabilities for structured data formats like CSV and JSON,
with robust error handling for malformed data and conversion to machine-readable formats.
"""

import csv
import json
import os
import re
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from io import StringIO

from ..models import (
    BaseExtractor, ExtractedContent, FileMetadata, DocumentStructure,
    ProcessingError, ErrorSeverity, TableData
)


class StructuredExtractor(BaseExtractor):
    """
    Extractor for structured data files (CSV, JSON).
    
    Handles parsing of structured data formats with robust error handling
    for malformed data and conversion to machine-readable internal formats.
    """
    
    SUPPORTED_FORMATS = {'csv', 'json', 'jsonl', 'tsv'}
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the StructuredExtractor.
        
        Args:
            config: Configuration dictionary with options like:
                - max_file_size: Maximum file size to process (default: 100MB)
                - csv_delimiter: CSV delimiter character (default: auto-detect)
                - json_max_depth: Maximum JSON nesting depth (default: 10)
                - encoding: File encoding (default: 'utf-8')
                - strict_parsing: Whether to use strict parsing (default: False)
        """
        super().__init__(config)
        self.max_file_size = self.config.get('max_file_size', 100 * 1024 * 1024)  # 100MB
        self.csv_delimiter = self.config.get('csv_delimiter', None)  # Auto-detect if None
        self.json_max_depth = self.config.get('json_max_depth', 10)
        self.encoding = self.config.get('encoding', 'utf-8')
        self.strict_parsing = self.config.get('strict_parsing', False)
    
    def extract(self, file_path: str) -> ExtractedContent:
        """
        Extract content from a structured data file.
        
        Args:
            file_path: Path to the structured data file to extract content from
            
        Returns:
            ExtractedContent object containing the extracted content and metadata
            
        Raises:
            ProcessingError: If extraction fails
        """
        try:
            # Validate file first
            self.validate_file(file_path)
            
            # Get basic file metadata
            file_metadata = self.get_metadata(file_path)
            
            # Check file size
            if file_metadata.size_bytes > self.max_file_size:
                raise self.create_processing_error(
                    stage="extraction",
                    error_type="FileTooLarge",
                    message=f"File size ({file_metadata.size_bytes} bytes) exceeds maximum ({self.max_file_size} bytes)",
                    severity=ErrorSeverity.HIGH
                )
            
            # Extract content based on file type
            if file_metadata.file_type in ['csv', 'tsv']:
                content, structure, tables = self._extract_csv_content(file_path, file_metadata.file_type)
            elif file_metadata.file_type in ['json', 'jsonl']:
                content, structure, tables = self._extract_json_content(file_path, file_metadata.file_type)
            else:
                raise self.create_processing_error(
                    stage="extraction",
                    error_type="UnsupportedFormat",
                    message=f"Unsupported structured data format: {file_metadata.file_type}",
                    severity=ErrorSeverity.HIGH
                )
            
            # Create extracted content object
            extracted = ExtractedContent(content, file_metadata)
            extracted.structure = structure
            extracted.tables = tables
            
            return extracted
            
        except ProcessingError:
            raise
        except Exception as e:
            raise self.create_processing_error(
                stage="extraction",
                error_type="StructuredExtractionError",
                message=f"Failed to extract content from structured file {file_path}: {str(e)}",
                severity=ErrorSeverity.HIGH,
                stack_trace=str(e)
            )
    
    def supports_format(self, file_type: str) -> bool:
        """
        Check if this extractor supports the given file type.
        
        Args:
            file_type: The file type to check
            
        Returns:
            True if the extractor supports this file type, False otherwise
        """
        return file_type.lower() in self.SUPPORTED_FORMATS
    
    def _extract_csv_content(self, file_path: str, file_type: str) -> Tuple[str, DocumentStructure, List[TableData]]:
        """
        Extract content from CSV/TSV files.
        
        Args:
            file_path: Path to the CSV/TSV file
            file_type: File type ('csv' or 'tsv')
            
        Returns:
            Tuple of (content, structure, tables)
            
        Raises:
            ProcessingError: If CSV parsing fails
        """
        try:
            # Determine delimiter
            delimiter = self._detect_csv_delimiter(file_path, file_type)
            
            # Read and parse CSV
            with open(file_path, 'r', encoding=self.encoding, newline='') as csvfile:
                # Detect dialect if not using strict parsing
                if not self.strict_parsing:
                    try:
                        sample = csvfile.read(1024)
                        csvfile.seek(0)
                        sniffer = csv.Sniffer()
                        dialect = sniffer.sniff(sample, delimiters=delimiter)
                        reader = csv.reader(csvfile, dialect)
                    except csv.Error:
                        # Fall back to simple reader with detected delimiter
                        csvfile.seek(0)
                        reader = csv.reader(csvfile, delimiter=delimiter)
                else:
                    reader = csv.reader(csvfile, delimiter=delimiter)
                
                # Read all rows
                rows = []
                headers = []
                row_count = 0
                
                for row_idx, row in enumerate(reader):
                    if row_idx == 0:
                        # First row as headers
                        headers = [str(cell).strip() for cell in row]
                    else:
                        # Data rows
                        clean_row = [str(cell).strip() for cell in row]
                        rows.append(clean_row)
                    row_count += 1
                
                if row_count == 0:
                    raise self.create_processing_error(
                        stage="extraction",
                        error_type="EmptyFile",
                        message=f"CSV file {file_path} is empty",
                        severity=ErrorSeverity.MEDIUM
                    )
                
                # Create table data
                table_data = TableData(
                    headers=headers,
                    rows=rows,
                    caption=f"Data from {os.path.basename(file_path)}"
                )
                
                # Convert to text representation
                content = self._csv_to_text(table_data)
                
                # Create structure
                structure = DocumentStructure(
                    headings=[f"Data from {os.path.basename(file_path)}"],
                    paragraphs=0,
                    tables=1,
                    images=0,
                    code_blocks=0,
                    lists=0
                )
                
                return content, structure, [table_data]
                
        except FileNotFoundError:
            raise self.create_processing_error(
                stage="extraction",
                error_type="FileNotFound",
                message=f"CSV file not found: {file_path}",
                severity=ErrorSeverity.HIGH
            )
        except PermissionError:
            raise self.create_processing_error(
                stage="extraction",
                error_type="PermissionError",
                message=f"Permission denied accessing CSV file: {file_path}",
                severity=ErrorSeverity.HIGH
            )
        except UnicodeDecodeError as e:
            raise self.create_processing_error(
                stage="extraction",
                error_type="EncodingError",
                message=f"Could not decode CSV file {file_path} with encoding {self.encoding}: {str(e)}",
                severity=ErrorSeverity.HIGH,
                stack_trace=str(e)
            )
        except csv.Error as e:
            raise self.create_processing_error(
                stage="extraction",
                error_type="CSVParsingError",
                message=f"Error parsing CSV file {file_path}: {str(e)}",
                severity=ErrorSeverity.HIGH,
                stack_trace=str(e)
            )
    
    def _extract_json_content(self, file_path: str, file_type: str) -> Tuple[str, DocumentStructure, List[TableData]]:
        """
        Extract content from JSON/JSONL files.
        
        Args:
            file_path: Path to the JSON/JSONL file
            file_type: File type ('json' or 'jsonl')
            
        Returns:
            Tuple of (content, structure, tables)
            
        Raises:
            ProcessingError: If JSON parsing fails
        """
        try:
            with open(file_path, 'r', encoding=self.encoding) as jsonfile:
                if file_type == 'jsonl':
                    # JSON Lines format - one JSON object per line
                    data = []
                    line_count = 0
                    for line_num, line in enumerate(jsonfile, 1):
                        line = line.strip()
                        if line:
                            try:
                                json_obj = json.loads(line)
                                data.append(json_obj)
                                line_count += 1
                            except json.JSONDecodeError as e:
                                if self.strict_parsing:
                                    raise self.create_processing_error(
                                        stage="extraction",
                                        error_type="JSONParsingError",
                                        message=f"Invalid JSON on line {line_num} in {file_path}: {str(e)}",
                                        severity=ErrorSeverity.HIGH,
                                        stack_trace=str(e)
                                    )
                                else:
                                    # Skip invalid lines in non-strict mode
                                    print(f"Warning: Skipping invalid JSON on line {line_num}: {str(e)}")
                                    continue
                    
                    if line_count == 0:
                        raise self.create_processing_error(
                            stage="extraction",
                            error_type="EmptyFile",
                            message=f"JSONL file {file_path} contains no valid JSON objects",
                            severity=ErrorSeverity.MEDIUM
                        )
                else:
                    # Regular JSON format
                    content_str = jsonfile.read()
                    if not content_str.strip():
                        raise self.create_processing_error(
                            stage="extraction",
                            error_type="EmptyFile",
                            message=f"JSON file {file_path} is empty",
                            severity=ErrorSeverity.MEDIUM
                        )
                    
                    data = json.loads(content_str)
                
                # Validate JSON depth
                max_depth = self._get_json_depth(data)
                if max_depth > self.json_max_depth:
                    raise self.create_processing_error(
                        stage="extraction",
                        error_type="JSONTooDeep",
                        message=f"JSON nesting depth ({max_depth}) exceeds maximum ({self.json_max_depth})",
                        severity=ErrorSeverity.MEDIUM
                    )
                
                # Convert JSON to text and extract tables
                content, tables = self._json_to_text_and_tables(data, os.path.basename(file_path))
                
                # Create structure
                structure = self._analyze_json_structure(data)
                
                return content, structure, tables
                
        except FileNotFoundError:
            raise self.create_processing_error(
                stage="extraction",
                error_type="FileNotFound",
                message=f"JSON file not found: {file_path}",
                severity=ErrorSeverity.HIGH
            )
        except PermissionError:
            raise self.create_processing_error(
                stage="extraction",
                error_type="PermissionError",
                message=f"Permission denied accessing JSON file: {file_path}",
                severity=ErrorSeverity.HIGH
            )
        except UnicodeDecodeError as e:
            raise self.create_processing_error(
                stage="extraction",
                error_type="EncodingError",
                message=f"Could not decode JSON file {file_path} with encoding {self.encoding}: {str(e)}",
                severity=ErrorSeverity.HIGH,
                stack_trace=str(e)
            )
        except json.JSONDecodeError as e:
            raise self.create_processing_error(
                stage="extraction",
                error_type="JSONParsingError",
                message=f"Invalid JSON in file {file_path}: {str(e)}",
                severity=ErrorSeverity.HIGH,
                stack_trace=str(e)
            )
    
    def _detect_csv_delimiter(self, file_path: str, file_type: str) -> str:
        """
        Detect CSV delimiter character.
        
        Args:
            file_path: Path to the CSV file
            file_type: File type ('csv' or 'tsv')
            
        Returns:
            Detected delimiter character
        """
        if self.csv_delimiter:
            return self.csv_delimiter
        
        if file_type == 'tsv':
            return '\t'
        
        # Auto-detect delimiter for CSV
        try:
            with open(file_path, 'r', encoding=self.encoding) as csvfile:
                sample = csvfile.read(1024)
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample, delimiters=',;\t|').delimiter
                return delimiter
        except (csv.Error, AttributeError):
            # Fall back to comma if detection fails
            return ','
    
    def _csv_to_text(self, table_data: TableData) -> str:
        """
        Convert CSV table data to text representation.
        
        Args:
            table_data: The table data to convert
            
        Returns:
            Text representation of the CSV data
        """
        lines = []
        
        if table_data.caption:
            lines.append(f"# {table_data.caption}")
            lines.append("")
        
        # Add headers
        if table_data.headers:
            lines.append("## Headers")
            lines.append(" | ".join(table_data.headers))
            lines.append(" | ".join(["---"] * len(table_data.headers)))
            lines.append("")
        
        # Add sample of data rows (first 10 rows)
        if table_data.rows:
            lines.append("## Data")
            sample_rows = table_data.rows[:10]  # Show first 10 rows
            for row in sample_rows:
                lines.append(" | ".join(row))
            
            if len(table_data.rows) > 10:
                lines.append(f"... and {len(table_data.rows) - 10} more rows")
            
            lines.append("")
            lines.append(f"Total rows: {len(table_data.rows)}")
            lines.append(f"Total columns: {len(table_data.headers) if table_data.headers else 0}")
        
        return "\n".join(lines)
    
    def _json_to_text_and_tables(self, data: Any, filename: str) -> Tuple[str, List[TableData]]:
        """
        Convert JSON data to text representation and extract tables.
        
        Args:
            data: The JSON data
            filename: Source filename for reference
            
        Returns:
            Tuple of (text content, list of tables)
        """
        lines = []
        tables = []
        
        lines.append(f"# Data from {filename}")
        lines.append("")
        
        # Handle different JSON structures
        if isinstance(data, dict):
            lines.append("## JSON Object")
            self._process_json_object(data, lines, tables, level=0)
        elif isinstance(data, list):
            lines.append("## JSON Array")
            # Check if root array is tabular
            if self._is_tabular_array(data):
                table = self._array_to_table(data, f"Root array from {filename}")
                if table:
                    tables.append(table)
                    lines.append(f"→ Converted to table with {len(table.rows)} rows")
                    # Add table preview to content
                    lines.append("")
                    lines.append("### Table Preview")
                    if table.headers:
                        lines.append(" | ".join(table.headers))
                        lines.append(" | ".join(["---"] * len(table.headers)))
                    # Show first few rows
                    preview_rows = table.rows[:3]
                    for row in preview_rows:
                        lines.append(" | ".join(row))
                    if len(table.rows) > 3:
                        lines.append(f"... and {len(table.rows) - 3} more rows")
                else:
                    self._process_json_array(data, lines, tables, level=0)
            else:
                self._process_json_array(data, lines, tables, level=0)
        else:
            lines.append(f"## JSON Value")
            lines.append(f"Value: {str(data)}")
        
        return "\n".join(lines), tables
    
    def _process_json_object(self, obj: Dict[str, Any], lines: List[str], 
                           tables: List[TableData], level: int = 0) -> None:
        """
        Process a JSON object and extract text/tables.
        
        Args:
            obj: JSON object to process
            lines: List to append text lines to
            tables: List to append extracted tables to
            level: Current nesting level
        """
        indent = "  " * level
        
        for key, value in obj.items():
            if isinstance(value, dict):
                lines.append(f"{indent}**{key}**: (object)")
                self._process_json_object(value, lines, tables, level + 1)
            elif isinstance(value, list):
                lines.append(f"{indent}**{key}**: (array with {len(value)} items)")
                # Check if this could be converted to a table
                if self._is_tabular_array(value):
                    table = self._array_to_table(value, f"{key} from level {level}")
                    if table:
                        tables.append(table)
                        lines.append(f"{indent}  → Converted to table with {len(table.rows)} rows")
                        # Add a preview of the table data
                        if table.rows and len(table.rows) > 0:
                            first_row = table.rows[0]
                            preview = " | ".join(first_row[:3])  # Show first 3 columns
                            if len(first_row) > 3:
                                preview += " | ..."
                            lines.append(f"{indent}    Preview: {preview}")
                else:
                    self._process_json_array(value, lines, tables, level + 1)
            else:
                # Simple value
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                lines.append(f"{indent}**{key}**: {value_str}")
    
    def _process_json_array(self, arr: List[Any], lines: List[str], 
                          tables: List[TableData], level: int = 0) -> None:
        """
        Process a JSON array and extract text/tables.
        
        Args:
            arr: JSON array to process
            lines: List to append text lines to
            tables: List to append extracted tables to
            level: Current nesting level
        """
        indent = "  " * level
        
        # Check if this array can be converted to a table
        if self._is_tabular_array(arr):
            table = self._array_to_table(arr, f"Array data (level {level})")
            if table:
                tables.append(table)
                lines.append(f"{indent}→ Converted to table with {len(table.rows)} rows")
                # Add a preview of the table data
                if table.rows and len(table.rows) > 0:
                    first_row = table.rows[0]
                    preview = " | ".join(first_row[:3])  # Show first 3 columns
                    if len(first_row) > 3:
                        preview += " | ..."
                    lines.append(f"{indent}  Preview: {preview}")
                return
        
        # Show first few items
        sample_size = min(5, len(arr))
        for i in range(sample_size):
            item = arr[i]
            if isinstance(item, dict):
                lines.append(f"{indent}Item {i + 1}: (object)")
                self._process_json_object(item, lines, tables, level + 1)
            elif isinstance(item, list):
                lines.append(f"{indent}Item {i + 1}: (array with {len(item)} items)")
                self._process_json_array(item, lines, tables, level + 1)
            else:
                item_str = str(item)
                if len(item_str) > 50:
                    item_str = item_str[:50] + "..."
                lines.append(f"{indent}Item {i + 1}: {item_str}")
        
        if len(arr) > sample_size:
            lines.append(f"{indent}... and {len(arr) - sample_size} more items")
    
    def _is_tabular_array(self, arr: List[Any]) -> bool:
        """
        Check if a JSON array can be converted to a table.
        
        Args:
            arr: Array to check
            
        Returns:
            True if array is tabular (list of objects with consistent keys)
        """
        if not arr or len(arr) < 2:
            return False
        
        # Check if all items are objects
        if not all(isinstance(item, dict) for item in arr):
            return False
        
        # Check if objects have consistent keys
        first_keys = set(arr[0].keys())
        return all(set(item.keys()) == first_keys for item in arr[1:])
    
    def _array_to_table(self, arr: List[Dict[str, Any]], caption: str) -> Optional[TableData]:
        """
        Convert a tabular JSON array to TableData.
        
        Args:
            arr: Array of objects to convert
            caption: Table caption
            
        Returns:
            TableData object or None if conversion fails
        """
        if not arr:
            return None
        
        try:
            # Get headers from first object
            headers = list(arr[0].keys())
            
            # Convert objects to rows
            rows = []
            for obj in arr:
                row = [str(obj.get(header, "")) for header in headers]
                rows.append(row)
            
            return TableData(
                headers=headers,
                rows=rows,
                caption=caption
            )
            
        except Exception:
            return None
    
    def _get_json_depth(self, obj: Any, current_depth: int = 0) -> int:
        """
        Calculate the maximum nesting depth of a JSON object.
        
        Args:
            obj: JSON object to analyze
            current_depth: Current depth level
            
        Returns:
            Maximum nesting depth
        """
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._get_json_depth(value, current_depth + 1) 
                      for value in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._get_json_depth(item, current_depth + 1) 
                      for item in obj)
        else:
            return current_depth
    
    def _analyze_json_structure(self, data: Any) -> DocumentStructure:
        """
        Analyze the structure of JSON data.
        
        Args:
            data: JSON data to analyze
            
        Returns:
            DocumentStructure object with analysis results
        """
        headings = []
        tables = 0
        
        # Count potential tables (arrays of objects)
        def count_tables(obj: Any) -> int:
            count = 0
            if isinstance(obj, dict):
                for value in obj.values():
                    if isinstance(value, list) and self._is_tabular_array(value):
                        count += 1
                    else:
                        count += count_tables(value)
            elif isinstance(obj, list):
                for item in obj:
                    count += count_tables(item)
            return count
        
        tables = count_tables(data)
        
        # Extract top-level keys as potential headings
        if isinstance(data, dict):
            headings = [str(key) for key in data.keys()]
        
        return DocumentStructure(
            headings=headings,
            paragraphs=1,  # JSON content as one logical paragraph
            tables=tables,
            images=0,
            code_blocks=0,
            lists=1 if isinstance(data, list) else 0
        )