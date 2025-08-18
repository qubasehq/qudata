"""
Unit tests for structured data extraction functionality.

Tests CSV, JSON, and JSONL parsing with various edge cases and error conditions.
"""

import csv
import json
import os
import tempfile
import unittest
from unittest.mock import patch, mock_open

from src.qudata.ingest.structured import StructuredExtractor
from src.qudata.models import ProcessingError, ErrorSeverity


class TestStructuredExtractor(unittest.TestCase):
    """Test cases for StructuredExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = StructuredExtractor()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_temp_file(self, content: str, filename: str) -> str:
        """Create a temporary file with given content."""
        file_path = os.path.join(self.temp_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return file_path
    
    def test_supports_format(self):
        """Test format support detection."""
        # Supported formats
        self.assertTrue(self.extractor.supports_format('csv'))
        self.assertTrue(self.extractor.supports_format('CSV'))
        self.assertTrue(self.extractor.supports_format('json'))
        self.assertTrue(self.extractor.supports_format('jsonl'))
        self.assertTrue(self.extractor.supports_format('tsv'))
        
        # Unsupported formats
        self.assertFalse(self.extractor.supports_format('pdf'))
        self.assertFalse(self.extractor.supports_format('docx'))
        self.assertFalse(self.extractor.supports_format('txt'))
    
    def test_csv_extraction_basic(self):
        """Test basic CSV extraction."""
        csv_content = """name,age,city
John,25,New York
Jane,30,Los Angeles
Bob,35,Chicago"""
        
        file_path = self._create_temp_file(csv_content, 'test.csv')
        
        result = self.extractor.extract(file_path)
        
        self.assertIsNotNone(result)
        self.assertIn('John', result.content)
        self.assertIn('New York', result.content)
        self.assertEqual(len(result.tables), 1)
        
        table = result.tables[0]
        self.assertEqual(table.headers, ['name', 'age', 'city'])
        self.assertEqual(len(table.rows), 3)
        self.assertEqual(table.rows[0], ['John', '25', 'New York'])
    
    def test_csv_extraction_with_quotes(self):
        """Test CSV extraction with quoted fields."""
        csv_content = '''name,description,price
"Product A","A great product, with features",19.99
"Product B","Another product ""with quotes""",29.99'''
        
        file_path = self._create_temp_file(csv_content, 'test_quotes.csv')
        
        result = self.extractor.extract(file_path)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result.tables), 1)
        
        table = result.tables[0]
        self.assertEqual(table.headers, ['name', 'description', 'price'])
        self.assertEqual(len(table.rows), 2)
        self.assertIn('A great product, with features', table.rows[0][1])
    
    def test_tsv_extraction(self):
        """Test TSV (tab-separated values) extraction."""
        tsv_content = """name\tage\tcity
John\t25\tNew York
Jane\t30\tLos Angeles"""
        
        file_path = self._create_temp_file(tsv_content, 'test.tsv')
        
        result = self.extractor.extract(file_path)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result.tables), 1)
        
        table = result.tables[0]
        self.assertEqual(table.headers, ['name', 'age', 'city'])
        self.assertEqual(len(table.rows), 2)
        self.assertEqual(table.rows[0], ['John', '25', 'New York'])
    
    def test_csv_empty_file(self):
        """Test handling of empty CSV file."""
        file_path = self._create_temp_file('', 'empty.csv')
        
        with self.assertRaises(ProcessingError) as context:
            self.extractor.extract(file_path)
        
        self.assertEqual(context.exception.error_type, 'EmptyFile')
        self.assertEqual(context.exception.severity, ErrorSeverity.MEDIUM)
    
    def test_csv_malformed_data(self):
        """Test handling of malformed CSV data."""
        # CSV with inconsistent number of columns
        csv_content = """name,age,city
John,25,New York
Jane,30
Bob,35,Chicago,Extra"""
        
        file_path = self._create_temp_file(csv_content, 'malformed.csv')
        
        # Should not raise error in non-strict mode
        result = self.extractor.extract(file_path)
        self.assertIsNotNone(result)
        
        # Test strict mode
        strict_extractor = StructuredExtractor({'strict_parsing': True})
        # Should still work as CSV reader handles inconsistent columns
        result = strict_extractor.extract(file_path)
        self.assertIsNotNone(result)
    
    def test_json_extraction_object(self):
        """Test JSON object extraction."""
        json_data = {
            "name": "John Doe",
            "age": 30,
            "address": {
                "street": "123 Main St",
                "city": "New York"
            },
            "hobbies": ["reading", "swimming"]
        }
        
        file_path = self._create_temp_file(json.dumps(json_data, indent=2), 'test.json')
        
        result = self.extractor.extract(file_path)
        
        self.assertIsNotNone(result)
        self.assertIn('John Doe', result.content)
        self.assertIn('123 Main St', result.content)
        self.assertIn('reading', result.content)
    
    def test_json_extraction_array(self):
        """Test JSON array extraction."""
        json_data = [
            {"name": "John", "age": 25},
            {"name": "Jane", "age": 30},
            {"name": "Bob", "age": 35}
        ]
        
        file_path = self._create_temp_file(json.dumps(json_data, indent=2), 'test_array.json')
        
        result = self.extractor.extract(file_path)
        
        self.assertIsNotNone(result)
        self.assertIn('John', result.content)
        self.assertEqual(len(result.tables), 1)  # Should convert to table
        
        table = result.tables[0]
        self.assertEqual(table.headers, ['name', 'age'])
        self.assertEqual(len(table.rows), 3)
    
    def test_jsonl_extraction(self):
        """Test JSONL (JSON Lines) extraction."""
        jsonl_content = '''{"name": "John", "age": 25}
{"name": "Jane", "age": 30}
{"name": "Bob", "age": 35}'''
        
        file_path = self._create_temp_file(jsonl_content, 'test.jsonl')
        
        result = self.extractor.extract(file_path)
        
        self.assertIsNotNone(result)
        self.assertIn('John', result.content)
        self.assertEqual(len(result.tables), 1)  # Should convert to table
        
        table = result.tables[0]
        self.assertEqual(table.headers, ['name', 'age'])
        self.assertEqual(len(table.rows), 3)
    
    def test_jsonl_with_invalid_lines(self):
        """Test JSONL with some invalid JSON lines."""
        jsonl_content = '''{"name": "John", "age": 25}
invalid json line
{"name": "Jane", "age": 30}
{"name": "Bob", "age": 35}'''
        
        file_path = self._create_temp_file(jsonl_content, 'test_invalid.jsonl')
        
        # Non-strict mode should skip invalid lines
        result = self.extractor.extract(file_path)
        self.assertIsNotNone(result)
        self.assertEqual(len(result.tables), 1)
        self.assertEqual(len(result.tables[0].rows), 3)  # Should have 3 valid rows
        
        # Strict mode should raise error
        strict_extractor = StructuredExtractor({'strict_parsing': True})
        with self.assertRaises(ProcessingError) as context:
            strict_extractor.extract(file_path)
        
        self.assertEqual(context.exception.error_type, 'JSONParsingError')
    
    def test_json_empty_file(self):
        """Test handling of empty JSON file."""
        file_path = self._create_temp_file('', 'empty.json')
        
        with self.assertRaises(ProcessingError) as context:
            self.extractor.extract(file_path)
        
        self.assertEqual(context.exception.error_type, 'EmptyFile')
    
    def test_json_invalid_syntax(self):
        """Test handling of invalid JSON syntax."""
        invalid_json = '{"name": "John", "age": 25'  # Missing closing brace
        
        file_path = self._create_temp_file(invalid_json, 'invalid.json')
        
        with self.assertRaises(ProcessingError) as context:
            self.extractor.extract(file_path)
        
        self.assertEqual(context.exception.error_type, 'JSONParsingError')
    
    def test_json_too_deep(self):
        """Test handling of deeply nested JSON."""
        # Create deeply nested JSON
        deep_json = {"level1": {"level2": {"level3": {"level4": {"level5": "value"}}}}}
        
        # Set max depth to 3
        extractor = StructuredExtractor({'json_max_depth': 3})
        file_path = self._create_temp_file(json.dumps(deep_json), 'deep.json')
        
        with self.assertRaises(ProcessingError) as context:
            extractor.extract(file_path)
        
        self.assertEqual(context.exception.error_type, 'JSONTooDeep')
    
    def test_file_too_large(self):
        """Test handling of files that exceed size limit."""
        # Create extractor with small size limit
        extractor = StructuredExtractor({'max_file_size': 100})  # 100 bytes
        
        large_content = "name,age\n" + "John,25\n" * 50  # Should exceed 100 bytes
        file_path = self._create_temp_file(large_content, 'large.csv')
        
        with self.assertRaises(ProcessingError) as context:
            extractor.extract(file_path)
        
        self.assertEqual(context.exception.error_type, 'FileTooLarge')
    
    def test_unsupported_format(self):
        """Test handling of unsupported file format."""
        file_path = self._create_temp_file('some content', 'test.xyz')
        
        with self.assertRaises(ProcessingError) as context:
            self.extractor.extract(file_path)
        
        self.assertEqual(context.exception.error_type, 'UnsupportedFormat')
    
    def test_file_not_found(self):
        """Test handling of non-existent file."""
        with self.assertRaises(ProcessingError) as context:
            self.extractor.extract('/nonexistent/file.csv')
        
        self.assertEqual(context.exception.error_type, 'FileNotFound')
    
    def test_encoding_error(self):
        """Test handling of encoding errors."""
        # Create file with non-UTF-8 content
        file_path = os.path.join(self.temp_dir, 'encoding_test.csv')
        with open(file_path, 'wb') as f:
            f.write(b'name,age\n\xff\xfe\x00\x00invalid')  # Invalid UTF-8
        
        with self.assertRaises(ProcessingError) as context:
            self.extractor.extract(file_path)
        
        self.assertEqual(context.exception.error_type, 'EncodingError')
    
    def test_csv_delimiter_detection(self):
        """Test CSV delimiter auto-detection."""
        # Test semicolon delimiter
        csv_content = """name;age;city
John;25;New York
Jane;30;Los Angeles"""
        
        file_path = self._create_temp_file(csv_content, 'semicolon.csv')
        
        result = self.extractor.extract(file_path)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result.tables), 1)
        
        table = result.tables[0]
        self.assertEqual(table.headers, ['name', 'age', 'city'])
        self.assertEqual(table.rows[0], ['John', '25', 'New York'])
    
    def test_custom_delimiter(self):
        """Test CSV extraction with custom delimiter."""
        csv_content = """name|age|city
John|25|New York
Jane|30|Los Angeles"""
        
        extractor = StructuredExtractor({'csv_delimiter': '|'})
        file_path = self._create_temp_file(csv_content, 'pipe.csv')
        
        result = extractor.extract(file_path)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result.tables), 1)
        
        table = result.tables[0]
        self.assertEqual(table.headers, ['name', 'age', 'city'])
        self.assertEqual(table.rows[0], ['John', '25', 'New York'])
    
    def test_json_structure_analysis(self):
        """Test JSON structure analysis."""
        json_data = {
            "users": [
                {"name": "John", "age": 25},
                {"name": "Jane", "age": 30}
            ],
            "metadata": {
                "version": "1.0",
                "created": "2023-01-01"
            }
        }
        
        file_path = self._create_temp_file(json.dumps(json_data), 'structure.json')
        
        result = self.extractor.extract(file_path)
        
        self.assertIsNotNone(result.structure)
        self.assertEqual(result.structure.tables, 1)  # users array should be a table
        self.assertIn('users', result.structure.headings)
        self.assertIn('metadata', result.structure.headings)
    
    def test_large_csv_content_truncation(self):
        """Test that large CSV content is properly truncated in text representation."""
        # Create CSV with many rows
        csv_content = "name,age\n" + "\n".join([f"User{i},{20+i}" for i in range(50)])
        
        file_path = self._create_temp_file(csv_content, 'large.csv')
        
        result = self.extractor.extract(file_path)
        
        self.assertIsNotNone(result)
        # Content should mention truncation
        self.assertIn('more rows', result.content)
        self.assertIn('Total rows: 50', result.content)
    
    def test_nested_json_arrays(self):
        """Test handling of nested JSON arrays."""
        json_data = {
            "departments": [
                {
                    "name": "Engineering",
                    "employees": [
                        {"name": "John", "role": "Developer"},
                        {"name": "Jane", "role": "Manager"}
                    ]
                },
                {
                    "name": "Sales",
                    "employees": [
                        {"name": "Bob", "role": "Rep"},
                        {"name": "Alice", "role": "Manager"}
                    ]
                }
            ]
        }
        
        file_path = self._create_temp_file(json.dumps(json_data, indent=2), 'nested.json')
        
        result = self.extractor.extract(file_path)
        
        self.assertIsNotNone(result)
        # Should extract multiple tables from nested arrays
        self.assertGreaterEqual(len(result.tables), 1)
        self.assertIn('Engineering', result.content)
        self.assertIn('Developer', result.content)


if __name__ == '__main__':
    unittest.main()