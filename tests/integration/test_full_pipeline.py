"""
Integration test for the complete QuData pipeline.

Tests the full end-to-end processing workflow from raw files to LLM-ready datasets.
"""

import os
import tempfile
import unittest
from pathlib import Path

from src.qudata.pipeline import QuDataPipeline
from src.qudata.config import PipelineConfig


class TestFullPipeline(unittest.TestCase):
    """Test complete pipeline integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_input_dir = tempfile.mkdtemp()
        self.temp_output_dir = tempfile.mkdtemp()
        
        # Create sample input files
        self._create_sample_files()
        
        # Create test configuration
        self.config = PipelineConfig()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_input_dir, ignore_errors=True)
        shutil.rmtree(self.temp_output_dir, ignore_errors=True)
    
    def _create_sample_files(self):
        """Create sample input files for testing."""
        # Create a sample text file
        with open(os.path.join(self.temp_input_dir, "sample.txt"), 'w', encoding='utf-8') as f:
            f.write("""
            This is a sample document for testing the QuData pipeline.
            
            It contains multiple paragraphs with various content types.
            The pipeline should be able to process this content and extract
            meaningful information for LLM training.
            
            Key features to test:
            - Text normalization
            - Content cleaning
            - Metadata extraction
            - Quality scoring
            """)
        
        # Create another sample file
        with open(os.path.join(self.temp_input_dir, "sample2.txt"), 'w', encoding='utf-8') as f:
            f.write("""
            This is a second sample document to test batch processing.
            
            The pipeline should handle multiple documents efficiently
            and provide comprehensive analysis across the entire dataset.
            
            This document has different content to test:
            - Topic diversity
            - Language detection
            - Deduplication
            - Quality assessment
            """)
    
    def test_full_pipeline_execution(self):
        """Test complete pipeline execution."""
        # Initialize pipeline
        pipeline = QuDataPipeline(config=self.config)
        
        # Run pipeline
        result = pipeline.process_directory(self.temp_input_dir, self.temp_output_dir)
        
        # Verify results
        self.assertTrue(result.success, "Pipeline should complete successfully")
        self.assertGreater(result.documents_processed, 0, "Should process at least one document")
        self.assertEqual(result.documents_failed, 0, "No documents should fail")
        self.assertGreater(result.processing_time, 0, "Should have positive processing time")
        
        # Verify stage results
        self.assertIn('ingestion', result.stage_results)
        self.assertIn('cleaning', result.stage_results)
        self.assertIn('annotation', result.stage_results)
        self.assertIn('scoring', result.stage_results)
        self.assertIn('analysis', result.stage_results)
        
        # Verify ingestion results
        ingestion_results = result.stage_results['ingestion']
        self.assertGreater(ingestion_results['documents_extracted'], 0)
        
        # Verify cleaning results
        cleaning_results = result.stage_results['cleaning']
        self.assertGreater(cleaning_results['documents_cleaned'], 0)
        
        # Verify output files were created
        output_path = Path(self.temp_output_dir)
        self.assertTrue(output_path.exists(), "Output directory should exist")
    
    def test_pipeline_with_empty_directory(self):
        """Test pipeline behavior with empty input directory."""
        empty_dir = tempfile.mkdtemp()
        
        try:
            pipeline = QuDataPipeline(config=self.config)
            result = pipeline.process_directory(empty_dir, self.temp_output_dir)
            
            # Should handle empty directory gracefully
            self.assertFalse(result.success, "Should fail gracefully with empty directory")
            self.assertEqual(result.documents_processed, 0)
            
        finally:
            import shutil
            shutil.rmtree(empty_dir, ignore_errors=True)
    
    def test_pipeline_error_handling(self):
        """Test pipeline error handling with invalid input."""
        # Test with non-existent directory
        pipeline = QuDataPipeline(config=self.config)
        
        with self.assertRaises(Exception):
            pipeline.process_directory("/non/existent/path", self.temp_output_dir)


if __name__ == '__main__':
    unittest.main()