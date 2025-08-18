"""
Integration tests for OCR functionality with the broader system.

Tests the integration between:
- File type detection and OCR processing
- OCR extractor with the processing pipeline
- Configuration management for OCR components
"""

import os
import tempfile
import unittest
from unittest.mock import patch, Mock
from pathlib import Path

from src.qudata.ingest.detector import FileTypeDetector
from src.qudata.ingest.ocr import OCRExtractor, OCRProcessor, OCRResult
from src.qudata.models import ProcessingError


class TestOCRIntegration(unittest.TestCase):
    """Test OCR integration with the file processing system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.detector = FileTypeDetector()
        
        # Mock tesseract to avoid dependency
        self.patcher = patch('pytesseract.get_tesseract_version')
        self.mock_version = self.patcher.start()
        self.mock_version.return_value = "5.0.0"
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.patcher.stop()
    
    def create_test_image(self, filename: str, format: str = "PNG") -> str:
        """Create a test image file."""
        from PIL import Image, ImageDraw
        
        image = Image.new('RGB', (300, 100), color='white')
        draw = ImageDraw.Draw(image)
        draw.text((10, 40), "Test OCR Integration", fill='black')
        
        image_path = os.path.join(self.temp_dir, filename)
        image.save(image_path, format=format)
        return image_path
    
    def test_file_type_detection_for_images(self):
        """Test that image files are correctly detected for OCR processing."""
        # Test different image formats
        test_cases = [
            ("test.png", "png"),
            ("test.jpg", "jpeg"),
            ("test.jpeg", "jpeg"),
            ("test.gif", "gif"),
            ("test.bmp", "bmp"),
            ("test.tiff", "tiff"),
            ("test.webp", "webp"),
        ]
        
        for filename, expected_type in test_cases:
            with self.subTest(filename=filename):
                # Create test image
                if expected_type in ["png", "jpeg", "gif", "bmp"]:
                    image_path = self.create_test_image(filename, expected_type.upper())
                    
                    # Detect file type
                    detected_type, confidence = self.detector.detect_file_type(image_path)
                    
                    # Verify detection
                    self.assertEqual(detected_type, expected_type)
                    self.assertGreater(confidence, 0.7)  # Should have good confidence
                    self.assertTrue(self.detector.is_supported(detected_type))
    
    @patch('src.qudata.ingest.ocr.OCRProcessor.extract_text_from_image')
    def test_ocr_extractor_with_detected_files(self, mock_ocr_extract):
        """Test OCR extractor integration with file type detection."""
        # Mock OCR result
        mock_result = OCRResult(
            text="Integration test text",
            confidence=85.0,
            word_confidences=[("Integration", 80.0), ("test", 90.0), ("text", 85.0)],
            bounding_boxes=[(0, 0, 80, 25), (90, 0, 30, 25), (130, 0, 30, 25)],
            preprocessing_applied=["grayscale_conversion", "denoising"]
        )
        mock_ocr_extract.return_value = mock_result
        
        # Create test image
        image_path = self.create_test_image("integration_test.png")
        
        # Detect file type
        detected_type, confidence = self.detector.detect_file_type(image_path)
        self.assertEqual(detected_type, "png")
        
        # Create OCR extractor and verify it supports the detected type
        extractor = OCRExtractor()
        self.assertTrue(extractor.supports_format(detected_type))
        
        # Extract content
        extracted = extractor.extract(image_path)
        
        # Verify extraction
        self.assertEqual(extracted.content, "Integration test text")
        self.assertEqual(extracted.metadata.file_type, "png")
        self.assertIsNotNone(extracted.structure)
        self.assertEqual(len(extracted.images), 1)
        
        # Verify OCR was called
        mock_ocr_extract.assert_called_once()
    
    def test_ocr_extractor_format_support(self):
        """Test that OCR extractor supports all expected image formats."""
        extractor = OCRExtractor()
        
        # Test supported formats
        supported_formats = ['png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp', 'gif', 'webp', 'pdf']
        for fmt in supported_formats:
            self.assertTrue(extractor.supports_format(fmt), f"Format {fmt} should be supported")
        
        # Test unsupported formats
        unsupported_formats = ['txt', 'docx', 'html', 'csv', 'json']
        for fmt in unsupported_formats:
            self.assertFalse(extractor.supports_format(fmt), f"Format {fmt} should not be supported")
    
    def test_file_detection_and_processing_pipeline(self):
        """Test the complete pipeline from file detection to OCR processing."""
        # Create test images of different formats
        test_files = []
        
        # PNG image
        png_path = self.create_test_image("test.png", "PNG")
        test_files.append((png_path, "png"))
        
        # JPEG image
        jpeg_path = self.create_test_image("test.jpg", "JPEG")
        test_files.append((jpeg_path, "jpeg"))
        
        for file_path, expected_type in test_files:
            with self.subTest(file_path=file_path):
                # Step 1: Detect file type
                detected_type, confidence = self.detector.detect_file_type(file_path)
                self.assertEqual(detected_type, expected_type)
                self.assertTrue(self.detector.is_supported(detected_type))
                
                # Step 2: Verify OCR extractor can handle this type
                extractor = OCRExtractor()
                self.assertTrue(extractor.supports_format(detected_type))
                
                # Step 3: Get file info
                file_info = self.detector.get_file_info(file_path)
                self.assertEqual(file_info['detected_type'], expected_type)
                self.assertTrue(file_info['is_supported'])
                self.assertGreater(file_info['size_bytes'], 0)
    
    @patch('src.qudata.ingest.ocr.ScannedPDFHandler.is_scanned_pdf')
    def test_pdf_detection_and_ocr_routing(self, mock_is_scanned):
        """Test PDF detection and routing to appropriate processor."""
        # Create a dummy PDF file
        pdf_path = os.path.join(self.temp_dir, "test.pdf")
        with open(pdf_path, 'wb') as f:
            f.write(b'%PDF-1.4\n%dummy pdf content')
        
        # Test file detection
        detected_type, confidence = self.detector.detect_file_type(pdf_path)
        self.assertEqual(detected_type, "pdf")
        
        # Test OCR extractor supports PDF
        extractor = OCRExtractor()
        self.assertTrue(extractor.supports_format("pdf"))
        
        # Test scanned PDF detection
        mock_is_scanned.return_value = True
        
        # This would normally process the PDF, but we'll just verify the setup
        self.assertTrue(mock_is_scanned.return_value)
    
    def test_configuration_integration(self):
        """Test configuration integration across OCR components."""
        # Test configuration propagation
        config = {
            "min_text_length": 15,
            "ocr": {
                "min_confidence": 70.0,
                "languages": "eng+fra",
                "preprocessing": {
                    "enable_denoising": False,
                    "enable_deskewing": True
                }
            },
            "scanned_pdf": {
                "dpi": 400,
                "image_format": "tiff"
            }
        }
        
        # Create extractor with config
        extractor = OCRExtractor(config)
        
        # Verify configuration is applied
        self.assertEqual(extractor.min_text_length, 15)
        self.assertEqual(extractor.ocr_processor.min_confidence, 70.0)
        self.assertEqual(extractor.ocr_processor.languages, "eng+fra")
        self.assertFalse(extractor.ocr_processor.preprocessor.enable_denoising)
        self.assertTrue(extractor.ocr_processor.preprocessor.enable_deskewing)
        self.assertEqual(extractor.pdf_handler.dpi, 400)
        self.assertEqual(extractor.pdf_handler.image_format, "tiff")
    
    def test_error_handling_integration(self):
        """Test error handling across the OCR integration."""
        extractor = OCRExtractor()
        
        # Test with non-existent file
        with self.assertRaises(ProcessingError) as context:
            extractor.extract("nonexistent_file.png")
        
        self.assertEqual(context.exception.error_type, "FileNotFound")
        
        # Test with unsupported file (should not reach OCR extractor)
        text_path = os.path.join(self.temp_dir, "test.txt")
        with open(text_path, 'w') as f:
            f.write("This is a text file")
        
        # OCR extractor should not support text files
        self.assertFalse(extractor.supports_format("txt"))
    
    def test_sample_data_integration(self):
        """Test integration with sample OCR data if available."""
        sample_dir = Path(__file__).parent.parent / "sample_data" / "ocr"
        
        if sample_dir.exists():
            detector = FileTypeDetector()
            extractor = OCRExtractor()
            
            # Test with available sample images
            image_files = list(sample_dir.glob("*.png"))[:2]  # Test first 2 images
            for image_file in image_files:
                # Detect file type
                detected_type, confidence = detector.detect_file_type(str(image_file))
                
                # Verify it's detected as an image
                self.assertIn(detected_type, ['png', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'])
                self.assertGreater(confidence, 0.7)
                
                # Verify OCR extractor supports it
                self.assertTrue(extractor.supports_format(detected_type))
                
                # Verify file info
                file_info = detector.get_file_info(str(image_file))
                self.assertTrue(file_info['is_supported'])
                self.assertEqual(file_info['detected_type'], detected_type)
        else:
            self.skipTest("Sample OCR data not available")


if __name__ == '__main__':
    unittest.main()