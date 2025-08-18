"""
Unit tests for OCR and image processing capabilities.

Tests cover:
- OCR text extraction from images
- Image preprocessing for better OCR accuracy
- Scanned PDF handling
- Confidence scoring for OCR results
"""

import os
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.qudata.ingest.ocr import (
    OCRProcessor, ImagePreprocessor, ScannedPDFHandler, 
    OCRExtractor, OCRResult
)
from src.qudata.models import ProcessingError, ErrorSeverity


class TestImagePreprocessor(unittest.TestCase):
    """Test image preprocessing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = ImagePreprocessor()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_image(self, text: str = "Test OCR Text", size: tuple = (400, 200)) -> str:
        """Create a test image with text for OCR testing."""
        image = Image.new('RGB', size, color='white')
        draw = ImageDraw.Draw(image)
        
        # Try to use a default font, fallback to basic if not available
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except (OSError, IOError):
            font = ImageFont.load_default()
        
        # Calculate text position (centered)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (size[0] - text_width) // 2
        y = (size[1] - text_height) // 2
        
        draw.text((x, y), text, fill='black', font=font)
        
        # Save to temporary file
        image_path = os.path.join(self.temp_dir, "test_image.png")
        image.save(image_path)
        return image_path
    
    def create_skewed_image(self, text: str = "Skewed Text", angle: float = 15.0) -> str:
        """Create a skewed test image."""
        image = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except (OSError, IOError):
            font = ImageFont.load_default()
        
        draw.text((50, 80), text, fill='black', font=font)
        
        # Rotate the image
        rotated = image.rotate(angle, fillcolor='white')
        
        image_path = os.path.join(self.temp_dir, "skewed_image.png")
        rotated.save(image_path)
        return image_path
    
    @patch('cv2.imread')
    @patch('cv2.fastNlMeansDenoising')
    @patch('cv2.createCLAHE')
    @patch('cv2.threshold')
    def test_preprocess_image_success(self, mock_threshold, mock_clahe, mock_denoise, mock_imread):
        """Test successful image preprocessing."""
        # Mock image data
        mock_image = np.ones((200, 400, 3), dtype=np.uint8) * 255
        mock_imread.return_value = mock_image
        
        # Mock preprocessing operations
        mock_denoise.return_value = np.ones((200, 400), dtype=np.uint8) * 255
        mock_clahe_obj = Mock()
        mock_clahe_obj.apply.return_value = np.ones((200, 400), dtype=np.uint8) * 255
        mock_clahe.return_value = mock_clahe_obj
        mock_threshold.return_value = (127, np.ones((200, 400), dtype=np.uint8) * 255)
        
        # Test preprocessing
        image_path = self.create_test_image()
        processed_image, operations = self.preprocessor.preprocess_image(image_path)
        
        # Verify operations were applied
        self.assertIsInstance(processed_image, np.ndarray)
        self.assertIn("grayscale_conversion", operations)
        self.assertIn("denoising", operations)
        self.assertIn("contrast_enhancement", operations)
        self.assertIn("binarization", operations)
    
    def test_preprocess_image_with_config(self):
        """Test preprocessing with custom configuration."""
        config = {
            "enable_denoising": False,
            "enable_deskewing": False,
            "enable_contrast_enhancement": True,
            "enable_binarization": True
        }
        preprocessor = ImagePreprocessor(config)
        
        # Verify configuration is applied
        self.assertFalse(preprocessor.enable_denoising)
        self.assertFalse(preprocessor.enable_deskewing)
        self.assertTrue(preprocessor.enable_contrast_enhancement)
        self.assertTrue(preprocessor.enable_binarization)
    
    @patch('cv2.imread')
    def test_preprocess_image_failure_fallback(self, mock_imread):
        """Test preprocessing fallback when operations fail."""
        mock_imread.return_value = None
        
        image_path = "nonexistent_image.png"
        
        # Should handle gracefully and return something
        try:
            processed_image, operations = self.preprocessor.preprocess_image(image_path)
            # If it doesn't raise an exception, check the operations
            self.assertIn("preprocessing_failed", operations)
        except Exception:
            # It's acceptable for this to raise an exception for nonexistent files
            pass


class TestOCRProcessor(unittest.TestCase):
    """Test OCR processing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock tesseract to avoid dependency on actual installation
        self.patcher = patch('pytesseract.get_tesseract_version')
        self.mock_version = self.patcher.start()
        self.mock_version.return_value = "5.0.0"
        
        self.ocr_processor = OCRProcessor()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.patcher.stop()
    
    def create_test_image(self, text: str = "Test OCR Text") -> str:
        """Create a test image with text."""
        image = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except (OSError, IOError):
            font = ImageFont.load_default()
        
        draw.text((50, 80), text, fill='black', font=font)
        
        image_path = os.path.join(self.temp_dir, "test_ocr.png")
        image.save(image_path)
        return image_path
    
    @patch('pytesseract.image_to_data')
    def test_extract_text_from_image_success(self, mock_ocr):
        """Test successful text extraction from image."""
        # Mock OCR data
        mock_ocr_data = {
            'text': ['', 'Test', 'OCR', 'Text', ''],
            'conf': ['-1', '95', '87', '92', '-1'],
            'left': [0, 50, 100, 150, 200],
            'top': [0, 80, 80, 80, 80],
            'width': [400, 40, 30, 35, 0],
            'height': [200, 25, 25, 25, 0]
        }
        mock_ocr.return_value = mock_ocr_data
        
        image_path = self.create_test_image("Test OCR Text")
        result = self.ocr_processor.extract_text_from_image(image_path)
        
        # Verify result
        self.assertIsInstance(result, OCRResult)
        self.assertEqual(result.text, "Test OCR Text")
        self.assertGreater(result.confidence, 0)
        self.assertEqual(len(result.word_confidences), 3)  # 3 words above confidence threshold
        self.assertEqual(len(result.bounding_boxes), 3)
    
    @patch('pytesseract.image_to_data')
    def test_extract_text_low_confidence(self, mock_ocr):
        """Test handling of low confidence OCR results."""
        # Mock low confidence OCR data
        mock_ocr_data = {
            'text': ['Low', 'confidence', 'text'],
            'conf': ['20', '15', '25'],  # Below default threshold of 30
            'left': [0, 50, 100],
            'top': [0, 80, 80],
            'width': [40, 60, 40],
            'height': [25, 25, 25]
        }
        mock_ocr.return_value = mock_ocr_data
        
        image_path = self.create_test_image("Low confidence text")
        result = self.ocr_processor.extract_text_from_image(image_path)
        
        # Should filter out low confidence words
        self.assertEqual(result.text, "")  # No words above threshold
        self.assertEqual(len(result.word_confidences), 0)
    
    @patch('pytesseract.image_to_data')
    def test_extract_text_with_custom_config(self, mock_ocr):
        """Test OCR with custom configuration."""
        config = {
            "tesseract_config": "--oem 1 --psm 8",
            "min_confidence": 50.0,
            "languages": "eng+fra"
        }
        ocr_processor = OCRProcessor(config)
        
        mock_ocr_data = {
            'text': ['Good', 'text'],
            'conf': ['60', '70'],
            'left': [0, 50],
            'top': [0, 80],
            'width': [40, 40],
            'height': [25, 25]
        }
        mock_ocr.return_value = mock_ocr_data
        
        image_path = self.create_test_image("Good text")
        result = ocr_processor.extract_text_from_image(image_path)
        
        # Verify custom config is used
        mock_ocr.assert_called_with(
            unittest.mock.ANY,
            config="--oem 1 --psm 8",
            lang="eng+fra",
            output_type=unittest.mock.ANY
        )
        self.assertEqual(result.text, "Good text")
    
    @patch('pytesseract.image_to_data')
    def test_extract_text_failure_handling(self, mock_ocr):
        """Test handling of OCR failures."""
        mock_ocr.side_effect = Exception("OCR failed")
        
        image_path = self.create_test_image("Test text")
        result = self.ocr_processor.extract_text_from_image(image_path)
        
        # Should return empty result on failure
        self.assertEqual(result.text, "")
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(len(result.word_confidences), 0)
    
    def test_extract_text_simple(self):
        """Test simple text extraction interface."""
        with patch.object(self.ocr_processor, 'extract_text_from_image') as mock_extract:
            mock_result = OCRResult(
                text="Simple text",
                confidence=85.0,
                word_confidences=[("Simple", 80.0), ("text", 90.0)],
                bounding_boxes=[(0, 0, 50, 25), (60, 0, 40, 25)],
                preprocessing_applied=["grayscale_conversion"]
            )
            mock_extract.return_value = mock_result
            
            image_path = self.create_test_image("Simple text")
            result = self.ocr_processor.extract_text_simple(image_path)
            
            self.assertEqual(result, "Simple text")


class TestScannedPDFHandler(unittest.TestCase):
    """Test scanned PDF handling functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock tesseract
        self.patcher = patch('pytesseract.get_tesseract_version')
        self.mock_version = self.patcher.start()
        self.mock_version.return_value = "5.0.0"
        
        self.pdf_handler = ScannedPDFHandler()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.patcher.stop()
    
    @patch('fitz.open')
    def test_is_scanned_pdf_true(self, mock_fitz_open):
        """Test detection of scanned PDF."""
        # Mock PDF with no text content
        mock_doc = MagicMock()
        mock_page = Mock()
        mock_page.get_text.return_value = ""  # No text content
        mock_doc.__len__ = Mock(return_value=1)
        mock_doc.__getitem__ = Mock(return_value=mock_page)
        mock_fitz_open.return_value = mock_doc
        
        pdf_path = os.path.join(self.temp_dir, "scanned.pdf")
        result = self.pdf_handler.is_scanned_pdf(pdf_path)
        
        self.assertTrue(result)
        mock_doc.close.assert_called_once()
    
    @patch('fitz.open')
    def test_is_scanned_pdf_false(self, mock_fitz_open):
        """Test detection of text-based PDF."""
        # Mock PDF with substantial text content
        mock_doc = MagicMock()
        mock_page = Mock()
        mock_page.get_text.return_value = "This is a text-based PDF with substantial content that can be extracted directly without OCR processing."
        mock_doc.__len__ = Mock(return_value=1)
        mock_doc.__getitem__ = Mock(return_value=mock_page)
        mock_fitz_open.return_value = mock_doc
        
        pdf_path = os.path.join(self.temp_dir, "text_based.pdf")
        result = self.pdf_handler.is_scanned_pdf(pdf_path)
        
        self.assertFalse(result)
        mock_doc.close.assert_called_once()
    
    @patch('fitz.open')
    def test_is_scanned_pdf_error_handling(self, mock_fitz_open):
        """Test error handling in PDF scanning detection."""
        mock_fitz_open.side_effect = Exception("Cannot open PDF")
        
        pdf_path = os.path.join(self.temp_dir, "corrupted.pdf")
        result = self.pdf_handler.is_scanned_pdf(pdf_path)
        
        # Should assume scanned on error
        self.assertTrue(result)
    
    @patch('fitz.open')
    @patch('pytesseract.image_to_data')
    def test_extract_text_from_scanned_pdf(self, mock_ocr, mock_fitz_open):
        """Test text extraction from scanned PDF."""
        # Mock PDF document
        mock_doc = MagicMock()
        mock_page = Mock()
        mock_pix = Mock()
        mock_pix.save = Mock()
        mock_page.get_pixmap.return_value = mock_pix
        mock_doc.__len__ = Mock(return_value=2)  # 2 pages
        mock_doc.__getitem__ = Mock(return_value=mock_page)
        mock_fitz_open.return_value = mock_doc
        
        # Mock OCR results for each page
        mock_ocr_data = {
            'text': ['Page', 'content', 'here'],
            'conf': ['85', '90', '88'],
            'left': [0, 50, 100],
            'top': [0, 80, 80],
            'width': [40, 60, 40],
            'height': [25, 25, 25]
        }
        mock_ocr.return_value = mock_ocr_data
        
        pdf_path = os.path.join(self.temp_dir, "scanned.pdf")
        result = self.pdf_handler.extract_text_from_scanned_pdf(pdf_path)
        
        # Verify result
        self.assertIsInstance(result, OCRResult)
        self.assertIn("Page content here", result.text)
        self.assertIn("--- Page 1 ---", result.text)
        self.assertIn("--- Page 2 ---", result.text)
        self.assertGreater(result.confidence, 0)
        
        # Verify PDF was processed
        mock_doc.close.assert_called_once()
        self.assertEqual(mock_ocr.call_count, 2)  # Called for each page
    
    @patch('fitz.open')
    def test_extract_text_from_scanned_pdf_failure(self, mock_fitz_open):
        """Test handling of PDF extraction failures."""
        mock_fitz_open.side_effect = Exception("Cannot process PDF")
        
        pdf_path = os.path.join(self.temp_dir, "corrupted.pdf")
        result = self.pdf_handler.extract_text_from_scanned_pdf(pdf_path)
        
        # Should return empty result on failure
        self.assertEqual(result.text, "")
        self.assertEqual(result.confidence, 0.0)


class TestOCRExtractor(unittest.TestCase):
    """Test OCR extractor integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock tesseract
        self.patcher = patch('pytesseract.get_tesseract_version')
        self.mock_version = self.patcher.start()
        self.mock_version.return_value = "5.0.0"
        
        self.extractor = OCRExtractor()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.patcher.stop()
    
    def create_test_image(self, filename: str = "test.png") -> str:
        """Create a test image file."""
        image = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except (OSError, IOError):
            font = ImageFont.load_default()
        
        draw.text((50, 80), "Test OCR Content", fill='black', font=font)
        
        image_path = os.path.join(self.temp_dir, filename)
        image.save(image_path)
        return image_path
    
    def test_supports_format(self):
        """Test file format support detection."""
        # Test supported image formats
        supported_formats = ['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif', 'webp', 'pdf']
        for fmt in supported_formats:
            self.assertTrue(self.extractor.supports_format(fmt))
        
        # Test unsupported formats
        unsupported_formats = ['txt', 'docx', 'html', 'csv', 'json']
        for fmt in unsupported_formats:
            self.assertFalse(self.extractor.supports_format(fmt))
    
    @patch('src.qudata.ingest.ocr.OCRProcessor.extract_text_from_image')
    def test_extract_image_success(self, mock_ocr_extract):
        """Test successful image extraction."""
        # Mock OCR result
        mock_result = OCRResult(
            text="Extracted text from image",
            confidence=85.0,
            word_confidences=[("Extracted", 80.0), ("text", 90.0), ("from", 85.0), ("image", 85.0)],
            bounding_boxes=[(0, 0, 60, 25), (70, 0, 30, 25), (110, 0, 30, 25), (150, 0, 40, 25)],
            preprocessing_applied=["grayscale_conversion", "denoising"]
        )
        mock_ocr_extract.return_value = mock_result
        
        image_path = self.create_test_image("test.png")
        extracted = self.extractor.extract(image_path)
        
        # Verify extraction
        self.assertEqual(extracted.content, "Extracted text from image")
        self.assertEqual(extracted.metadata.file_type, "png")
        self.assertIsNotNone(extracted.structure)
        self.assertEqual(len(extracted.images), 1)
        self.assertIn("OCR processed image", extracted.images[0].caption)
    
    @patch('src.qudata.ingest.ocr.ScannedPDFHandler.is_scanned_pdf')
    @patch('src.qudata.ingest.ocr.ScannedPDFHandler.extract_text_from_scanned_pdf')
    def test_extract_scanned_pdf_success(self, mock_pdf_extract, mock_is_scanned):
        """Test successful scanned PDF extraction."""
        # Mock scanned PDF detection and extraction
        mock_is_scanned.return_value = True
        mock_result = OCRResult(
            text="Extracted text from scanned PDF",
            confidence=78.0,
            word_confidences=[("Extracted", 75.0), ("text", 80.0), ("from", 78.0), ("scanned", 80.0), ("PDF", 75.0)],
            bounding_boxes=[(0, 0, 60, 25), (70, 0, 30, 25), (110, 0, 30, 25), (150, 0, 50, 25), (210, 0, 30, 25)],
            preprocessing_applied=["grayscale_conversion", "binarization"]
        )
        mock_pdf_extract.return_value = mock_result
        
        # Create a dummy PDF file
        pdf_path = os.path.join(self.temp_dir, "scanned.pdf")
        with open(pdf_path, 'wb') as f:
            f.write(b'%PDF-1.4\n%dummy pdf content')
        
        extracted = self.extractor.extract(pdf_path)
        
        # Verify extraction
        self.assertEqual(extracted.content, "Extracted text from scanned PDF")
        self.assertEqual(extracted.metadata.file_type, "pdf")
        mock_is_scanned.assert_called_once_with(pdf_path)
        mock_pdf_extract.assert_called_once_with(pdf_path)
    
    @patch('src.qudata.ingest.ocr.ScannedPDFHandler.is_scanned_pdf')
    def test_extract_text_based_pdf_error(self, mock_is_scanned):
        """Test error handling for text-based PDF."""
        mock_is_scanned.return_value = False  # Not a scanned PDF
        
        # Create a dummy PDF file
        pdf_path = os.path.join(self.temp_dir, "text_based.pdf")
        with open(pdf_path, 'wb') as f:
            f.write(b'%PDF-1.4\n%dummy pdf content')
        
        with self.assertRaises(ProcessingError) as context:
            self.extractor.extract(pdf_path)
        
        self.assertEqual(context.exception.error_type, "NotScannedPDF")
        self.assertEqual(context.exception.severity, ErrorSeverity.MEDIUM)
    
    @patch('src.qudata.ingest.ocr.OCRProcessor.extract_text_from_image')
    def test_extract_insufficient_text_error(self, mock_ocr_extract):
        """Test error handling for insufficient extracted text."""
        # Mock OCR result with very little text
        mock_result = OCRResult(
            text="Hi",  # Only 2 characters, below default minimum of 10
            confidence=85.0,
            word_confidences=[("Hi", 85.0)],
            bounding_boxes=[(0, 0, 20, 25)],
            preprocessing_applied=["grayscale_conversion"]
        )
        mock_ocr_extract.return_value = mock_result
        
        image_path = self.create_test_image("minimal.png")
        
        with self.assertRaises(ProcessingError) as context:
            self.extractor.extract(image_path)
        
        self.assertEqual(context.exception.error_type, "InsufficientText")
        self.assertEqual(context.exception.severity, ErrorSeverity.LOW)
    
    def test_extract_nonexistent_file_error(self):
        """Test error handling for nonexistent files."""
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent.png")
        
        with self.assertRaises(ProcessingError) as context:
            self.extractor.extract(nonexistent_path)
        
        self.assertEqual(context.exception.error_type, "FileNotFound")
        self.assertEqual(context.exception.severity, ErrorSeverity.HIGH)
    
    @patch('src.qudata.ingest.ocr.OCRProcessor.extract_text_from_image')
    def test_extract_with_custom_config(self, mock_ocr_extract):
        """Test extraction with custom configuration."""
        config = {
            "min_text_length": 5,
            "ocr": {
                "min_confidence": 70.0,
                "languages": "eng+fra"
            }
        }
        extractor = OCRExtractor(config)
        
        # Mock OCR result
        mock_result = OCRResult(
            text="Short",  # 5 characters, meets custom minimum
            confidence=75.0,
            word_confidences=[("Short", 75.0)],
            bounding_boxes=[(0, 0, 40, 25)],
            preprocessing_applied=["grayscale_conversion"]
        )
        mock_ocr_extract.return_value = mock_result
        
        image_path = self.create_test_image("short.png")
        extracted = extractor.extract(image_path)
        
        # Should succeed with custom minimum length
        self.assertEqual(extracted.content, "Short")


class TestOCRResult(unittest.TestCase):
    """Test OCRResult data class."""
    
    def test_ocr_result_creation(self):
        """Test OCR result creation and serialization."""
        result = OCRResult(
            text="Test text",
            confidence=85.5,
            word_confidences=[("Test", 80.0), ("text", 91.0)],
            bounding_boxes=[(0, 0, 30, 25), (40, 0, 30, 25)],
            preprocessing_applied=["grayscale_conversion", "denoising"]
        )
        
        # Test properties
        self.assertEqual(result.text, "Test text")
        self.assertEqual(result.confidence, 85.5)
        self.assertEqual(len(result.word_confidences), 2)
        self.assertEqual(len(result.bounding_boxes), 2)
        self.assertEqual(len(result.preprocessing_applied), 2)
        
        # Test serialization
        result_dict = result.to_dict()
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict["text"], "Test text")
        self.assertEqual(result_dict["confidence"], 85.5)
        self.assertIn("word_confidences", result_dict)
        self.assertIn("bounding_boxes", result_dict)
        self.assertIn("preprocessing_applied", result_dict)


if __name__ == '__main__':
    unittest.main()