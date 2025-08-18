"""
OCR and image processing capabilities for QuData.

This module provides OCR text extraction from images and scanned PDFs,
with image preprocessing for better OCR accuracy and confidence scoring.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter
    import pytesseract
    import fitz  # PyMuPDF for PDF handling
except ImportError as e:
    raise ImportError(
        f"OCR dependencies not installed: {e}. "
        "Install with: pip install pytesseract opencv-python Pillow PyMuPDF"
    )

from ..models import (
    BaseExtractor, ExtractedContent, FileMetadata, ProcessingError, 
    ErrorSeverity, DocumentStructure, ImageData
)


logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Result of OCR processing with confidence scores."""
    text: str
    confidence: float
    word_confidences: List[Tuple[str, float]]
    bounding_boxes: List[Tuple[int, int, int, int]]  # x, y, width, height
    preprocessing_applied: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert OCR result to dictionary."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "word_confidences": self.word_confidences,
            "bounding_boxes": self.bounding_boxes,
            "preprocessing_applied": self.preprocessing_applied
        }


class ImagePreprocessor:
    """Image preprocessing for improved OCR accuracy."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize image preprocessor.
        
        Args:
            config: Configuration for preprocessing operations
        """
        self.config = config or {}
        self.enable_denoising = self.config.get("enable_denoising", True)
        self.enable_deskewing = self.config.get("enable_deskewing", True)
        self.enable_contrast_enhancement = self.config.get("enable_contrast_enhancement", True)
        self.enable_binarization = self.config.get("enable_binarization", True)
        self.target_dpi = self.config.get("target_dpi", 300)
    
    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, List[str]]:
        """
        Preprocess an image for better OCR results.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (processed_image_array, list_of_applied_operations)
        """
        applied_operations = []
        
        try:
            # Load image
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
                if image is None:
                    # Try with PIL for better format support
                    pil_image = Image.open(image_path)
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            else:
                image = image_path  # Already a numpy array
            
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            original_image = image.copy()
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                applied_operations.append("grayscale_conversion")
            else:
                gray = image
            
            # Resize if DPI is too low (estimate based on image size)
            height, width = gray.shape
            if width < 1000 or height < 1000:  # Likely low DPI
                scale_factor = max(1000 / width, 1000 / height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                applied_operations.append(f"upscaling_{scale_factor:.2f}x")
            
            # Denoising
            if self.enable_denoising:
                gray = cv2.fastNlMeansDenoising(gray)
                applied_operations.append("denoising")
            
            # Deskewing
            if self.enable_deskewing:
                gray = self._deskew_image(gray)
                applied_operations.append("deskewing")
            
            # Contrast enhancement
            if self.enable_contrast_enhancement:
                gray = self._enhance_contrast(gray)
                applied_operations.append("contrast_enhancement")
            
            # Binarization (convert to black and white)
            if self.enable_binarization:
                gray = self._binarize_image(gray)
                applied_operations.append("binarization")
            
            return gray, applied_operations
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            # Return original image if preprocessing fails
            if isinstance(image_path, str):
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            return image, ["preprocessing_failed"]
    
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Correct skew in the image."""
        try:
            # Find edges
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            
            # Find lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                # Calculate the most common angle
                angles = []
                for rho, theta in lines[:, 0]:
                    angle = theta * 180 / np.pi
                    if angle < 45:
                        angles.append(angle)
                    elif angle > 135:
                        angles.append(angle - 180)
                
                if angles:
                    # Use median angle for deskewing
                    median_angle = np.median(angles)
                    
                    # Only deskew if angle is significant (> 0.5 degrees)
                    if abs(median_angle) > 0.5:
                        # Rotate image
                        (h, w) = image.shape[:2]
                        center = (w // 2, h // 2)
                        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                        image = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                             flags=cv2.INTER_CUBIC, 
                                             borderMode=cv2.BORDER_REPLICATE)
            
            return image
        except Exception as e:
            logger.warning(f"Deskewing failed: {e}")
            return image
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE."""
        try:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
        except Exception as e:
            logger.warning(f"Contrast enhancement failed: {e}")
            return image
    
    def _binarize_image(self, image: np.ndarray) -> np.ndarray:
        """Convert image to binary (black and white)."""
        try:
            # Use Otsu's thresholding for automatic threshold selection
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        except Exception as e:
            logger.warning(f"Binarization failed: {e}")
            return image


class OCRProcessor:
    """Main OCR processor for text extraction from images."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize OCR processor.
        
        Args:
            config: Configuration for OCR processing
        """
        self.config = config or {}
        self.preprocessor = ImagePreprocessor(self.config.get("preprocessing", {}))
        self.tesseract_config = self.config.get("tesseract_config", "--oem 3 --psm 6")
        self.min_confidence = self.config.get("min_confidence", 30.0)
        self.languages = self.config.get("languages", "eng")
        
        # Verify tesseract installation
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            raise RuntimeError(f"Tesseract not found or not properly installed: {e}")
    
    def extract_text_from_image(self, image_path: str) -> OCRResult:
        """
        Extract text from an image using OCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            OCRResult with extracted text and confidence scores
        """
        try:
            # Preprocess image
            processed_image, applied_operations = self.preprocessor.preprocess_image(image_path)
            
            # Perform OCR with detailed output
            ocr_data = pytesseract.image_to_data(
                processed_image,
                config=self.tesseract_config,
                lang=self.languages,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text and confidence scores
            text_parts = []
            word_confidences = []
            bounding_boxes = []
            
            for i in range(len(ocr_data['text'])):
                word = ocr_data['text'][i].strip()
                confidence = float(ocr_data['conf'][i])
                
                if word and confidence > self.min_confidence:
                    text_parts.append(word)
                    word_confidences.append((word, confidence))
                    
                    # Extract bounding box
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    bounding_boxes.append((x, y, w, h))
            
            # Combine text
            extracted_text = ' '.join(text_parts)
            
            # Calculate overall confidence
            if word_confidences:
                overall_confidence = sum(conf for _, conf in word_confidences) / len(word_confidences)
            else:
                overall_confidence = 0.0
            
            return OCRResult(
                text=extracted_text,
                confidence=overall_confidence,
                word_confidences=word_confidences,
                bounding_boxes=bounding_boxes,
                preprocessing_applied=applied_operations
            )
            
        except Exception as e:
            logger.error(f"OCR extraction failed for {image_path}: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                word_confidences=[],
                bounding_boxes=[],
                preprocessing_applied=[]
            )
    
    def extract_text_simple(self, image_path: str) -> str:
        """
        Simple text extraction without detailed confidence data.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text as string
        """
        result = self.extract_text_from_image(image_path)
        return result.text


class ScannedPDFHandler:
    """Handler for processing scanned PDFs with OCR."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize scanned PDF handler.
        
        Args:
            config: Configuration for PDF processing
        """
        self.config = config or {}
        self.ocr_processor = OCRProcessor(self.config.get("ocr", {}))
        self.dpi = self.config.get("dpi", 300)
        self.image_format = self.config.get("image_format", "png")
    
    def is_scanned_pdf(self, pdf_path: str) -> bool:
        """
        Determine if a PDF is scanned (image-based) or text-based.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            True if PDF appears to be scanned, False if it contains extractable text
        """
        try:
            doc = fitz.open(pdf_path)
            
            # Check first few pages for text content
            pages_to_check = min(3, len(doc))
            text_found = False
            
            for page_num in range(pages_to_check):
                page = doc[page_num]
                text = page.get_text().strip()
                
                # If we find substantial text, it's likely not scanned
                if len(text) > 100:  # Arbitrary threshold
                    text_found = True
                    break
            
            doc.close()
            return not text_found
            
        except Exception as e:
            logger.error(f"Error checking if PDF is scanned: {e}")
            # Assume it's scanned if we can't determine
            return True
    
    def extract_text_from_scanned_pdf(self, pdf_path: str) -> OCRResult:
        """
        Extract text from a scanned PDF using OCR.
        
        Args:
            pdf_path: Path to the scanned PDF file
            
        Returns:
            OCRResult with extracted text from all pages
        """
        try:
            doc = fitz.open(pdf_path)
            all_text = []
            all_word_confidences = []
            all_bounding_boxes = []
            all_preprocessing = []
            total_confidence = 0.0
            page_count = 0
            
            with tempfile.TemporaryDirectory() as temp_dir:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    
                    # Convert page to image
                    mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)  # Scale for DPI
                    pix = page.get_pixmap(matrix=mat)
                    
                    # Save as temporary image
                    temp_image_path = os.path.join(temp_dir, f"page_{page_num}.{self.image_format}")
                    pix.save(temp_image_path)
                    
                    # Perform OCR on the page
                    page_result = self.ocr_processor.extract_text_from_image(temp_image_path)
                    
                    if page_result.text.strip():
                        all_text.append(f"--- Page {page_num + 1} ---")
                        all_text.append(page_result.text)
                        all_word_confidences.extend(page_result.word_confidences)
                        all_bounding_boxes.extend(page_result.bounding_boxes)
                        all_preprocessing.extend(page_result.preprocessing_applied)
                        total_confidence += page_result.confidence
                        page_count += 1
            
            doc.close()
            
            # Combine results
            combined_text = '\n'.join(all_text)
            average_confidence = total_confidence / page_count if page_count > 0 else 0.0
            
            return OCRResult(
                text=combined_text,
                confidence=average_confidence,
                word_confidences=all_word_confidences,
                bounding_boxes=all_bounding_boxes,
                preprocessing_applied=list(set(all_preprocessing))  # Remove duplicates
            )
            
        except Exception as e:
            logger.error(f"Error extracting text from scanned PDF {pdf_path}: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                word_confidences=[],
                bounding_boxes=[],
                preprocessing_applied=[]
            )


class OCRExtractor(BaseExtractor):
    """Extractor for image files and scanned PDFs using OCR."""
    
    SUPPORTED_IMAGE_FORMATS = {
        'png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp', 'gif', 'webp'
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize OCR extractor.
        
        Args:
            config: Configuration for OCR extraction
        """
        super().__init__(config)
        self.ocr_processor = OCRProcessor(self.config.get("ocr", {}))
        self.pdf_handler = ScannedPDFHandler(self.config.get("scanned_pdf", {}))
        self.min_text_length = self.config.get("min_text_length", 10)
    
    def supports_format(self, file_type: str) -> bool:
        """
        Check if this extractor supports the given file type.
        
        Args:
            file_type: The file type to check
            
        Returns:
            True if the extractor supports this file type
        """
        return file_type.lower() in self.SUPPORTED_IMAGE_FORMATS or file_type.lower() == 'pdf'
    
    def extract(self, file_path: str) -> ExtractedContent:
        """
        Extract text content from image files or scanned PDFs using OCR.
        
        Args:
            file_path: Path to the image or PDF file
            
        Returns:
            ExtractedContent with OCR-extracted text
        """
        try:
            self.validate_file(file_path)
            metadata = self.get_metadata(file_path)
            
            # Determine file type and process accordingly
            file_extension = Path(file_path).suffix.lower().lstrip('.')
            
            if file_extension == 'pdf':
                # Check if it's a scanned PDF
                if self.pdf_handler.is_scanned_pdf(file_path):
                    ocr_result = self.pdf_handler.extract_text_from_scanned_pdf(file_path)
                else:
                    # Not a scanned PDF, should be handled by regular PDF extractor
                    raise ProcessingError(
                        stage="ocr_extraction",
                        error_type="NotScannedPDF",
                        message=f"PDF {file_path} contains extractable text and should not be processed with OCR",
                        severity=ErrorSeverity.MEDIUM
                    )
            else:
                # Regular image file
                ocr_result = self.ocr_processor.extract_text_from_image(file_path)
            
            # Check if we extracted enough text
            if len(ocr_result.text.strip()) < self.min_text_length:
                raise ProcessingError(
                    stage="ocr_extraction",
                    error_type="InsufficientText",
                    message=f"OCR extracted only {len(ocr_result.text)} characters from {file_path}",
                    severity=ErrorSeverity.LOW
                )
            
            # Create document structure
            structure = DocumentStructure()
            structure.paragraphs = len([p for p in ocr_result.text.split('\n') if p.strip()])
            
            # Create image data entry
            images = [ImageData(
                path=file_path,
                caption=f"OCR processed image (confidence: {ocr_result.confidence:.1f}%)"
            )]
            
            # Create extracted content
            extracted = ExtractedContent(
                content=ocr_result.text,
                metadata=metadata
            )
            extracted.structure = structure
            extracted.images = images
            
            # Add OCR-specific metadata
            if hasattr(extracted, 'ocr_result'):
                extracted.ocr_result = ocr_result
            
            return extracted
            
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(
                stage="ocr_extraction",
                error_type="ExtractionError",
                message=f"Failed to extract text from {file_path}: {str(e)}",
                severity=ErrorSeverity.HIGH,
                stack_trace=str(e)
            )