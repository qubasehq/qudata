"""
Image content extraction module for QuData.

This module provides extraction capabilities for image files, including SVG files
which can contain textual content that should be extracted for analysis.
"""

import os
import re
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

# Optional dependency for image processing
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Required dependency for XML/SVG processing
try:
    from xml.etree import ElementTree as ET
    HAS_XML = True
except ImportError:
    HAS_XML = False

from ..models import (
    BaseExtractor, ExtractedContent, FileMetadata, DocumentStructure,
    ProcessingError, ErrorSeverity, ImageData, parse_file_size
)


class ImageExtractor(BaseExtractor):
    """
    Extractor for image files, with special handling for SVG files.
    
    SVG files can contain textual content that should be extracted for analysis.
    Other image formats are processed for metadata only.
    """
    
    SUPPORTED_FORMATS = {'svg', 'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'tiff', 'ico', 'heic', 'heif'}
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the ImageExtractor.
        
        Args:
            config: Configuration dictionary with options like:
                - max_file_size: Maximum file size to process (default: 50MB)
                - extract_svg_text: Whether to extract text from SVG files (default: True)
                - extract_metadata: Whether to extract image metadata (default: True)
        """
        super().__init__(config)
        
        self.max_file_size = parse_file_size(self.config.get('max_file_size', 50 * 1024 * 1024))  # 50MB
        self.extract_svg_text = self.config.get('extract_svg_text', True)
        self.extract_metadata = self.config.get('extract_metadata', True)
    
    def extract(self, file_path: str) -> ExtractedContent:
        """
        Extract content from an image file.
        
        Args:
            file_path: Path to the image file to extract content from
            
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
            if file_metadata.file_type == 'svg':
                content, structure, images = self._extract_svg_content(file_path)
            else:
                content, structure, images = self._extract_image_metadata(file_path, file_metadata)
            
            # Create extracted content object
            extracted = ExtractedContent(content, file_metadata)
            extracted.structure = structure
            extracted.images = images
            
            return extracted
            
        except ProcessingError:
            raise
        except Exception as e:
            raise self.create_processing_error(
                stage="extraction",
                error_type="ImageExtractionError",
                message=f"Failed to extract content from image {file_path}: {str(e)}",
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
    
    def _extract_svg_content(self, file_path: str) -> Tuple[str, DocumentStructure, List[ImageData]]:
        """
        Extract text content from SVG files.
        
        Args:
            file_path: Path to the SVG file
            
        Returns:
            Tuple of (content, structure, images)
        """
        try:
            if not HAS_XML:
                # If XML parsing is not available, treat as regular image
                return self._extract_image_metadata(file_path, self.get_metadata(file_path))
            
            # Parse SVG as XML
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Extract text elements
            text_elements = []
            self._extract_svg_text_recursive(root, text_elements)
            
            # Create content
            content_parts = []
            content_parts.append(f"# SVG Content from {os.path.basename(file_path)}")
            content_parts.append("")
            
            if text_elements:
                content_parts.append("## Text Content")
                for text in text_elements:
                    if text.strip():
                        content_parts.append(f"- {text.strip()}")
                content_parts.append("")
            
            # Extract SVG metadata
            width = root.get('width', 'unknown')
            height = root.get('height', 'unknown')
            viewbox = root.get('viewBox', 'unknown')
            
            content_parts.append("## SVG Metadata")
            content_parts.append(f"- Width: {width}")
            content_parts.append(f"- Height: {height}")
            content_parts.append(f"- ViewBox: {viewbox}")
            
            content = "\n".join(content_parts)
            
            # Create structure
            structure = DocumentStructure(
                headings=["SVG Content", "Text Content", "SVG Metadata"] if text_elements else ["SVG Content", "SVG Metadata"],
                paragraphs=len(text_elements) if text_elements else 1,
                tables=0,
                images=1,
                code_blocks=0,
                lists=1
            )
            
            # Create image data
            try:
                width_int = int(re.sub(r'[^\d]', '', str(width))) if width != 'unknown' else None
            except (ValueError, TypeError):
                width_int = None
                
            try:
                height_int = int(re.sub(r'[^\d]', '', str(height))) if height != 'unknown' else None
            except (ValueError, TypeError):
                height_int = None
            
            image_data = ImageData(
                path=file_path,
                caption=f"SVG: {os.path.basename(file_path)}",
                alt_text="SVG image with extracted text content",
                width=width_int,
                height=height_int
            )
            
            return content, structure, [image_data]
            
        except ET.ParseError as e:
            raise self.create_processing_error(
                stage="extraction",
                error_type="SVGParsingError",
                message=f"Error parsing SVG file {file_path}: {str(e)}",
                severity=ErrorSeverity.HIGH,
                stack_trace=str(e)
            )
        except Exception as e:
            # Fall back to regular image metadata extraction
            return self._extract_image_metadata(file_path, self.get_metadata(file_path))
    
    def _extract_svg_text_recursive(self, element, text_elements: List[str]):
        """
        Recursively extract text from SVG elements.
        
        Args:
            element: XML element to process
            text_elements: List to append found text to
        """
        # Extract text from text elements
        if element.tag.endswith('text') or element.tag.endswith('tspan'):
            if element.text:
                text_elements.append(element.text)
        
        # Extract title and desc elements
        if element.tag.endswith('title') or element.tag.endswith('desc'):
            if element.text:
                text_elements.append(element.text)
        
        # Recursively process child elements
        for child in element:
            self._extract_svg_text_recursive(child, text_elements)
    
    def _extract_image_metadata(self, file_path: str, file_metadata: FileMetadata) -> Tuple[str, DocumentStructure, List[ImageData]]:
        """
        Extract metadata from image files.
        
        Args:
            file_path: Path to the image file
            file_metadata: File metadata
            
        Returns:
            Tuple of (content, structure, images)
        """
        content_parts = []
        content_parts.append(f"# Image: {os.path.basename(file_path)}")
        content_parts.append("")
        content_parts.append("## Image Metadata")
        content_parts.append(f"- File Type: {file_metadata.file_type.upper()}")
        content_parts.append(f"- File Size: {file_metadata.size_bytes} bytes")
        
        width = None
        height = None
        
        # Try to get image dimensions using PIL if available
        if HAS_PIL and self.extract_metadata:
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    content_parts.append(f"- Dimensions: {width} x {height} pixels")
                    content_parts.append(f"- Format: {img.format}")
                    if hasattr(img, 'mode'):
                        content_parts.append(f"- Mode: {img.mode}")
            except Exception as e:
                content_parts.append(f"- Dimensions: Could not determine ({str(e)})")
        else:
            content_parts.append("- Dimensions: PIL not available for metadata extraction")
        
        content = "\n".join(content_parts)
        
        # Create structure
        structure = DocumentStructure(
            headings=["Image", "Image Metadata"],
            paragraphs=1,
            tables=0,
            images=1,
            code_blocks=0,
            lists=1
        )
        
        # Create image data
        image_data = ImageData(
            path=file_path,
            caption=f"{file_metadata.file_type.upper()}: {os.path.basename(file_path)}",
            alt_text=f"Image file of type {file_metadata.file_type}",
            width=width,
            height=height
        )
        
        return content, structure, [image_data]
