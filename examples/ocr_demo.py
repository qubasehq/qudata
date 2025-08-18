#!/usr/bin/env python3
"""
OCR Processing Demo

This script demonstrates the OCR capabilities of QuData, including:
- Text extraction from images
- Image preprocessing for better OCR accuracy
- Scanned PDF processing
- Confidence scoring
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qudata.ingest.ocr import OCRProcessor, ImagePreprocessor, ScannedPDFHandler, OCRExtractor


def demo_image_preprocessing():
    """Demonstrate image preprocessing capabilities."""
    print("=== Image Preprocessing Demo ===")
    
    # Create preprocessor with different configurations
    configs = [
        {"enable_denoising": True, "enable_deskewing": True, "enable_contrast_enhancement": True},
        {"enable_denoising": False, "enable_deskewing": False, "enable_contrast_enhancement": False},
    ]
    
    for i, config in enumerate(configs):
        print(f"\nConfiguration {i+1}: {config}")
        preprocessor = ImagePreprocessor(config)
        
        # Test with sample images if they exist
        sample_dir = Path(__file__).parent.parent / "tests" / "sample_data" / "ocr"
        if sample_dir.exists():
            for image_file in sample_dir.glob("*.png"):
                try:
                    processed_image, operations = preprocessor.preprocess_image(str(image_file))
                    print(f"  {image_file.name}: Applied operations: {operations}")
                    print(f"    Processed image shape: {processed_image.shape}")
                except Exception as e:
                    print(f"  {image_file.name}: Error - {e}")
        else:
            print("  No sample images found. Run tests/create_sample_ocr_data.py first.")


def demo_ocr_processing():
    """Demonstrate OCR text extraction."""
    print("\n=== OCR Processing Demo ===")
    
    # Create OCR processor with different configurations
    configs = [
        {"min_confidence": 30.0, "languages": "eng"},
        {"min_confidence": 50.0, "languages": "eng", "tesseract_config": "--oem 3 --psm 8"},
    ]
    
    for i, config in enumerate(configs):
        print(f"\nOCR Configuration {i+1}: {config}")
        try:
            ocr_processor = OCRProcessor(config)
            
            # Test with sample images
            sample_dir = Path(__file__).parent.parent / "tests" / "sample_data" / "ocr"
            if sample_dir.exists():
                for image_file in sample_dir.glob("*.png")[:3]:  # Test first 3 images
                    try:
                        result = ocr_processor.extract_text_from_image(str(image_file))
                        print(f"  {image_file.name}:")
                        print(f"    Extracted text: '{result.text[:100]}{'...' if len(result.text) > 100 else ''}'")
                        print(f"    Confidence: {result.confidence:.1f}%")
                        print(f"    Words found: {len(result.word_confidences)}")
                        print(f"    Preprocessing: {result.preprocessing_applied}")
                    except Exception as e:
                        print(f"  {image_file.name}: Error - {e}")
            else:
                print("  No sample images found. Run tests/create_sample_ocr_data.py first.")
        except Exception as e:
            print(f"  OCR Processor initialization failed: {e}")


def demo_scanned_pdf_handling():
    """Demonstrate scanned PDF processing."""
    print("\n=== Scanned PDF Processing Demo ===")
    
    try:
        pdf_handler = ScannedPDFHandler()
        
        # Create a simple test to show the interface
        print("PDF Handler initialized successfully")
        print("Features:")
        print("  - Automatic detection of scanned vs text-based PDFs")
        print("  - Multi-page OCR processing")
        print("  - Configurable DPI and image format")
        print("  - Page-by-page confidence scoring")
        
        # Note: We can't easily create a real PDF for demo without additional dependencies
        print("\nTo test with real PDFs:")
        print("  1. Place scanned PDF files in the tests/sample_data/ocr/ directory")
        print("  2. Use pdf_handler.is_scanned_pdf(pdf_path) to check if OCR is needed")
        print("  3. Use pdf_handler.extract_text_from_scanned_pdf(pdf_path) to extract text")
        
    except Exception as e:
        print(f"PDF Handler initialization failed: {e}")


def demo_ocr_extractor():
    """Demonstrate the complete OCR extractor."""
    print("\n=== OCR Extractor Demo ===")
    
    try:
        extractor = OCRExtractor()
        
        print("Supported formats:")
        formats_to_test = ['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif', 'webp', 'pdf', 'txt', 'docx']
        for fmt in formats_to_test:
            supported = extractor.supports_format(fmt)
            print(f"  {fmt}: {'✓' if supported else '✗'}")
        
        # Test with sample images
        sample_dir = Path(__file__).parent.parent / "tests" / "sample_data" / "ocr"
        if sample_dir.exists():
            print(f"\nTesting with sample images from {sample_dir}:")
            for image_file in sample_dir.glob("*.png")[:2]:  # Test first 2 images
                try:
                    extracted = extractor.extract(str(image_file))
                    print(f"\n  {image_file.name}:")
                    print(f"    Content length: {len(extracted.content)} characters")
                    print(f"    File type: {extracted.metadata.file_type}")
                    print(f"    File size: {extracted.metadata.size_bytes} bytes")
                    print(f"    Structure - Paragraphs: {extracted.structure.paragraphs}")
                    print(f"    Images: {len(extracted.images)}")
                    if extracted.content:
                        preview = extracted.content[:150].replace('\n', ' ')
                        print(f"    Preview: '{preview}{'...' if len(extracted.content) > 150 else ''}'")
                except Exception as e:
                    print(f"  {image_file.name}: Error - {e}")
        else:
            print("  No sample images found. Run tests/create_sample_ocr_data.py first.")
            
    except Exception as e:
        print(f"OCR Extractor initialization failed: {e}")


def demo_configuration_options():
    """Demonstrate various configuration options."""
    print("\n=== Configuration Options Demo ===")
    
    print("Available configuration options:")
    
    print("\n1. ImagePreprocessor config:")
    print("   - enable_denoising: bool (default: True)")
    print("   - enable_deskewing: bool (default: True)")
    print("   - enable_contrast_enhancement: bool (default: True)")
    print("   - enable_binarization: bool (default: True)")
    print("   - target_dpi: int (default: 300)")
    
    print("\n2. OCRProcessor config:")
    print("   - tesseract_config: str (default: '--oem 3 --psm 6')")
    print("   - min_confidence: float (default: 30.0)")
    print("   - languages: str (default: 'eng')")
    print("   - preprocessing: dict (ImagePreprocessor config)")
    
    print("\n3. ScannedPDFHandler config:")
    print("   - dpi: int (default: 300)")
    print("   - image_format: str (default: 'png')")
    print("   - ocr: dict (OCRProcessor config)")
    
    print("\n4. OCRExtractor config:")
    print("   - min_text_length: int (default: 10)")
    print("   - ocr: dict (OCRProcessor config)")
    print("   - scanned_pdf: dict (ScannedPDFHandler config)")
    
    # Example configuration
    example_config = {
        "min_text_length": 20,
        "ocr": {
            "min_confidence": 60.0,
            "languages": "eng+fra",
            "tesseract_config": "--oem 1 --psm 8",
            "preprocessing": {
                "enable_denoising": True,
                "enable_deskewing": False,
                "target_dpi": 400
            }
        },
        "scanned_pdf": {
            "dpi": 400,
            "image_format": "tiff"
        }
    }
    
    print(f"\nExample configuration:")
    import json
    print(json.dumps(example_config, indent=2))


def main():
    """Run all OCR demos."""
    print("QuData OCR Processing Demo")
    print("=" * 50)
    
    try:
        demo_image_preprocessing()
        demo_ocr_processing()
        demo_scanned_pdf_handling()
        demo_ocr_extractor()
        demo_configuration_options()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nNext steps:")
        print("1. Install Tesseract OCR on your system if not already installed")
        print("2. Run tests/create_sample_ocr_data.py to create test images")
        print("3. Test with your own images and scanned PDFs")
        print("4. Experiment with different configuration options")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()