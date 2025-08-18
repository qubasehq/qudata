# OCR Test Sample Data

This directory contains sample images and documents for testing OCR functionality.

## Image Files

### Simple Text Images
- `simple_text.png` - Basic "Hello World" text
- `pangram.png` - Complete alphabet pangram
- `numbers_symbols.png` - Numbers and special characters

### Challenging Images
- `skewed_15deg.png` - Text rotated 15 degrees clockwise
- `skewed_neg10deg.png` - Text rotated 10 degrees counterclockwise
- `noisy_text.png` - Text with noise and blur artifacts
- `low_resolution.png` - Low resolution image requiring upscaling

### Complex Content
- `multi_line_text.png` - Multiple lines of text
- `table_data.png` - Tabular data with headers and rows

### Format Variations
- `format_test.jpg` - JPEG format
- `format_test.tiff` - TIFF format
- `format_test.bmp` - BMP format

## Usage

These images are used by the OCR unit tests to verify:
- Text extraction accuracy
- Image preprocessing effectiveness
- Confidence scoring
- Format support
- Error handling

## Notes

- All images contain readable text for OCR testing
- Images are designed to test different OCR challenges
- Font sizes and styles are optimized for OCR recognition
