#!/usr/bin/env python3
"""
Create sample images and scanned PDFs for OCR testing.

This script generates test data including:
- Simple text images
- Skewed/rotated images
- Low quality/noisy images
- Multi-page scanned PDFs
"""

import os
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

def create_sample_directory():
    """Create sample data directory structure."""
    base_dir = Path(__file__).parent / "sample_data" / "ocr"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir

def create_simple_text_image(text: str, filename: str, size: tuple = (600, 200)) -> str:
    """Create a simple text image."""
    image = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(image)
    
    # Try to use a system font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 32)  # macOS
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)  # Linux
            except (OSError, IOError):
                font = ImageFont.load_default()
    
    # Calculate text position (centered)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    draw.text((x, y), text, fill='black', font=font)
    
    sample_dir = create_sample_directory()
    image_path = sample_dir / filename
    image.save(image_path)
    return str(image_path)

def create_skewed_image(text: str, filename: str, angle: float = 15.0) -> str:
    """Create a skewed/rotated text image."""
    image = Image.new('RGB', (600, 300), color='white')
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except (OSError, IOError):
        font = ImageFont.load_default()
    
    # Draw text
    draw.text((50, 120), text, fill='black', font=font)
    
    # Rotate the image
    rotated = image.rotate(angle, fillcolor='white', expand=True)
    
    sample_dir = create_sample_directory()
    image_path = sample_dir / filename
    rotated.save(image_path)
    return str(image_path)

def create_noisy_image(text: str, filename: str) -> str:
    """Create a noisy/low quality text image."""
    image = Image.new('RGB', (600, 200), color='white')
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except (OSError, IOError):
        font = ImageFont.load_default()
    
    # Draw text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    x = (600 - text_width) // 2
    draw.text((x, 80), text, fill='black', font=font)
    
    # Add noise
    image_array = np.array(image)
    noise = np.random.randint(0, 50, image_array.shape, dtype=np.uint8)
    noisy_array = np.clip(image_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    noisy_image = Image.fromarray(noisy_array)
    
    # Apply blur
    blurred = noisy_image.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    sample_dir = create_sample_directory()
    image_path = sample_dir / filename
    blurred.save(image_path)
    return str(image_path)

def create_multi_line_image(lines: list, filename: str) -> str:
    """Create an image with multiple lines of text."""
    line_height = 40
    padding = 20
    height = len(lines) * line_height + 2 * padding
    image = Image.new('RGB', (800, height), color='white')
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except (OSError, IOError):
        font = ImageFont.load_default()
    
    # Draw each line
    for i, line in enumerate(lines):
        y = padding + i * line_height
        draw.text((padding, y), line, fill='black', font=font)
    
    sample_dir = create_sample_directory()
    image_path = sample_dir / filename
    image.save(image_path)
    return str(image_path)

def create_table_image(filename: str) -> str:
    """Create an image with tabular data."""
    image = Image.new('RGB', (700, 400), color='white')
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        header_font = ImageFont.truetype("arial.ttf", 22)
    except (OSError, IOError):
        font = ImageFont.load_default()
        header_font = font
    
    # Table data
    headers = ["Name", "Age", "City", "Occupation"]
    rows = [
        ["John Doe", "30", "New York", "Engineer"],
        ["Jane Smith", "25", "Los Angeles", "Designer"],
        ["Bob Johnson", "35", "Chicago", "Manager"],
        ["Alice Brown", "28", "Boston", "Developer"]
    ]
    
    # Column positions
    col_positions = [50, 200, 300, 450]
    row_height = 35
    start_y = 50
    
    # Draw headers
    for i, header in enumerate(headers):
        draw.text((col_positions[i], start_y), header, fill='black', font=header_font)
    
    # Draw horizontal line under headers
    draw.line([(40, start_y + 25), (650, start_y + 25)], fill='black', width=2)
    
    # Draw data rows
    for row_idx, row in enumerate(rows):
        y = start_y + (row_idx + 1) * row_height + 10
        for col_idx, cell in enumerate(row):
            draw.text((col_positions[col_idx], y), cell, fill='black', font=font)
    
    sample_dir = create_sample_directory()
    image_path = sample_dir / filename
    image.save(image_path)
    return str(image_path)

def create_low_resolution_image(text: str, filename: str) -> str:
    """Create a low resolution image that needs upscaling."""
    # Create small image first
    small_image = Image.new('RGB', (200, 100), color='white')
    draw = ImageDraw.Draw(small_image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()
    
    draw.text((20, 40), text, fill='black', font=font)
    
    sample_dir = create_sample_directory()
    image_path = sample_dir / filename
    small_image.save(image_path)
    return str(image_path)

def create_sample_images():
    """Create all sample images for testing."""
    print("Creating sample OCR test images...")
    
    # Simple text images
    create_simple_text_image("Hello World OCR Test", "simple_text.png")
    create_simple_text_image("The quick brown fox jumps over the lazy dog", "pangram.png")
    create_simple_text_image("1234567890 !@#$%^&*()", "numbers_symbols.png")
    
    # Skewed images
    create_skewed_image("This text is rotated 15 degrees", "skewed_15deg.png", 15.0)
    create_skewed_image("This text is rotated -10 degrees", "skewed_neg10deg.png", -10.0)
    
    # Noisy/low quality images
    create_noisy_image("Noisy text with artifacts", "noisy_text.png")
    
    # Multi-line text
    lines = [
        "This is line one of the document.",
        "Here is the second line with more text.",
        "The third line contains different content.",
        "Finally, this is the last line."
    ]
    create_multi_line_image(lines, "multi_line_text.png")
    
    # Table image
    create_table_image("table_data.png")
    
    # Low resolution image
    create_low_resolution_image("Low res text", "low_resolution.png")
    
    # Different formats
    # Convert one image to different formats
    simple_path = create_simple_text_image("Format test image", "temp_format.png")
    simple_image = Image.open(simple_path)
    
    sample_dir = create_sample_directory()
    simple_image.save(sample_dir / "format_test.jpg", "JPEG")
    simple_image.save(sample_dir / "format_test.tiff", "TIFF")
    simple_image.save(sample_dir / "format_test.bmp", "BMP")
    
    # Remove temporary file
    os.remove(simple_path)
    
    print(f"Sample images created in: {sample_dir}")
    return sample_dir

def create_readme():
    """Create a README file describing the test data."""
    sample_dir = create_sample_directory()
    readme_path = sample_dir / "README.md"
    
    readme_content = """# OCR Test Sample Data

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
"""
    
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"README created: {readme_path}")

if __name__ == "__main__":
    sample_dir = create_sample_images()
    create_readme()
    print(f"\nAll OCR test data created successfully in: {sample_dir}")
    print("You can now run the OCR tests with: python -m pytest tests/unit/test_ocr_processing.py")