"""
Utility script to create sample PDF files for testing.

This script creates various types of PDF files to test the PDF extractor,
including simple text PDFs, PDFs with tables, and corrupted PDFs.
"""

import os
from pathlib import Path

# Optional dependency for PDF creation
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


def create_sample_pdfs():
    """Create sample PDF files for testing."""
    if not HAS_REPORTLAB:
        print("reportlab not available. Install with: pip install reportlab")
        return
    
    # Create sample_data directory if it doesn't exist
    sample_dir = Path("tests/sample_data/pdfs")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Create simple text PDF
    create_simple_text_pdf(sample_dir / "sample_simple.pdf")
    
    # Create PDF with tables
    create_table_pdf(sample_dir / "sample_table.pdf")
    
    # Create multi-page PDF
    create_multipage_pdf(sample_dir / "sample_multipage.pdf")
    
    # Create PDF with headings
    create_structured_pdf(sample_dir / "sample_structured.pdf")
    
    # Create corrupted PDF (just invalid content)
    create_corrupted_pdf(sample_dir / "sample_corrupted.pdf")
    
    print("Sample PDF files created successfully!")


def create_simple_text_pdf(filename):
    """Create a simple PDF with text content."""
    c = canvas.Canvas(str(filename), pagesize=letter)
    width, height = letter
    
    # Add title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 100, "Sample PDF Document")
    
    # Add content
    c.setFont("Helvetica", 12)
    y_position = height - 150
    
    content = [
        "This is a sample PDF document created for testing the LLMDataForge PDF extractor.",
        "",
        "The document contains multiple paragraphs to test structure analysis.",
        "",
        "It includes various types of content:",
        "• Plain text paragraphs",
        "• Bullet points",
        "• Different formatting",
        "",
        "This helps verify that the PDF extractor can properly analyze document",
        "structure and extract meaningful content for LLM training datasets."
    ]
    
    for line in content:
        if line:
            c.drawString(100, y_position, line)
        y_position -= 20
    
    c.save()


def create_table_pdf(filename):
    """Create a PDF with tables."""
    doc = SimpleDocTemplate(str(filename), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Add title
    title = Paragraph("PDF with Tables", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Add description
    desc = Paragraph("This PDF contains tables to test table extraction capabilities.", styles['Normal'])
    story.append(desc)
    story.append(Spacer(1, 12))
    
    # Create a table
    data = [
        ['Name', 'Age', 'City', 'Occupation'],
        ['John Doe', '28', 'New York', 'Engineer'],
        ['Jane Smith', '34', 'Los Angeles', 'Designer'],
        ['Bob Johnson', '45', 'Chicago', 'Manager'],
        ['Alice Brown', '29', 'Boston', 'Analyst']
    ]
    
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(table)
    story.append(Spacer(1, 12))
    
    # Add another paragraph
    para = Paragraph("The table above demonstrates the extractor's ability to parse structured data.", styles['Normal'])
    story.append(para)
    
    doc.build(story)


def create_multipage_pdf(filename):
    """Create a multi-page PDF."""
    c = canvas.Canvas(str(filename), pagesize=letter)
    width, height = letter
    
    # Page 1
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 100, "Multi-Page PDF - Page 1")
    
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 150, "This is the content of the first page.")
    c.drawString(100, height - 170, "It contains some introductory text.")
    
    c.showPage()  # Start new page
    
    # Page 2
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 100, "Multi-Page PDF - Page 2")
    
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 150, "This is the content of the second page.")
    c.drawString(100, height - 170, "It demonstrates multi-page extraction capabilities.")
    
    c.showPage()  # Start new page
    
    # Page 3
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 100, "Multi-Page PDF - Page 3")
    
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 150, "This is the final page of the document.")
    c.drawString(100, height - 170, "All pages should be processed correctly.")
    
    c.save()


def create_structured_pdf(filename):
    """Create a PDF with clear structure and headings."""
    doc = SimpleDocTemplate(str(filename), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom heading style
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading1'],
        fontSize=14,
        spaceAfter=12,
        textColor=colors.black
    )
    
    # Title
    title = Paragraph("STRUCTURED DOCUMENT", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 20))
    
    # Introduction
    intro_heading = Paragraph("1. INTRODUCTION", heading_style)
    story.append(intro_heading)
    
    intro_text = Paragraph(
        "This document demonstrates structured content with clear headings and sections. "
        "The PDF extractor should be able to identify these structural elements.",
        styles['Normal']
    )
    story.append(intro_text)
    story.append(Spacer(1, 12))
    
    # Methodology
    method_heading = Paragraph("2. METHODOLOGY", heading_style)
    story.append(method_heading)
    
    method_text = Paragraph(
        "The methodology section contains information about the approach used. "
        "This includes various steps and procedures:",
        styles['Normal']
    )
    story.append(method_text)
    
    # List items
    list_items = [
        "• Data collection and preprocessing",
        "• Feature extraction and analysis",
        "• Model training and validation",
        "• Results evaluation and interpretation"
    ]
    
    for item in list_items:
        item_para = Paragraph(item, styles['Normal'])
        story.append(item_para)
    
    story.append(Spacer(1, 12))
    
    # Results
    results_heading = Paragraph("3. RESULTS", heading_style)
    story.append(results_heading)
    
    results_text = Paragraph(
        "The results section presents the findings of the analysis. "
        "Key metrics and observations are summarized below.",
        styles['Normal']
    )
    story.append(results_text)
    story.append(Spacer(1, 12))
    
    # Conclusion
    conclusion_heading = Paragraph("4. CONCLUSION", heading_style)
    story.append(conclusion_heading)
    
    conclusion_text = Paragraph(
        "In conclusion, this structured document format allows the PDF extractor "
        "to properly identify and extract hierarchical content organization.",
        styles['Normal']
    )
    story.append(conclusion_text)
    
    doc.build(story)


def create_corrupted_pdf(filename):
    """Create a corrupted PDF file for error testing."""
    with open(filename, 'wb') as f:
        # Write invalid PDF content
        f.write(b"This is not a valid PDF file content")
        f.write(b"It should cause parsing errors")
        f.write(b"When the PDF extractor tries to process it")


if __name__ == "__main__":
    create_sample_pdfs()