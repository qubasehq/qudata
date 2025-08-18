#!/usr/bin/env python3
"""
Content Segmentation Demo

This script demonstrates the content segmentation functionality for creating
LLM training datasets in various formats (ChatML, JSONL, Alpaca).
"""

import json
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qudata.export.segmenter import (
    ContentSegmenter,
    OutputFormat,
    SegmentationType,
)
from qudata.models import Document, DocumentMetadata


def create_sample_document(content: str, doc_id: str = "sample") -> Document:
    """Create a sample document for testing."""
    metadata = DocumentMetadata(
        file_type="txt",
        size_bytes=len(content),
        language="en",
        domain="demo"
    )
    
    return Document(
        id=doc_id,
        source_path=f"{doc_id}.txt",
        content=content,
        metadata=metadata
    )


def demo_qa_segmentation():
    """Demonstrate Q&A content segmentation."""
    print("=== Q&A Content Segmentation Demo ===")
    
    qa_content = """
    Q: What is machine learning?
    A: Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.
    
    Q: How does deep learning work?
    A: Deep learning uses neural networks with multiple layers to automatically discover patterns in data, mimicking how the human brain processes information.
    
    Q: What are some common ML algorithms?
    A: Common algorithms include linear regression, decision trees, random forests, support vector machines, and neural networks.
    """
    
    document = create_sample_document(qa_content, "qa_demo")
    segmenter = ContentSegmenter()
    
    # Segment using Q&A strategy
    segments = segmenter.segment_document(document, strategy="qa")
    
    print(f"Found {len(segments)} Q&A segments:")
    for i, segment in enumerate(segments, 1):
        print(f"\nSegment {i}:")
        print(f"  Question: {segment.context}")
        print(f"  Answer: {segment.output}")
    
    # Export to different formats
    print("\n--- Alpaca Format ---")
    alpaca_format = segmenter.export_segments(segments, OutputFormat.ALPACA)
    print(json.dumps(alpaca_format[0], indent=2))
    
    print("\n--- ChatML Format ---")
    chatml_format = segmenter.export_segments(segments, OutputFormat.CHATML)
    print(json.dumps(chatml_format[0], indent=2))
    
    print("\n--- JSONL Format ---")
    jsonl_format = segmenter.export_segments(segments, OutputFormat.JSONL)
    print(json.dumps(jsonl_format[0], indent=2))


def demo_instruction_segmentation():
    """Demonstrate instruction content segmentation."""
    print("\n\n=== Instruction Content Segmentation Demo ===")
    
    instruction_content = """
    Task: Translate the following text to Spanish
    Input: Hello, how are you today?
    Output: Hola, ¿cómo estás hoy?
    
    Task: Summarize the following paragraph
    Input: Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.
    Output: Machine learning is an AI method that enables systems to automatically learn from data and make decisions with minimal human input.
    """
    
    document = create_sample_document(instruction_content, "instruction_demo")
    segmenter = ContentSegmenter()
    
    # Segment using instruction strategy
    segments = segmenter.segment_document(document, strategy="instruction")
    
    print(f"Found {len(segments)} instruction segments:")
    for i, segment in enumerate(segments, 1):
        print(f"\nSegment {i}:")
        print(f"  Instruction: {segment.instruction}")
        print(f"  Input: {segment.context}")
        print(f"  Output: {segment.output}")
    
    # Export to Alpaca format (popular for instruction tuning)
    if segments:
        alpaca_format = segmenter.export_segments(segments, OutputFormat.ALPACA)
        print(f"\n--- Alpaca Format (First Segment) ---")
        print(json.dumps(alpaca_format[0], indent=2))
    else:
        print("\nNo instruction segments found. Trying auto-detection...")
        segments = segmenter.segment_document(document, strategy="auto")
        if segments:
            alpaca_format = segmenter.export_segments(segments, OutputFormat.ALPACA)
            print(f"--- Auto-detected Format (First Segment) ---")
            print(json.dumps(alpaca_format[0], indent=2))


def demo_conversation_segmentation():
    """Demonstrate conversation content segmentation."""
    print("\n\n=== Conversation Content Segmentation Demo ===")
    
    conversation_content = """
    User: Hello, can you help me with Python programming?
    Assistant: Of course! I'd be happy to help you with Python programming. What specific topic or problem would you like assistance with?
    
    User: How do I create a list in Python?
    Assistant: You can create a list in Python using square brackets. For example: my_list = [1, 2, 3, 'hello', 'world']. Lists can contain different data types and are mutable.
    
    User: What's the difference between a list and a tuple?
    Assistant: The main differences are: 1) Lists are mutable (can be changed) while tuples are immutable, 2) Lists use square brackets [] while tuples use parentheses (), 3) Lists are generally used for homogeneous data while tuples for heterogeneous data.
    """
    
    document = create_sample_document(conversation_content, "conversation_demo")
    segmenter = ContentSegmenter()
    
    # Segment using conversation strategy
    segments = segmenter.segment_document(document, strategy="conversation")
    
    print(f"Found {len(segments)} conversation segments:")
    for i, segment in enumerate(segments, 1):
        print(f"\nSegment {i}:")
        print(f"  User: {segment.context}")
        print(f"  Assistant: {segment.output}")
    
    # Export to ChatML format (ideal for conversational AI)
    chatml_format = segmenter.export_segments(segments, OutputFormat.CHATML)
    print(f"\n--- ChatML Format (First Segment) ---")
    print(json.dumps(chatml_format[0], indent=2))


def demo_auto_segmentation():
    """Demonstrate automatic segmentation strategy."""
    print("\n\n=== Auto Segmentation Demo ===")
    
    mixed_content = """
    Q: What is the capital of France?
    A: The capital of France is Paris.
    
    Task: Convert temperature
    Input: 32°F
    Output: 0°C
    
    User: What's 2 + 2?
    Assistant: 2 + 2 equals 4.
    
    This is some unstructured text that doesn't follow any particular pattern. 
    It should be handled by the fallback completion strategy.
    """
    
    document = create_sample_document(mixed_content, "mixed_demo")
    segmenter = ContentSegmenter()
    
    # Use auto strategy to detect the best approach
    segments = segmenter.segment_document(document, strategy="auto")
    
    print(f"Found {len(segments)} segments using auto-detection:")
    for i, segment in enumerate(segments, 1):
        print(f"\nSegment {i} (Type: {segment.segment_type.value}):")
        if segment.instruction:
            print(f"  Instruction: {segment.instruction}")
        if segment.context:
            print(f"  Context/Input: {segment.context[:100]}...")
        if segment.output:
            print(f"  Output: {segment.output[:100]}...")
    
    # Get statistics
    stats = segmenter.get_segment_statistics(segments)
    print(f"\n--- Segmentation Statistics ---")
    print(f"Total segments: {stats['total_segments']}")
    print(f"Segment types: {stats['segment_types']}")
    print(f"Average context length: {stats['avg_context_length']:.1f}")
    print(f"Average output length: {stats['avg_output_length']:.1f}")


def demo_format_comparison():
    """Compare different output formats for the same content."""
    print("\n\n=== Format Comparison Demo ===")
    
    content = "Q: What is Python? A: Python is a high-level programming language."
    document = create_sample_document(content, "format_demo")
    segmenter = ContentSegmenter()
    
    segments = segmenter.segment_document(document, strategy="qa")
    
    if segments:
        segment = segments[0]
        
        print("Same content in different formats:")
        
        print("\n1. JSONL Format:")
        jsonl = segmenter.export_segments([segment], OutputFormat.JSONL)
        print(json.dumps(jsonl[0], indent=2))
        
        print("\n2. Alpaca Format:")
        alpaca = segmenter.export_segments([segment], OutputFormat.ALPACA)
        print(json.dumps(alpaca[0], indent=2))
        
        print("\n3. ChatML Format:")
        chatml = segmenter.export_segments([segment], OutputFormat.CHATML)
        print(json.dumps(chatml[0], indent=2))
        
        print("\n4. Plain Text Format:")
        plain = segmenter.export_segments([segment], OutputFormat.PLAIN_TEXT)
        print(plain[0]["text"])


def main():
    """Run all demos."""
    print("Content Segmentation for LLM Training Data")
    print("=" * 50)
    
    try:
        demo_qa_segmentation()
        demo_instruction_segmentation()
        demo_conversation_segmentation()
        demo_auto_segmentation()
        demo_format_comparison()
        
        print("\n\n=== Demo Complete ===")
        print("The ContentSegmenter successfully:")
        print("✓ Segmented Q&A content into instruction-context-output blocks")
        print("✓ Handled instruction-based content with task definitions")
        print("✓ Processed conversational content for chat model training")
        print("✓ Auto-detected content types and applied appropriate strategies")
        print("✓ Exported to multiple formats: JSONL, Alpaca, ChatML, Plain Text")
        print("✓ Provided detailed statistics about segmented content")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()