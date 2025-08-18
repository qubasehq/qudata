"""
Export module for QuData.

This module provides functionality for exporting processed data in various formats
suitable for LLM training and analysis.
"""

from .segmenter import (
    ContentSegmenter,
    TrainingSegment,
    SegmentationType,
    OutputFormat,
    SegmentationRule,
    BaseSegmentationStrategy,
    QuestionAnswerStrategy,
    InstructionStrategy,
    ConversationStrategy,
)

from .formats import (
    FormatConverter,
    ChatMLFormatter,
    AlpacaFormatter,
    JSONLFormatter,
    ConversationTurn,
    InstructionExample,
    FormatConversionResult,
    OutputFormat as FormatType,
    convert_to_chatml,
    convert_to_alpaca,
    convert_to_jsonl
)

__all__ = [
    # Content segmentation
    "ContentSegmenter",
    "TrainingSegment", 
    "SegmentationType",
    "OutputFormat",
    "SegmentationRule",
    "BaseSegmentationStrategy",
    "QuestionAnswerStrategy",
    "InstructionStrategy", 
    "ConversationStrategy",
    
    # Format conversion
    "FormatConverter",
    "ChatMLFormatter",
    "AlpacaFormatter", 
    "JSONLFormatter",
    "ConversationTurn",
    "InstructionExample",
    "FormatConversionResult",
    "FormatType",
    "convert_to_chatml",
    "convert_to_alpaca",
    "convert_to_jsonl"
]