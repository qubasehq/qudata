"""
Content segmentation for LLM training formats.

This module provides functionality to segment content into structured training formats
like Instruction-Context-Output blocks and export to various LLM training formats
including ChatML, JSONL, and Alpaca.
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from ..models import Document, DocumentMetadata


class SegmentationType(Enum):
    """Types of content segmentation."""
    INSTRUCTION_CONTEXT_OUTPUT = "instruction_context_output"
    QUESTION_ANSWER = "question_answer"
    CONVERSATION = "conversation"
    COMPLETION = "completion"


class OutputFormat(Enum):
    """Supported output formats for training data."""
    JSONL = "jsonl"
    CHATML = "chatml"
    ALPACA = "alpaca"
    PLAIN_TEXT = "plain_text"


@dataclass
class TrainingSegment:
    """Represents a segment of content structured for LLM training."""
    instruction: Optional[str] = None
    context: Optional[str] = None
    output: Optional[str] = None
    input: Optional[str] = None  # For Alpaca format compatibility
    response: Optional[str] = None  # Alternative to output
    metadata: Dict[str, Any] = field(default_factory=dict)
    labels: List[str] = field(default_factory=list)
    segment_type: SegmentationType = SegmentationType.INSTRUCTION_CONTEXT_OUTPUT
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert segment to dictionary."""
        result = {}
        
        # Add non-None fields
        if self.instruction is not None:
            result["instruction"] = self.instruction
        if self.context is not None:
            result["context"] = self.context
        if self.output is not None:
            result["output"] = self.output
        if self.input is not None:
            result["input"] = self.input
        if self.response is not None:
            result["response"] = self.response
        
        # Add metadata and labels if present
        if self.metadata:
            result["metadata"] = self.metadata
        if self.labels:
            result["labels"] = self.labels
        
        result["segment_type"] = self.segment_type.value
        
        return result
    
    def to_alpaca_format(self) -> Dict[str, str]:
        """Convert to Alpaca format."""
        result = {}
        
        if self.instruction:
            result["instruction"] = self.instruction
        
        # Use input field or context for Alpaca input
        if self.input:
            result["input"] = self.input
        elif self.context:
            result["input"] = self.context
        else:
            result["input"] = ""
        
        # Use output or response for Alpaca output
        if self.output:
            result["output"] = self.output
        elif self.response:
            result["output"] = self.response
        else:
            result["output"] = ""
        
        return result
    
    def to_chatml_format(self) -> List[Dict[str, str]]:
        """Convert to ChatML format."""
        messages = []
        
        if self.instruction:
            messages.append({"role": "system", "content": self.instruction})
        
        if self.context or self.input:
            user_content = self.context or self.input or ""
            messages.append({"role": "user", "content": user_content})
        
        if self.output or self.response:
            assistant_content = self.output or self.response or ""
            messages.append({"role": "assistant", "content": assistant_content})
        
        return messages


@dataclass
class SegmentationRule:
    """Rule for segmenting content based on patterns."""
    name: str
    pattern: str
    segment_type: SegmentationType
    instruction_template: Optional[str] = None
    context_extractor: Optional[str] = None
    output_extractor: Optional[str] = None
    priority: int = 0  # Higher priority rules are applied first
    
    def matches(self, content: str) -> bool:
        """Check if this rule matches the content."""
        return bool(re.search(self.pattern, content, re.IGNORECASE | re.MULTILINE))
    
    def extract_segments(self, content: str) -> List[TrainingSegment]:
        """Extract segments from content using this rule."""
        matches = list(re.finditer(self.pattern, content, re.IGNORECASE | re.MULTILINE))
        segments = []
        
        for match in matches:
            segment = TrainingSegment(segment_type=self.segment_type)
            
            # Extract instruction
            if self.instruction_template:
                segment.instruction = self.instruction_template.format(**match.groupdict())
            elif "instruction" in match.groupdict():
                segment.instruction = match.group("instruction").strip()
            
            # Extract context
            if self.context_extractor:
                segment.context = self._extract_with_pattern(content, self.context_extractor, match)
            elif "context" in match.groupdict():
                segment.context = match.group("context").strip()
            
            # Extract output
            if self.output_extractor:
                segment.output = self._extract_with_pattern(content, self.output_extractor, match)
            elif "output" in match.groupdict():
                segment.output = match.group("output").strip()
            
            segments.append(segment)
        
        return segments
    
    def _extract_with_pattern(self, content: str, pattern: str, match: re.Match) -> str:
        """Extract content using a pattern relative to a match."""
        # Simple implementation - can be extended for more complex extraction
        try:
            return pattern.format(**match.groupdict())
        except (KeyError, ValueError):
            return pattern


class BaseSegmentationStrategy(ABC):
    """Abstract base class for content segmentation strategies."""
    
    @abstractmethod
    def segment_content(self, document: Document) -> List[TrainingSegment]:
        """Segment document content into training segments."""
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[str]:
        """Get list of supported content types."""
        pass


class QuestionAnswerStrategy(BaseSegmentationStrategy):
    """Strategy for segmenting Q&A content."""
    
    def __init__(self):
        self.qa_patterns = [
            # Q: ... A: ... format
            r"Q:\s*(?P<question>.*?)\s*A:\s*(?P<answer>.*?)(?=Q:|$)",
            # Question: ... Answer: ... format
            r"Question:\s*(?P<question>.*?)\s*Answer:\s*(?P<answer>.*?)(?=Question:|$)",
            # FAQ format with numbers
            r"\d+\.\s*(?P<question>.*?\?)\s*(?P<answer>.*?)(?=\d+\.|$)",
        ]
    
    def segment_content(self, document: Document) -> List[TrainingSegment]:
        """Segment Q&A content."""
        segments = []
        content = document.content
        
        for pattern in self.qa_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            
            for match in matches:
                groups = match.groupdict()
                
                if "question" in groups and "answer" in groups:
                    segment = TrainingSegment(
                        instruction="Answer the following question based on the provided context.",
                        context=groups["question"].strip(),
                        output=groups["answer"].strip(),
                        segment_type=SegmentationType.QUESTION_ANSWER,
                        metadata={
                            "source_document": document.id,
                            "domain": document.metadata.domain,
                            "extraction_method": "qa_pattern"
                        }
                    )
                    segments.append(segment)
        
        return segments
    
    def get_supported_types(self) -> List[str]:
        """Get supported content types."""
        return ["faq", "qa", "question_answer"]


class InstructionStrategy(BaseSegmentationStrategy):
    """Strategy for segmenting instructional content."""
    
    def __init__(self):
        self.instruction_patterns = [
            # Task descriptions with Input/Output
            r"Task:\s*(?P<instruction>[^\n]+)\s*\n\s*Input:\s*(?P<context>[^\n]+)\s*\n\s*Output:\s*(?P<output>[^\n]+)",
            # "How to" instructions
            r"(?P<instruction>How to [^\n]+)\n(?P<context>[^\n]+)\n(?P<output>[^\n]+)",
            # Step-by-step instructions
            r"(?P<instruction>[^:]+:)\s*(?P<context>[^\n]+)\s*(?P<output>Steps?:[^\n]+)",
        ]
    
    def segment_content(self, document: Document) -> List[TrainingSegment]:
        """Segment instructional content."""
        segments = []
        content = document.content
        
        for pattern in self.instruction_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            
            for match in matches:
                groups = match.groupdict()
                
                segment = TrainingSegment(
                    instruction=groups.get("instruction", "").strip(),
                    context=groups.get("context", "").strip(),
                    output=groups.get("output", "").strip(),
                    segment_type=SegmentationType.INSTRUCTION_CONTEXT_OUTPUT,
                    metadata={
                        "source_document": document.id,
                        "domain": document.metadata.domain,
                        "extraction_method": "instruction_pattern"
                    }
                )
                
                # Only add if we have meaningful content
                if segment.instruction and (segment.context or segment.output):
                    segments.append(segment)
        
        return segments
    
    def get_supported_types(self) -> List[str]:
        """Get supported content types."""
        return ["instruction", "tutorial", "howto", "guide"]


class ConversationStrategy(BaseSegmentationStrategy):
    """Strategy for segmenting conversational content."""
    
    def __init__(self):
        self.conversation_patterns = [
            # User: ... Assistant: ... format (more specific)
            r"User:\s*(?P<user>[^\n]+)\s*\n\s*Assistant:\s*(?P<assistant>[^\n]+)",
            # Human: ... AI: ... format (more specific)
            r"Human:\s*(?P<user>[^\n]+)\s*\n\s*AI:\s*(?P<assistant>[^\n]+)",
        ]
    
    def segment_content(self, document: Document) -> List[TrainingSegment]:
        """Segment conversational content."""
        segments = []
        content = document.content
        
        for pattern in self.conversation_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            
            for match in matches:
                groups = match.groupdict()
                
                if "user" in groups and "assistant" in groups:
                    segment = TrainingSegment(
                        context=groups["user"].strip(),
                        output=groups["assistant"].strip(),
                        segment_type=SegmentationType.CONVERSATION,
                        metadata={
                            "source_document": document.id,
                            "domain": document.metadata.domain,
                            "extraction_method": "conversation_pattern"
                        }
                    )
                    segments.append(segment)
        
        return segments
    
    def get_supported_types(self) -> List[str]:
        """Get supported content types."""
        return ["conversation", "dialogue", "chat"]


class ContentSegmenter:
    """Main class for segmenting content into training formats."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize content segmenter.
        
        Args:
            config: Configuration for segmentation behavior
        """
        self.config = config or {}
        self.strategies: Dict[str, BaseSegmentationStrategy] = {
            "qa": QuestionAnswerStrategy(),
            "instruction": InstructionStrategy(),
            "conversation": ConversationStrategy(),
        }
        self.rules: List[SegmentationRule] = []
        self._load_default_rules()
    
    def _load_default_rules(self):
        """Load default segmentation rules."""
        # Q&A rules
        self.rules.append(SegmentationRule(
            name="qa_format",
            pattern=r"Q:\s*(?P<context>.*?)\s*A:\s*(?P<output>.*?)(?=Q:|$)",
            segment_type=SegmentationType.QUESTION_ANSWER,
            instruction_template="Answer the following question.",
            priority=10
        ))
        
        # Instruction rules
        self.rules.append(SegmentationRule(
            name="task_format",
            pattern=r"Task:\s*(?P<instruction>[^\n]+)\s*\n\s*Input:\s*(?P<context>[^\n]+)\s*\n\s*Output:\s*(?P<output>[^\n]+)",
            segment_type=SegmentationType.INSTRUCTION_CONTEXT_OUTPUT,
            priority=9
        ))
        
        # Conversation rules
        self.rules.append(SegmentationRule(
            name="user_assistant",
            pattern=r"User:\s*(?P<context>.*?)\s*Assistant:\s*(?P<output>.*?)(?=User:|$)",
            segment_type=SegmentationType.CONVERSATION,
            priority=8
        ))
        
        # Sort rules by priority (highest first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def add_rule(self, rule: SegmentationRule):
        """Add a custom segmentation rule."""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def segment_document(self, document: Document, strategy: str = "auto") -> List[TrainingSegment]:
        """
        Segment a document into training segments.
        
        Args:
            document: Document to segment
            strategy: Segmentation strategy to use ("auto", "qa", "instruction", "conversation")
            
        Returns:
            List of training segments
        """
        if strategy == "auto":
            return self._auto_segment(document)
        elif strategy in self.strategies:
            return self.strategies[strategy].segment_content(document)
        else:
            raise ValueError(f"Unknown segmentation strategy: {strategy}")
    
    def _auto_segment(self, document: Document) -> List[TrainingSegment]:
        """Automatically determine the best segmentation strategy."""
        segments = []
        
        # Try rule-based segmentation first
        for rule in self.rules:
            if rule.matches(document.content):
                rule_segments = rule.extract_segments(document.content)
                segments.extend(rule_segments)
        
        # If no rule-based segments found, try strategies
        if not segments:
            # Try each strategy and use the one that produces the most segments
            best_segments = []
            best_count = 0
            
            for strategy_name, strategy in self.strategies.items():
                try:
                    strategy_segments = strategy.segment_content(document)
                    if len(strategy_segments) > best_count:
                        best_segments = strategy_segments
                        best_count = len(strategy_segments)
                except Exception:
                    continue
            
            segments = best_segments
        
        # If still no segments, create a simple completion segment
        if not segments:
            segments = [self._create_completion_segment(document)]
        
        return segments
    
    def _create_completion_segment(self, document: Document) -> TrainingSegment:
        """Create a simple completion segment from document content."""
        # Split content into chunks if it's too long
        content = document.content.strip()
        
        # Simple heuristic: if content has clear structure, try to split
        if len(content) > 1000:
            # Try to find a natural break point
            sentences = content.split('. ')
            if len(sentences) > 2:
                mid_point = len(sentences) // 2
                context = '. '.join(sentences[:mid_point]) + '.'
                output = '. '.join(sentences[mid_point:])
            else:
                # Split at halfway point
                mid_point = len(content) // 2
                context = content[:mid_point]
                output = content[mid_point:]
        else:
            context = content
            output = ""
        
        return TrainingSegment(
            instruction="Continue or complete the following text.",
            context=context,
            output=output,
            segment_type=SegmentationType.COMPLETION,
            metadata={
                "source_document": document.id,
                "domain": document.metadata.domain,
                "extraction_method": "completion_fallback"
            }
        )
    
    def export_segments(self, segments: List[TrainingSegment], 
                       output_format: OutputFormat) -> List[Dict[str, Any]]:
        """
        Export segments to specified format.
        
        Args:
            segments: List of training segments to export
            output_format: Target output format
            
        Returns:
            List of formatted training examples
        """
        if output_format == OutputFormat.JSONL:
            return [segment.to_dict() for segment in segments]
        elif output_format == OutputFormat.ALPACA:
            return [segment.to_alpaca_format() for segment in segments]
        elif output_format == OutputFormat.CHATML:
            return [{"messages": segment.to_chatml_format()} for segment in segments]
        elif output_format == OutputFormat.PLAIN_TEXT:
            return [self._to_plain_text(segment) for segment in segments]
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _to_plain_text(self, segment: TrainingSegment) -> Dict[str, str]:
        """Convert segment to plain text format."""
        text_parts = []
        
        if segment.instruction:
            text_parts.append(f"Instruction: {segment.instruction}")
        if segment.context:
            text_parts.append(f"Input: {segment.context}")
        if segment.output:
            text_parts.append(f"Output: {segment.output}")
        
        return {"text": "\n\n".join(text_parts)}
    
    def segment_and_export(self, document: Document, output_format: OutputFormat,
                          strategy: str = "auto") -> List[Dict[str, Any]]:
        """
        Segment document and export in specified format.
        
        Args:
            document: Document to segment
            output_format: Target output format
            strategy: Segmentation strategy to use
            
        Returns:
            List of formatted training examples
        """
        segments = self.segment_document(document, strategy)
        return self.export_segments(segments, output_format)
    
    def get_segment_statistics(self, segments: List[TrainingSegment]) -> Dict[str, Any]:
        """Get statistics about segmented content."""
        if not segments:
            return {"total_segments": 0}
        
        stats = {
            "total_segments": len(segments),
            "segment_types": {},
            "avg_instruction_length": 0,
            "avg_context_length": 0,
            "avg_output_length": 0,
            "segments_with_instruction": 0,
            "segments_with_context": 0,
            "segments_with_output": 0,
        }
        
        instruction_lengths = []
        context_lengths = []
        output_lengths = []
        
        for segment in segments:
            # Count segment types
            seg_type = segment.segment_type.value
            stats["segment_types"][seg_type] = stats["segment_types"].get(seg_type, 0) + 1
            
            # Track lengths
            if segment.instruction:
                instruction_lengths.append(len(segment.instruction))
                stats["segments_with_instruction"] += 1
            if segment.context:
                context_lengths.append(len(segment.context))
                stats["segments_with_context"] += 1
            if segment.output:
                output_lengths.append(len(segment.output))
                stats["segments_with_output"] += 1
        
        # Calculate averages
        if instruction_lengths:
            stats["avg_instruction_length"] = sum(instruction_lengths) / len(instruction_lengths)
        if context_lengths:
            stats["avg_context_length"] = sum(context_lengths) / len(context_lengths)
        if output_lengths:
            stats["avg_output_length"] = sum(output_lengths) / len(output_lengths)
        
        return stats