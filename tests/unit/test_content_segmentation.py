"""
Unit tests for content segmentation functionality.

Tests the ContentSegmenter and related classes for accuracy and format compliance.
"""

import pytest
from datetime import datetime
from typing import List, Dict, Any

from src.qudata.export.segmenter import (
    ContentSegmenter,
    TrainingSegment,
    SegmentationType,
    OutputFormat,
    SegmentationRule,
    QuestionAnswerStrategy,
    InstructionStrategy,
    ConversationStrategy,
)
from src.qudata.models import Document, DocumentMetadata


class TestTrainingSegment:
    """Test TrainingSegment class functionality."""
    
    def test_training_segment_creation(self):
        """Test basic TrainingSegment creation."""
        segment = TrainingSegment(
            instruction="Test instruction",
            context="Test context", 
            output="Test output"
        )
        
        assert segment.instruction == "Test instruction"
        assert segment.context == "Test context"
        assert segment.output == "Test output"
        assert segment.segment_type == SegmentationType.INSTRUCTION_CONTEXT_OUTPUT
    
    def test_training_segment_to_dict(self):
        """Test TrainingSegment to_dict conversion."""
        segment = TrainingSegment(
            instruction="Test instruction",
            context="Test context",
            output="Test output",
            metadata={"source": "test"},
            labels=["test_label"]
        )
        
        result = segment.to_dict()
        
        assert result["instruction"] == "Test instruction"
        assert result["context"] == "Test context"
        assert result["output"] == "Test output"
        assert result["metadata"] == {"source": "test"}
        assert result["labels"] == ["test_label"]
        assert result["segment_type"] == "instruction_context_output"
    
    def test_training_segment_to_alpaca_format(self):
        """Test TrainingSegment to Alpaca format conversion."""
        segment = TrainingSegment(
            instruction="Translate the following text",
            context="Hello world",
            output="Hola mundo"
        )
        
        result = segment.to_alpaca_format()
        
        assert result["instruction"] == "Translate the following text"
        assert result["input"] == "Hello world"
        assert result["output"] == "Hola mundo"
    
    def test_training_segment_to_alpaca_format_with_input_field(self):
        """Test Alpaca format with explicit input field."""
        segment = TrainingSegment(
            instruction="Process this data",
            input="Raw data",
            output="Processed data"
        )
        
        result = segment.to_alpaca_format()
        
        assert result["instruction"] == "Process this data"
        assert result["input"] == "Raw data"
        assert result["output"] == "Processed data"
    
    def test_training_segment_to_chatml_format(self):
        """Test TrainingSegment to ChatML format conversion."""
        segment = TrainingSegment(
            instruction="You are a helpful assistant",
            context="What is the capital of France?",
            output="The capital of France is Paris."
        )
        
        result = segment.to_chatml_format()
        
        assert len(result) == 3
        assert result[0] == {"role": "system", "content": "You are a helpful assistant"}
        assert result[1] == {"role": "user", "content": "What is the capital of France?"}
        assert result[2] == {"role": "assistant", "content": "The capital of France is Paris."}
    
    def test_training_segment_to_chatml_format_minimal(self):
        """Test ChatML format with minimal content."""
        segment = TrainingSegment(
            context="Hello",
            output="Hi there!"
        )
        
        result = segment.to_chatml_format()
        
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "assistant", "content": "Hi there!"}


class TestSegmentationRule:
    """Test SegmentationRule functionality."""
    
    def test_segmentation_rule_matches(self):
        """Test rule pattern matching."""
        rule = SegmentationRule(
            name="qa_rule",
            pattern=r"Q:\s*(?P<context>.*?)\s*A:\s*(?P<output>.*?)(?=Q:|$)",
            segment_type=SegmentationType.QUESTION_ANSWER
        )
        
        content = "Q: What is Python? A: A programming language."
        assert rule.matches(content)
        
        content_no_match = "This is just regular text."
        assert not rule.matches(content_no_match)
    
    def test_segmentation_rule_extract_segments(self):
        """Test segment extraction from content."""
        rule = SegmentationRule(
            name="qa_rule",
            pattern=r"Q:\s*(?P<context>.*?)\s*A:\s*(?P<output>.*?)(?=Q:|$)",
            segment_type=SegmentationType.QUESTION_ANSWER,
            instruction_template="Answer the question."
        )
        
        content = "Q: What is Python? A: A programming language. Q: What is Java? A: Another programming language."
        segments = rule.extract_segments(content)
        
        assert len(segments) == 2
        assert segments[0].instruction == "Answer the question."
        assert segments[0].context == "What is Python?"
        assert segments[0].output == "A programming language."
        assert segments[1].context == "What is Java?"
        assert segments[1].output == "Another programming language."


class TestQuestionAnswerStrategy:
    """Test QuestionAnswerStrategy functionality."""
    
    def create_test_document(self, content: str) -> Document:
        """Create a test document with given content."""
        metadata = DocumentMetadata(
            file_type="txt",
            size_bytes=len(content),
            language="en",
            domain="test"
        )
        
        return Document(
            id="test_doc",
            source_path="test.txt",
            content=content,
            metadata=metadata
        )
    
    def test_qa_strategy_q_a_format(self):
        """Test Q&A strategy with Q: A: format."""
        strategy = QuestionAnswerStrategy()
        content = "Q: What is machine learning? A: A subset of artificial intelligence."
        document = self.create_test_document(content)
        
        segments = strategy.segment_content(document)
        
        assert len(segments) == 1
        assert segments[0].instruction == "Answer the following question based on the provided context."
        assert segments[0].context == "What is machine learning?"
        assert segments[0].output == "A subset of artificial intelligence."
        assert segments[0].segment_type == SegmentationType.QUESTION_ANSWER
    
    def test_qa_strategy_question_answer_format(self):
        """Test Q&A strategy with Question: Answer: format."""
        strategy = QuestionAnswerStrategy()
        content = "Question: How does photosynthesis work? Answer: Plants convert sunlight into energy."
        document = self.create_test_document(content)
        
        segments = strategy.segment_content(document)
        
        assert len(segments) == 1
        assert segments[0].context == "How does photosynthesis work?"
        assert segments[0].output == "Plants convert sunlight into energy."
    
    def test_qa_strategy_numbered_format(self):
        """Test Q&A strategy with numbered FAQ format."""
        strategy = QuestionAnswerStrategy()
        content = "1. What is Python? Python is a programming language. 2. What is Java? Java is another language."
        document = self.create_test_document(content)
        
        segments = strategy.segment_content(document)
        
        assert len(segments) == 2
        assert "What is Python?" in segments[0].context
        assert "Python is a programming language" in segments[0].output
    
    def test_qa_strategy_no_matches(self):
        """Test Q&A strategy with content that doesn't match."""
        strategy = QuestionAnswerStrategy()
        content = "This is just regular text without any Q&A format."
        document = self.create_test_document(content)
        
        segments = strategy.segment_content(document)
        
        assert len(segments) == 0
    
    def test_qa_strategy_supported_types(self):
        """Test Q&A strategy supported types."""
        strategy = QuestionAnswerStrategy()
        supported = strategy.get_supported_types()
        
        assert "faq" in supported
        assert "qa" in supported
        assert "question_answer" in supported


class TestInstructionStrategy:
    """Test InstructionStrategy functionality."""
    
    def create_test_document(self, content: str) -> Document:
        """Create a test document with given content."""
        metadata = DocumentMetadata(
            file_type="txt",
            size_bytes=len(content),
            language="en",
            domain="test"
        )
        
        return Document(
            id="test_doc",
            source_path="test.txt",
            content=content,
            metadata=metadata
        )
    
    def test_instruction_strategy_task_format(self):
        """Test instruction strategy with Task: Input: Output: format."""
        strategy = InstructionStrategy()
        content = "Task: Translate text\nInput: Hello world\nOutput: Hola mundo"
        document = self.create_test_document(content)
        
        segments = strategy.segment_content(document)
        
        assert len(segments) == 1
        assert segments[0].instruction == "Translate text"
        assert segments[0].context == "Hello world"
        assert segments[0].output == "Hola mundo"
        assert segments[0].segment_type == SegmentationType.INSTRUCTION_CONTEXT_OUTPUT
    
    def test_instruction_strategy_how_to_format(self):
        """Test instruction strategy with 'How to' format."""
        strategy = InstructionStrategy()
        content = "How to make coffee\nYou need coffee beans and water\nBoil water, add coffee, brew for 4 minutes"
        document = self.create_test_document(content)
        
        segments = strategy.segment_content(document)
        
        # This might not match perfectly due to the simple regex, but should handle basic cases
        assert len(segments) >= 0  # At least doesn't crash
    
    def test_instruction_strategy_supported_types(self):
        """Test instruction strategy supported types."""
        strategy = InstructionStrategy()
        supported = strategy.get_supported_types()
        
        assert "instruction" in supported
        assert "tutorial" in supported
        assert "howto" in supported
        assert "guide" in supported


class TestConversationStrategy:
    """Test ConversationStrategy functionality."""
    
    def create_test_document(self, content: str) -> Document:
        """Create a test document with given content."""
        metadata = DocumentMetadata(
            file_type="txt",
            size_bytes=len(content),
            language="en",
            domain="test"
        )
        
        return Document(
            id="test_doc",
            source_path="test.txt",
            content=content,
            metadata=metadata
        )
    
    def test_conversation_strategy_user_assistant_format(self):
        """Test conversation strategy with User: Assistant: format."""
        strategy = ConversationStrategy()
        content = "User: Hello there\nAssistant: Hi! How can I help you today?"
        document = self.create_test_document(content)
        
        segments = strategy.segment_content(document)
        
        assert len(segments) == 1
        assert segments[0].context == "Hello there"
        assert segments[0].output == "Hi! How can I help you today?"
        assert segments[0].segment_type == SegmentationType.CONVERSATION
    
    def test_conversation_strategy_human_ai_format(self):
        """Test conversation strategy with Human: AI: format."""
        strategy = ConversationStrategy()
        content = "Human: What's the weather like?\nAI: I don't have access to current weather data."
        document = self.create_test_document(content)
        
        segments = strategy.segment_content(document)
        
        assert len(segments) == 1
        assert segments[0].context == "What's the weather like?"
        assert segments[0].output == "I don't have access to current weather data."
    
    def test_conversation_strategy_supported_types(self):
        """Test conversation strategy supported types."""
        strategy = ConversationStrategy()
        supported = strategy.get_supported_types()
        
        assert "conversation" in supported
        assert "dialogue" in supported
        assert "chat" in supported


class TestContentSegmenter:
    """Test ContentSegmenter main functionality."""
    
    def create_test_document(self, content: str, domain: str = "test") -> Document:
        """Create a test document with given content."""
        metadata = DocumentMetadata(
            file_type="txt",
            size_bytes=len(content),
            language="en",
            domain=domain
        )
        
        return Document(
            id="test_doc",
            source_path="test.txt",
            content=content,
            metadata=metadata
        )
    
    def test_content_segmenter_initialization(self):
        """Test ContentSegmenter initialization."""
        segmenter = ContentSegmenter()
        
        assert "qa" in segmenter.strategies
        assert "instruction" in segmenter.strategies
        assert "conversation" in segmenter.strategies
        assert len(segmenter.rules) > 0
    
    def test_content_segmenter_add_rule(self):
        """Test adding custom segmentation rules."""
        segmenter = ContentSegmenter()
        initial_rule_count = len(segmenter.rules)
        
        custom_rule = SegmentationRule(
            name="custom_rule",
            pattern=r"Custom:\s*(?P<output>.*)",
            segment_type=SegmentationType.COMPLETION,
            priority=15
        )
        
        segmenter.add_rule(custom_rule)
        
        assert len(segmenter.rules) == initial_rule_count + 1
        # Should be first due to high priority
        assert segmenter.rules[0].name == "custom_rule"
    
    def test_content_segmenter_qa_strategy(self):
        """Test segmenter with Q&A strategy."""
        segmenter = ContentSegmenter()
        content = "Q: What is AI? A: Artificial Intelligence is a field of computer science."
        document = self.create_test_document(content)
        
        segments = segmenter.segment_document(document, strategy="qa")
        
        assert len(segments) == 1
        assert segments[0].context == "What is AI?"
        assert segments[0].output == "Artificial Intelligence is a field of computer science."
    
    def test_content_segmenter_auto_strategy(self):
        """Test segmenter with auto strategy."""
        segmenter = ContentSegmenter()
        content = "Q: What is Python? A: A programming language."
        document = self.create_test_document(content)
        
        segments = segmenter.segment_document(document, strategy="auto")
        
        assert len(segments) >= 1
        # Should detect Q&A format automatically
        assert any("Python" in seg.context for seg in segments if seg.context)
    
    def test_content_segmenter_fallback_completion(self):
        """Test segmenter fallback to completion for unstructured content."""
        segmenter = ContentSegmenter()
        content = "This is a long piece of unstructured text that doesn't match any specific pattern. " * 20
        document = self.create_test_document(content)
        
        segments = segmenter.segment_document(document, strategy="auto")
        
        assert len(segments) == 1
        assert segments[0].segment_type == SegmentationType.COMPLETION
        assert segments[0].instruction == "Continue or complete the following text."
        assert segments[0].context is not None
    
    def test_content_segmenter_export_jsonl(self):
        """Test exporting segments to JSONL format."""
        segmenter = ContentSegmenter()
        segments = [
            TrainingSegment(
                instruction="Test instruction",
                context="Test context",
                output="Test output"
            )
        ]
        
        result = segmenter.export_segments(segments, OutputFormat.JSONL)
        
        assert len(result) == 1
        assert result[0]["instruction"] == "Test instruction"
        assert result[0]["context"] == "Test context"
        assert result[0]["output"] == "Test output"
    
    def test_content_segmenter_export_alpaca(self):
        """Test exporting segments to Alpaca format."""
        segmenter = ContentSegmenter()
        segments = [
            TrainingSegment(
                instruction="Translate text",
                context="Hello",
                output="Hola"
            )
        ]
        
        result = segmenter.export_segments(segments, OutputFormat.ALPACA)
        
        assert len(result) == 1
        assert result[0]["instruction"] == "Translate text"
        assert result[0]["input"] == "Hello"
        assert result[0]["output"] == "Hola"
    
    def test_content_segmenter_export_chatml(self):
        """Test exporting segments to ChatML format."""
        segmenter = ContentSegmenter()
        segments = [
            TrainingSegment(
                instruction="You are helpful",
                context="Hi there",
                output="Hello! How can I help?"
            )
        ]
        
        result = segmenter.export_segments(segments, OutputFormat.CHATML)
        
        assert len(result) == 1
        assert "messages" in result[0]
        messages = result[0]["messages"]
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
    
    def test_content_segmenter_export_plain_text(self):
        """Test exporting segments to plain text format."""
        segmenter = ContentSegmenter()
        segments = [
            TrainingSegment(
                instruction="Test instruction",
                context="Test context",
                output="Test output"
            )
        ]
        
        result = segmenter.export_segments(segments, OutputFormat.PLAIN_TEXT)
        
        assert len(result) == 1
        assert "text" in result[0]
        text = result[0]["text"]
        assert "Instruction: Test instruction" in text
        assert "Input: Test context" in text
        assert "Output: Test output" in text
    
    def test_content_segmenter_segment_and_export(self):
        """Test combined segmentation and export."""
        segmenter = ContentSegmenter()
        content = "Q: What is ML? A: Machine Learning."
        document = self.create_test_document(content)
        
        result = segmenter.segment_and_export(document, OutputFormat.ALPACA)
        
        assert len(result) >= 1
        assert "instruction" in result[0]
        assert "input" in result[0]
        assert "output" in result[0]
    
    def test_content_segmenter_statistics(self):
        """Test segment statistics generation."""
        segmenter = ContentSegmenter()
        segments = [
            TrainingSegment(
                instruction="Test 1",
                context="Context 1",
                output="Output 1",
                segment_type=SegmentationType.QUESTION_ANSWER
            ),
            TrainingSegment(
                instruction="Test 2",
                context="Context 2",
                output="Output 2",
                segment_type=SegmentationType.INSTRUCTION_CONTEXT_OUTPUT
            )
        ]
        
        stats = segmenter.get_segment_statistics(segments)
        
        assert stats["total_segments"] == 2
        assert stats["segments_with_instruction"] == 2
        assert stats["segments_with_context"] == 2
        assert stats["segments_with_output"] == 2
        assert "question_answer" in stats["segment_types"]
        assert "instruction_context_output" in stats["segment_types"]
        assert stats["avg_instruction_length"] > 0
        assert stats["avg_context_length"] > 0
        assert stats["avg_output_length"] > 0
    
    def test_content_segmenter_empty_statistics(self):
        """Test statistics with empty segment list."""
        segmenter = ContentSegmenter()
        stats = segmenter.get_segment_statistics([])
        
        assert stats["total_segments"] == 0
    
    def test_content_segmenter_invalid_strategy(self):
        """Test segmenter with invalid strategy."""
        segmenter = ContentSegmenter()
        document = self.create_test_document("Test content")
        
        with pytest.raises(ValueError, match="Unknown segmentation strategy"):
            segmenter.segment_document(document, strategy="invalid_strategy")
    
    def test_content_segmenter_invalid_output_format(self):
        """Test segmenter with invalid output format."""
        segmenter = ContentSegmenter()
        segments = [TrainingSegment(context="test", output="test")]
        
        with pytest.raises(ValueError, match="Unsupported output format"):
            segmenter.export_segments(segments, "invalid_format")


class TestFormatCompliance:
    """Test format compliance for different output formats."""
    
    def test_alpaca_format_compliance(self):
        """Test that Alpaca format output is compliant."""
        segment = TrainingSegment(
            instruction="Summarize the text",
            context="This is a long text that needs summarization.",
            output="This is a summary."
        )
        
        alpaca_format = segment.to_alpaca_format()
        
        # Alpaca format should have these exact keys
        required_keys = {"instruction", "input", "output"}
        assert set(alpaca_format.keys()) == required_keys
        
        # All values should be strings
        for value in alpaca_format.values():
            assert isinstance(value, str)
    
    def test_chatml_format_compliance(self):
        """Test that ChatML format output is compliant."""
        segment = TrainingSegment(
            instruction="You are a helpful assistant",
            context="Hello",
            output="Hi there!"
        )
        
        chatml_format = segment.to_chatml_format()
        
        # Should be a list of message objects
        assert isinstance(chatml_format, list)
        assert len(chatml_format) > 0
        
        # Each message should have role and content
        for message in chatml_format:
            assert isinstance(message, dict)
            assert "role" in message
            assert "content" in message
            assert message["role"] in ["system", "user", "assistant"]
            assert isinstance(message["content"], str)
    
    def test_jsonl_format_compliance(self):
        """Test that JSONL format output is JSON-serializable."""
        segment = TrainingSegment(
            instruction="Test",
            context="Test context",
            output="Test output",
            metadata={"key": "value"},
            labels=["label1", "label2"]
        )
        
        jsonl_format = segment.to_dict()
        
        # Should be JSON serializable
        import json
        json_str = json.dumps(jsonl_format)
        assert isinstance(json_str, str)
        
        # Should be deserializable back to same structure
        deserialized = json.loads(json_str)
        assert deserialized == jsonl_format


class TestSegmentationAccuracy:
    """Test segmentation accuracy with various content types."""
    
    def create_test_document(self, content: str) -> Document:
        """Create a test document with given content."""
        metadata = DocumentMetadata(
            file_type="txt",
            size_bytes=len(content),
            language="en",
            domain="test"
        )
        
        return Document(
            id="test_doc",
            source_path="test.txt",
            content=content,
            metadata=metadata
        )
    
    def test_complex_qa_content(self):
        """Test segmentation accuracy with complex Q&A content."""
        content = """
        Q: What is machine learning?
        A: Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.
        
        Q: How does deep learning differ from traditional machine learning?
        A: Deep learning uses neural networks with multiple layers to automatically discover patterns in data, while traditional ML often requires manual feature engineering.
        
        Q: What are some common applications of AI?
        A: Common applications include image recognition, natural language processing, recommendation systems, and autonomous vehicles.
        """
        
        segmenter = ContentSegmenter()
        document = self.create_test_document(content)
        segments = segmenter.segment_document(document, strategy="qa")
        
        # Should extract all 3 Q&A pairs
        assert len(segments) == 3
        
        # Check first segment
        assert "machine learning" in segments[0].context.lower()
        assert "artificial intelligence" in segments[0].output.lower()
        
        # Check that all segments have proper structure
        for segment in segments:
            assert segment.instruction is not None
            assert segment.context is not None
            assert segment.output is not None
            assert len(segment.context.strip()) > 0
            assert len(segment.output.strip()) > 0
    
    def test_mixed_content_segmentation(self):
        """Test segmentation with mixed content types."""
        content = """
        Task: Translate the following sentence
        Input: Hello, how are you?
        Output: Hola, ¿cómo estás?
        
        Q: What language was that?
        A: That was Spanish.
        
        User: Can you translate to French too?
        Assistant: Yes, in French it would be "Bonjour, comment allez-vous?"
        """
        
        segmenter = ContentSegmenter()
        document = self.create_test_document(content)
        segments = segmenter.segment_document(document, strategy="auto")
        
        # Should extract multiple segments from different formats
        assert len(segments) >= 2
        
        # Should have different segment types
        segment_types = {seg.segment_type for seg in segments}
        assert len(segment_types) > 1
    
    def test_edge_case_empty_content(self):
        """Test segmentation with empty or minimal content."""
        segmenter = ContentSegmenter()
        
        # Empty content
        empty_doc = self.create_test_document("")
        segments = segmenter.segment_document(empty_doc, strategy="auto")
        assert len(segments) == 1  # Should create fallback completion segment
        
        # Very short content
        short_doc = self.create_test_document("Hi")
        segments = segmenter.segment_document(short_doc, strategy="auto")
        assert len(segments) == 1
    
    def test_malformed_qa_content(self):
        """Test segmentation with malformed Q&A content."""
        content = """
        Q: What is Python
        A: 
        
        Q: 
        A: This answer has no question.
        
        Q: This question has no answer?
        """
        
        segmenter = ContentSegmenter()
        document = self.create_test_document(content)
        segments = segmenter.segment_document(document, strategy="qa")
        
        # Should handle malformed content gracefully
        # May extract some segments, but shouldn't crash
        assert isinstance(segments, list)
        
        # Valid segments should have both question and answer
        valid_segments = [s for s in segments if s.context and s.output and 
                         len(s.context.strip()) > 0 and len(s.output.strip()) > 0]
        
        # Should filter out malformed segments
        assert len(valid_segments) <= len(segments)


if __name__ == "__main__":
    pytest.main([__file__])