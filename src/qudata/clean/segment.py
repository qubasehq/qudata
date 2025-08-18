"""
Sentence segmentation and text normalization module.

This module provides comprehensive sentence segmentation capabilities including:
- Sentence boundary detection
- Text normalization and cleanup
- Paragraph and section segmentation
- Language-aware segmentation
"""

import re
import nltk
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import unicodedata


@dataclass
class SegmentationResult:
    """Result of text segmentation operation."""
    sentences: List[str]
    paragraphs: List[List[str]]
    word_count: int
    sentence_count: int
    average_sentence_length: float
    language: str
    
    @property
    def total_characters(self) -> int:
        """Total character count across all sentences."""
        return sum(len(sentence) for sentence in self.sentences)


class SentenceSegmenter:
    """
    Comprehensive sentence segmenter for text normalization.
    
    This class provides methods to segment text into sentences and paragraphs
    while performing normalization and cleanup operations.
    """
    
    def __init__(self, 
                 language: str = 'english',
                 min_sentence_length: int = 10,
                 max_sentence_length: int = 1000,
                 normalize_whitespace: bool = True,
                 remove_empty_sentences: bool = True):
        """
        Initialize SentenceSegmenter with configuration options.
        
        Args:
            language: Language for sentence tokenization
            min_sentence_length: Minimum sentence length to keep
            max_sentence_length: Maximum sentence length to keep
            normalize_whitespace: Whether to normalize whitespace
            remove_empty_sentences: Whether to remove empty sentences
        """
        self.language = language
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length
        self.normalize_whitespace = normalize_whitespace
        self.remove_empty_sentences = remove_empty_sentences
        
        # Download required NLTK data
        self._ensure_nltk_data()
        
        # Compile regex patterns for text cleaning
        self._whitespace_pattern = re.compile(r'\s+')
        self._sentence_end_pattern = re.compile(r'[.!?]+\s*')
        self._bullet_pattern = re.compile(r'^[\s]*[•\-\*\+]\s+')
        self._number_pattern = re.compile(r'^\d+[\.\)]\s+')
    
    def _ensure_nltk_data(self):
        """Ensure required NLTK data is downloaded."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
    
    def segment_text(self, text: str) -> SegmentationResult:
        """
        Segment text into sentences with normalization.
        
        Args:
            text: Input text to segment
            
        Returns:
            SegmentationResult with segmented and normalized text
        """
        if not text or not text.strip():
            return SegmentationResult(
                sentences=[],
                paragraphs=[],
                word_count=0,
                sentence_count=0,
                average_sentence_length=0.0,
                language=self.language
            )
        
        # Normalize text
        normalized_text = self._normalize_text(text)
        
        # Split into paragraphs first
        paragraphs = self._split_paragraphs(normalized_text)
        
        # Segment each paragraph into sentences
        all_sentences = []
        paragraph_sentences = []
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
                
            sentences = self._segment_paragraph(paragraph)
            if sentences:
                all_sentences.extend(sentences)
                paragraph_sentences.append(sentences)
        
        # Filter sentences by length
        filtered_sentences = self._filter_sentences(all_sentences)
        
        # Calculate statistics
        word_count = sum(len(word_tokenize(sentence)) for sentence in filtered_sentences)
        sentence_count = len(filtered_sentences)
        avg_length = sum(len(sentence) for sentence in filtered_sentences) / max(sentence_count, 1)
        
        return SegmentationResult(
            sentences=filtered_sentences,
            paragraphs=paragraph_sentences,
            word_count=word_count,
            sentence_count=sentence_count,
            average_sentence_length=avg_length,
            language=self.language
        )
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text by cleaning whitespace and characters."""
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Normalize whitespace if enabled
        if self.normalize_whitespace:
            text = self._whitespace_pattern.sub(' ', text)
        
        # Remove excessive line breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Clean up common formatting issues
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)  # Ensure space after sentence end
        text = re.sub(r'([a-z])([A-Z])', r'\1. \2', text)    # Add period between sentences
        
        return text.strip()
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double line breaks
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Clean and filter paragraphs
        cleaned_paragraphs = []
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if paragraph:
                cleaned_paragraphs.append(paragraph)
        
        return cleaned_paragraphs
    
    def _segment_paragraph(self, paragraph: str) -> List[str]:
        """Segment a paragraph into sentences."""
        try:
            # Use NLTK sentence tokenizer
            sentences = sent_tokenize(paragraph, language=self.language)
        except Exception:
            # Fallback to simple sentence splitting
            sentences = self._simple_sentence_split(paragraph)
        
        # Clean and normalize sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Remove bullet points and numbering
                sentence = self._bullet_pattern.sub('', sentence)
                sentence = self._number_pattern.sub('', sentence)
                sentence = sentence.strip()
                
                if sentence:
                    cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _simple_sentence_split(self, text: str) -> List[str]:
        """Simple fallback sentence splitting."""
        # Split on sentence endings
        sentences = self._sentence_end_pattern.split(text)
        
        # Clean up sentences
        cleaned = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and not sentence.endswith(('.', '!', '?')):
                sentence += '.'
            if sentence:
                cleaned.append(sentence)
        
        return cleaned
    
    def _filter_sentences(self, sentences: List[str]) -> List[str]:
        """Filter sentences by length and quality."""
        filtered = []
        
        for sentence in sentences:
            # Skip empty sentences if configured
            if self.remove_empty_sentences and not sentence.strip():
                continue
            
            # Check length constraints
            if len(sentence) < self.min_sentence_length:
                continue
            if len(sentence) > self.max_sentence_length:
                continue
            
            # Skip sentences that are mostly punctuation or numbers
            word_chars = sum(1 for c in sentence if c.isalnum())
            if word_chars < len(sentence) * 0.3:  # Less than 30% word characters
                continue
            
            filtered.append(sentence)
        
        return filtered
    
    def segment_by_length(self, text: str, target_length: int = 512) -> List[str]:
        """
        Segment text into chunks of approximately target length.
        
        Args:
            text: Input text to segment
            target_length: Target length for each segment
            
        Returns:
            List of text segments
        """
        # First segment into sentences
        result = self.segment_text(text)
        sentences = result.sentences
        
        if not sentences:
            return []
        
        # Group sentences into chunks of target length
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed target, start new chunk
            if current_length + sentence_length > target_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk if not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def batch_segment(self, texts: List[str]) -> List[SegmentationResult]:
        """
        Segment multiple texts in batch.
        
        Args:
            texts: List of text strings to segment
            
        Returns:
            List of SegmentationResult objects
        """
        return [self.segment_text(text) for text in texts]
    
    def get_segmentation_stats(self, results: List[SegmentationResult]) -> Dict[str, any]:
        """
        Get statistics from multiple segmentation results.
        
        Args:
            results: List of SegmentationResult objects
            
        Returns:
            Dictionary with segmentation statistics
        """
        if not results:
            return {}
        
        total_sentences = sum(r.sentence_count for r in results)
        total_words = sum(r.word_count for r in results)
        total_chars = sum(r.total_characters for r in results)
        
        return {
            'total_documents': len(results),
            'total_sentences': total_sentences,
            'total_words': total_words,
            'total_characters': total_chars,
            'average_sentences_per_document': total_sentences / len(results),
            'average_words_per_document': total_words / len(results),
            'average_sentence_length': total_chars / max(total_sentences, 1),
            'average_words_per_sentence': total_words / max(total_sentences, 1),
        }


def segment_text_simple(text: str, language: str = 'english') -> List[str]:
    """
    Convenience function to segment text into sentences with default settings.
    
    Args:
        text: Text to segment
        language: Language for tokenization
        
    Returns:
        List of sentences
    """
    segmenter = SentenceSegmenter(language=language)
    result = segmenter.segment_text(text)
    return result.sentences
