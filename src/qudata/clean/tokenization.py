"""
Tokenization preview module for LLM compatibility.

This module provides tokenization preview capabilities including:
- Token count estimation for various LLM models
- Character-to-token ratio analysis
- Batch processing for datasets
- Model-specific tokenization patterns
"""

import re
import math
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from collections import Counter

# Make tiktoken optional to avoid hard failure when not installed
try:
    import tiktoken  # type: ignore
    _HAS_TIKTOKEN = True
except Exception:
    tiktoken = None  # type: ignore
    _HAS_TIKTOKEN = False

# Simple fallback encoder if tiktoken is unavailable
class _FallbackEncoding:
    """Lightweight fallback tokenizer approximation when tiktoken is missing."""
    def encode(self, text: str):
        # Approximate tokens via word and punctuation segmentation
        tokens = re.findall(r"\b\w+\b|[^\w\s]", text, flags=re.UNICODE)
        if tokens:
            return tokens
        # Final fallback: approx 1 token per 4 chars
        return [""] * max(1, len(text) // 4)


@dataclass
class TokenizationResult:
    """Result of tokenization analysis."""
    text: str
    token_count: int
    character_count: int
    word_count: int
    char_to_token_ratio: float
    word_to_token_ratio: float
    model_name: str
    estimated_cost: Optional[float] = None
    
    @property
    def tokens_per_word(self) -> float:
        """Average tokens per word."""
        return self.token_count / max(self.word_count, 1)
    
    @property
    def chars_per_token(self) -> float:
        """Average characters per token."""
        return self.character_count / max(self.token_count, 1)


@dataclass
class BatchTokenizationStats:
    """Statistics for batch tokenization results."""
    total_documents: int
    total_tokens: int
    total_characters: int
    total_words: int
    average_tokens_per_document: float
    average_char_to_token_ratio: float
    token_distribution: Dict[str, int]  # Ranges like "0-100", "100-500", etc.
    estimated_total_cost: Optional[float] = None


class TokenizationPreview:
    """
    Tokenization preview for LLM compatibility analysis.
    
    This class provides methods to estimate token counts and analyze
    text compatibility with various LLM models.
    """
    
    # Model configurations with approximate token costs (per 1K tokens)
    MODEL_CONFIGS = {
        'gpt-4': {
            'encoding': 'cl100k_base',
            'cost_per_1k_tokens': 0.03,
            'max_context': 8192
        },
        'gpt-3.5-turbo': {
            'encoding': 'cl100k_base', 
            'cost_per_1k_tokens': 0.002,
            'max_context': 4096
        },
        'text-davinci-003': {
            'encoding': 'p50k_base',
            'cost_per_1k_tokens': 0.02,
            'max_context': 4097
        },
        'claude-2': {
            'encoding': 'cl100k_base',  # Approximation
            'cost_per_1k_tokens': 0.008,
            'max_context': 100000
        },
        'llama-2-7b': {
            'encoding': 'cl100k_base',  # Approximation
            'cost_per_1k_tokens': 0.0,  # Open source
            'max_context': 4096
        },
        'llama-2-13b': {
            'encoding': 'cl100k_base',  # Approximation
            'cost_per_1k_tokens': 0.0,  # Open source
            'max_context': 4096
        }
    }
    
    def __init__(self, model_name: str = 'gpt-3.5-turbo'):
        """
        Initialize TokenizationPreview for a specific model.
        
        Args:
            model_name: Name of the LLM model to analyze for
        """
        self.model_name = model_name
        self.model_config = self.MODEL_CONFIGS.get(model_name, self.MODEL_CONFIGS['gpt-3.5-turbo'])
        
        # Initialize tokenizer
        if _HAS_TIKTOKEN:
            try:
                self.encoding = tiktoken.get_encoding(self.model_config['encoding'])
            except Exception:
                # Fallback to default encoding
                self.encoding = tiktoken.get_encoding('cl100k_base')
        else:
            self.encoding = _FallbackEncoding()
        
        # Compile regex patterns for fallback tokenization
        self._word_pattern = re.compile(r'\b\w+\b')
        self._whitespace_pattern = re.compile(r'\s+')
    
    def analyze_text(self, text: str, include_cost: bool = True) -> TokenizationResult:
        """
        Analyze text tokenization for the configured model.
        
        Args:
            text: Input text to analyze
            include_cost: Whether to include cost estimation
            
        Returns:
            TokenizationResult with tokenization analysis
        """
        if not text:
            return TokenizationResult(
                text="",
                token_count=0,
                character_count=0,
                word_count=0,
                char_to_token_ratio=0.0,
                word_to_token_ratio=0.0,
                model_name=self.model_name,
                estimated_cost=0.0 if include_cost else None
            )
        
        # Count characters and words
        character_count = len(text)
        word_count = len(self._word_pattern.findall(text))
        
        # Count tokens
        try:
            tokens = self.encoding.encode(text)
            token_count = len(tokens)
        except Exception:
            # Fallback estimation: roughly 4 characters per token
            token_count = max(1, character_count // 4)
        
        # Calculate ratios
        char_to_token_ratio = character_count / max(token_count, 1)
        word_to_token_ratio = word_count / max(token_count, 1)
        
        # Estimate cost
        estimated_cost = None
        if include_cost and self.model_config['cost_per_1k_tokens'] > 0:
            estimated_cost = (token_count / 1000) * self.model_config['cost_per_1k_tokens']
        
        return TokenizationResult(
            text=text,
            token_count=token_count,
            character_count=character_count,
            word_count=word_count,
            char_to_token_ratio=char_to_token_ratio,
            word_to_token_ratio=word_to_token_ratio,
            model_name=self.model_name,
            estimated_cost=estimated_cost
        )
    
    def check_context_limit(self, text: str) -> Dict[str, any]:
        """
        Check if text fits within model's context limit.
        
        Args:
            text: Input text to check
            
        Returns:
            Dictionary with context limit analysis
        """
        result = self.analyze_text(text, include_cost=False)
        max_context = self.model_config['max_context']
        
        fits_in_context = result.token_count <= max_context
        utilization_ratio = result.token_count / max_context
        
        return {
            'token_count': result.token_count,
            'max_context': max_context,
            'fits_in_context': fits_in_context,
            'utilization_ratio': utilization_ratio,
            'tokens_over_limit': max(0, result.token_count - max_context),
            'suggested_chunks': math.ceil(result.token_count / max_context) if not fits_in_context else 1
        }
    
    def suggest_chunking(self, text: str, 
                        target_tokens: Optional[int] = None,
                        overlap_tokens: int = 50) -> List[str]:
        """
        Suggest text chunking to fit within context limits.
        
        Args:
            text: Input text to chunk
            target_tokens: Target tokens per chunk (defaults to 80% of max context)
            overlap_tokens: Number of tokens to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if target_tokens is None:
            target_tokens = int(self.model_config['max_context'] * 0.8)
        
        # Split text into sentences for better chunking
        sentences = re.split(r'[.!?]+\s+', text)
        if not sentences:
            return [text]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Estimate tokens for this sentence
            sentence_tokens = self.analyze_text(sentence, include_cost=False).token_count
            
            # If adding this sentence would exceed target, start new chunk
            if current_tokens + sentence_tokens > target_tokens and current_chunk:
                chunk_text = '. '.join(current_chunk) + '.'
                chunks.append(chunk_text)
                
                # Start new chunk with overlap if specified
                if overlap_tokens > 0 and len(current_chunk) > 1:
                    # Take last few sentences for overlap
                    overlap_sentences = []
                    overlap_token_count = 0
                    
                    for prev_sentence in reversed(current_chunk):
                        prev_tokens = self.analyze_text(prev_sentence, include_cost=False).token_count
                        if overlap_token_count + prev_tokens <= overlap_tokens:
                            overlap_sentences.insert(0, prev_sentence)
                            overlap_token_count += prev_tokens
                        else:
                            break
                    
                    current_chunk = overlap_sentences + [sentence]
                    current_tokens = overlap_token_count + sentence_tokens
                else:
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            chunks.append(chunk_text)
        
        return chunks
    
    def batch_analyze(self, texts: List[str], include_cost: bool = True) -> BatchTokenizationStats:
        """
        Analyze multiple texts in batch.
        
        Args:
            texts: List of texts to analyze
            include_cost: Whether to include cost estimation
            
        Returns:
            BatchTokenizationStats with aggregate statistics
        """
        if not texts:
            return BatchTokenizationStats(
                total_documents=0,
                total_tokens=0,
                total_characters=0,
                total_words=0,
                average_tokens_per_document=0.0,
                average_char_to_token_ratio=0.0,
                token_distribution={},
                estimated_total_cost=0.0 if include_cost else None
            )
        
        results = [self.analyze_text(text, include_cost) for text in texts]
        
        # Calculate totals
        total_tokens = sum(r.token_count for r in results)
        total_characters = sum(r.character_count for r in results)
        total_words = sum(r.word_count for r in results)
        total_cost = sum(r.estimated_cost or 0 for r in results) if include_cost else None
        
        # Calculate averages
        avg_tokens_per_doc = total_tokens / len(results)
        avg_char_to_token = sum(r.char_to_token_ratio for r in results) / len(results)
        
        # Calculate token distribution
        token_ranges = [
            ('0-100', 0, 100),
            ('100-500', 100, 500),
            ('500-1000', 500, 1000),
            ('1000-2000', 1000, 2000),
            ('2000-4000', 2000, 4000),
            ('4000+', 4000, float('inf'))
        ]
        
        distribution = {}
        for range_name, min_tokens, max_tokens in token_ranges:
            count = sum(1 for r in results 
                       if min_tokens <= r.token_count < max_tokens)
            distribution[range_name] = count
        
        return BatchTokenizationStats(
            total_documents=len(results),
            total_tokens=total_tokens,
            total_characters=total_characters,
            total_words=total_words,
            average_tokens_per_document=avg_tokens_per_doc,
            average_char_to_token_ratio=avg_char_to_token,
            token_distribution=distribution,
            estimated_total_cost=total_cost
        )
    
    def compare_models(self, text: str, models: List[str]) -> Dict[str, TokenizationResult]:
        """
        Compare tokenization across multiple models.
        
        Args:
            text: Input text to analyze
            models: List of model names to compare
            
        Returns:
            Dictionary mapping model names to TokenizationResult
        """
        results = {}
        original_model = self.model_name
        
        for model in models:
            if model in self.MODEL_CONFIGS:
                self.model_name = model
                self.model_config = self.MODEL_CONFIGS[model]
                if _HAS_TIKTOKEN:
                    try:
                        self.encoding = tiktoken.get_encoding(self.model_config['encoding'])
                    except Exception:
                        self.encoding = tiktoken.get_encoding('cl100k_base')
                else:
                    self.encoding = _FallbackEncoding()
                
                results[model] = self.analyze_text(text)
        
        # Restore original model
        self.model_name = original_model
        self.model_config = self.MODEL_CONFIGS[original_model]
        if _HAS_TIKTOKEN:
            try:
                self.encoding = tiktoken.get_encoding(self.model_config['encoding'])
            except Exception:
                self.encoding = tiktoken.get_encoding('cl100k_base')
        else:
            self.encoding = _FallbackEncoding()
        
        return results
    
    def estimate_training_tokens(self, texts: List[str], epochs: int = 1) -> Dict[str, any]:
        """
        Estimate total tokens needed for training.
        
        Args:
            texts: Training texts
            epochs: Number of training epochs
            
        Returns:
            Dictionary with training token estimates
        """
        stats = self.batch_analyze(texts, include_cost=True)
        
        tokens_per_epoch = stats.total_tokens
        total_training_tokens = tokens_per_epoch * epochs
        
        # Estimate training time (rough approximation)
        # Assume ~1000 tokens/second processing speed
        estimated_seconds = total_training_tokens / 1000
        estimated_hours = estimated_seconds / 3600
        
        return {
            'total_documents': stats.total_documents,
            'tokens_per_epoch': tokens_per_epoch,
            'total_training_tokens': total_training_tokens,
            'epochs': epochs,
            'estimated_training_hours': estimated_hours,
            'estimated_cost_per_epoch': stats.estimated_total_cost,
            'estimated_total_cost': (stats.estimated_total_cost or 0) * epochs,
            'token_distribution': stats.token_distribution
        }


def quick_token_count(text: str, model: str = 'gpt-3.5-turbo') -> int:
    """
    Quick token count estimation for text.
    
    Args:
        text: Text to count tokens for
        model: Model name for tokenization
        
    Returns:
        Estimated token count
    """
    preview = TokenizationPreview(model)
    result = preview.analyze_text(text, include_cost=False)
    return result.token_count


def check_context_fit(text: str, model: str = 'gpt-3.5-turbo') -> bool:
    """
    Check if text fits in model's context window.
    
    Args:
        text: Text to check
        model: Model name to check against
        
    Returns:
        True if text fits in context window
    """
    preview = TokenizationPreview(model)
    context_check = preview.check_context_limit(text)
    return context_check['fits_in_context']