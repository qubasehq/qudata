"""
Stopword removal module with configurable word lists.

This module provides comprehensive stopword removal capabilities including:
- Multi-language stopword support
- Custom stopword lists
- Configurable removal strategies
- Frequency-based filtering
"""

import re
from typing import List, Dict, Set, Optional, Union
from dataclasses import dataclass
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


@dataclass
class StopwordRemovalResult:
    """Result of stopword removal operation."""
    cleaned_text: str
    removed_words: List[str]
    word_count_before: int
    word_count_after: int
    removal_ratio: float
    
    @property
    def words_removed_count(self) -> int:
        """Number of words removed."""
        return len(self.removed_words)


class StopwordRemover:
    """
    Comprehensive stopword remover with configurable word lists.
    
    This class provides methods to remove stopwords from text using
    built-in language stopwords, custom lists, and frequency-based filtering.
    """
    
    def __init__(self, 
                 languages: Union[str, List[str]] = 'english',
                 custom_stopwords: Optional[Set[str]] = None,
                 min_word_length: int = 2,
                 preserve_case: bool = False,
                 remove_numbers: bool = False,
                 remove_punctuation: bool = True):
        """
        Initialize StopwordRemover with configuration options.
        
        Args:
            languages: Language(s) for built-in stopwords
            custom_stopwords: Additional custom stopwords to remove
            min_word_length: Minimum word length to preserve
            preserve_case: Whether to preserve original case
            remove_numbers: Whether to remove numeric tokens
            remove_punctuation: Whether to remove punctuation tokens
        """
        self.languages = [languages] if isinstance(languages, str) else languages
        self.custom_stopwords = custom_stopwords or set()
        self.min_word_length = min_word_length
        self.preserve_case = preserve_case
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        
        # Download required NLTK data
        self._ensure_nltk_data()
        
        # Build combined stopword set
        self.stopwords = self._build_stopword_set()
        
        # Compile regex patterns
        self._number_pattern = re.compile(r'^\d+$')
        self._punctuation_pattern = re.compile(r'^[^\w\s]+$')
        self._word_pattern = re.compile(r'\b\w+\b')
    
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
    
    def _build_stopword_set(self) -> Set[str]:
        """Build combined stopword set from languages and custom words."""
        combined_stopwords = set()
        
        # Add stopwords from specified languages
        for language in self.languages:
            try:
                lang_stopwords = set(stopwords.words(language))
                combined_stopwords.update(lang_stopwords)
                
                # Add case variations if not preserving case
                if not self.preserve_case:
                    combined_stopwords.update(word.lower() for word in lang_stopwords)
                    combined_stopwords.update(word.upper() for word in lang_stopwords)
                    combined_stopwords.update(word.capitalize() for word in lang_stopwords)
            except OSError:
                # Language not available, skip
                continue
        
        # Add custom stopwords
        combined_stopwords.update(self.custom_stopwords)
        
        return combined_stopwords
    
    def remove_stopwords(self, text: str) -> StopwordRemovalResult:
        """
        Remove stopwords from text.
        
        Args:
            text: Input text to process
            
        Returns:
            StopwordRemovalResult with cleaned text and metadata
        """
        if not text or not text.strip():
            return StopwordRemovalResult(
                cleaned_text="",
                removed_words=[],
                word_count_before=0,
                word_count_after=0,
                removal_ratio=0.0
            )
        
        # Tokenize text
        try:
            tokens = word_tokenize(text)
        except Exception:
            # Fallback to simple tokenization
            tokens = self._word_pattern.findall(text)
        
        original_word_count = len(tokens)
        removed_words = []
        kept_tokens = []
        
        for token in tokens:
            should_remove = False
            
            # Check if token is a stopword
            check_token = token if self.preserve_case else token.lower()
            if check_token in self.stopwords:
                should_remove = True
            
            # Check minimum word length
            elif len(token) < self.min_word_length:
                should_remove = True
            
            # Check if token is a number
            elif self.remove_numbers and self._number_pattern.match(token):
                should_remove = True
            
            # Check if token is punctuation
            elif self.remove_punctuation and self._punctuation_pattern.match(token):
                should_remove = True
            
            if should_remove:
                removed_words.append(token)
            else:
                kept_tokens.append(token)
        
        # Reconstruct text
        cleaned_text = ' '.join(kept_tokens)
        final_word_count = len(kept_tokens)
        
        removal_ratio = (original_word_count - final_word_count) / max(original_word_count, 1)
        
        return StopwordRemovalResult(
            cleaned_text=cleaned_text,
            removed_words=removed_words,
            word_count_before=original_word_count,
            word_count_after=final_word_count,
            removal_ratio=removal_ratio
        )
    
    def remove_by_frequency(self, text: str, 
                          min_frequency: int = 2,
                          max_frequency_ratio: float = 0.1) -> StopwordRemovalResult:
        """
        Remove words based on frequency thresholds.
        
        Args:
            text: Input text to process
            min_frequency: Minimum frequency to keep a word
            max_frequency_ratio: Maximum frequency ratio to keep a word
            
        Returns:
            StopwordRemovalResult with frequency-filtered text
        """
        if not text or not text.strip():
            return StopwordRemovalResult(
                cleaned_text="",
                removed_words=[],
                word_count_before=0,
                word_count_after=0,
                removal_ratio=0.0
            )
        
        # Tokenize and count words
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = self._word_pattern.findall(text)
        
        original_word_count = len(tokens)
        word_counts = Counter(token.lower() for token in tokens)
        total_words = sum(word_counts.values())
        
        # Determine words to remove based on frequency
        words_to_remove = set()
        for word, count in word_counts.items():
            frequency_ratio = count / total_words
            
            # Remove if too infrequent or too frequent
            if count < min_frequency or frequency_ratio > max_frequency_ratio:
                words_to_remove.add(word)
        
        # Filter tokens
        removed_words = []
        kept_tokens = []
        
        for token in tokens:
            check_token = token.lower()
            if check_token in words_to_remove:
                removed_words.append(token)
            else:
                kept_tokens.append(token)
        
        # Reconstruct text
        cleaned_text = ' '.join(kept_tokens)
        final_word_count = len(kept_tokens)
        
        removal_ratio = (original_word_count - final_word_count) / max(original_word_count, 1)
        
        return StopwordRemovalResult(
            cleaned_text=cleaned_text,
            removed_words=removed_words,
            word_count_before=original_word_count,
            word_count_after=final_word_count,
            removal_ratio=removal_ratio
        )
    
    def add_custom_stopwords(self, words: Union[str, List[str], Set[str]]):
        """
        Add custom stopwords to the removal list.
        
        Args:
            words: Word(s) to add as stopwords
        """
        if isinstance(words, str):
            words = [words]
        
        self.custom_stopwords.update(words)
        self.stopwords = self._build_stopword_set()
    
    def remove_custom_stopwords(self, words: Union[str, List[str], Set[str]]):
        """
        Remove words from custom stopwords list.
        
        Args:
            words: Word(s) to remove from stopwords
        """
        if isinstance(words, str):
            words = [words]
        
        for word in words:
            self.custom_stopwords.discard(word)
        
        self.stopwords = self._build_stopword_set()
    
    def get_stopwords(self) -> Set[str]:
        """Get current set of stopwords."""
        return self.stopwords.copy()
    
    def batch_remove(self, texts: List[str]) -> List[StopwordRemovalResult]:
        """
        Remove stopwords from multiple texts in batch.
        
        Args:
            texts: List of text strings to process
            
        Returns:
            List of StopwordRemovalResult objects
        """
        return [self.remove_stopwords(text) for text in texts]
    
    def get_removal_stats(self, results: List[StopwordRemovalResult]) -> Dict[str, any]:
        """
        Get statistics from multiple removal results.
        
        Args:
            results: List of StopwordRemovalResult objects
            
        Returns:
            Dictionary with removal statistics
        """
        if not results:
            return {}
        
        total_words_before = sum(r.word_count_before for r in results)
        total_words_after = sum(r.word_count_after for r in results)
        total_removed = sum(r.words_removed_count for r in results)
        
        # Count most common removed words
        all_removed_words = []
        for result in results:
            all_removed_words.extend(result.removed_words)
        
        most_common_removed = Counter(all_removed_words).most_common(10)
        
        return {
            'total_documents': len(results),
            'total_words_before': total_words_before,
            'total_words_after': total_words_after,
            'total_words_removed': total_removed,
            'average_removal_ratio': sum(r.removal_ratio for r in results) / len(results),
            'average_words_removed_per_document': total_removed / len(results),
            'most_common_removed_words': most_common_removed,
        }


def remove_stopwords_simple(text: str, 
                           language: str = 'english',
                           custom_stopwords: Optional[Set[str]] = None) -> str:
    """
    Convenience function to remove stopwords with default settings.
    
    Args:
        text: Text to process
        language: Language for built-in stopwords
        custom_stopwords: Additional custom stopwords
        
    Returns:
        Text with stopwords removed
    """
    remover = StopwordRemover(
        languages=language,
        custom_stopwords=custom_stopwords
    )
    result = remover.remove_stopwords(text)
    return result.cleaned_text


def load_stopwords_from_file(file_path: str) -> Set[str]:
    """
    Load custom stopwords from a file.
    
    Args:
        file_path: Path to file containing stopwords (one per line)
        
    Returns:
        Set of stopwords
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            stopwords = {line.strip().lower() for line in f if line.strip()}
        return stopwords
    except Exception as e:
        raise ValueError(f"Failed to load stopwords from {file_path}: {e}")