"""
Text analysis module for statistical analysis and keyword extraction.

This module provides comprehensive text statistics including word counts, token analysis,
keyword extraction, and document length distributions.
"""

import re
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Set
from ..models import Document


@dataclass
class TextStatistics:
    """Comprehensive text statistics for a dataset."""
    total_documents: int
    total_words: int
    total_characters: int
    unique_tokens: int
    avg_document_length: float
    median_document_length: float
    min_document_length: int
    max_document_length: int
    top_keywords: List[Tuple[str, int]]
    token_length_distribution: Dict[int, int]
    word_frequency_distribution: Dict[str, int]
    vocabulary_size: int
    type_token_ratio: float  # Vocabulary richness measure
    avg_sentence_length: float
    readability_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            "total_documents": self.total_documents,
            "total_words": self.total_words,
            "total_characters": self.total_characters,
            "unique_tokens": self.unique_tokens,
            "avg_document_length": self.avg_document_length,
            "median_document_length": self.median_document_length,
            "min_document_length": self.min_document_length,
            "max_document_length": self.max_document_length,
            "top_keywords": self.top_keywords,
            "token_length_distribution": self.token_length_distribution,
            "vocabulary_size": self.vocabulary_size,
            "type_token_ratio": self.type_token_ratio,
            "avg_sentence_length": self.avg_sentence_length,
            "readability_score": self.readability_score
        }


@dataclass
class KeywordResult:
    """Result of keyword extraction."""
    keyword: str
    frequency: int
    tf_idf_score: float
    positions: List[int]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "keyword": self.keyword,
            "frequency": self.frequency,
            "tf_idf_score": self.tf_idf_score,
            "positions": self.positions
        }


class TextAnalyzer:
    """
    Comprehensive text analyzer for statistical analysis and keyword extraction.
    
    Provides methods for analyzing text statistics, extracting keywords,
    and computing various text metrics for datasets.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize text analyzer.
        
        Args:
            config: Configuration dictionary with analysis parameters
        """
        self.config = config or {}
        self.min_keyword_length = self.config.get("min_keyword_length", 3)
        self.max_keyword_length = self.config.get("max_keyword_length", 50)
        self.top_keywords_count = self.config.get("top_keywords_count", 50)
        self.stopwords = self._load_stopwords()
        
    def _load_stopwords(self) -> Set[str]:
        """Load stopwords for keyword filtering."""
        # Basic English stopwords - in production, this could load from NLTK
        basic_stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'would', 'you', 'your', 'have', 'had',
            'this', 'they', 'them', 'their', 'there', 'then', 'than', 'these',
            'those', 'when', 'where', 'who', 'what', 'why', 'how', 'can', 'could',
            'should', 'would', 'may', 'might', 'must', 'shall', 'will', 'do', 'did',
            'does', 'done', 'been', 'being', 'am', 'is', 'are', 'was', 'were'
        }
        
        # Allow custom stopwords from config
        custom_stopwords = set(self.config.get("stopwords", []))
        return basic_stopwords.union(custom_stopwords)
    
    def analyze_text_statistics(self, documents: List[Document]) -> TextStatistics:
        """
        Analyze comprehensive text statistics for a list of documents.
        
        Args:
            documents: List of documents to analyze
            
        Returns:
            TextStatistics object with comprehensive metrics
        """
        if not documents:
            return self._empty_statistics()
        
        # Collect basic metrics
        document_lengths = []
        all_words = []
        all_tokens = []
        total_characters = 0
        sentence_lengths = []
        
        for doc in documents:
            content = doc.content
            words = self._tokenize_words(content)
            tokens = self._tokenize_tokens(content)
            sentences = self._split_sentences(content)
            
            doc_length = len(words)
            document_lengths.append(doc_length)
            all_words.extend(words)
            all_tokens.extend(tokens)
            total_characters += len(content)
            
            # Sentence length analysis
            for sentence in sentences:
                sentence_words = self._tokenize_words(sentence)
                if sentence_words:
                    sentence_lengths.append(len(sentence_words))
        
        # Calculate statistics
        total_documents = len(documents)
        total_words = len(all_words)
        unique_tokens = len(set(all_tokens))
        vocabulary_size = len(set(all_words))
        
        # Document length statistics
        avg_doc_length = sum(document_lengths) / len(document_lengths)
        sorted_lengths = sorted(document_lengths)
        median_doc_length = self._calculate_median(sorted_lengths)
        min_doc_length = min(document_lengths)
        max_doc_length = max(document_lengths)
        
        # Token length distribution
        token_length_dist = Counter(len(token) for token in all_tokens)
        
        # Word frequency distribution and top keywords
        word_freq = Counter(all_words)
        top_keywords = self._extract_top_keywords(word_freq)
        
        # Advanced metrics
        type_token_ratio = vocabulary_size / total_words if total_words > 0 else 0.0
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0.0
        readability_score = self._calculate_readability_score(documents)
        
        return TextStatistics(
            total_documents=total_documents,
            total_words=total_words,
            total_characters=total_characters,
            unique_tokens=unique_tokens,
            avg_document_length=avg_doc_length,
            median_document_length=median_doc_length,
            min_document_length=min_doc_length,
            max_document_length=max_doc_length,
            top_keywords=top_keywords,
            token_length_distribution=dict(token_length_dist),
            word_frequency_distribution=dict(word_freq.most_common(1000)),  # Top 1000 words
            vocabulary_size=vocabulary_size,
            type_token_ratio=type_token_ratio,
            avg_sentence_length=avg_sentence_length,
            readability_score=readability_score
        )
    
    def extract_keywords(self, documents: List[Document], method: str = "tf_idf") -> List[KeywordResult]:
        """
        Extract keywords from documents using specified method.
        
        Args:
            documents: List of documents to extract keywords from
            method: Extraction method ("tf_idf", "frequency", "combined")
            
        Returns:
            List of KeywordResult objects sorted by relevance
        """
        if not documents:
            return []
        
        if method == "tf_idf":
            return self._extract_keywords_tfidf(documents)
        elif method == "frequency":
            return self._extract_keywords_frequency(documents)
        elif method == "combined":
            return self._extract_keywords_combined(documents)
        else:
            raise ValueError(f"Unknown keyword extraction method: {method}")
    
    def _extract_keywords_tfidf(self, documents: List[Document]) -> List[KeywordResult]:
        """Extract keywords using TF-IDF scoring."""
        # Calculate term frequencies for each document
        doc_term_freqs = []
        all_terms = set()
        
        for doc in documents:
            words = self._tokenize_words(doc.content)
            filtered_words = [w for w in words if self._is_valid_keyword(w)]
            term_freq = Counter(filtered_words)
            doc_term_freqs.append(term_freq)
            all_terms.update(filtered_words)
        
        # Calculate document frequencies
        doc_freq = defaultdict(int)
        for term_freq in doc_term_freqs:
            for term in term_freq:
                doc_freq[term] += 1
        
        # Calculate TF-IDF scores
        num_docs = len(documents)
        keyword_scores = defaultdict(float)
        keyword_positions = defaultdict(list)
        
        for doc_idx, (doc, term_freq) in enumerate(zip(documents, doc_term_freqs)):
            doc_length = sum(term_freq.values())
            
            for term, freq in term_freq.items():
                tf = freq / doc_length if doc_length > 0 else 0
                idf = math.log(num_docs / doc_freq[term]) if doc_freq[term] > 0 else 0
                tfidf = tf * idf
                
                keyword_scores[term] += tfidf
                
                # Find positions of this term in the document
                positions = self._find_term_positions(doc.content, term)
                keyword_positions[term].extend(positions)
        
        # Create keyword results
        results = []
        for term in sorted(keyword_scores.keys(), key=lambda x: keyword_scores[x], reverse=True):
            total_freq = sum(tf[term] for tf in doc_term_freqs if term in tf)
            results.append(KeywordResult(
                keyword=term,
                frequency=total_freq,
                tf_idf_score=keyword_scores[term],
                positions=keyword_positions[term][:10]  # Limit positions
            ))
        
        return results[:self.top_keywords_count]
    
    def _extract_keywords_frequency(self, documents: List[Document]) -> List[KeywordResult]:
        """Extract keywords based on frequency."""
        all_words = []
        word_positions = defaultdict(list)
        
        for doc_idx, doc in enumerate(documents):
            words = self._tokenize_words(doc.content)
            filtered_words = [w for w in words if self._is_valid_keyword(w)]
            all_words.extend(filtered_words)
            
            # Track positions
            for word in filtered_words:
                positions = self._find_term_positions(doc.content, word)
                word_positions[word].extend(positions)
        
        word_freq = Counter(all_words)
        
        results = []
        for word, freq in word_freq.most_common(self.top_keywords_count):
            results.append(KeywordResult(
                keyword=word,
                frequency=freq,
                tf_idf_score=0.0,  # Not calculated for frequency method
                positions=word_positions[word][:10]
            ))
        
        return results
    
    def _extract_keywords_combined(self, documents: List[Document]) -> List[KeywordResult]:
        """Extract keywords using combined TF-IDF and frequency scoring."""
        tfidf_results = self._extract_keywords_tfidf(documents)
        freq_results = self._extract_keywords_frequency(documents)
        
        # Create combined scoring
        combined_scores = {}
        
        # Normalize TF-IDF scores
        max_tfidf = max(r.tf_idf_score for r in tfidf_results) if tfidf_results else 1.0
        for result in tfidf_results:
            normalized_tfidf = result.tf_idf_score / max_tfidf if max_tfidf > 0 else 0
            combined_scores[result.keyword] = {
                'tfidf': normalized_tfidf,
                'frequency': result.frequency,
                'positions': result.positions
            }
        
        # Normalize frequency scores
        max_freq = max(r.frequency for r in freq_results) if freq_results else 1.0
        for result in freq_results:
            normalized_freq = result.frequency / max_freq if max_freq > 0 else 0
            if result.keyword in combined_scores:
                combined_scores[result.keyword]['norm_freq'] = normalized_freq
            else:
                combined_scores[result.keyword] = {
                    'tfidf': 0.0,
                    'frequency': result.frequency,
                    'norm_freq': normalized_freq,
                    'positions': result.positions
                }
        
        # Calculate combined scores (weighted average)
        tfidf_weight = self.config.get("tfidf_weight", 0.7)
        freq_weight = self.config.get("frequency_weight", 0.3)
        
        results = []
        for keyword, scores in combined_scores.items():
            tfidf_score = scores.get('tfidf', 0.0)
            norm_freq = scores.get('norm_freq', 0.0)
            combined_score = (tfidf_weight * tfidf_score) + (freq_weight * norm_freq)
            
            results.append(KeywordResult(
                keyword=keyword,
                frequency=scores['frequency'],
                tf_idf_score=combined_score,
                positions=scores['positions']
            ))
        
        # Sort by combined score
        results.sort(key=lambda x: x.tf_idf_score, reverse=True)
        return results[:self.top_keywords_count]
    
    def _tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple word tokenization - in production, could use NLTK
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return [w for w in words if len(w) >= 2]
    
    def _tokenize_tokens(self, text: str) -> List[str]:
        """Tokenize text into all tokens (including punctuation)."""
        return re.findall(r'\S+', text)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - in production, could use NLTK
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _is_valid_keyword(self, word: str) -> bool:
        """Check if a word is valid for keyword extraction."""
        return (
            len(word) >= self.min_keyword_length and
            len(word) <= self.max_keyword_length and
            word.lower() not in self.stopwords and
            word.isalpha()
        )
    
    def _find_term_positions(self, text: str, term: str) -> List[int]:
        """Find all positions of a term in text."""
        positions = []
        text_lower = text.lower()
        term_lower = term.lower()
        start = 0
        
        while True:
            pos = text_lower.find(term_lower, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        
        return positions
    
    def _extract_top_keywords(self, word_freq: Counter) -> List[Tuple[str, int]]:
        """Extract top keywords from word frequency counter."""
        filtered_words = [
            (word, freq) for word, freq in word_freq.items()
            if self._is_valid_keyword(word)
        ]
        
        # Sort by frequency and return top N
        filtered_words.sort(key=lambda x: x[1], reverse=True)
        return filtered_words[:self.top_keywords_count]
    
    def _calculate_median(self, sorted_values: List[float]) -> float:
        """Calculate median of sorted values."""
        n = len(sorted_values)
        if n == 0:
            return 0.0
        elif n % 2 == 0:
            return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
        else:
            return sorted_values[n//2]
    
    def _calculate_readability_score(self, documents: List[Document]) -> float:
        """
        Calculate a simple readability score based on sentence and word length.
        
        This is a simplified version of readability metrics like Flesch-Kincaid.
        """
        total_sentences = 0
        total_words = 0
        total_syllables = 0
        
        for doc in documents:
            sentences = self._split_sentences(doc.content)
            total_sentences += len(sentences)
            
            words = self._tokenize_words(doc.content)
            total_words += len(words)
            
            # Estimate syllables (simple heuristic)
            for word in words:
                syllables = max(1, len(re.findall(r'[aeiouAEIOU]', word)))
                total_syllables += syllables
        
        if total_sentences == 0 or total_words == 0:
            return 0.0
        
        # Simplified Flesch Reading Ease formula
        avg_sentence_length = total_words / total_sentences
        avg_syllables_per_word = total_syllables / total_words
        
        # Flesch Reading Ease = 206.835 - (1.015 × ASL) - (84.6 × ASW)
        # Where ASL = average sentence length, ASW = average syllables per word
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Normalize to 0-100 range
        return max(0.0, min(100.0, score))
    
    def _empty_statistics(self) -> TextStatistics:
        """Return empty statistics for edge cases."""
        return TextStatistics(
            total_documents=0,
            total_words=0,
            total_characters=0,
            unique_tokens=0,
            avg_document_length=0.0,
            median_document_length=0.0,
            min_document_length=0,
            max_document_length=0,
            top_keywords=[],
            token_length_distribution={},
            word_frequency_distribution={},
            vocabulary_size=0,
            type_token_ratio=0.0,
            avg_sentence_length=0.0,
            readability_score=0.0
        )
    
    def analyze_document_statistics(self, document: Document) -> Dict[str, Any]:
        """
        Analyze statistics for a single document.
        
        Args:
            document: Document to analyze
            
        Returns:
            Dictionary with document statistics
        """
        words = self._tokenize_words(document.content)
        tokens = self._tokenize_tokens(document.content)
        sentences = self._split_sentences(document.content)
        
        word_freq = Counter(words)
        top_words = word_freq.most_common(20)
        
        return {
            "word_count": len(words),
            "character_count": len(document.content),
            "token_count": len(tokens),
            "sentence_count": len(sentences),
            "unique_words": len(set(words)),
            "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
            "top_words": top_words,
            "vocabulary_richness": len(set(words)) / len(words) if words else 0
        }