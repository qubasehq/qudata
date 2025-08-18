"""
Unit tests for the analysis engine and its components.

Tests cover text analysis, topic modeling, sentiment analysis, language analysis,
and quality analysis functionality.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from src.qudata.models import Document, DocumentMetadata, DocumentStructure
from src.qudata.analyze import (
    TextAnalyzer, TopicModeler, SentimentAnalyzer, 
    LanguageAnalyzer, QualityAnalyzer, AnalysisEngine
)


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    docs = []
    
    # Document 1: English, positive sentiment, good quality
    doc1 = Document(
        id="doc1",
        source_path="test1.txt",
        content="This is a wonderful example of high-quality content. "
                "It contains multiple sentences with varied vocabulary. "
                "The text demonstrates excellent readability and structure. "
                "Machine learning and artificial intelligence are fascinating topics.",
        metadata=DocumentMetadata(
            file_type="txt",
            size_bytes=200,
            language="en"
        ),
        structure=DocumentStructure(
            headings=["Introduction"],
            paragraphs=2,
            tables=0,
            images=0
        )
    )
    
    # Document 2: English, negative sentiment, medium quality
    doc2 = Document(
        id="doc2", 
        source_path="test2.txt",
        content="This content is terrible and awful. "
                "The quality is very poor and disappointing. "
                "I hate this bad example of writing. "
                "It lacks proper structure and coherence.",
        metadata=DocumentMetadata(
            file_type="txt",
            size_bytes=150,
            language="en"
        ),
        structure=DocumentStructure(
            paragraphs=1
        )
    )
    
    # Document 3: Short content, neutral sentiment
    doc3 = Document(
        id="doc3",
        source_path="test3.txt", 
        content="Short text example.",
        metadata=DocumentMetadata(
            file_type="txt",
            size_bytes=20,
            language="en"
        ),
        structure=DocumentStructure(
            paragraphs=1
        )
    )
    
    docs.extend([doc1, doc2, doc3])
    return docs


class TestTextAnalyzer:
    """Test cases for TextAnalyzer."""
    
    def test_initialization(self):
        """Test TextAnalyzer initialization."""
        analyzer = TextAnalyzer()
        assert analyzer.min_keyword_length == 3
        assert analyzer.max_keyword_length == 50
        assert analyzer.top_keywords_count == 50
        assert len(analyzer.stopwords) > 0
    
    def test_initialization_with_config(self):
        """Test TextAnalyzer initialization with custom config."""
        config = {
            "min_keyword_length": 5,
            "top_keywords_count": 20,
            "stopwords": ["custom", "stop"]
        }
        analyzer = TextAnalyzer(config)
        assert analyzer.min_keyword_length == 5
        assert analyzer.top_keywords_count == 20
        assert "custom" in analyzer.stopwords
    
    def test_analyze_text_statistics(self, sample_documents):
        """Test text statistics analysis."""
        analyzer = TextAnalyzer()
        stats = analyzer.analyze_text_statistics(sample_documents)
        
        assert stats.total_documents == 3
        assert stats.total_words > 0
        assert stats.unique_tokens > 0
        assert stats.vocabulary_size > 0
        assert 0 <= stats.type_token_ratio <= 1
        assert 0 <= stats.readability_score <= 100
        assert len(stats.top_keywords) > 0
    
    def test_analyze_empty_documents(self):
        """Test analysis with empty document list."""
        analyzer = TextAnalyzer()
        stats = analyzer.analyze_text_statistics([])
        
        assert stats.total_documents == 0
        assert stats.total_words == 0
        assert stats.vocabulary_size == 0
    
    def test_extract_keywords_tfidf(self, sample_documents):
        """Test TF-IDF keyword extraction."""
        analyzer = TextAnalyzer()
        keywords = analyzer.extract_keywords(sample_documents, method="tf_idf")
        
        assert len(keywords) > 0
        assert all(kw.tf_idf_score >= 0 for kw in keywords)
        assert all(kw.frequency > 0 for kw in keywords)
    
    def test_extract_keywords_frequency(self, sample_documents):
        """Test frequency-based keyword extraction."""
        analyzer = TextAnalyzer()
        keywords = analyzer.extract_keywords(sample_documents, method="frequency")
        
        assert len(keywords) > 0
        assert all(kw.frequency > 0 for kw in keywords)
    
    def test_analyze_document_statistics(self, sample_documents):
        """Test single document statistics."""
        analyzer = TextAnalyzer()
        doc_stats = analyzer.analyze_document_statistics(sample_documents[0])
        
        assert doc_stats["word_count"] > 0
        assert doc_stats["character_count"] > 0
        assert doc_stats["unique_words"] > 0
        assert doc_stats["vocabulary_richness"] > 0


class TestTopicModeler:
    """Test cases for TopicModeler."""
    
    def test_initialization(self):
        """Test TopicModeler initialization."""
        modeler = TopicModeler()
        assert modeler.min_word_length == 3
        assert modeler.max_word_length == 50
        assert len(modeler.stopwords) > 0
    
    def test_perform_simple_modeling(self, sample_documents):
        """Test simple topic modeling (fallback method)."""
        modeler = TopicModeler()
        result = modeler.perform_topic_modeling(sample_documents, method="simple", num_topics=2)
        
        assert result.method == "simple"
        assert len(result.topics) <= 2
        assert len(result.document_topics) <= len(sample_documents)
    
    def test_perform_modeling_empty_documents(self):
        """Test topic modeling with empty document list."""
        modeler = TopicModeler()
        result = modeler.perform_topic_modeling([], method="simple")
        
        assert result.num_topics == 0
        assert len(result.topics) == 0
        assert len(result.document_topics) == 0
    
    @patch('src.qudata.analyze.topic_modeler.TopicModeler._check_sklearn')
    def test_lda_modeling_unavailable(self, mock_sklearn, sample_documents):
        """Test LDA modeling when sklearn is unavailable."""
        mock_sklearn.return_value = False
        modeler = TopicModeler()
        result = modeler.perform_topic_modeling(sample_documents, method="lda")
        
        # Should fallback to simple method
        assert result.method == "simple"
    
    def test_get_topic_summary(self, sample_documents):
        """Test topic modeling summary generation."""
        modeler = TopicModeler()
        result = modeler.perform_topic_modeling(sample_documents, method="simple", num_topics=2)
        summary = modeler.get_topic_summary(result)
        
        # Check if we got topics or empty result
        if result.topics:
            assert "method" in summary
            assert "num_topics" in summary
            assert "topics_summary" in summary
        else:
            assert "message" in summary


class TestSentimentAnalyzer:
    """Test cases for SentimentAnalyzer."""
    
    def test_initialization(self):
        """Test SentimentAnalyzer initialization."""
        analyzer = SentimentAnalyzer()
        assert analyzer.confidence_threshold == 0.5
        assert len(analyzer.positive_words) > 0
        assert len(analyzer.negative_words) > 0
        assert len(analyzer.emotion_lexicon) > 0
    
    def test_analyze_sentiment(self, sample_documents):
        """Test sentiment analysis on documents."""
        analyzer = SentimentAnalyzer()
        analysis = analyzer.analyze_sentiment(sample_documents)
        
        assert len(analysis.document_sentiments) == len(sample_documents)
        assert -1 <= analysis.overall_polarity <= 1
        assert 0 <= analysis.overall_subjectivity <= 1
        assert "positive" in analysis.polarity_distribution
        assert "negative" in analysis.polarity_distribution
        assert "neutral" in analysis.polarity_distribution
    
    def test_analyze_document_sentiment(self, sample_documents):
        """Test single document sentiment analysis."""
        analyzer = SentimentAnalyzer()
        doc_sentiment = analyzer.analyze_document_sentiment(sample_documents[0])
        
        assert doc_sentiment.document_id == sample_documents[0].id
        assert -1 <= doc_sentiment.overall_sentiment.polarity <= 1
        assert 0 <= doc_sentiment.overall_sentiment.subjectivity <= 1
        assert doc_sentiment.overall_sentiment.emotion in [
            'joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'trust', 'anticipation', 'neutral'
        ]
    
    def test_analyze_empty_documents(self):
        """Test sentiment analysis with empty document list."""
        analyzer = SentimentAnalyzer()
        analysis = analyzer.analyze_sentiment([])
        
        assert len(analysis.document_sentiments) == 0
        assert analysis.overall_polarity == 0.0
        assert analysis.overall_subjectivity == 0.0
    
    def test_get_sentiment_summary(self, sample_documents):
        """Test sentiment analysis summary generation."""
        analyzer = SentimentAnalyzer()
        analysis = analyzer.analyze_sentiment(sample_documents)
        summary = analyzer.get_sentiment_summary(analysis)
        
        assert "total_documents" in summary
        assert "overall_sentiment" in summary
        assert "polarity_distribution" in summary


class TestLanguageAnalyzer:
    """Test cases for LanguageAnalyzer."""
    
    def test_initialization(self):
        """Test LanguageAnalyzer initialization."""
        analyzer = LanguageAnalyzer()
        assert analyzer.min_text_length == 20
        assert analyzer.confidence_threshold == 0.7
        assert len(analyzer.language_names) > 0
        assert len(analyzer.script_patterns) > 0
    
    def test_analyze_languages(self, sample_documents):
        """Test language analysis on documents."""
        analyzer = LanguageAnalyzer()
        distribution = analyzer.analyze_languages(sample_documents)
        
        assert distribution.total_documents == len(sample_documents)
        assert len(distribution.language_counts) > 0
        assert len(distribution.language_percentages) > 0
        assert distribution.dominant_language is not None
        assert 0 <= distribution.language_diversity_index
    
    def test_analyze_document_language(self, sample_documents):
        """Test single document language analysis."""
        analyzer = LanguageAnalyzer()
        doc_lang = analyzer.analyze_document_language(sample_documents[0])
        
        assert doc_lang.document_id == sample_documents[0].id
        assert doc_lang.primary_language.language is not None
        assert 0 <= doc_lang.primary_language.confidence <= 1
        assert doc_lang.primary_language.iso_code is not None
    
    def test_analyze_empty_documents(self):
        """Test language analysis with empty document list."""
        analyzer = LanguageAnalyzer()
        distribution = analyzer.analyze_languages([])
        
        assert distribution.total_documents == 0
        assert len(distribution.language_counts) == 0
        assert distribution.dominant_language == 'unknown'
    
    def test_get_language_summary(self, sample_documents):
        """Test language analysis summary generation."""
        analyzer = LanguageAnalyzer()
        distribution = analyzer.analyze_languages(sample_documents)
        summary = analyzer.get_language_summary(distribution)
        
        assert "total_documents" in summary
        assert "dominant_language" in summary
        assert "top_languages" in summary


class TestQualityAnalyzer:
    """Test cases for QualityAnalyzer."""
    
    def test_initialization(self):
        """Test QualityAnalyzer initialization."""
        analyzer = QualityAnalyzer()
        assert analyzer.min_length == 50
        assert analyzer.max_length == 50000
        assert len(analyzer.weights) == 7
        assert sum(analyzer.weights.values()) == pytest.approx(1.0, rel=1e-2)
    
    def test_analyze_quality(self, sample_documents):
        """Test quality analysis on documents."""
        analyzer = QualityAnalyzer()
        report = analyzer.analyze_quality(sample_documents)
        
        assert len(report.document_qualities) == len(sample_documents)
        assert 0 <= report.overall_quality_score <= 1
        assert len(report.quality_distribution) > 0
        assert "mean_score" in report.quality_statistics
    
    def test_analyze_document_quality(self, sample_documents):
        """Test single document quality analysis."""
        analyzer = QualityAnalyzer()
        doc_quality = analyzer.analyze_document_quality(sample_documents[0])
        
        assert doc_quality.document_id == sample_documents[0].id
        assert 0 <= doc_quality.quality_score.overall_score <= 1
        assert 0 <= doc_quality.quality_score.length_score <= 1
        assert 0 <= doc_quality.quality_score.language_score <= 1
        assert doc_quality.quality_grade in ['A', 'B', 'C', 'D', 'F']
    
    def test_analyze_empty_documents(self):
        """Test quality analysis with empty document list."""
        analyzer = QualityAnalyzer()
        report = analyzer.analyze_quality([])
        
        assert len(report.document_qualities) == 0
        assert report.overall_quality_score == 0.0
        assert len(report.quality_distribution) == 0
    
    def test_get_quality_summary(self, sample_documents):
        """Test quality analysis summary generation."""
        analyzer = QualityAnalyzer()
        report = analyzer.analyze_quality(sample_documents)
        summary = analyzer.get_quality_summary(report)
        
        assert "total_documents" in summary
        assert "overall_quality_score" in summary
        assert "quality_level" in summary


class TestAnalysisEngine:
    """Test cases for AnalysisEngine."""
    
    def test_initialization(self):
        """Test AnalysisEngine initialization."""
        engine = AnalysisEngine()
        assert engine.text_analyzer is not None
        assert engine.topic_modeler is not None
        assert engine.sentiment_analyzer is not None
        assert engine.language_analyzer is not None
        assert engine.quality_analyzer is not None
    
    def test_initialization_with_config(self):
        """Test AnalysisEngine initialization with custom config."""
        config = {
            "enabled_analyses": {
                "text_statistics": True,
                "topic_modeling": False,
                "sentiment_analysis": True,
                "language_analysis": False,
                "quality_analysis": True
            }
        }
        engine = AnalysisEngine(config)
        assert engine.enabled_analyses["text_statistics"] is True
        assert engine.enabled_analyses["topic_modeling"] is False
    
    def test_analyze_dataset(self, sample_documents):
        """Test comprehensive dataset analysis."""
        engine = AnalysisEngine()
        result = engine.analyze_dataset(sample_documents)
        
        assert result.text_statistics is not None
        assert result.topic_model_result is not None
        assert result.sentiment_analysis is not None
        assert result.language_distribution is not None
        assert result.quality_report is not None
        assert result.analysis_metadata["document_count"] == len(sample_documents)
    
    def test_analyze_empty_dataset(self):
        """Test analysis with empty dataset."""
        engine = AnalysisEngine()
        result = engine.analyze_dataset([])
        
        assert result.analysis_metadata["document_count"] == 0
    
    def test_create_comprehensive_report(self, sample_documents):
        """Test comprehensive report generation."""
        engine = AnalysisEngine()
        report = engine.create_comprehensive_report(sample_documents)
        
        assert report.analysis_result is not None
        assert "dataset_size" in report.executive_summary
        assert len(report.detailed_insights) > 0
        assert len(report.recommendations) > 0
    
    def test_get_analysis_summary(self, sample_documents):
        """Test analysis summary generation."""
        engine = AnalysisEngine()
        result = engine.analyze_dataset(sample_documents)
        summary = engine.get_analysis_summary(result)
        
        assert "document_count" in summary
        assert "enabled_analyses" in summary
        assert "text_summary" in summary
    
    def test_selective_analysis(self, sample_documents):
        """Test analysis with selective components enabled."""
        config = {
            "enabled_analyses": {
                "text_statistics": True,
                "topic_modeling": False,
                "sentiment_analysis": True,
                "language_analysis": False,
                "quality_analysis": False
            }
        }
        engine = AnalysisEngine(config)
        result = engine.analyze_dataset(sample_documents)
        
        assert result.text_statistics is not None
        assert result.topic_model_result is None
        assert result.sentiment_analysis is not None
        assert result.language_distribution is None
        assert result.quality_report is None


class TestAnalysisIntegration:
    """Integration tests for analysis components."""
    
    def test_full_analysis_pipeline(self, sample_documents):
        """Test complete analysis pipeline integration."""
        engine = AnalysisEngine()
        
        # Run full analysis
        result = engine.analyze_dataset(sample_documents)
        
        # Verify all components ran successfully
        assert result.text_statistics is not None
        assert result.topic_model_result is not None
        assert result.sentiment_analysis is not None
        assert result.language_distribution is not None
        assert result.quality_report is not None
        
        # Verify data consistency
        assert result.text_statistics.total_documents == len(sample_documents)
        assert len(result.sentiment_analysis.document_sentiments) == len(sample_documents)
        assert result.language_distribution.total_documents == len(sample_documents)
        assert len(result.quality_report.document_qualities) == len(sample_documents)
    
    def test_analysis_result_serialization(self, sample_documents):
        """Test analysis result serialization to dictionary."""
        engine = AnalysisEngine()
        result = engine.analyze_dataset(sample_documents)
        
        # Convert to dictionary
        result_dict = result.to_dict()
        
        # Verify structure
        assert "analysis_metadata" in result_dict
        assert "text_statistics" in result_dict
        assert "topic_model_result" in result_dict
        assert "sentiment_analysis" in result_dict
        assert "language_distribution" in result_dict
        assert "quality_report" in result_dict
    
    def test_comprehensive_report_generation(self, sample_documents):
        """Test comprehensive report generation and structure."""
        engine = AnalysisEngine()
        report = engine.create_comprehensive_report(sample_documents)
        
        # Verify report structure
        assert report.analysis_result is not None
        assert "dataset_size" in report.executive_summary
        assert len(report.detailed_insights) > 0
        assert len(report.recommendations) > 0
        
        # Verify report serialization
        report_dict = report.to_dict()
        assert "analysis_result" in report_dict
        assert "executive_summary" in report_dict
        assert "detailed_insights" in report_dict
        assert "recommendations" in report_dict


if __name__ == "__main__":
    pytest.main([__file__])