"""
Main analysis engine that coordinates all analysis components.

This module provides a unified interface for comprehensive data analysis including
text statistics, topic modeling, sentiment analysis, language analysis, and quality scoring.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from ..models import Document

from .text_analyzer import TextAnalyzer, TextStatistics
from .topic_modeler import TopicModeler, TopicModelResult
from .sentiment_analyzer import SentimentAnalyzer, SentimentAnalysis
from .language_analyzer import LanguageAnalyzer, LanguageDistribution
from .quality_analyzer import QualityAnalyzer, QualityReport

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Comprehensive analysis result containing all analysis types."""
    text_statistics: Optional[TextStatistics] = None
    topic_model_result: Optional[TopicModelResult] = None
    sentiment_analysis: Optional[SentimentAnalysis] = None
    language_distribution: Optional[LanguageDistribution] = None
    quality_report: Optional[QualityReport] = None
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis result to dictionary."""
        result = {
            "analysis_metadata": self.analysis_metadata
        }
        
        if self.text_statistics:
            result["text_statistics"] = self.text_statistics.to_dict()
        
        if self.topic_model_result:
            result["topic_model_result"] = self.topic_model_result.to_dict()
        
        if self.sentiment_analysis:
            result["sentiment_analysis"] = self.sentiment_analysis.to_dict()
        
        if self.language_distribution:
            result["language_distribution"] = self.language_distribution.to_dict()
        
        if self.quality_report:
            result["quality_report"] = self.quality_report.to_dict()
        
        return result


@dataclass
class AnalysisReport:
    """Comprehensive analysis report with summaries and insights."""
    analysis_result: AnalysisResult
    executive_summary: Dict[str, Any]
    detailed_insights: Dict[str, Any]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis report to dictionary."""
        return {
            "analysis_result": self.analysis_result.to_dict(),
            "executive_summary": self.executive_summary,
            "detailed_insights": self.detailed_insights,
            "recommendations": self.recommendations
        }


class AnalysisEngine:
    """
    Comprehensive analysis engine coordinating all analysis components.
    
    Provides a unified interface for performing text statistics, topic modeling,
    sentiment analysis, language analysis, and quality assessment on document collections.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize analysis engine.
        
        Args:
            config: Configuration dictionary for all analysis components
        """
        self.config = config or {}
        
        # Initialize individual analyzers
        self.text_analyzer = TextAnalyzer(self.config.get("text_analysis", {}))
        self.topic_modeler = TopicModeler(self.config.get("topic_modeling", {}))
        self.sentiment_analyzer = SentimentAnalyzer(self.config.get("sentiment_analysis", {}))
        self.language_analyzer = LanguageAnalyzer(self.config.get("language_analysis", {}))
        self.quality_analyzer = QualityAnalyzer(self.config.get("quality_analysis", {}))
        
        # Analysis configuration
        self.enabled_analyses = self.config.get("enabled_analyses", {
            "text_statistics": True,
            "topic_modeling": True,
            "sentiment_analysis": True,
            "language_analysis": True,
            "quality_analysis": True
        })
        
        logger.info("Analysis engine initialized with components: %s", 
                   [name for name, enabled in self.enabled_analyses.items() if enabled])
    
    def analyze_dataset(self, documents: List[Document]) -> AnalysisResult:
        """
        Perform comprehensive analysis on a dataset.
        
        Args:
            documents: List of documents to analyze
            
        Returns:
            AnalysisResult with all enabled analysis results
        """
        if not documents:
            logger.warning("No documents provided for analysis")
            result = AnalysisResult()
            result.analysis_metadata = {
                "document_count": 0,
                "enabled_analyses": self.enabled_analyses,
                "analysis_timestamp": self._get_timestamp()
            }
            return result
        
        logger.info(f"Starting comprehensive analysis of {len(documents)} documents")
        
        result = AnalysisResult()
        result.analysis_metadata = {
            "document_count": len(documents),
            "enabled_analyses": self.enabled_analyses,
            "analysis_timestamp": self._get_timestamp()
        }
        
        # Perform text statistics analysis
        if self.enabled_analyses.get("text_statistics", True):
            try:
                logger.info("Performing text statistics analysis...")
                result.text_statistics = self.text_analyzer.analyze_text_statistics(documents)
                logger.info("Text statistics analysis completed")
            except Exception as e:
                logger.error(f"Text statistics analysis failed: {e}")
                result.analysis_metadata["text_statistics_error"] = str(e)
        
        # Perform topic modeling
        if self.enabled_analyses.get("topic_modeling", True):
            try:
                logger.info("Performing topic modeling...")
                topic_method = self.config.get("topic_modeling", {}).get("method", "lda")
                num_topics = self.config.get("topic_modeling", {}).get("num_topics")
                result.topic_model_result = self.topic_modeler.perform_topic_modeling(
                    documents, method=topic_method, num_topics=num_topics
                )
                logger.info(f"Topic modeling completed with {len(result.topic_model_result.topics)} topics")
            except Exception as e:
                logger.error(f"Topic modeling failed: {e}")
                result.analysis_metadata["topic_modeling_error"] = str(e)
        
        # Perform sentiment analysis
        if self.enabled_analyses.get("sentiment_analysis", True):
            try:
                logger.info("Performing sentiment analysis...")
                result.sentiment_analysis = self.sentiment_analyzer.analyze_sentiment(documents)
                logger.info("Sentiment analysis completed")
            except Exception as e:
                logger.error(f"Sentiment analysis failed: {e}")
                result.analysis_metadata["sentiment_analysis_error"] = str(e)
        
        # Perform language analysis
        if self.enabled_analyses.get("language_analysis", True):
            try:
                logger.info("Performing language analysis...")
                result.language_distribution = self.language_analyzer.analyze_languages(documents)
                logger.info(f"Language analysis completed, found {len(result.language_distribution.language_counts)} languages")
            except Exception as e:
                logger.error(f"Language analysis failed: {e}")
                result.analysis_metadata["language_analysis_error"] = str(e)
        
        # Perform quality analysis
        if self.enabled_analyses.get("quality_analysis", True):
            try:
                logger.info("Performing quality analysis...")
                result.quality_report = self.quality_analyzer.analyze_quality(documents)
                logger.info(f"Quality analysis completed, overall score: {result.quality_report.overall_quality_score:.3f}")
            except Exception as e:
                logger.error(f"Quality analysis failed: {e}")
                result.analysis_metadata["quality_analysis_error"] = str(e)
        
        logger.info("Comprehensive analysis completed")
        return result
    
    def create_comprehensive_report(self, documents: List[Document]) -> AnalysisReport:
        """
        Create a comprehensive analysis report with insights and recommendations.
        
        Args:
            documents: List of documents to analyze
            
        Returns:
            AnalysisReport with detailed insights and recommendations
        """
        # Perform analysis
        analysis_result = self.analyze_dataset(documents)
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(analysis_result)
        
        # Generate detailed insights
        detailed_insights = self._generate_detailed_insights(analysis_result)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(analysis_result)
        
        return AnalysisReport(
            analysis_result=analysis_result,
            executive_summary=executive_summary,
            detailed_insights=detailed_insights,
            recommendations=recommendations
        )
    
    def _generate_executive_summary(self, result: AnalysisResult) -> Dict[str, Any]:
        """Generate executive summary from analysis results."""
        summary = {
            "dataset_size": result.analysis_metadata.get("document_count", 0),
            "analysis_timestamp": result.analysis_metadata.get("analysis_timestamp")
        }
        
        # Text statistics summary
        if result.text_statistics:
            summary["text_overview"] = {
                "total_words": result.text_statistics.total_words,
                "avg_document_length": round(result.text_statistics.avg_document_length, 1),
                "vocabulary_size": result.text_statistics.vocabulary_size,
                "readability_score": round(result.text_statistics.readability_score, 1)
            }
        
        # Topic modeling summary
        if result.topic_model_result:
            summary["topic_overview"] = {
                "num_topics": result.topic_model_result.num_topics,
                "method": result.topic_model_result.method,
                "coherence_score": round(result.topic_model_result.coherence_score, 3)
            }
        
        # Sentiment summary
        if result.sentiment_analysis:
            polarity = result.sentiment_analysis.overall_polarity
            sentiment_label = "Positive" if polarity > 0.1 else "Negative" if polarity < -0.1 else "Neutral"
            
            summary["sentiment_overview"] = {
                "overall_sentiment": sentiment_label,
                "polarity_score": round(polarity, 3),
                "subjectivity_score": round(result.sentiment_analysis.overall_subjectivity, 3)
            }
        
        # Language summary
        if result.language_distribution:
            summary["language_overview"] = {
                "dominant_language": result.language_distribution.dominant_language,
                "language_count": len(result.language_distribution.language_counts),
                "multilingual_percentage": round(result.language_distribution.multilingual_percentage, 1)
            }
        
        # Quality summary
        if result.quality_report:
            score = result.quality_report.overall_quality_score
            quality_level = "High" if score >= 0.8 else "Medium" if score >= 0.6 else "Low"
            
            summary["quality_overview"] = {
                "overall_quality": quality_level,
                "quality_score": round(score, 3),
                "high_quality_percentage": round(
                    result.quality_report.quality_statistics.get("high_quality_percentage", 0), 1
                )
            }
        
        return summary
    
    def _generate_detailed_insights(self, result: AnalysisResult) -> Dict[str, Any]:
        """Generate detailed insights from analysis results."""
        insights = {}
        
        # Text analysis insights
        if result.text_statistics:
            insights["text_insights"] = {
                "vocabulary_richness": round(result.text_statistics.type_token_ratio, 3),
                "document_length_variance": {
                    "min": result.text_statistics.min_document_length,
                    "max": result.text_statistics.max_document_length,
                    "median": result.text_statistics.median_document_length
                },
                "top_keywords": result.text_statistics.top_keywords[:10]
            }
        
        # Topic modeling insights
        if result.topic_model_result:
            topic_summary = self.topic_modeler.get_topic_summary(result.topic_model_result)
            insights["topic_insights"] = {
                "topic_distribution": topic_summary.get("topic_distribution", {}),
                "most_common_topic": topic_summary.get("most_common_topic"),
                "topics_summary": topic_summary.get("topics_summary", [])[:5]
            }
        
        # Sentiment insights
        if result.sentiment_analysis:
            sentiment_summary = self.sentiment_analyzer.get_sentiment_summary(result.sentiment_analysis)
            insights["sentiment_insights"] = {
                "polarity_distribution": sentiment_summary.get("polarity_distribution", {}),
                "emotion_distribution": sentiment_summary.get("emotion_distribution", {}),
                "dominant_emotion": sentiment_summary.get("dominant_emotion")
            }
        
        # Language insights
        if result.language_distribution:
            language_summary = self.language_analyzer.get_language_summary(result.language_distribution)
            insights["language_insights"] = {
                "top_languages": language_summary.get("top_languages", []),
                "language_diversity": language_summary.get("language_diversity", {}),
                "script_distribution": language_summary.get("script_distribution", {})
            }
        
        # Quality insights
        if result.quality_report:
            quality_summary = self.quality_analyzer.get_quality_summary(result.quality_report)
            insights["quality_insights"] = {
                "grade_distribution": quality_summary.get("grade_distribution", {}),
                "top_issues": quality_summary.get("top_issues", []),
                "quality_statistics": quality_summary.get("quality_statistics", {})
            }
        
        return insights
    
    def _generate_recommendations(self, result: AnalysisResult) -> List[str]:
        """Generate actionable recommendations based on analysis results."""
        recommendations = []
        
        # Text statistics recommendations
        if result.text_statistics:
            if result.text_statistics.type_token_ratio < 0.3:
                recommendations.append("Consider increasing vocabulary diversity in the dataset")
            
            if result.text_statistics.readability_score < 50:
                recommendations.append("Improve text readability through preprocessing or filtering")
        
        # Topic modeling recommendations
        if result.topic_model_result:
            if result.topic_model_result.coherence_score < 0.4:
                recommendations.append("Consider adjusting topic modeling parameters for better coherence")
            
            if result.topic_model_result.num_topics < 3:
                recommendations.append("Dataset may benefit from more diverse content to increase topic variety")
        
        # Sentiment recommendations
        if result.sentiment_analysis:
            polarity_dist = result.sentiment_analysis.polarity_distribution
            total_docs = sum(polarity_dist.values())
            
            if total_docs > 0:
                negative_ratio = polarity_dist.get('negative', 0) / total_docs
                if negative_ratio > 0.6:
                    recommendations.append("High negative sentiment detected - consider content filtering")
                
                neutral_ratio = polarity_dist.get('neutral', 0) / total_docs
                if neutral_ratio > 0.8:
                    recommendations.append("Most content is neutral - consider adding more expressive content")
        
        # Language recommendations
        if result.language_distribution:
            if result.language_distribution.multilingual_percentage > 30:
                recommendations.append("High multilingual content - consider language-specific processing")
            
            if len(result.language_distribution.language_counts) > 10:
                recommendations.append("Many languages detected - consider focusing on primary languages")
        
        # Quality recommendations
        if result.quality_report:
            recommendations.extend(result.quality_report.recommendations)
            
            if result.quality_report.overall_quality_score < 0.6:
                recommendations.append("Overall quality is low - implement stricter quality filters")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Dataset analysis shows good overall characteristics")
        
        return recommendations
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as ISO string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_analysis_summary(self, result: AnalysisResult) -> Dict[str, Any]:
        """
        Get a concise summary of analysis results.
        
        Args:
            result: AnalysisResult to summarize
            
        Returns:
            Dictionary with concise analysis summary
        """
        summary = {
            "document_count": result.analysis_metadata.get("document_count", 0),
            "enabled_analyses": list(result.analysis_metadata.get("enabled_analyses", {}).keys())
        }
        
        # Add component summaries
        if result.text_statistics:
            summary["text_summary"] = {
                "total_words": result.text_statistics.total_words,
                "vocabulary_size": result.text_statistics.vocabulary_size,
                "avg_length": round(result.text_statistics.avg_document_length, 1)
            }
        
        if result.topic_model_result:
            summary["topic_summary"] = {
                "num_topics": result.topic_model_result.num_topics,
                "method": result.topic_model_result.method
            }
        
        if result.sentiment_analysis:
            polarity = result.sentiment_analysis.overall_polarity
            sentiment = "Positive" if polarity > 0.1 else "Negative" if polarity < -0.1 else "Neutral"
            summary["sentiment_summary"] = {
                "overall_sentiment": sentiment,
                "polarity": round(polarity, 3)
            }
        
        if result.language_distribution:
            summary["language_summary"] = {
                "dominant_language": result.language_distribution.dominant_language,
                "language_count": len(result.language_distribution.language_counts)
            }
        
        if result.quality_report:
            score = result.quality_report.overall_quality_score
            quality = "High" if score >= 0.8 else "Medium" if score >= 0.6 else "Low"
            summary["quality_summary"] = {
                "quality_level": quality,
                "score": round(score, 3)
            }
        
        return summary