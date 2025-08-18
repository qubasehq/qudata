"""
Sentiment analysis module for polarity scoring and emotion detection.

This module provides sentiment analysis capabilities including polarity scoring,
emotion detection, and sentiment distribution analysis across document collections.
"""

import re
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
from ..models import Document

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class SentimentScore:
    """Represents sentiment analysis results for text."""
    polarity: float  # -1.0 (negative) to 1.0 (positive)
    subjectivity: float  # 0.0 (objective) to 1.0 (subjective)
    confidence: float  # 0.0 to 1.0
    emotion: str  # Primary detected emotion
    emotion_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "polarity": self.polarity,
            "subjectivity": self.subjectivity,
            "confidence": self.confidence,
            "emotion": self.emotion,
            "emotion_scores": self.emotion_scores
        }


@dataclass
class DocumentSentiment:
    """Sentiment analysis result for a document."""
    document_id: str
    overall_sentiment: SentimentScore
    sentence_sentiments: List[SentimentScore] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "overall_sentiment": self.overall_sentiment.to_dict(),
            "sentence_sentiments": [s.to_dict() for s in self.sentence_sentiments]
        }


@dataclass
class SentimentAnalysis:
    """Complete sentiment analysis results for a document collection."""
    document_sentiments: List[DocumentSentiment]
    overall_polarity: float
    overall_subjectivity: float
    polarity_distribution: Dict[str, int]  # positive, negative, neutral counts
    emotion_distribution: Dict[str, int]
    confidence_stats: Dict[str, float]  # min, max, avg confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_sentiments": [ds.to_dict() for ds in self.document_sentiments],
            "overall_polarity": self.overall_polarity,
            "overall_subjectivity": self.overall_subjectivity,
            "polarity_distribution": self.polarity_distribution,
            "emotion_distribution": self.emotion_distribution,
            "confidence_stats": self.confidence_stats
        }


class SentimentAnalyzer:
    """
    Sentiment analyzer for polarity scoring and emotion detection.
    
    Provides methods for analyzing sentiment in text using lexicon-based approaches
    and optional machine learning models when available.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize sentiment analyzer.
        
        Args:
            config: Configuration dictionary with analysis parameters
        """
        self.config = config or {}
        self.use_sentence_level = self.config.get("sentence_level_analysis", True)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)
        
        # Load sentiment lexicons
        self.positive_words = self._load_positive_words()
        self.negative_words = self._load_negative_words()
        self.emotion_lexicon = self._load_emotion_lexicon()
        self.intensifiers = self._load_intensifiers()
        self.negation_words = self._load_negation_words()
        
        # Check for optional dependencies
        self.textblob_available = self._check_textblob()
        self.vader_available = self._check_vader()
    
    def _check_textblob(self) -> bool:
        """Check if TextBlob is available."""
        try:
            from textblob import TextBlob
            return True
        except ImportError:
            logger.info("TextBlob not available. Using lexicon-based sentiment analysis.")
            return False
    
    def _check_vader(self) -> bool:
        """Check if VADER sentiment is available."""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            return True
        except ImportError:
            logger.info("VADER sentiment not available. Using lexicon-based sentiment analysis.")
            return False
    
    def _load_positive_words(self) -> set:
        """Load positive sentiment words."""
        # Basic positive words - in production, could load from external lexicon
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'awesome', 'brilliant', 'outstanding', 'superb', 'magnificent',
            'perfect', 'beautiful', 'lovely', 'nice', 'pleasant', 'delightful',
            'happy', 'joy', 'love', 'like', 'enjoy', 'pleased', 'satisfied',
            'excited', 'thrilled', 'glad', 'cheerful', 'positive', 'optimistic',
            'success', 'successful', 'win', 'winner', 'victory', 'triumph',
            'best', 'better', 'superior', 'top', 'first', 'leading', 'premium',
            'quality', 'valuable', 'useful', 'helpful', 'beneficial', 'advantage',
            'recommend', 'approve', 'praise', 'compliment', 'congratulate',
            'thank', 'thanks', 'grateful', 'appreciate', 'admire', 'respect'
        }
        
        # Add custom positive words from config
        custom_positive = set(self.config.get("positive_words", []))
        return positive_words.union(custom_positive)
    
    def _load_negative_words(self) -> set:
        """Load negative sentiment words."""
        # Basic negative words - in production, could load from external lexicon
        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate',
            'dislike', 'despise', 'loathe', 'detest', 'abhor', 'reject',
            'refuse', 'deny', 'oppose', 'against', 'wrong', 'incorrect',
            'false', 'fake', 'lie', 'lying', 'dishonest', 'fraud', 'scam',
            'sad', 'unhappy', 'depressed', 'disappointed', 'frustrated',
            'angry', 'mad', 'furious', 'annoyed', 'irritated', 'upset',
            'worried', 'concerned', 'anxious', 'nervous', 'scared', 'afraid',
            'fear', 'panic', 'stress', 'pressure', 'problem', 'issue',
            'trouble', 'difficulty', 'challenge', 'obstacle', 'barrier',
            'fail', 'failure', 'lose', 'loss', 'defeat', 'disaster',
            'crisis', 'emergency', 'danger', 'risk', 'threat', 'warning',
            'worst', 'worse', 'inferior', 'poor', 'low', 'weak', 'broken',
            'damage', 'harm', 'hurt', 'pain', 'suffer', 'victim', 'attack'
        }
        
        # Add custom negative words from config
        custom_negative = set(self.config.get("negative_words", []))
        return negative_words.union(custom_negative)
    
    def _load_emotion_lexicon(self) -> Dict[str, str]:
        """Load emotion lexicon mapping words to emotions."""
        # Basic emotion mapping - in production, could use NRC Emotion Lexicon
        emotion_lexicon = {
            # Joy/Happiness
            'happy': 'joy', 'joy': 'joy', 'cheerful': 'joy', 'excited': 'joy',
            'thrilled': 'joy', 'delighted': 'joy', 'pleased': 'joy', 'glad': 'joy',
            'elated': 'joy', 'euphoric': 'joy', 'blissful': 'joy', 'content': 'joy',
            
            # Sadness
            'sad': 'sadness', 'unhappy': 'sadness', 'depressed': 'sadness',
            'melancholy': 'sadness', 'gloomy': 'sadness', 'miserable': 'sadness',
            'sorrowful': 'sadness', 'mournful': 'sadness', 'dejected': 'sadness',
            
            # Anger
            'angry': 'anger', 'mad': 'anger', 'furious': 'anger', 'rage': 'anger',
            'irritated': 'anger', 'annoyed': 'anger', 'frustrated': 'anger',
            'outraged': 'anger', 'livid': 'anger', 'irate': 'anger',
            
            # Fear
            'afraid': 'fear', 'scared': 'fear', 'frightened': 'fear', 'terrified': 'fear',
            'anxious': 'fear', 'worried': 'fear', 'nervous': 'fear', 'panic': 'fear',
            'alarmed': 'fear', 'apprehensive': 'fear', 'uneasy': 'fear',
            
            # Surprise
            'surprised': 'surprise', 'amazed': 'surprise', 'astonished': 'surprise',
            'shocked': 'surprise', 'stunned': 'surprise', 'bewildered': 'surprise',
            'startled': 'surprise', 'astounded': 'surprise',
            
            # Disgust
            'disgusted': 'disgust', 'revolted': 'disgust', 'repulsed': 'disgust',
            'sickened': 'disgust', 'nauseated': 'disgust', 'appalled': 'disgust',
            
            # Trust
            'trust': 'trust', 'confident': 'trust', 'secure': 'trust', 'safe': 'trust',
            'reliable': 'trust', 'dependable': 'trust', 'faithful': 'trust',
            
            # Anticipation
            'excited': 'anticipation', 'eager': 'anticipation', 'hopeful': 'anticipation',
            'optimistic': 'anticipation', 'expectant': 'anticipation'
        }
        
        return emotion_lexicon
    
    def _load_intensifiers(self) -> Dict[str, float]:
        """Load intensifier words and their multipliers."""
        return {
            'very': 1.5, 'extremely': 2.0, 'incredibly': 2.0, 'absolutely': 2.0,
            'completely': 1.8, 'totally': 1.8, 'really': 1.3, 'quite': 1.2,
            'rather': 1.1, 'somewhat': 0.8, 'slightly': 0.7, 'barely': 0.5,
            'hardly': 0.5, 'scarcely': 0.5, 'almost': 0.9, 'nearly': 0.9
        }
    
    def _load_negation_words(self) -> set:
        """Load negation words."""
        return {
            'not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither',
            'nor', 'none', 'cannot', 'cant', 'couldnt', 'shouldnt', 'wouldnt',
            'dont', 'doesnt', 'didnt', 'isnt', 'arent', 'wasnt', 'werent',
            'havent', 'hasnt', 'hadnt', 'wont', 'without', 'lack', 'lacking'
        }
    
    def analyze_sentiment(self, documents: List[Document]) -> SentimentAnalysis:
        """
        Analyze sentiment for a collection of documents.
        
        Args:
            documents: List of documents to analyze
            
        Returns:
            SentimentAnalysis with comprehensive sentiment results
        """
        if not documents:
            return self._empty_analysis()
        
        document_sentiments = []
        all_polarities = []
        all_subjectivities = []
        all_confidences = []
        polarity_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        emotion_counts = defaultdict(int)
        
        for doc in documents:
            doc_sentiment = self.analyze_document_sentiment(doc)
            document_sentiments.append(doc_sentiment)
            
            # Collect statistics
            polarity = doc_sentiment.overall_sentiment.polarity
            subjectivity = doc_sentiment.overall_sentiment.subjectivity
            confidence = doc_sentiment.overall_sentiment.confidence
            emotion = doc_sentiment.overall_sentiment.emotion
            
            all_polarities.append(polarity)
            all_subjectivities.append(subjectivity)
            all_confidences.append(confidence)
            
            # Categorize polarity
            if polarity > 0.1:
                polarity_counts['positive'] += 1
            elif polarity < -0.1:
                polarity_counts['negative'] += 1
            else:
                polarity_counts['neutral'] += 1
            
            emotion_counts[emotion] += 1
        
        # Calculate overall statistics
        overall_polarity = sum(all_polarities) / len(all_polarities)
        overall_subjectivity = sum(all_subjectivities) / len(all_subjectivities)
        
        confidence_stats = {
            'min': min(all_confidences),
            'max': max(all_confidences),
            'avg': sum(all_confidences) / len(all_confidences)
        }
        
        return SentimentAnalysis(
            document_sentiments=document_sentiments,
            overall_polarity=overall_polarity,
            overall_subjectivity=overall_subjectivity,
            polarity_distribution=polarity_counts,
            emotion_distribution=dict(emotion_counts),
            confidence_stats=confidence_stats
        )
    
    def analyze_document_sentiment(self, document: Document) -> DocumentSentiment:
        """
        Analyze sentiment for a single document.
        
        Args:
            document: Document to analyze
            
        Returns:
            DocumentSentiment with document-level and sentence-level results
        """
        # Analyze overall document sentiment
        overall_sentiment = self._analyze_text_sentiment(document.content)
        
        # Analyze sentence-level sentiment if enabled
        sentence_sentiments = []
        if self.use_sentence_level:
            sentences = self._split_sentences(document.content)
            for sentence in sentences:
                if sentence.strip():
                    sent_sentiment = self._analyze_text_sentiment(sentence)
                    sentence_sentiments.append(sent_sentiment)
        
        return DocumentSentiment(
            document_id=document.id,
            overall_sentiment=overall_sentiment,
            sentence_sentiments=sentence_sentiments
        )
    
    def _analyze_text_sentiment(self, text: str) -> SentimentScore:
        """
        Analyze sentiment for a piece of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            SentimentScore with polarity, subjectivity, and emotion
        """
        # Try advanced methods first if available
        if self.vader_available:
            return self._analyze_with_vader(text)
        elif self.textblob_available:
            return self._analyze_with_textblob(text)
        else:
            return self._analyze_with_lexicon(text)
    
    def _analyze_with_vader(self, text: str) -> SentimentScore:
        """Analyze sentiment using VADER."""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            
            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores(text)
            
            # VADER returns compound score (-1 to 1), positive, negative, neutral
            polarity = scores['compound']
            
            # Estimate subjectivity (VADER doesn't provide this directly)
            subjectivity = 1.0 - scores['neu']  # Less neutral = more subjective
            
            # Confidence based on the strength of the compound score
            confidence = abs(polarity)
            
            # Determine primary emotion
            emotion = self._determine_emotion_from_scores(scores)
            
            return SentimentScore(
                polarity=polarity,
                subjectivity=subjectivity,
                confidence=confidence,
                emotion=emotion,
                emotion_scores={
                    'positive': scores['pos'],
                    'negative': scores['neg'],
                    'neutral': scores['neu']
                }
            )
            
        except Exception as e:
            logger.error(f"Error with VADER analysis: {e}")
            return self._analyze_with_lexicon(text)
    
    def _analyze_with_textblob(self, text: str) -> SentimentScore:
        """Analyze sentiment using TextBlob."""
        try:
            from textblob import TextBlob
            
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Confidence based on absolute polarity and subjectivity
            confidence = (abs(polarity) + subjectivity) / 2
            
            # Determine emotion from polarity and text analysis
            emotion = self._determine_emotion_from_text(text, polarity)
            
            return SentimentScore(
                polarity=polarity,
                subjectivity=subjectivity,
                confidence=confidence,
                emotion=emotion
            )
            
        except Exception as e:
            logger.error(f"Error with TextBlob analysis: {e}")
            return self._analyze_with_lexicon(text)
    
    def _analyze_with_lexicon(self, text: str) -> SentimentScore:
        """Analyze sentiment using lexicon-based approach."""
        words = self._tokenize_text(text)
        
        positive_score = 0.0
        negative_score = 0.0
        emotion_scores = defaultdict(float)
        total_words = len(words)
        
        i = 0
        while i < len(words):
            word = words[i].lower()
            
            # Check for negation in previous words
            negation_multiplier = 1.0
            if i > 0 and words[i-1].lower() in self.negation_words:
                negation_multiplier = -0.8
            elif i > 1 and words[i-2].lower() in self.negation_words:
                negation_multiplier = -0.6
            
            # Check for intensifiers
            intensifier_multiplier = 1.0
            if i > 0 and words[i-1].lower() in self.intensifiers:
                intensifier_multiplier = self.intensifiers[words[i-1].lower()]
            
            # Calculate sentiment score
            word_score = 0.0
            if word in self.positive_words:
                word_score = 1.0
            elif word in self.negative_words:
                word_score = -1.0
            
            # Apply modifiers
            final_score = word_score * intensifier_multiplier * negation_multiplier
            
            if final_score > 0:
                positive_score += final_score
            elif final_score < 0:
                negative_score += abs(final_score)
            
            # Track emotions
            if word in self.emotion_lexicon:
                emotion = self.emotion_lexicon[word]
                emotion_scores[emotion] += abs(final_score)
            
            i += 1
        
        # Normalize scores
        if total_words > 0:
            positive_score /= total_words
            negative_score /= total_words
        
        # Calculate polarity (-1 to 1)
        polarity = positive_score - negative_score
        
        # Calculate subjectivity (0 to 1)
        subjectivity = min(1.0, positive_score + negative_score)
        
        # Calculate confidence
        confidence = min(1.0, abs(polarity) + subjectivity * 0.5)
        
        # Determine primary emotion
        primary_emotion = 'neutral'
        if emotion_scores:
            primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        elif polarity > 0.1:
            primary_emotion = 'joy'
        elif polarity < -0.1:
            primary_emotion = 'sadness'
        
        return SentimentScore(
            polarity=polarity,
            subjectivity=subjectivity,
            confidence=confidence,
            emotion=primary_emotion,
            emotion_scores=dict(emotion_scores)
        )
    
    def _determine_emotion_from_scores(self, vader_scores: Dict[str, float]) -> str:
        """Determine primary emotion from VADER scores."""
        if vader_scores['compound'] >= 0.5:
            return 'joy'
        elif vader_scores['compound'] <= -0.5:
            return 'sadness'
        elif vader_scores['neg'] > vader_scores['pos']:
            return 'anger'
        elif vader_scores['pos'] > 0.3:
            return 'joy'
        else:
            return 'neutral'
    
    def _determine_emotion_from_text(self, text: str, polarity: float) -> str:
        """Determine emotion from text analysis and polarity."""
        words = self._tokenize_text(text)
        emotion_counts = defaultdict(int)
        
        for word in words:
            word_lower = word.lower()
            if word_lower in self.emotion_lexicon:
                emotion = self.emotion_lexicon[word_lower]
                emotion_counts[emotion] += 1
        
        if emotion_counts:
            return max(emotion_counts.items(), key=lambda x: x[1])[0]
        elif polarity > 0.1:
            return 'joy'
        elif polarity < -0.1:
            return 'sadness'
        else:
            return 'neutral'
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple tokenization - in production, could use NLTK
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - in production, could use NLTK
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _empty_analysis(self) -> SentimentAnalysis:
        """Return empty analysis for edge cases."""
        return SentimentAnalysis(
            document_sentiments=[],
            overall_polarity=0.0,
            overall_subjectivity=0.0,
            polarity_distribution={'positive': 0, 'negative': 0, 'neutral': 0},
            emotion_distribution={},
            confidence_stats={'min': 0.0, 'max': 0.0, 'avg': 0.0}
        )
    
    def get_sentiment_summary(self, analysis: SentimentAnalysis) -> Dict[str, Any]:
        """
        Generate a summary of sentiment analysis results.
        
        Args:
            analysis: SentimentAnalysis to summarize
            
        Returns:
            Dictionary with sentiment analysis summary
        """
        if not analysis.document_sentiments:
            return {"message": "No sentiment data available"}
        
        total_docs = len(analysis.document_sentiments)
        
        # Calculate percentages
        polarity_percentages = {
            category: (count / total_docs) * 100
            for category, count in analysis.polarity_distribution.items()
        }
        
        # Find dominant emotion
        dominant_emotion = None
        if analysis.emotion_distribution:
            dominant_emotion = max(analysis.emotion_distribution.items(), key=lambda x: x[1])
        
        # Sentiment interpretation
        if analysis.overall_polarity > 0.1:
            overall_sentiment = "Positive"
        elif analysis.overall_polarity < -0.1:
            overall_sentiment = "Negative"
        else:
            overall_sentiment = "Neutral"
        
        return {
            "total_documents": total_docs,
            "overall_sentiment": overall_sentiment,
            "overall_polarity": round(analysis.overall_polarity, 3),
            "overall_subjectivity": round(analysis.overall_subjectivity, 3),
            "polarity_distribution": analysis.polarity_distribution,
            "polarity_percentages": {k: round(v, 1) for k, v in polarity_percentages.items()},
            "emotion_distribution": analysis.emotion_distribution,
            "dominant_emotion": {
                "emotion": dominant_emotion[0],
                "count": dominant_emotion[1],
                "percentage": round((dominant_emotion[1] / total_docs) * 100, 1)
            } if dominant_emotion else None,
            "confidence_stats": {
                k: round(v, 3) for k, v in analysis.confidence_stats.items()
            }
        }