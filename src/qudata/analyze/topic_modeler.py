"""
Topic modeling module using LDA and BERTopic for topic discovery.

This module provides topic modeling capabilities to discover latent topics
in document collections using both traditional LDA and modern BERTopic approaches.
"""

import re
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Set
from ..models import Document

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class Topic:
    """Represents a discovered topic."""
    id: int
    name: str
    keywords: List[Tuple[str, float]]  # (word, weight) pairs
    document_count: int
    coherence_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert topic to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "keywords": self.keywords,
            "document_count": self.document_count,
            "coherence_score": self.coherence_score
        }


@dataclass
class DocumentTopic:
    """Represents topic assignment for a document."""
    document_id: str
    topic_id: int
    probability: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "topic_id": self.topic_id,
            "probability": self.probability
        }


@dataclass
class TopicModelResult:
    """Result of topic modeling analysis."""
    topics: List[Topic]
    document_topics: List[DocumentTopic]
    method: str
    num_topics: int
    coherence_score: float
    perplexity: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "topics": [topic.to_dict() for topic in self.topics],
            "document_topics": [dt.to_dict() for dt in self.document_topics],
            "method": self.method,
            "num_topics": self.num_topics,
            "coherence_score": self.coherence_score,
            "perplexity": self.perplexity
        }


class TopicModeler:
    """
    Topic modeling engine supporting LDA and BERTopic methods.
    
    Provides methods for discovering topics in document collections using
    various algorithms and preprocessing techniques.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize topic modeler.
        
        Args:
            config: Configuration dictionary with modeling parameters
        """
        self.config = config or {}
        self.min_word_length = self.config.get("min_word_length", 3)
        self.max_word_length = self.config.get("max_word_length", 50)
        self.min_document_frequency = self.config.get("min_document_frequency", 2)
        self.max_document_frequency = self.config.get("max_document_frequency", 0.8)
        self.stopwords = self._load_stopwords()
        
        # Try to import optional dependencies
        self.sklearn_available = self._check_sklearn()
        self.bertopic_available = self._check_bertopic()
        
    def _check_sklearn(self) -> bool:
        """Check if scikit-learn is available."""
        try:
            import sklearn
            return True
        except ImportError:
            logger.warning("scikit-learn not available. LDA modeling will be limited.")
            return False
    
    def _check_bertopic(self) -> bool:
        """Check if BERTopic is available."""
        try:
            import bertopic
            return True
        except ImportError:
            logger.warning("BERTopic not available. Advanced topic modeling will be limited.")
            return False
    
    def _load_stopwords(self) -> Set[str]:
        """Load stopwords for preprocessing."""
        # Basic English stopwords
        basic_stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'would', 'you', 'your', 'have', 'had',
            'this', 'they', 'them', 'their', 'there', 'then', 'than', 'these',
            'those', 'when', 'where', 'who', 'what', 'why', 'how', 'can', 'could',
            'should', 'would', 'may', 'might', 'must', 'shall', 'will', 'do', 'did',
            'does', 'done', 'been', 'being', 'am', 'is', 'are', 'was', 'were',
            'also', 'but', 'or', 'so', 'if', 'no', 'not', 'only', 'own', 'same',
            'such', 'than', 'too', 'very', 'can', 'will', 'just', 'don', 'should',
            'now', 'use', 'used', 'using', 'each', 'which', 'she', 'do', 'get',
            'all', 'any', 'been', 'had', 'her', 'his', 'how', 'man', 'new',
            'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its',
            'let', 'put', 'say', 'she', 'too', 'use'
        }
        
        # Allow custom stopwords from config
        custom_stopwords = set(self.config.get("stopwords", []))
        return basic_stopwords.union(custom_stopwords)
    
    def perform_topic_modeling(self, documents: List[Document], method: str = "lda", 
                             num_topics: int = None) -> TopicModelResult:
        """
        Perform topic modeling on documents using specified method.
        
        Args:
            documents: List of documents to analyze
            method: Modeling method ("lda", "bertopic", "simple")
            num_topics: Number of topics to discover (auto-detected if None)
            
        Returns:
            TopicModelResult with discovered topics and assignments
        """
        if not documents:
            return self._empty_result(method)
        
        # Auto-detect number of topics if not specified
        if num_topics is None:
            num_topics = self._estimate_num_topics(len(documents))
        
        if method == "lda" and self.sklearn_available:
            return self._perform_lda_modeling(documents, num_topics)
        elif method == "bertopic" and self.bertopic_available:
            return self._perform_bertopic_modeling(documents, num_topics)
        else:
            # Fallback to simple keyword-based clustering
            logger.info(f"Using simple topic modeling (method '{method}' not available)")
            return self._perform_simple_modeling(documents, num_topics)
    
    def _perform_lda_modeling(self, documents: List[Document], num_topics: int) -> TopicModelResult:
        """Perform LDA topic modeling using scikit-learn."""
        try:
            from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
            from sklearn.decomposition import LatentDirichletAllocation
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
        except ImportError:
            logger.error("scikit-learn not available for LDA modeling")
            return self._perform_simple_modeling(documents, num_topics)
        
        # Preprocess documents
        processed_docs = [self._preprocess_text(doc.content) for doc in documents]
        
        # Create document-term matrix
        vectorizer = CountVectorizer(
            max_features=self.config.get("max_features", 1000),
            min_df=self.min_document_frequency,
            max_df=self.max_document_frequency,
            stop_words=list(self.stopwords),
            ngram_range=(1, self.config.get("max_ngram", 2))
        )
        
        try:
            doc_term_matrix = vectorizer.fit_transform(processed_docs)
            feature_names = vectorizer.get_feature_names_out()
        except ValueError as e:
            logger.error(f"Error creating document-term matrix: {e}")
            return self._perform_simple_modeling(documents, num_topics)
        
        # Fit LDA model
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            max_iter=self.config.get("max_iter", 100),
            learning_method='batch'
        )
        
        try:
            lda.fit(doc_term_matrix)
            doc_topic_probs = lda.transform(doc_term_matrix)
        except Exception as e:
            logger.error(f"Error fitting LDA model: {e}")
            return self._perform_simple_modeling(documents, num_topics)
        
        # Extract topics
        topics = []
        words_per_topic = self.config.get("words_per_topic", 10)
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-words_per_topic:][::-1]
            keywords = [(feature_names[i], float(topic[i])) for i in top_words_idx]
            
            # Count documents assigned to this topic
            doc_count = sum(1 for probs in doc_topic_probs if np.argmax(probs) == topic_idx)
            
            # Generate topic name from top keywords
            topic_name = f"Topic_{topic_idx}: {', '.join([kw[0] for kw in keywords[:3]])}"
            
            topics.append(Topic(
                id=topic_idx,
                name=topic_name,
                keywords=keywords,
                document_count=doc_count,
                coherence_score=0.0  # Could implement coherence calculation
            ))
        
        # Create document-topic assignments
        document_topics = []
        for doc_idx, (doc, probs) in enumerate(zip(documents, doc_topic_probs)):
            best_topic = int(np.argmax(probs))
            probability = float(probs[best_topic])
            
            document_topics.append(DocumentTopic(
                document_id=doc.id,
                topic_id=best_topic,
                probability=probability
            ))
        
        # Calculate perplexity
        try:
            perplexity = lda.perplexity(doc_term_matrix)
        except:
            perplexity = 0.0
        
        return TopicModelResult(
            topics=topics,
            document_topics=document_topics,
            method="lda",
            num_topics=num_topics,
            coherence_score=0.0,  # Could implement coherence calculation
            perplexity=perplexity
        )
    
    def _perform_bertopic_modeling(self, documents: List[Document], num_topics: int) -> TopicModelResult:
        """Perform BERTopic modeling."""
        try:
            from bertopic import BERTopic
            from sklearn.feature_extraction.text import CountVectorizer
        except ImportError:
            logger.error("BERTopic not available")
            return self._perform_simple_modeling(documents, num_topics)
        
        # Preprocess documents
        processed_docs = [self._preprocess_text(doc.content) for doc in documents]
        
        # Configure BERTopic
        vectorizer_model = CountVectorizer(
            stop_words=list(self.stopwords),
            min_df=self.min_document_frequency,
            ngram_range=(1, 2)
        )
        
        try:
            # Initialize BERTopic model
            topic_model = BERTopic(
                nr_topics=num_topics,
                vectorizer_model=vectorizer_model,
                verbose=False
            )
            
            # Fit model and get topics
            topics_per_doc, probabilities = topic_model.fit_transform(processed_docs)
            
        except Exception as e:
            logger.error(f"Error with BERTopic modeling: {e}")
            return self._perform_simple_modeling(documents, num_topics)
        
        # Extract topic information
        topics = []
        topic_info = topic_model.get_topic_info()
        
        for _, row in topic_info.iterrows():
            topic_id = int(row['Topic'])
            if topic_id == -1:  # Skip outlier topic
                continue
                
            topic_words = topic_model.get_topic(topic_id)
            keywords = [(word, float(score)) for word, score in topic_words[:10]]
            
            # Generate topic name
            top_words = [kw[0] for kw in keywords[:3]]
            topic_name = f"Topic_{topic_id}: {', '.join(top_words)}"
            
            topics.append(Topic(
                id=topic_id,
                name=topic_name,
                keywords=keywords,
                document_count=int(row['Count']),
                coherence_score=0.0
            ))
        
        # Create document-topic assignments
        document_topics = []
        for doc_idx, (doc, topic_id) in enumerate(zip(documents, topics_per_doc)):
            if topic_id != -1:  # Skip outliers
                probability = probabilities[doc_idx] if probabilities is not None else 1.0
                document_topics.append(DocumentTopic(
                    document_id=doc.id,
                    topic_id=topic_id,
                    probability=float(probability)
                ))
        
        return TopicModelResult(
            topics=topics,
            document_topics=document_topics,
            method="bertopic",
            num_topics=len(topics),
            coherence_score=0.0,
            perplexity=0.0
        )
    
    def _perform_simple_modeling(self, documents: List[Document], num_topics: int) -> TopicModelResult:
        """
        Perform simple keyword-based topic clustering as fallback.
        
        This method uses TF-IDF and simple clustering to identify topics
        when advanced libraries are not available.
        """
        # Extract keywords from all documents
        all_keywords = []
        doc_keywords = []
        
        for doc in documents:
            keywords = self._extract_document_keywords(doc.content)
            doc_keywords.append(keywords)
            all_keywords.extend(keywords)
        
        # Find most common keywords
        keyword_freq = Counter(all_keywords)
        top_keywords = [kw for kw, freq in keyword_freq.most_common(num_topics * 5)]
        
        # Create simple topics based on keyword co-occurrence
        topics = []
        document_topics = []
        
        # Group keywords into topics
        keywords_per_topic = max(1, len(top_keywords) // num_topics)
        
        for topic_id in range(num_topics):
            start_idx = topic_id * keywords_per_topic
            end_idx = min(start_idx + keywords_per_topic, len(top_keywords))
            
            if start_idx >= len(top_keywords):
                break
                
            topic_keywords = top_keywords[start_idx:end_idx]
            
            # Calculate keyword weights (simple frequency-based)
            keywords_with_weights = []
            for kw in topic_keywords:
                weight = keyword_freq[kw] / len(documents)
                keywords_with_weights.append((kw, weight))
            
            # Generate topic name
            topic_name = f"Topic_{topic_id}: {', '.join(topic_keywords[:3])}"
            
            # Count documents containing these keywords
            doc_count = 0
            for doc_idx, doc in enumerate(documents):
                doc_kw = doc_keywords[doc_idx]
                if any(kw in doc_kw for kw in topic_keywords):
                    doc_count += 1
                    
                    # Assign document to topic with highest keyword overlap
                    overlap = len(set(topic_keywords) & set(doc_kw))
                    if overlap > 0:
                        probability = overlap / len(topic_keywords)
                        document_topics.append(DocumentTopic(
                            document_id=doc.id,
                            topic_id=topic_id,
                            probability=probability
                        ))
            
            topics.append(Topic(
                id=topic_id,
                name=topic_name,
                keywords=keywords_with_weights,
                document_count=doc_count,
                coherence_score=0.0
            ))
        
        return TopicModelResult(
            topics=topics,
            document_topics=document_topics,
            method="simple",
            num_topics=len(topics),
            coherence_score=0.0,
            perplexity=0.0
        )
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for topic modeling."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and filter words
        words = text.split()
        filtered_words = [
            word for word in words
            if (len(word) >= self.min_word_length and 
                len(word) <= self.max_word_length and
                word not in self.stopwords)
        ]
        
        return ' '.join(filtered_words)
    
    def _extract_document_keywords(self, text: str) -> List[str]:
        """Extract keywords from a single document."""
        processed_text = self._preprocess_text(text)
        words = processed_text.split()
        
        # Simple keyword extraction based on word frequency and length
        word_freq = Counter(words)
        
        # Filter keywords
        keywords = []
        for word, freq in word_freq.items():
            if (freq >= 2 and  # Appears at least twice
                len(word) >= self.min_word_length and
                word not in self.stopwords):
                keywords.append(word)
        
        return keywords
    
    def _estimate_num_topics(self, num_documents: int) -> int:
        """Estimate optimal number of topics based on document count."""
        if num_documents < 10:
            return 2
        elif num_documents < 50:
            return min(5, num_documents // 5)
        elif num_documents < 200:
            return min(10, num_documents // 10)
        else:
            return min(20, num_documents // 20)
    
    def _empty_result(self, method: str) -> TopicModelResult:
        """Return empty result for edge cases."""
        return TopicModelResult(
            topics=[],
            document_topics=[],
            method=method,
            num_topics=0,
            coherence_score=0.0,
            perplexity=0.0
        )
    
    def get_topic_summary(self, result: TopicModelResult) -> Dict[str, Any]:
        """
        Generate a summary of topic modeling results.
        
        Args:
            result: TopicModelResult to summarize
            
        Returns:
            Dictionary with topic modeling summary
        """
        if not result.topics:
            return {"message": "No topics found"}
        
        # Calculate topic distribution
        topic_distribution = {}
        for dt in result.document_topics:
            topic_id = dt.topic_id
            if topic_id not in topic_distribution:
                topic_distribution[topic_id] = 0
            topic_distribution[topic_id] += 1
        
        # Find most and least common topics
        sorted_topics = sorted(topic_distribution.items(), key=lambda x: x[1], reverse=True)
        most_common_topic = sorted_topics[0] if sorted_topics else None
        least_common_topic = sorted_topics[-1] if sorted_topics else None
        
        return {
            "method": result.method,
            "num_topics": result.num_topics,
            "total_documents": len(result.document_topics),
            "coherence_score": result.coherence_score,
            "perplexity": result.perplexity,
            "topic_distribution": topic_distribution,
            "most_common_topic": {
                "id": most_common_topic[0],
                "document_count": most_common_topic[1]
            } if most_common_topic else None,
            "least_common_topic": {
                "id": least_common_topic[0], 
                "document_count": least_common_topic[1]
            } if least_common_topic else None,
            "topics_summary": [
                {
                    "id": topic.id,
                    "name": topic.name,
                    "top_keywords": topic.keywords[:5],
                    "document_count": topic.document_count
                }
                for topic in result.topics
            ]
        }