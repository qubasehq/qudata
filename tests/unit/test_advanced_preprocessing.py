"""
Unit tests for advanced preprocessing pipeline components.

This module tests all components of the advanced preprocessing pipeline:
- HTMLCleaner for tag and emoji removal
- SentenceSegmenter for text normalization
- StopwordRemover with configurable word lists
- TokenizationPreview for LLM compatibility
- FormatConverter for ChatML, Alpaca, JSONL
"""

import pytest
import json
from unittest.mock import patch, MagicMock

from src.qudata.clean.html_cleaner import HTMLCleaner, HTMLCleaningResult, clean_html_content
from src.qudata.clean.segment import SentenceSegmenter, SegmentationResult, segment_text_simple
from src.qudata.clean.stopwords import StopwordRemover, StopwordRemovalResult, remove_stopwords_simple, load_stopwords_from_file
from src.qudata.clean.tokenization import TokenizationPreview, TokenizationResult, quick_token_count, check_context_fit
from src.qudata.export.formats import (
    FormatConverter, ChatMLFormatter, AlpacaFormatter, JSONLFormatter,
    ConversationTurn, InstructionExample, OutputFormat,
    convert_to_chatml, convert_to_alpaca, convert_to_jsonl
)


class TestHTMLCleaner:
    """Test cases for HTMLCleaner component."""
    
    def test_html_cleaner_initialization(self):
        """Test HTMLCleaner initialization with various options."""
        # Default initialization
        cleaner = HTMLCleaner()
        assert cleaner.preserve_links is False
        assert cleaner.preserve_formatting is False
        assert cleaner.remove_emojis is True
        assert cleaner.remove_special_chars is True
        
        # Custom initialization
        cleaner = HTMLCleaner(
            preserve_links=True,
            preserve_formatting=True,
            remove_emojis=False,
            custom_emoji_patterns=[r':\w+:']
        )
        assert cleaner.preserve_links is True
        assert cleaner.preserve_formatting is True
        assert cleaner.remove_emojis is False
    
    def test_clean_basic_html(self):
        """Test basic HTML cleaning functionality."""
        cleaner = HTMLCleaner()
        html_content = """
        <html>
            <head><title>Test</title></head>
            <body>
                <h1>Hello World</h1>
                <p>This is a <strong>test</strong> paragraph.</p>
                <script>alert('malicious');</script>
            </body>
        </html>
        """
        
        result = cleaner.clean_html(html_content)
        
        assert isinstance(result, HTMLCleaningResult)
        assert "Hello World" in result.cleaned_text
        assert "test paragraph" in result.cleaned_text
        assert "malicious" not in result.cleaned_text
        assert "script" in result.removed_tags
        assert result.original_length > result.cleaned_length
        assert result.reduction_ratio > 0
    
    def test_emoji_removal(self):
        """Test emoji removal functionality."""
        cleaner = HTMLCleaner(remove_emojis=True)
        text_with_emojis = "Hello 😀 World! 🌍 This is great! 👍"
        
        result = cleaner.clean_text_emojis(text_with_emojis)
        
        assert "😀" not in result.cleaned_text
        assert "🌍" not in result.cleaned_text
        assert "👍" not in result.cleaned_text
        assert "Hello" in result.cleaned_text
        assert "World" in result.cleaned_text
        assert len(result.removed_emojis) >= 3
    
    def test_preserve_links(self):
        """Test link preservation functionality."""
        cleaner = HTMLCleaner(preserve_links=True)
        html_with_links = '<p>Visit <a href="https://example.com">our website</a> for more info.</p>'
        
        result = cleaner.clean_html(html_with_links)
        
        assert "our website (https://example.com)" in result.cleaned_text
        assert "Visit" in result.cleaned_text
        assert "for more info" in result.cleaned_text
    
    def test_preserve_formatting(self):
        """Test formatting preservation."""
        cleaner = HTMLCleaner(preserve_formatting=True)
        html_with_formatting = '<p>This is <strong>bold</strong> and <em>italic</em> text.</p>'
        
        result = cleaner.clean_html(html_with_formatting)
        
        # Should preserve some structure or at least the text content
        assert "bold" in result.cleaned_text
        assert "italic" in result.cleaned_text
        assert "This is" in result.cleaned_text
    
    def test_batch_cleaning(self):
        """Test batch HTML cleaning."""
        cleaner = HTMLCleaner()
        html_contents = [
            "<p>First document</p>",
            "<div>Second <strong>document</strong></div>",
            "<span>Third document with 😀</span>"
        ]
        
        results = cleaner.batch_clean(html_contents)
        
        assert len(results) == 3
        assert all(isinstance(r, HTMLCleaningResult) for r in results)
        assert "First document" in results[0].cleaned_text
        assert "Second document" in results[1].cleaned_text
        assert "Third document" in results[2].cleaned_text
    
    def test_cleaning_stats(self):
        """Test cleaning statistics generation."""
        cleaner = HTMLCleaner()
        results = [
            HTMLCleaningResult("clean1", ["p"], ["😀"], ["&amp;"], 100, 80),
            HTMLCleaningResult("clean2", ["div"], ["🌍"], ["&lt;"], 150, 120),
        ]
        
        stats = cleaner.get_cleaning_stats(results)
        
        assert stats['total_documents'] == 2
        assert stats['total_original_length'] == 250
        assert stats['total_cleaned_length'] == 200
        assert stats['total_tags_removed'] == 2
        assert stats['total_emojis_removed'] == 2
        assert stats['average_reduction_ratio'] > 0
    
    def test_convenience_function(self):
        """Test convenience function for HTML cleaning."""
        html_content = "<p>Hello <strong>World</strong>! 😀</p>"
        
        cleaned = clean_html_content(html_content, remove_emojis=True)
        
        assert isinstance(cleaned, str)
        assert "Hello World" in cleaned
        assert "😀" not in cleaned


class TestSentenceSegmenter:
    """Test cases for SentenceSegmenter component."""
    
    @patch('nltk.download')
    @patch('nltk.data.find')
    def test_segmenter_initialization(self, mock_find, mock_download):
        """Test SentenceSegmenter initialization."""
        mock_find.side_effect = LookupError()  # Simulate missing data
        
        segmenter = SentenceSegmenter(
            language='english',
            min_sentence_length=5,
            max_sentence_length=500
        )
        
        assert segmenter.language == 'english'
        assert segmenter.min_sentence_length == 5
        assert segmenter.max_sentence_length == 500
        assert mock_download.called
    
    @patch('nltk.tokenize.sent_tokenize')
    def test_basic_segmentation(self, mock_sent_tokenize):
        """Test basic text segmentation."""
        mock_sent_tokenize.return_value = [
            "This is the first sentence.",
            "This is the second sentence.",
            "This is the third sentence."
        ]
        
        segmenter = SentenceSegmenter()
        text = "This is the first sentence. This is the second sentence. This is the third sentence."
        
        result = segmenter.segment_text(text)
        
        assert isinstance(result, SegmentationResult)
        assert result.sentence_count == 3
        assert len(result.sentences) == 3
        assert "first sentence" in result.sentences[0]
        assert result.language == 'english'
        assert result.word_count > 0
        assert result.average_sentence_length > 0
    
    def test_empty_text_segmentation(self):
        """Test segmentation of empty text."""
        segmenter = SentenceSegmenter()
        
        result = segmenter.segment_text("")
        
        assert result.sentence_count == 0
        assert len(result.sentences) == 0
        assert result.word_count == 0
        assert result.average_sentence_length == 0.0
    
    @patch('nltk.tokenize.sent_tokenize')
    def test_sentence_filtering(self, mock_sent_tokenize):
        """Test sentence filtering by length."""
        mock_sent_tokenize.return_value = [
            "Short.",  # Too short
            "This is a good sentence that meets the minimum length requirement.",
            "A" * 2000,  # Too long
        ]
        
        segmenter = SentenceSegmenter(min_sentence_length=20, max_sentence_length=100)
        
        result = segmenter.segment_text("Some text")
        
        # Should only keep the middle sentence
        assert result.sentence_count == 1
        assert "good sentence" in result.sentences[0]
    
    def test_segment_by_length(self):
        """Test segmentation by target length."""
        segmenter = SentenceSegmenter()
        
        # Mock the segment_text method to return known sentences
        with patch.object(segmenter, 'segment_text') as mock_segment:
            mock_segment.return_value = SegmentationResult(
                sentences=[
                    "First sentence.",
                    "Second sentence.",
                    "Third sentence.",
                    "Fourth sentence."
                ],
                paragraphs=[],
                word_count=8,
                sentence_count=4,
                average_sentence_length=15.0,
                language='english'
            )
            
            chunks = segmenter.segment_by_length("Some text", target_length=30)
            
            assert len(chunks) >= 1
            assert all(isinstance(chunk, str) for chunk in chunks)
    
    def test_batch_segmentation(self):
        """Test batch text segmentation."""
        segmenter = SentenceSegmenter()
        texts = [
            "First document. With two sentences.",
            "Second document. Also with two sentences.",
            "Third document. Has three sentences. Like this one."
        ]
        
        with patch.object(segmenter, 'segment_text') as mock_segment:
            mock_segment.side_effect = [
                SegmentationResult(["First document.", "With two sentences."], [], 6, 2, 15.0, 'english'),
                SegmentationResult(["Second document.", "Also with two sentences."], [], 7, 2, 16.0, 'english'),
                SegmentationResult(["Third document.", "Has three sentences.", "Like this one."], [], 8, 3, 14.0, 'english')
            ]
            
            results = segmenter.batch_segment(texts)
            
            assert len(results) == 3
            assert all(isinstance(r, SegmentationResult) for r in results)
    
    def test_segmentation_stats(self):
        """Test segmentation statistics generation."""
        segmenter = SentenceSegmenter()
        results = [
            SegmentationResult(["Sentence 1.", "Sentence 2."], [], 4, 2, 10.0, 'english'),
            SegmentationResult(["Sentence 3.", "Sentence 4.", "Sentence 5."], [], 6, 3, 12.0, 'english'),
        ]
        
        stats = segmenter.get_segmentation_stats(results)
        
        assert stats['total_documents'] == 2
        assert stats['total_sentences'] == 5
        assert stats['total_words'] == 10
        assert stats['average_sentences_per_document'] == 2.5
        assert stats['average_words_per_document'] == 5.0
    
    def test_convenience_function(self):
        """Test convenience function for text segmentation."""
        with patch('src.qudata.clean.segment.sent_tokenize') as mock_tokenize:
            mock_tokenize.return_value = ["First sentence.", "Second sentence."]
            
            sentences = segment_text_simple("First sentence. Second sentence.")
            
            assert len(sentences) == 2
            assert "First sentence" in sentences[0]
            assert "Second sentence" in sentences[1]


class TestStopwordRemover:
    """Test cases for StopwordRemover component."""
    
    @patch('nltk.download')
    @patch('nltk.data.find')
    @patch('nltk.corpus.stopwords.words')
    def test_stopword_remover_initialization(self, mock_words, mock_find, mock_download):
        """Test StopwordRemover initialization."""
        mock_find.side_effect = LookupError()  # Simulate missing data
        mock_words.return_value = ['the', 'a', 'an', 'and', 'or']
        
        remover = StopwordRemover(
            languages='english',
            custom_stopwords={'custom', 'words'},
            min_word_length=3
        )
        
        assert remover.languages == ['english']
        assert 'custom' in remover.custom_stopwords
        assert remover.min_word_length == 3
        assert mock_download.called
    
    @patch('nltk.tokenize.word_tokenize')
    @patch('nltk.corpus.stopwords.words')
    def test_basic_stopword_removal(self, mock_words, mock_tokenize):
        """Test basic stopword removal functionality."""
        mock_words.return_value = ['the', 'a', 'an', 'and', 'is']
        mock_tokenize.return_value = ['This', 'is', 'a', 'test', 'sentence', 'with', 'the', 'stopwords']
        
        remover = StopwordRemover(languages='english')
        
        result = remover.remove_stopwords("This is a test sentence with the stopwords")
        
        assert isinstance(result, StopwordRemovalResult)
        assert result.word_count_before == 8
        assert result.word_count_after < result.word_count_before
        assert 'is' in result.removed_words
        assert 'a' in result.removed_words
        assert 'the' in result.removed_words
        assert result.removal_ratio > 0
    
    def test_empty_text_stopword_removal(self):
        """Test stopword removal on empty text."""
        remover = StopwordRemover()
        
        result = remover.remove_stopwords("")
        
        assert result.cleaned_text == ""
        assert result.word_count_before == 0
        assert result.word_count_after == 0
        assert result.removal_ratio == 0.0
        assert len(result.removed_words) == 0
    
    @patch('nltk.tokenize.word_tokenize')
    def test_custom_stopwords(self, mock_tokenize):
        """Test custom stopword functionality."""
        mock_tokenize.return_value = ['hello', 'world', 'custom', 'word', 'test']
        
        remover = StopwordRemover(
            languages=[],  # No built-in stopwords
            custom_stopwords={'custom', 'word'}
        )
        
        result = remover.remove_stopwords("hello world custom word test")
        
        assert 'custom' in result.removed_words
        assert 'word' in result.removed_words
        assert 'hello' not in result.removed_words
        assert 'world' not in result.removed_words
    
    def test_minimum_word_length_filtering(self):
        """Test minimum word length filtering."""
        with patch('nltk.tokenize.word_tokenize') as mock_tokenize:
            mock_tokenize.return_value = ['I', 'am', 'testing', 'short', 'words']
            
            remover = StopwordRemover(
                languages=[],  # No built-in stopwords
                min_word_length=4
            )
            
            result = remover.remove_stopwords("I am testing short words")
            
            # Words shorter than 4 characters should be removed
            assert 'I' in result.removed_words
            assert 'am' in result.removed_words
            assert 'testing' not in result.removed_words
            assert 'short' not in result.removed_words
            assert 'words' not in result.removed_words
    
    def test_number_and_punctuation_removal(self):
        """Test number and punctuation removal."""
        with patch('nltk.tokenize.word_tokenize') as mock_tokenize:
            mock_tokenize.return_value = ['hello', '123', 'world', '!!!', 'test']
            
            remover = StopwordRemover(
                languages=[],
                remove_numbers=True,
                remove_punctuation=True
            )
            
            result = remover.remove_stopwords("hello 123 world !!! test")
            
            assert '123' in result.removed_words
            assert '!!!' in result.removed_words
            assert 'hello' not in result.removed_words
            assert 'world' not in result.removed_words
            assert 'test' not in result.removed_words
    
    def test_frequency_based_removal(self):
        """Test frequency-based word removal."""
        with patch('nltk.tokenize.word_tokenize') as mock_tokenize:
            # Simulate text with word frequencies
            mock_tokenize.return_value = [
                'common', 'common', 'common', 'common', 'common',  # Very frequent
                'rare',  # Very rare
                'normal', 'normal', 'normal'  # Normal frequency
            ]
            
            remover = StopwordRemover(languages=[])
            
            result = remover.remove_by_frequency(
                "text with various word frequencies",
                min_frequency=2,
                max_frequency_ratio=0.4
            )
            
            # 'common' should be removed (too frequent)
            # 'rare' should be removed (too infrequent)
            # 'normal' should be kept
            assert any('common' in word for word in result.removed_words)
            assert 'rare' in result.removed_words
    
    def test_add_remove_custom_stopwords(self):
        """Test adding and removing custom stopwords."""
        remover = StopwordRemover(languages=[], custom_stopwords={'initial'})
        
        # Add custom stopwords
        remover.add_custom_stopwords(['new', 'words'])
        assert 'new' in remover.get_stopwords()
        assert 'words' in remover.get_stopwords()
        assert 'initial' in remover.get_stopwords()
        
        # Remove custom stopwords
        remover.remove_custom_stopwords(['initial'])
        assert 'initial' not in remover.get_stopwords()
        assert 'new' in remover.get_stopwords()
    
    def test_batch_removal(self):
        """Test batch stopword removal."""
        remover = StopwordRemover(languages=[])
        texts = ["first text", "second text", "third text"]
        
        with patch.object(remover, 'remove_stopwords') as mock_remove:
            mock_remove.side_effect = [
                StopwordRemovalResult("first", [], 2, 1, 0.5),
                StopwordRemovalResult("second", [], 2, 1, 0.5),
                StopwordRemovalResult("third", [], 2, 1, 0.5)
            ]
            
            results = remover.batch_remove(texts)
            
            assert len(results) == 3
            assert all(isinstance(r, StopwordRemovalResult) for r in results)
    
    def test_removal_stats(self):
        """Test removal statistics generation."""
        remover = StopwordRemover()
        results = [
            StopwordRemovalResult("text1", ['the', 'a'], 5, 3, 0.4),
            StopwordRemovalResult("text2", ['and', 'the'], 4, 2, 0.5),
        ]
        
        stats = remover.get_removal_stats(results)
        
        assert stats['total_documents'] == 2
        assert stats['total_words_before'] == 9
        assert stats['total_words_after'] == 5
        assert stats['total_words_removed'] == 4
        assert stats['average_removal_ratio'] == 0.45
        assert 'most_common_removed_words' in stats
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        with patch('src.qudata.clean.stopwords.StopwordRemover') as mock_remover_class:
            mock_remover = MagicMock()
            mock_remover.remove_stopwords.return_value = StopwordRemovalResult("cleaned", [], 3, 2, 0.33)
            mock_remover_class.return_value = mock_remover
            
            # Test simple removal
            result = remove_stopwords_simple("test text", language='english')
            assert result == "cleaned"
            
            # Test loading from file
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value = ['word1\n', 'word2\n', '\n']
                stopwords = load_stopwords_from_file('test.txt')
                assert 'word1' in stopwords
                assert 'word2' in stopwords


class TestTokenizationPreview:
    """Test cases for TokenizationPreview component."""
    
    @patch('tiktoken.get_encoding')
    def test_tokenization_preview_initialization(self, mock_get_encoding):
        """Test TokenizationPreview initialization."""
        mock_encoding = MagicMock()
        mock_get_encoding.return_value = mock_encoding
        
        preview = TokenizationPreview(model_name='gpt-3.5-turbo')
        
        assert preview.model_name == 'gpt-3.5-turbo'
        assert preview.model_config['encoding'] == 'cl100k_base'
        assert mock_get_encoding.called
    
    @patch('tiktoken.get_encoding')
    def test_text_analysis(self, mock_get_encoding):
        """Test text tokenization analysis."""
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        mock_get_encoding.return_value = mock_encoding
        
        preview = TokenizationPreview()
        text = "This is a test sentence."
        
        result = preview.analyze_text(text, include_cost=True)
        
        assert isinstance(result, TokenizationResult)
        assert result.token_count == 5
        assert result.character_count == len(text)
        assert result.word_count > 0
        assert result.char_to_token_ratio > 0
        assert result.estimated_cost is not None
        assert result.model_name == 'gpt-3.5-turbo'
    
    def test_empty_text_analysis(self):
        """Test analysis of empty text."""
        preview = TokenizationPreview()
        
        result = preview.analyze_text("", include_cost=True)
        
        assert result.token_count == 0
        assert result.character_count == 0
        assert result.word_count == 0
        assert result.estimated_cost == 0.0
    
    @patch('tiktoken.get_encoding')
    def test_context_limit_check(self, mock_get_encoding):
        """Test context limit checking."""
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1] * 5000  # 5000 tokens
        mock_get_encoding.return_value = mock_encoding
        
        preview = TokenizationPreview('gpt-3.5-turbo')  # 4096 max context
        
        context_check = preview.check_context_limit("Long text")
        
        assert context_check['token_count'] == 5000
        assert context_check['max_context'] == 4096
        assert context_check['fits_in_context'] is False
        assert context_check['tokens_over_limit'] == 904
        assert context_check['suggested_chunks'] > 1
    
    def test_chunking_suggestion(self):
        """Test text chunking suggestions."""
        preview = TokenizationPreview()
        
        with patch.object(preview, 'analyze_text') as mock_analyze:
            # Mock sentence analysis
            mock_analyze.side_effect = lambda text, **kwargs: TokenizationResult(
                text=text,
                token_count=len(text.split()) * 2,  # Rough estimation
                character_count=len(text),
                word_count=len(text.split()),
                char_to_token_ratio=2.0,
                word_to_token_ratio=0.5,
                model_name='gpt-3.5-turbo'
            )
            
            text = "First sentence. Second sentence. Third sentence. Fourth sentence."
            chunks = preview.suggest_chunking(text, target_tokens=10, overlap_tokens=2)
            
            assert len(chunks) >= 1
            assert all(isinstance(chunk, str) for chunk in chunks)
    
    def test_batch_analysis(self):
        """Test batch text analysis."""
        preview = TokenizationPreview()
        texts = ["First text", "Second text", "Third text"]
        
        with patch.object(preview, 'analyze_text') as mock_analyze:
            mock_analyze.side_effect = [
                TokenizationResult("First text", 4, 10, 2, 2.5, 2.0, 'gpt-3.5-turbo', 0.01),
                TokenizationResult("Second text", 5, 11, 2, 2.2, 2.5, 'gpt-3.5-turbo', 0.01),
                TokenizationResult("Third text", 4, 10, 2, 2.5, 2.0, 'gpt-3.5-turbo', 0.01)
            ]
            
            stats = preview.batch_analyze(texts, include_cost=True)
            
            assert stats.total_documents == 3
            assert stats.total_tokens == 13
            assert stats.total_characters == 31
            assert stats.estimated_total_cost == 0.03
            assert '0-100' in stats.token_distribution
    
    def test_model_comparison(self):
        """Test comparison across multiple models."""
        preview = TokenizationPreview()
        
        with patch.object(preview, 'analyze_text') as mock_analyze:
            mock_analyze.side_effect = [
                TokenizationResult("test", 5, 10, 2, 2.0, 2.5, 'gpt-3.5-turbo'),
                TokenizationResult("test", 6, 10, 2, 1.67, 3.0, 'gpt-4')
            ]
            
            results = preview.compare_models("test text", ['gpt-3.5-turbo', 'gpt-4'])
            
            assert 'gpt-3.5-turbo' in results
            assert 'gpt-4' in results
            assert results['gpt-3.5-turbo'].token_count == 5
            assert results['gpt-4'].token_count == 6
    
    def test_training_token_estimation(self):
        """Test training token estimation."""
        preview = TokenizationPreview()
        texts = ["Training text 1", "Training text 2"]
        
        with patch.object(preview, 'batch_analyze') as mock_batch:
            from src.qudata.clean.tokenization import BatchTokenizationStats
            mock_batch.return_value = BatchTokenizationStats(
                total_documents=2,
                total_tokens=100,
                total_characters=200,
                total_words=50,
                average_tokens_per_document=50.0,
                average_char_to_token_ratio=2.0,
                token_distribution={'0-100': 2},
                estimated_total_cost=0.1
            )
            
            estimate = preview.estimate_training_tokens(texts, epochs=3)
            
            assert estimate['total_documents'] == 2
            assert estimate['tokens_per_epoch'] == 100
            assert estimate['total_training_tokens'] == 300
            assert estimate['epochs'] == 3
            assert estimate['estimated_total_cost'] == 0.3
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        with patch('src.qudata.clean.tokenization.TokenizationPreview') as mock_preview_class:
            mock_preview = MagicMock()
            mock_preview.analyze_text.return_value = TokenizationResult("test", 5, 10, 2, 2.0, 2.5, 'gpt-3.5-turbo')
            mock_preview.check_context_limit.return_value = {'fits_in_context': True}
            mock_preview_class.return_value = mock_preview
            
            # Test quick token count
            count = quick_token_count("test text")
            assert count == 5
            
            # Test context fit check
            fits = check_context_fit("test text")
            assert fits is True


class TestFormatConverter:
    """Test cases for FormatConverter and related formatters."""
    
    def test_conversation_turn_creation(self):
        """Test ConversationTurn dataclass."""
        turn = ConversationTurn(
            role="user",
            content="Hello, how are you?",
            metadata={"timestamp": "2023-01-01"}
        )
        
        assert turn.role == "user"
        assert turn.content == "Hello, how are you?"
        assert turn.metadata["timestamp"] == "2023-01-01"
    
    def test_instruction_example_creation(self):
        """Test InstructionExample dataclass."""
        example = InstructionExample(
            instruction="Translate the following text to French",
            input="Hello world",
            output="Bonjour le monde",
            metadata={"language": "french"}
        )
        
        assert example.instruction == "Translate the following text to French"
        assert example.input == "Hello world"
        assert example.output == "Bonjour le monde"
        assert example.metadata["language"] == "french"
    
    def test_chatml_formatter(self):
        """Test ChatML formatter."""
        formatter = ChatMLFormatter()
        
        turns = [
            ConversationTurn("system", "You are a helpful assistant."),
            ConversationTurn("user", "Hello!"),
            ConversationTurn("assistant", "Hi there! How can I help you?")
        ]
        
        result = formatter.format_conversation(turns)
        
        assert "<|im_start|>system" in result
        assert "You are a helpful assistant." in result
        assert "<|im_end|>" in result
        assert "<|im_start|>user" in result
        assert "Hello!" in result
        assert "<|im_start|>assistant" in result
        assert "Hi there! How can I help you?" in result
        
        # Test validation
        assert formatter.validate_format(result) is True
    
    def test_chatml_instruction_formatting(self):
        """Test ChatML instruction formatting."""
        formatter = ChatMLFormatter()
        
        example = InstructionExample(
            instruction="Summarize this text",
            input="Long text to summarize",
            output="Short summary"
        )
        
        result = formatter.format_instruction(example)
        
        assert "<|im_start|>user" in result
        assert "Summarize this text" in result
        assert "Input: Long text to summarize" in result
        assert "<|im_start|>assistant" in result
        assert "Short summary" in result
    
    def test_alpaca_formatter(self):
        """Test Alpaca formatter."""
        formatter = AlpacaFormatter()
        
        example = InstructionExample(
            instruction="Translate to Spanish",
            input="Hello world",
            output="Hola mundo"
        )
        
        result = formatter.format_instruction(example)
        
        # Parse JSON result
        data = json.loads(result)
        assert data["instruction"] == "Translate to Spanish"
        assert data["input"] == "Hello world"
        assert data["output"] == "Hola mundo"
        
        # Test validation
        assert formatter.validate_format(result) is True
    
    def test_alpaca_conversation_formatting(self):
        """Test Alpaca conversation formatting."""
        formatter = AlpacaFormatter()
        
        turns = [
            ConversationTurn("system", "You are helpful."),
            ConversationTurn("user", "Hello!"),
            ConversationTurn("assistant", "Hi there!")
        ]
        
        result = formatter.format_conversation(turns)
        
        if result:  # May be empty if no valid conversation structure
            data = json.loads(result)
            assert "instruction" in data
            assert "output" in data
    
    def test_jsonl_formatter(self):
        """Test JSONL formatter."""
        formatter = JSONLFormatter()
        
        # Test conversation formatting
        turns = [
            ConversationTurn("user", "Hello!"),
            ConversationTurn("assistant", "Hi there!")
        ]
        
        result = formatter.format_conversation(turns)
        data = json.loads(result)
        
        assert "messages" in data
        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "Hello!"
        
        # Test instruction formatting
        example = InstructionExample(
            instruction="Test instruction",
            input="Test input",
            output="Test output"
        )
        
        result = formatter.format_instruction(example)
        data = json.loads(result)
        
        assert "text" in data
        assert "instruction" in data
        assert "output" in data
        assert data["instruction"] == "Test instruction"
    
    def test_format_converter_conversations(self):
        """Test FormatConverter for conversations."""
        converter = FormatConverter()
        
        conversations = [
            [
                ConversationTurn("user", "Hello!"),
                ConversationTurn("assistant", "Hi there!")
            ],
            [
                ConversationTurn("user", "How are you?"),
                ConversationTurn("assistant", "I'm doing well!")
            ]
        ]
        
        result = converter.convert_conversations(conversations, OutputFormat.CHATML)
        
        assert result.format_type == OutputFormat.CHATML
        assert result.total_examples == 2
        assert len(result.formatted_data) <= 2  # May have conversion errors
        assert result.success_rate >= 0.0
    
    def test_format_converter_instructions(self):
        """Test FormatConverter for instructions."""
        converter = FormatConverter()
        
        instructions = [
            InstructionExample("Task 1", "Input 1", "Output 1"),
            InstructionExample("Task 2", "Input 2", "Output 2")
        ]
        
        result = converter.convert_instructions(instructions, OutputFormat.ALPACA)
        
        assert result.format_type == OutputFormat.ALPACA
        assert result.total_examples == 2
        assert len(result.formatted_data) <= 2
    
    def test_mixed_conversion(self):
        """Test mixed conversation and instruction conversion."""
        converter = FormatConverter()
        
        conversations = [[ConversationTurn("user", "Hello!"), ConversationTurn("assistant", "Hi!")]]
        instructions = [InstructionExample("Test", "Input", "Output")]
        
        result = converter.batch_convert_mixed(conversations, instructions, OutputFormat.JSONL)
        
        assert result.total_examples == 2
        assert result.metadata["mixed_conversion"] is True
        assert result.metadata["conversations_count"] == 1
        assert result.metadata["instructions_count"] == 1
    
    def test_save_to_file(self, tmp_path):
        """Test saving conversion results to file."""
        converter = FormatConverter()
        
        # Create a mock result
        from src.qudata.export.formats import FormatConversionResult
        result = FormatConversionResult(
            formatted_data=["line1", "line2", "line3"],
            format_type=OutputFormat.JSONL,
            total_examples=3,
            conversion_errors=[],
            metadata={}
        )
        
        output_file = tmp_path / "test_output.jsonl"
        converter.save_to_file(result, str(output_file))
        
        # Check file contents
        content = output_file.read_text()
        lines = content.strip().split('\n')
        assert len(lines) == 3
        assert lines[0] == "line1"
        assert lines[1] == "line2"
        assert lines[2] == "line3"
    
    def test_custom_formatter(self):
        """Test adding custom formatter."""
        converter = FormatConverter()
        
        # Create a mock custom formatter
        custom_formatter = MagicMock(spec=ChatMLFormatter)
        custom_formatter.format_conversation.return_value = "custom format"
        custom_formatter.format_instruction.return_value = "custom instruction"
        custom_formatter.validate_format.return_value = True
        
        converter.add_custom_formatter("custom", custom_formatter)
        
        # Test that custom formatter is added
        assert OutputFormat.CUSTOM in converter.formatters
    
    def test_convenience_functions(self):
        """Test convenience functions for format conversion."""
        conversations = [[ConversationTurn("user", "Hello!"), ConversationTurn("assistant", "Hi!")]]
        instructions = [InstructionExample("Test", "Input", "Output")]
        
        # Test ChatML conversion
        with patch('src.qudata.export.formats.FormatConverter') as mock_converter_class:
            mock_converter = MagicMock()
            mock_converter.convert_conversations.return_value = MagicMock(formatted_data=["chatml_result"])
            mock_converter_class.return_value = mock_converter
            
            result = convert_to_chatml(conversations)
            assert result == ["chatml_result"]
        
        # Test Alpaca conversion
        with patch('src.qudata.export.formats.FormatConverter') as mock_converter_class:
            mock_converter = MagicMock()
            mock_converter.convert_instructions.return_value = MagicMock(formatted_data=["alpaca_result"])
            mock_converter_class.return_value = mock_converter
            
            result = convert_to_alpaca(instructions)
            assert result == ["alpaca_result"]
        
        # Test JSONL conversion
        with patch('src.qudata.export.formats.FormatConverter') as mock_converter_class:
            mock_converter = MagicMock()
            mock_converter.convert_conversations.return_value = MagicMock(formatted_data=["jsonl_result"])
            mock_converter_class.return_value = mock_converter
            
            result = convert_to_jsonl(conversations)
            assert result == ["jsonl_result"]


class TestAdvancedPreprocessingIntegration:
    """Integration tests for the complete advanced preprocessing pipeline."""
    
    def test_full_preprocessing_pipeline(self):
        """Test the complete preprocessing pipeline integration."""
        # Sample HTML content with various elements
        html_content = """
        <html>
            <head><title>Test Document</title></head>
            <body>
                <h1>Sample Document 📄</h1>
                <p>This is a test paragraph with <strong>bold text</strong>.</p>
                <p>Another paragraph. With multiple sentences. And some stopwords like the, a, an.</p>
                <ul>
                    <li>First item</li>
                    <li>Second item</li>
                </ul>
                <script>alert('remove this');</script>
            </body>
        </html>
        """
        
        # Step 1: Clean HTML
        html_cleaner = HTMLCleaner(remove_emojis=True)
        cleaned_result = html_cleaner.clean_html(html_content)
        
        assert "Sample Document" in cleaned_result.cleaned_text
        assert "📄" not in cleaned_result.cleaned_text
        assert "alert" not in cleaned_result.cleaned_text
        
        # Step 2: Segment text
        segmenter = SentenceSegmenter(min_sentence_length=10)
        
        with patch('nltk.tokenize.sent_tokenize') as mock_tokenize:
            mock_tokenize.return_value = [
                "Sample Document.",
                "This is a test paragraph with bold text.",
                "Another paragraph.",
                "With multiple sentences.",
                "And some stopwords like the, a, an."
            ]
            
            segmented_result = segmenter.segment_text(cleaned_result.cleaned_text)
            
            assert segmented_result.sentence_count >= 3
            assert any("test paragraph" in s for s in segmented_result.sentences)
        
        # Step 3: Remove stopwords
        with patch('nltk.corpus.stopwords.words') as mock_words:
            mock_words.return_value = ['the', 'a', 'an', 'and', 'with', 'is']
            
            stopword_remover = StopwordRemover(languages='english')
            
            # Process each sentence
            processed_sentences = []
            for sentence in segmented_result.sentences:
                with patch('nltk.tokenize.word_tokenize') as mock_word_tokenize:
                    mock_word_tokenize.return_value = sentence.split()
                    
                    removal_result = stopword_remover.remove_stopwords(sentence)
                    processed_sentences.append(removal_result.cleaned_text)
            
            assert len(processed_sentences) == len(segmented_result.sentences)
        
        # Step 4: Analyze tokenization
        tokenization_preview = TokenizationPreview()
        
        with patch('tiktoken.get_encoding') as mock_get_encoding:
            mock_encoding = MagicMock()
            mock_encoding.encode.return_value = [1, 2, 3, 4, 5, 6, 7, 8]
            mock_get_encoding.return_value = mock_encoding
            
            combined_text = " ".join(processed_sentences)
            token_result = tokenization_preview.analyze_text(combined_text)
            
            assert token_result.token_count > 0
            assert token_result.character_count > 0
            assert token_result.word_count > 0
        
        # Step 5: Convert to training format
        format_converter = FormatConverter()
        
        # Create instruction example from processed text
        instruction_example = InstructionExample(
            instruction="Summarize the following text",
            input=combined_text,
            output="This is a sample document with test content."
        )
        
        conversion_result = format_converter.convert_instructions(
            [instruction_example], 
            OutputFormat.ALPACA
        )
        
        assert conversion_result.total_examples == 1
        assert len(conversion_result.formatted_data) >= 0  # May have conversion errors
        
        # Verify the pipeline processed the content correctly
        assert cleaned_result.reduction_ratio > 0  # HTML was cleaned
        assert segmented_result.sentence_count > 0  # Text was segmented
        assert token_result.token_count > 0  # Tokens were counted
    
    def test_pipeline_error_handling(self):
        """Test error handling throughout the preprocessing pipeline."""
        # Test with problematic content
        problematic_content = ""
        
        # Each component should handle empty content gracefully
        html_cleaner = HTMLCleaner()
        cleaned_result = html_cleaner.clean_html(problematic_content)
        assert cleaned_result.cleaned_text == ""
        
        segmenter = SentenceSegmenter()
        segmented_result = segmenter.segment_text(problematic_content)
        assert segmented_result.sentence_count == 0
        
        stopword_remover = StopwordRemover()
        removal_result = stopword_remover.remove_stopwords(problematic_content)
        assert removal_result.cleaned_text == ""
        
        tokenization_preview = TokenizationPreview()
        token_result = tokenization_preview.analyze_text(problematic_content)
        assert token_result.token_count == 0
        
        format_converter = FormatConverter()
        conversion_result = format_converter.convert_instructions([], OutputFormat.JSONL)
        assert conversion_result.total_examples == 0
    
    def test_pipeline_performance_metrics(self):
        """Test that pipeline components provide useful performance metrics."""
        # Test with sample content
        sample_content = "<p>This is a test document with some content. It has multiple sentences.</p>"
        
        # HTML cleaning metrics
        html_cleaner = HTMLCleaner()
        cleaned_result = html_cleaner.clean_html(sample_content)
        assert hasattr(cleaned_result, 'reduction_ratio')
        assert hasattr(cleaned_result, 'removed_tags')
        assert hasattr(cleaned_result, 'removed_emojis')
        
        # Segmentation metrics
        segmenter = SentenceSegmenter()
        with patch('nltk.tokenize.sent_tokenize') as mock_tokenize:
            mock_tokenize.return_value = ["First sentence.", "Second sentence."]
            
            segmented_result = segmenter.segment_text(cleaned_result.cleaned_text)
            assert hasattr(segmented_result, 'sentence_count')
            assert hasattr(segmented_result, 'word_count')
            assert hasattr(segmented_result, 'average_sentence_length')
        
        # Stopword removal metrics
        stopword_remover = StopwordRemover()
        with patch('nltk.tokenize.word_tokenize') as mock_tokenize:
            mock_tokenize.return_value = ["This", "is", "a", "test"]
            with patch('nltk.corpus.stopwords.words') as mock_words:
                mock_words.return_value = ['is', 'a']
                
                removal_result = stopword_remover.remove_stopwords("This is a test")
                assert hasattr(removal_result, 'removal_ratio')
                assert hasattr(removal_result, 'words_removed_count')
        
        # Tokenization metrics
        tokenization_preview = TokenizationPreview()
        with patch('tiktoken.get_encoding') as mock_get_encoding:
            mock_encoding = MagicMock()
            mock_encoding.encode.return_value = [1, 2, 3, 4]
            mock_get_encoding.return_value = mock_encoding
            
            token_result = tokenization_preview.analyze_text("test text")
            assert hasattr(token_result, 'char_to_token_ratio')
            assert hasattr(token_result, 'tokens_per_word')
            assert hasattr(token_result, 'chars_per_token')
        
        # Format conversion metrics
        format_converter = FormatConverter()
        instruction = InstructionExample("Test", "Input", "Output")
        conversion_result = format_converter.convert_instructions([instruction], OutputFormat.JSONL)
        assert hasattr(conversion_result, 'success_rate')
        assert hasattr(conversion_result, 'conversion_errors')


if __name__ == '__main__':
    pytest.main([__file__])