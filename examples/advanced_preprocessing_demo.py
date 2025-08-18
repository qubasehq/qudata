#!/usr/bin/env python3
"""
Advanced Preprocessing Pipeline Demo

This script demonstrates the complete advanced preprocessing pipeline including:
- HTML cleaning and emoji removal
- Text segmentation and normalization
- Stopword removal with configurable lists
- Tokenization preview for LLM compatibility
- Format conversion to ChatML, Alpaca, and JSONL
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.qudata.clean.html_cleaner import HTMLCleaner
from src.qudata.clean.segment import SentenceSegmenter
from src.qudata.clean.stopwords import StopwordRemover
from src.qudata.clean.tokenization import TokenizationPreview
from src.qudata.export.formats import (
    FormatConverter, ConversationTurn, InstructionExample, OutputFormat
)


def main():
    """Demonstrate the advanced preprocessing pipeline."""
    print("🔧 Advanced Preprocessing Pipeline Demo")
    print("=" * 50)
    
    # Sample HTML content with various elements to clean
    sample_html = """
    <html>
        <head><title>Sample Document</title></head>
        <body>
            <h1>Welcome to Our Platform! 🚀</h1>
            <p>This is a <strong>comprehensive</strong> guide to using our system.</p>
            <p>Here are the key features:</p>
            <ul>
                <li>Advanced text processing 📝</li>
                <li>Multi-format support 📄</li>
                <li>Real-time analytics 📊</li>
            </ul>
            <p>For more information, visit <a href="https://example.com">our website</a>.</p>
            <script>
                // This script should be removed
                alert('This is unwanted content');
            </script>
            <div class="footer">
                Copyright © 2024 Example Corp. All rights reserved.
            </div>
        </body>
    </html>
    """
    
    print("📄 Original HTML Content:")
    print(sample_html[:200] + "..." if len(sample_html) > 200 else sample_html)
    print()
    
    # Step 1: HTML Cleaning
    print("🧹 Step 1: HTML Cleaning and Emoji Removal")
    print("-" * 40)
    
    html_cleaner = HTMLCleaner(
        preserve_links=True,
        remove_emojis=True,
        remove_special_chars=False
    )
    
    cleaning_result = html_cleaner.clean_html(sample_html)
    
    print(f"✅ Cleaned text: {cleaning_result.cleaned_text}")
    print(f"📊 Reduction ratio: {cleaning_result.reduction_ratio:.2%}")
    print(f"🏷️  Removed tags: {cleaning_result.removed_tags}")
    print(f"😀 Removed emojis: {cleaning_result.removed_emojis}")
    print()
    
    # Step 2: Text Segmentation
    print("✂️  Step 2: Text Segmentation and Normalization")
    print("-" * 40)
    
    segmenter = SentenceSegmenter(
        min_sentence_length=10,
        max_sentence_length=500,
        normalize_whitespace=True
    )
    
    try:
        segmentation_result = segmenter.segment_text(cleaning_result.cleaned_text)
        
        print(f"📝 Total sentences: {segmentation_result.sentence_count}")
        print(f"📊 Average sentence length: {segmentation_result.average_sentence_length:.1f} chars")
        print(f"🔤 Total words: {segmentation_result.word_count}")
        print("📋 Sample sentences:")
        for i, sentence in enumerate(segmentation_result.sentences[:3]):
            print(f"  {i+1}. {sentence}")
        print()
    except Exception as e:
        print(f"⚠️  Segmentation failed (likely missing NLTK data): {e}")
        # Use simple fallback segmentation
        sentences = cleaning_result.cleaned_text.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        print(f"📝 Fallback segmentation: {len(sentences)} sentences")
        segmentation_result = type('MockResult', (), {
            'sentences': sentences,
            'sentence_count': len(sentences),
            'word_count': len(cleaning_result.cleaned_text.split()),
            'average_sentence_length': len(cleaning_result.cleaned_text) / max(len(sentences), 1)
        })()
        print()
    
    # Step 3: Stopword Removal
    print("🚫 Step 3: Stopword Removal")
    print("-" * 40)
    
    try:
        stopword_remover = StopwordRemover(
            languages='english',
            custom_stopwords={'example', 'corp', 'copyright'},
            min_word_length=3
        )
        
        # Process first sentence as example
        if segmentation_result.sentences:
            sample_sentence = segmentation_result.sentences[0]
            removal_result = stopword_remover.remove_stopwords(sample_sentence)
            
            print(f"📝 Original: {sample_sentence}")
            print(f"✅ Cleaned: {removal_result.cleaned_text}")
            print(f"📊 Removal ratio: {removal_result.removal_ratio:.2%}")
            print(f"🚫 Removed words: {removal_result.removed_words}")
        print()
    except Exception as e:
        print(f"⚠️  Stopword removal failed (likely missing NLTK data): {e}")
        print()
    
    # Step 4: Tokenization Preview
    print("🔢 Step 4: Tokenization Preview for LLM Compatibility")
    print("-" * 40)
    
    try:
        tokenization_preview = TokenizationPreview(model_name='gpt-3.5-turbo')
        
        combined_text = ' '.join(segmentation_result.sentences[:3])  # Use first 3 sentences
        token_result = tokenization_preview.analyze_text(combined_text, include_cost=True)
        
        print(f"📝 Text: {combined_text[:100]}...")
        print(f"🔢 Token count: {token_result.token_count}")
        print(f"📊 Char-to-token ratio: {token_result.char_to_token_ratio:.2f}")
        print(f"💰 Estimated cost: ${token_result.estimated_cost:.4f}" if token_result.estimated_cost else "💰 Cost: N/A")
        
        # Check context limits
        context_check = tokenization_preview.check_context_limit(combined_text)
        print(f"✅ Fits in context: {context_check['fits_in_context']}")
        print(f"📊 Context utilization: {context_check['utilization_ratio']:.2%}")
        print()
    except Exception as e:
        print(f"⚠️  Tokenization preview failed: {e}")
        print()
    
    # Step 5: Format Conversion
    print("🔄 Step 5: Format Conversion (ChatML, Alpaca, JSONL)")
    print("-" * 40)
    
    format_converter = FormatConverter()
    
    # Create sample conversation
    conversation = [
        ConversationTurn("system", "You are a helpful assistant that explains text processing."),
        ConversationTurn("user", "What is text preprocessing?"),
        ConversationTurn("assistant", "Text preprocessing is the process of cleaning and preparing raw text data for analysis or machine learning tasks.")
    ]
    
    # Create sample instruction
    instruction = InstructionExample(
        instruction="Explain the importance of text preprocessing",
        output="Text preprocessing is crucial because it removes noise, standardizes format, and improves the quality of data used for training language models."
    )
    
    # Convert to different formats
    formats_to_test = [OutputFormat.CHATML, OutputFormat.ALPACA, OutputFormat.JSONL]
    
    for format_type in formats_to_test:
        try:
            print(f"📋 {format_type.value.upper()} Format:")
            
            if format_type == OutputFormat.ALPACA:
                # Use instruction for Alpaca
                result = format_converter.convert_instructions([instruction], format_type)
            else:
                # Use conversation for ChatML and JSONL
                result = format_converter.convert_conversations([conversation], format_type)
            
            if result.formatted_data:
                sample_output = result.formatted_data[0]
                print(f"✅ Success rate: {result.success_rate:.2%}")
                print(f"📝 Sample output: {sample_output[:150]}...")
            else:
                print(f"❌ Conversion failed: {result.conversion_errors}")
            print()
        except Exception as e:
            print(f"❌ Format conversion failed for {format_type.value}: {e}")
            print()
    
    # Summary
    print("📊 Pipeline Summary")
    print("-" * 40)
    print(f"✅ HTML cleaning: Removed {len(cleaning_result.removed_tags)} tags, {len(cleaning_result.removed_emojis)} emojis")
    print(f"✅ Text segmentation: {segmentation_result.sentence_count} sentences extracted")
    print(f"✅ Tokenization: Ready for LLM processing")
    print(f"✅ Format conversion: Multiple training formats generated")
    print()
    print("🎉 Advanced preprocessing pipeline completed successfully!")


if __name__ == "__main__":
    main()