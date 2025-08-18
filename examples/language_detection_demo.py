#!/usr/bin/env python3
"""
Language Detection and Content Filtering Demo

This script demonstrates the language detection and content filtering capabilities
of the QuData system, showing how to detect languages, filter content, and
handle multi-language documents.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qudata.clean.language import LanguageDetector
from qudata.clean.pipeline import ComprehensiveCleaningPipeline


def demo_basic_language_detection():
    """Demonstrate basic language detection functionality."""
    print("=" * 60)
    print("BASIC LANGUAGE DETECTION DEMO")
    print("=" * 60)
    
    detector = LanguageDetector()
    
    # Sample texts in different languages
    sample_texts = {
        'English': "This is a comprehensive test of the English language detection system. It should work properly and be detected as English language with high confidence.",
        'Spanish': "Este es un texto de muestra en español. Contiene múltiples oraciones para asegurar una detección adecuada del idioma y verificar que el sistema funciona correctamente.",
        'French': "Ceci est un exemple de texte en français. Il contient plusieurs phrases pour assurer une détection linguistique appropriée et tester la précision du système.",
        'German': "Dies ist ein Beispieltext auf Deutsch. Er enthält mehrere Sätze, um eine ordnungsgemäße Spracherkennung zu gewährleisten und die Genauigkeit zu testen.",
        'Italian': "Questo è un testo di esempio in italiano. Contiene più frasi per garantire un rilevamento linguistico adeguato e testare l'accuratezza del sistema.",
        'Portuguese': "Este é um texto de amostra em português. Contém várias frases para garantir a detecção adequada do idioma e testar a precisão do sistema.",
        'Russian': "Это образец текста на русском языке. Он содержит несколько предложений для обеспечения правильного определения языка и проверки точности системы.",
        'Japanese': "これは日本語のサンプルテキストです。適切な言語検出を確実にするために複数の文が含まれており、システムの精度をテストします。",
        'Chinese': "这是中文的示例文本。它包含多个句子以确保正确的语言检测并测试系统的准确性。",
        'Arabic': "هذا نص عينة باللغة العربية. يحتوي على جمل متعددة لضمان الكشف المناسب عن اللغة واختبار دقة النظام."
    }
    
    for language_name, text in sample_texts.items():
        print(f"\n{language_name} Text:")
        print(f"Text: {text[:80]}...")
        
        result = detector.detect_language(text)
        
        print(f"Detected Language: {result.language}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Is Reliable: {result.is_reliable}")
        print(f"Language Name: {detector.get_language_name(result.language)}")
        
        if len(result.detected_languages) > 1:
            print("Alternative Languages:")
            for lang, conf in result.detected_languages[1:3]:  # Show top 3
                print(f"  - {detector.get_language_name(lang)}: {conf:.3f}")


def demo_content_filtering():
    """Demonstrate content filtering based on language."""
    print("\n" + "=" * 60)
    print("CONTENT FILTERING DEMO")
    print("=" * 60)
    
    # Configure detector with language restrictions
    config = {
        "allowed_languages": ["en", "es", "fr"],
        "min_confidence": 0.8
    }
    detector = LanguageDetector(config)
    
    test_texts = [
        ("English (allowed)", "This is an English text that should pass the filter with high confidence."),
        ("Spanish (allowed)", "Este es un texto en español que debería pasar el filtro con alta confianza."),
        ("French (allowed)", "Ceci est un texte français qui devrait passer le filtre avec une grande confiance."),
        ("German (not allowed)", "Dies ist ein deutscher Text, der vom Filter blockiert werden sollte."),
        ("Italian (not allowed)", "Questo è un testo italiano che dovrebbe essere bloccato dal filtro."),
        ("Short text (low confidence)", "Hello world"),
        ("Mixed content", "This is English text. Pero también tiene español. Et un peu de français.")
    ]
    
    for description, text in test_texts:
        print(f"\n{description}:")
        print(f"Text: {text}")
        
        filter_result = detector.filter_by_language(text)
        
        print(f"Should Keep: {filter_result.should_keep}")
        print(f"Reason: {filter_result.reason}")
        print(f"Detected Language: {filter_result.language_result.language}")
        print(f"Confidence: {filter_result.language_result.confidence:.3f}")


def demo_multilingual_normalization():
    """Demonstrate multi-language content normalization."""
    print("\n" + "=" * 60)
    print("MULTI-LANGUAGE NORMALIZATION DEMO")
    print("=" * 60)
    
    detector = LanguageDetector()
    
    # Multi-language document
    multilingual_text = """
    Welcome to our international company. We serve customers worldwide.
    
    Bienvenidos a nuestra empresa internacional. Servimos a clientes en todo el mundo.
    
    Bienvenue dans notre entreprise internationale. Nous servons des clients dans le monde entier.
    
    Willkommen in unserem internationalen Unternehmen. Wir bedienen Kunden weltweit.
    
    Benvenuti nella nostra azienda internazionale. Serviamo clienti in tutto il mondo.
    """
    
    print("Multi-language Document:")
    print(multilingual_text)
    
    result = detector.normalize_multilingual_content(multilingual_text)
    
    print(f"\nPrimary Language: {detector.get_language_name(result['primary_language'])}")
    print(f"Is Multilingual: {result['is_multilingual']}")
    print(f"Total Segments: {result['total_segments']}")
    print(f"Reliable Segments: {result['reliable_segments']}")
    
    print("\nDetected Languages:")
    for lang in result['detected_languages']:
        percentage = result['language_distribution'].get(lang, 0) * 100
        print(f"  - {detector.get_language_name(lang)}: {percentage:.1f}%")
    
    print("\nLanguage Segments:")
    for segment in result['language_segments']:
        if segment['is_reliable']:
            lang_name = detector.get_language_name(segment['language'])
            text_preview = segment['text'][:60] + "..." if len(segment['text']) > 60 else segment['text']
            print(f"  Segment {segment['segment_index']}: {lang_name} ({segment['confidence']:.3f})")
            print(f"    Text: {text_preview}")


def demo_pipeline_integration():
    """Demonstrate language detection integration in the cleaning pipeline."""
    print("\n" + "=" * 60)
    print("PIPELINE INTEGRATION DEMO")
    print("=" * 60)
    
    # Configure pipeline with language detection
    config = {
        'language_detection': {
            'enabled': True,
            'min_confidence': 0.7,
            'allowed_languages': ['en', 'es']
        },
        'language_filtering': {
            'enabled': True
        },
        'quality_scoring': {
            'enabled': True
        }
    }
    
    pipeline = ComprehensiveCleaningPipeline(config)
    
    test_documents = {
        'doc1': """
        Tlie Document Title
        
        This is tlie main content witli some OCR errors in English.
        Navigation: Home | About | Contact
        Advertisement: Buy our products now!
        
        More valuable content here with proper English grammar and structure.
        Cookie policy: We use cookies for analytics.
        """,
        'doc2': """
        Título del Documento
        
        Este es el contenido principal con algunos errores de OCR en español.
        Navegación: Inicio | Acerca de | Contacto
        Publicidad: ¡Compre nuestros productos ahora!
        
        Más contenido valioso aquí con gramática y estructura española adecuadas.
        Política de cookies: Utilizamos cookies para análisis.
        """,
        'doc3': """
        Titre du Document
        
        Ceci est le contenu principal avec quelques erreurs OCR en français.
        Navigation: Accueil | À propos | Contact
        Publicité: Achetez nos produits maintenant!
        
        Plus de contenu précieux ici avec une grammaire et une structure françaises appropriées.
        """,
    }
    
    for doc_id, content in test_documents.items():
        print(f"\nProcessing {doc_id}:")
        print(f"Original length: {len(content)} characters")
        
        result = pipeline.clean_text(content, doc_id)
        
        print(f"Cleaned length: {len(result.cleaned_text)} characters")
        print(f"Operations applied: {', '.join(result.operations_applied)}")
        
        if result.language_result:
            lang_name = pipeline.language_detector.get_language_name(result.language_result.language)
            print(f"Detected Language: {lang_name} ({result.language_result.confidence:.3f})")
        
        if result.language_filter_result:
            print(f"Language Filter: {'PASS' if result.language_filter_result.should_keep else 'FAIL'}")
            if not result.language_filter_result.should_keep:
                print(f"Filter Reason: {result.language_filter_result.reason}")
        
        print(f"Quality Score: {result.quality_score:.3f}")
        
        if result.warnings:
            print(f"Warnings: {', '.join(result.warnings)}")


def demo_language_statistics():
    """Demonstrate language statistics for a collection of texts."""
    print("\n" + "=" * 60)
    print("LANGUAGE STATISTICS DEMO")
    print("=" * 60)
    
    detector = LanguageDetector()
    
    # Collection of texts in various languages
    text_collection = [
        "This is the first English document in our collection.",
        "This is another English document with different content.",
        "Este es un documento en español con contenido único.",
        "Ceci est un document français avec un contenu spécifique.",
        "Dies ist ein deutsches Dokument mit einzigartigem Inhalt.",
        "This is a third English document to show distribution.",
        "Este es otro documento español para mostrar la distribución.",
        "Short text",  # This should have low confidence
        "Another English text to demonstrate the language distribution analysis.",
        "Un autre document français pour compléter la collection."
    ]
    
    print(f"Analyzing {len(text_collection)} documents...")
    
    stats = detector.get_language_statistics(text_collection)
    
    print(f"\nTotal Texts: {stats['total_texts']}")
    print(f"Reliable Detections: {stats['reliable_detections']}")
    print(f"Reliability Rate: {stats['reliability_rate']:.1f}%")
    print(f"Most Common Language: {detector.get_language_name(stats['most_common_language']) if stats['most_common_language'] else 'None'}")
    
    print("\nLanguage Distribution:")
    for lang, percentage in stats['language_percentages'].items():
        lang_name = detector.get_language_name(lang)
        count = stats['language_counts'][lang]
        print(f"  - {lang_name}: {count} documents ({percentage:.1f}%)")


def main():
    """Run all language detection demos."""
    print("QuData Language Detection and Content Filtering Demo")
    print("=" * 60)
    
    try:
        demo_basic_language_detection()
        demo_content_filtering()
        demo_multilingual_normalization()
        demo_pipeline_integration()
        demo_language_statistics()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during demo: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()