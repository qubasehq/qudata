#!/usr/bin/env python3
"""
Web Scraping Pipeline Demo

This example demonstrates how to use QuData to scrape web content,
process it, and generate training datasets from web sources.
"""

import os
import sys
from pathlib import Path
from typing import List

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qudata import QuDataPipeline, load_config
from qudata.ingest import WebScraper, WebExtractor
from qudata.models import Document, DocumentMetadata


def main():
    """Demonstrate web scraping pipeline."""
    print("QuData Web Scraping Demo")
    print("=" * 40)
    
    # Sample URLs for demonstration (using public, scrapable content)
    sample_urls = [
        "https://httpbin.org/html",  # Simple HTML test page
        "https://example.com",       # Basic example page
        # Add more URLs as needed for testing
    ]
    
    # 1. Initialize web scraper
    print("\n1. Initializing web scraper...")
    try:
        scraper = WebScraper()
        extractor = WebExtractor()
        print("✓ Web scraper initialized")
    except Exception as e:
        print(f"✗ Failed to initialize scraper: {e}")
        return
    
    # 2. Scrape content from URLs
    print("\n2. Scraping web content...")
    scraped_documents = []
    
    for i, url in enumerate(sample_urls, 1):
        print(f"  Scraping URL {i}/{len(sample_urls)}: {url}")
        try:
            # Extract content from URL
            result = extractor.extract_from_url(url)
            
            if result and result.content:
                document = Document(
                    id=f"web_doc_{i}",
                    source_path=url,
                    content=result.content,
                    metadata=result.metadata
                )
                scraped_documents.append(document)
                print(f"    ✓ Extracted {len(result.content)} characters")
            else:
                print(f"    ✗ No content extracted")
                
        except Exception as e:
            print(f"    ✗ Failed to scrape {url}: {e}")
    
    print(f"✓ Successfully scraped {len(scraped_documents)} documents")
    
    if not scraped_documents:
        print("No documents were scraped. Creating sample documents for demo...")
        scraped_documents = create_sample_web_documents()
    
    # 3. Load pipeline configuration
    print("\n3. Loading pipeline configuration...")
    config_path = "configs/templates/basic-pipeline.yaml"
    
    try:
        config = load_config(config_path)
        
        # Modify config for web content processing
        config.clean.remove_html_tags = True
        config.clean.remove_boilerplate = True
        config.annotate.enable_classification = True
        
        print("✓ Configuration loaded and optimized for web content")
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return
    
    # 4. Initialize and run pipeline
    print("\n4. Processing scraped content...")
    try:
        pipeline = QuDataPipeline(config)
        results = pipeline.process_documents(scraped_documents)
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        print(f"✓ Processing completed:")
        print(f"  - Successful: {len(successful_results)}")
        print(f"  - Failed: {len(failed_results)}")
        
        if successful_results:
            avg_quality = sum(r.document.quality_score for r in successful_results) / len(successful_results)
            print(f"  - Average quality score: {avg_quality:.2f}")
        
    except Exception as e:
        print(f"✗ Pipeline processing failed: {e}")
        return
    
    # 5. Display detailed results
    print("\n5. Processing Results:")
    print("-" * 30)
    
    for i, result in enumerate(results, 1):
        print(f"\nDocument {i} ({scraped_documents[i-1].source_path}):")
        if result.success:
            doc = result.document
            print(f"  ✓ Status: Success")
            print(f"  ✓ Quality Score: {doc.quality_score:.2f}")
            print(f"  ✓ Language: {doc.metadata.language}")
            print(f"  ✓ Original Length: {len(scraped_documents[i-1].content)} chars")
            print(f"  ✓ Cleaned Length: {len(doc.content)} chars")
            print(f"  ✓ Processing Time: {result.processing_time:.2f}s")
            
            if doc.metadata.title:
                print(f"  ✓ Title: {doc.metadata.title}")
            
            if doc.metadata.topics:
                print(f"  ✓ Topics: {', '.join(doc.metadata.topics)}")
            
            # Show content preview
            preview = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
            print(f"  ✓ Content Preview: {preview}")
            
        else:
            print(f"  ✗ Status: Failed")
            for error in result.errors:
                print(f"    - {error.message}")
    
    # 6. Export results
    if successful_results:
        print("\n6. Exporting web content dataset...")
        try:
            processed_docs = [r.document for r in successful_results]
            
            # Export to multiple formats
            from qudata.pack import JSONLFormatter, ChatMLFormatter
            
            # Ensure output directory exists
            os.makedirs("output", exist_ok=True)
            
            # Export to JSONL
            jsonl_formatter = JSONLFormatter()
            jsonl_path = "output/web_content.jsonl"
            jsonl_formatter.export_to_file(
                documents=processed_docs,
                output_path=jsonl_path,
                fields=["text", "metadata", "quality_score"]
            )
            print(f"✓ JSONL export: {jsonl_path}")
            
            # Export to ChatML for conversational training
            chatml_formatter = ChatMLFormatter()
            chatml_data = chatml_formatter.format_documents(
                processed_docs,
                system_message="You are a helpful assistant with knowledge from web content."
            )
            
            chatml_path = "output/web_content_chatml.json"
            import json
            with open(chatml_path, 'w', encoding='utf-8') as f:
                json.dump(chatml_data, f, indent=2, ensure_ascii=False)
            print(f"✓ ChatML export: {chatml_path}")
            
        except Exception as e:
            print(f"✗ Export failed: {e}")
    
    # 7. Generate quality report
    print("\n7. Quality Analysis:")
    print("-" * 20)
    
    if successful_results:
        quality_scores = [r.document.quality_score for r in successful_results]
        languages = [r.document.metadata.language for r in successful_results]
        content_lengths = [len(r.document.content) for r in successful_results]
        
        print(f"Quality Statistics:")
        print(f"  - Average: {sum(quality_scores) / len(quality_scores):.2f}")
        print(f"  - Min: {min(quality_scores):.2f}")
        print(f"  - Max: {max(quality_scores):.2f}")
        
        print(f"\nContent Statistics:")
        print(f"  - Average length: {sum(content_lengths) / len(content_lengths):.0f} chars")
        print(f"  - Min length: {min(content_lengths)} chars")
        print(f"  - Max length: {max(content_lengths)} chars")
        
        print(f"\nLanguage Distribution:")
        from collections import Counter
        lang_counts = Counter(languages)
        for lang, count in lang_counts.items():
            print(f"  - {lang}: {count} documents")
    
    print("\n" + "=" * 40)
    print("Web scraping demo completed!")
    print("\nNext steps:")
    print("- Review the exported files in the output/ directory")
    print("- Try scraping your own URLs")
    print("- Experiment with different web scraping configurations")


def create_sample_web_documents() -> List[Document]:
    """Create sample web-like documents for demonstration."""
    
    sample_web_content = [
        {
            "url": "https://example.com/ai-article",
            "content": """
            <html>
            <head><title>The Future of Artificial Intelligence</title></head>
            <body>
                <nav>Navigation menu</nav>
                <header>Site Header</header>
                
                <main>
                    <h1>The Future of Artificial Intelligence</h1>
                    <p>Published on March 15, 2024 by Dr. Sarah Johnson</p>
                    
                    <p>Artificial Intelligence (AI) is rapidly transforming our world, 
                    from healthcare and finance to transportation and entertainment. 
                    As we look toward the future, several key trends are emerging.</p>
                    
                    <h2>Key Developments</h2>
                    <ul>
                        <li>Large Language Models like GPT-4 and Claude</li>
                        <li>Computer Vision advances in autonomous vehicles</li>
                        <li>AI-powered drug discovery and medical diagnosis</li>
                        <li>Robotics and automation in manufacturing</li>
                    </ul>
                    
                    <p>Companies like OpenAI, Google DeepMind, and Anthropic are 
                    pushing the boundaries of what's possible with AI technology.</p>
                    
                    <blockquote>
                        "AI will be the most important general-purpose technology 
                        of our era." - Sundar Pichai, CEO of Google
                    </blockquote>
                </main>
                
                <footer>Copyright 2024 Example.com</footer>
                <script>Analytics code here</script>
            </body>
            </html>
            """,
            "title": "The Future of Artificial Intelligence",
            "domain": "technology"
        },
        {
            "url": "https://example.com/climate-science",
            "content": """
            <html>
            <head><title>Climate Change Research Update</title></head>
            <body>
                <div class="ads">Advertisement</div>
                <nav>Site navigation</nav>
                
                <article>
                    <h1>Climate Change Research Update</h1>
                    <div class="metadata">
                        <span>By Dr. Michael Chen</span>
                        <span>Environmental Science Institute</span>
                        <span>February 28, 2024</span>
                    </div>
                    
                    <p>Recent studies from NASA and NOAA show accelerating trends 
                    in global climate patterns. The latest data reveals significant 
                    changes in ocean temperatures and ice sheet dynamics.</p>
                    
                    <h2>Key Findings</h2>
                    <p>The research, published in Nature Climate Change, highlights:</p>
                    <ol>
                        <li>Arctic sea ice decline of 13% per decade</li>
                        <li>Ocean acidification increasing by 0.02 pH units per decade</li>
                        <li>Extreme weather events becoming more frequent</li>
                        <li>Renewable energy adoption accelerating globally</li>
                    </ol>
                    
                    <p>The United Nations Framework Convention on Climate Change 
                    emphasizes the urgent need for coordinated global action.</p>
                </article>
                
                <aside>Related articles sidebar</aside>
                <footer>Site footer</footer>
            </body>
            </html>
            """,
            "title": "Climate Change Research Update",
            "domain": "science"
        },
        {
            "url": "https://example.com/tech-news",
            "content": """
            <html>
            <head><title>Latest Technology News</title></head>
            <body>
                <header class="site-header">
                    <h1>TechNews Daily</h1>
                    <nav>Home | News | Reviews | About</nav>
                </header>
                
                <main>
                    <article>
                        <h1>Breakthrough in Quantum Computing</h1>
                        <p class="byline">By Alex Rodriguez | March 10, 2024</p>
                        
                        <p>IBM and Google have announced significant advances in 
                        quantum computing technology, bringing us closer to 
                        practical quantum applications.</p>
                        
                        <h2>Technical Achievements</h2>
                        <p>The new quantum processors demonstrate:</p>
                        <ul>
                            <li>Improved qubit stability and coherence times</li>
                            <li>Better error correction algorithms</li>
                            <li>Increased quantum volume metrics</li>
                            <li>Enhanced quantum-classical hybrid algorithms</li>
                        </ul>
                        
                        <p>Companies like Microsoft, Amazon, and startups like 
                        Rigetti Computing are also making significant contributions 
                        to the quantum computing ecosystem.</p>
                        
                        <h2>Future Applications</h2>
                        <p>Potential applications include cryptography, drug discovery, 
                        financial modeling, and optimization problems that are 
                        intractable for classical computers.</p>
                    </article>
                </main>
                
                <div class="sidebar">
                    <h3>Trending Topics</h3>
                    <ul>
                        <li>AI Ethics</li>
                        <li>5G Networks</li>
                        <li>Cybersecurity</li>
                    </ul>
                </div>
                
                <footer>© 2024 TechNews Daily</footer>
            </body>
            </html>
            """,
            "title": "Latest Technology News",
            "domain": "technology"
        }
    ]
    
    documents = []
    for i, sample in enumerate(sample_web_content, 1):
        # Simulate basic HTML cleaning for metadata extraction
        import re
        
        # Extract title from HTML
        title_match = re.search(r'<title>(.*?)</title>', sample["content"])
        title = title_match.group(1) if title_match else sample["title"]
        
        # Extract main content (simplified)
        content = re.sub(r'<script.*?</script>', '', sample["content"], flags=re.DOTALL)
        content = re.sub(r'<style.*?</style>', '', content, flags=re.DOTALL)
        content = re.sub(r'<[^>]+>', ' ', content)
        content = re.sub(r'\s+', ' ', content).strip()
        
        metadata = DocumentMetadata(
            title=title,
            domain=sample["domain"],
            language="en",
            file_type="html"
        )
        
        document = Document(
            id=f"web_doc_{i}",
            source_path=sample["url"],
            content=content,
            metadata=metadata
        )
        
        documents.append(document)
    
    return documents


if __name__ == "__main__":
    main()