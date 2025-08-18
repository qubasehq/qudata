#!/usr/bin/env python3
"""
Basic Pipeline Demo

This example demonstrates the basic usage of QuData pipeline for processing
documents and generating training datasets.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qudata import QuDataPipeline, load_config
from qudata.models import Document, DocumentMetadata


def main():
    """Demonstrate basic pipeline usage."""
    print("QuData Basic Pipeline Demo")
    print("=" * 40)
    
    # 1. Load configuration
    print("\n1. Loading configuration...")
    config_path = "configs/templates/basic-pipeline.yaml"
    
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        print("Please ensure you're running from the project root directory.")
        return
    
    try:
        config = load_config(config_path)
        print(f"✓ Configuration loaded from {config_path}")
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return
    
    # 2. Initialize pipeline
    print("\n2. Initializing pipeline...")
    try:
        pipeline = QuDataPipeline(config)
        print("✓ Pipeline initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize pipeline: {e}")
        return
    
    # 3. Create sample documents for demonstration
    print("\n3. Creating sample documents...")
    sample_docs = create_sample_documents()
    print(f"✓ Created {len(sample_docs)} sample documents")
    
    # 4. Process documents
    print("\n4. Processing documents...")
    try:
        results = pipeline.process_documents(sample_docs)
        print(f"✓ Processed {len(results)} documents")
        
        # Display results
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        print(f"  - Successful: {len(successful_results)}")
        print(f"  - Failed: {len(failed_results)}")
        
        if successful_results:
            avg_quality = sum(r.document.quality_score for r in successful_results) / len(successful_results)
            print(f"  - Average quality score: {avg_quality:.2f}")
        
    except Exception as e:
        print(f"✗ Processing failed: {e}")
        return
    
    # 5. Display detailed results
    print("\n5. Processing Results:")
    print("-" * 30)
    
    for i, result in enumerate(results, 1):
        print(f"\nDocument {i}:")
        if result.success:
            doc = result.document
            print(f"  ✓ Status: Success")
            print(f"  ✓ Quality Score: {doc.quality_score:.2f}")
            print(f"  ✓ Language: {doc.metadata.language}")
            print(f"  ✓ Content Length: {len(doc.content)} characters")
            print(f"  ✓ Processing Time: {result.processing_time:.2f}s")
            
            if doc.metadata.topics:
                print(f"  ✓ Topics: {', '.join(doc.metadata.topics)}")
            
            if doc.metadata.entities:
                print(f"  ✓ Entities: {len(doc.metadata.entities)} found")
                for entity in doc.metadata.entities[:3]:  # Show first 3
                    print(f"    - {entity.text} ({entity.label})")
        else:
            print(f"  ✗ Status: Failed")
            print(f"  ✗ Errors: {len(result.errors)}")
            for error in result.errors:
                print(f"    - {error.message}")
    
    # 6. Export results (if any successful)
    if successful_results:
        print("\n6. Exporting results...")
        try:
            # Extract successful documents
            processed_docs = [r.document for r in successful_results]
            
            # Export to JSONL format
            from qudata.pack import JSONLFormatter
            
            formatter = JSONLFormatter()
            output_path = "output/basic_demo_results.jsonl"
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            formatter.export_to_file(
                documents=processed_docs,
                output_path=output_path,
                fields=["text", "metadata", "quality_score"]
            )
            
            print(f"✓ Results exported to {output_path}")
            print(f"✓ Exported {len(processed_docs)} documents")
            
        except Exception as e:
            print(f"✗ Export failed: {e}")
    
    print("\n" + "=" * 40)
    print("Demo completed successfully!")
    print("\nNext steps:")
    print("- Review the exported JSONL file")
    print("- Experiment with different configurations")
    print("- Try processing your own documents")


def create_sample_documents():
    """Create sample documents for demonstration."""
    
    sample_texts = [
        {
            "content": """
            Machine Learning and Artificial Intelligence
            
            Machine learning is a subset of artificial intelligence (AI) that focuses on 
            algorithms that can learn and make decisions from data. It has revolutionized 
            many industries including healthcare, finance, and technology.
            
            Key concepts in machine learning include:
            - Supervised learning: Learning from labeled examples
            - Unsupervised learning: Finding patterns in unlabeled data
            - Reinforcement learning: Learning through interaction and feedback
            
            Popular algorithms include neural networks, decision trees, and support vector machines.
            Companies like Google, Microsoft, and OpenAI are leading the development of 
            advanced AI systems.
            """,
            "metadata": {
                "title": "Introduction to Machine Learning",
                "domain": "technology",
                "language": "en"
            }
        },
        {
            "content": """
            Climate Change and Environmental Impact
            
            Climate change refers to long-term shifts in global temperatures and weather patterns.
            While climate variations are natural, scientific evidence shows that human activities
            have been the main driver of climate change since the 1800s.
            
            The primary cause is the emission of greenhouse gases, particularly carbon dioxide
            from burning fossil fuels. This has led to:
            - Rising global temperatures
            - Melting ice caps and glaciers
            - Rising sea levels
            - More frequent extreme weather events
            
            Organizations like the United Nations and NASA are working to address these challenges
            through research, policy, and international cooperation.
            """,
            "metadata": {
                "title": "Understanding Climate Change",
                "domain": "science",
                "language": "en"
            }
        },
        {
            "content": """
            The History of the Internet
            
            The Internet began as ARPANET in the late 1960s, a project funded by the 
            U.S. Department of Defense. It was designed to create a decentralized 
            communication network that could survive partial outages.
            
            Key milestones include:
            - 1969: First ARPANET connection between UCLA and Stanford
            - 1989: Tim Berners-Lee invents the World Wide Web at CERN
            - 1993: The first web browser, Mosaic, is released
            - 1995: Commercial use of the Internet begins
            
            Today, the Internet connects billions of people worldwide and has transformed
            communication, commerce, and information sharing. Companies like Amazon, 
            Facebook, and Google have built their empires on Internet technologies.
            """,
            "metadata": {
                "title": "History of the Internet",
                "domain": "technology",
                "language": "en"
            }
        },
        {
            "content": """
            Short text example.
            """,
            "metadata": {
                "title": "Short Document",
                "domain": "test",
                "language": "en"
            }
        },
        {
            "content": """
            Este es un documento en español para probar la detección de idiomas.
            
            La inteligencia artificial es una rama de la informática que se centra en 
            crear sistemas capaces de realizar tareas que normalmente requieren 
            inteligencia humana.
            
            Algunas aplicaciones incluyen:
            - Reconocimiento de voz
            - Procesamiento de lenguaje natural
            - Visión por computadora
            - Sistemas de recomendación
            
            Empresas como IBM, Microsoft y Google están invirtiendo fuertemente en 
            esta tecnología.
            """,
            "metadata": {
                "title": "Inteligencia Artificial",
                "domain": "technology",
                "language": "es"
            }
        }
    ]
    
    documents = []
    for i, sample in enumerate(sample_texts, 1):
        metadata = DocumentMetadata(
            title=sample["metadata"]["title"],
            domain=sample["metadata"]["domain"],
            language=sample["metadata"]["language"],
            file_type="text"
        )
        
        document = Document(
            id=f"sample_doc_{i}",
            source_path=f"sample_{i}.txt",
            content=sample["content"].strip(),
            metadata=metadata
        )
        
        documents.append(document)
    
    return documents


if __name__ == "__main__":
    main()