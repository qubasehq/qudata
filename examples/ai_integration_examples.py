#!/usr/bin/env python3
"""
AI Framework Integration Examples

This script demonstrates how to integrate QuData with popular AI frameworks
and training systems like Hugging Face, OpenAI, LangChain, and others.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def print_integration(title: str, description: str):
    """Print an integration example header."""
    print("\n" + "🔗" + "=" * 60)
    print(f"INTEGRATION: {title}")
    print(f"DESCRIPTION: {description}")
    print("=" * 61)

def print_code_block(title: str, code: str):
    """Print a formatted code block."""
    print(f"\n📝 {title}")
    print("```python")
    print(code.strip())
    print("```")

def print_install_note(packages: List[str]):
    """Print installation requirements."""
    print(f"\n📦 Required packages: {', '.join(packages)}")
    print(f"💻 Install with: pip install {' '.join(packages)}")

def hugging_face_integration():
    """Demonstrate Hugging Face integration."""
    print_integration(
        "Hugging Face Transformers",
        "Process data with QuData and train models with Hugging Face"
    )
    
    print_install_note(["transformers", "datasets", "torch"])
    
    print_code_block("1. Process Data with QuData", """
from qudata import QuDataPipeline
import json

# Process documents with QuData
pipeline = QuDataPipeline()
results = pipeline.process_directory("raw_data", "processed_data")

# Export to JSONL for Hugging Face
successful_docs = [r.document for r in results if r.success]
training_data = []

for doc in successful_docs:
    training_data.append({
        "text": doc.content,
        "label": doc.metadata.domain or "general",
        "quality_score": doc.quality_score
    })

# Save training data
with open("hf_training_data.jsonl", "w") as f:
    for item in training_data:
        f.write(json.dumps(item) + "\\n")

print(f"Prepared {len(training_data)} examples for Hugging Face")
    """)
    
    print_code_block("2. Load Data into Hugging Face Datasets", """
from datasets import Dataset, DatasetDict
import pandas as pd

# Load QuData processed data
data = []
with open("hf_training_data.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

# Create train/validation split
train_size = int(0.8 * len(df))
train_df = df[:train_size]
val_df = df[train_size:]

# Create Hugging Face dataset
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "validation": Dataset.from_pandas(val_df)
})

print(f"Created dataset with {len(dataset['train'])} training examples")
print(f"Validation set: {len(dataset['validation'])} examples")
    """)
    
    print_code_block("3. Fine-tune a Model", """
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)

# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=len(set(df['label']))
)

# Tokenize data
def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding=True, 
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
    """)

def openai_integration():
    """Demonstrate OpenAI API integration."""
    print_integration(
        "OpenAI API",
        "Use QuData processed content for OpenAI fine-tuning and embeddings"
    )
    
    print_install_note(["openai"])
    
    print_code_block("1. Prepare Data for OpenAI Fine-tuning", """
from qudata import QuDataPipeline
import openai
import json

# Process documents with QuData
pipeline = QuDataPipeline()
results = pipeline.process_directory("raw_data", "processed_data")

# Convert to OpenAI fine-tuning format
openai_data = []
successful_docs = [r.document for r in results if r.success]

for doc in successful_docs:
    # Create instruction-following examples
    openai_data.append({
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided content."
            },
            {
                "role": "user", 
                "content": f"Summarize this content: {doc.content[:1000]}..."
            },
            {
                "role": "assistant",
                "content": f"This content discusses {doc.metadata.domain or 'general topics'} and covers key points about the subject matter."
            }
        ]
    })

# Save in OpenAI format
with open("openai_training_data.jsonl", "w") as f:
    for item in openai_data:
        f.write(json.dumps(item) + "\\n")

print(f"Prepared {len(openai_data)} examples for OpenAI fine-tuning")
    """)
    
    print_code_block("2. Create Embeddings for Semantic Search", """
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set up OpenAI client
client = openai.OpenAI(api_key="your-api-key-here")

# Create embeddings for processed documents
embeddings = []
documents = []

for doc in successful_docs:
    # Get embedding for document content
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=doc.content[:8000]  # Limit to token limit
    )
    
    embeddings.append(response.data[0].embedding)
    documents.append({
        "content": doc.content,
        "title": doc.metadata.title,
        "quality": doc.quality_score
    })

# Convert to numpy array for similarity search
embeddings_array = np.array(embeddings)

def semantic_search(query, top_k=5):
    # Get query embedding
    query_response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=query
    )
    query_embedding = np.array([query_response.data[0].embedding])
    
    # Calculate similarities
    similarities = cosine_similarity(query_embedding, embeddings_array)[0]
    
    # Get top results
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            "document": documents[idx],
            "similarity": similarities[idx]
        })
    
    return results

# Example search
results = semantic_search("artificial intelligence in healthcare")
for i, result in enumerate(results, 1):
    print(f"{i}. Similarity: {result['similarity']:.3f}")
    print(f"   Title: {result['document']['title']}")
    print(f"   Quality: {result['document']['quality']:.2f}")
    """)

def langchain_integration():
    """Demonstrate LangChain integration."""
    print_integration(
        "LangChain",
        "Build RAG systems using QuData processed documents with LangChain"
    )
    
    print_install_note(["langchain", "langchain-openai", "chromadb", "tiktoken"])
    
    print_code_block("1. Create Vector Store from QuData Output", """
from qudata import QuDataPipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document as LangChainDocument

# Process documents with QuData
pipeline = QuDataPipeline()
results = pipeline.process_directory("raw_data", "processed_data")

# Convert QuData documents to LangChain documents
langchain_docs = []
for result in results:
    if result.success:
        doc = result.document
        langchain_doc = LangChainDocument(
            page_content=doc.content,
            metadata={
                "title": doc.metadata.title or "Untitled",
                "domain": doc.metadata.domain or "general",
                "quality_score": doc.quality_score,
                "language": doc.metadata.language,
                "source": doc.source_path
            }
        )
        langchain_docs.append(langchain_doc)

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

splits = text_splitter.split_documents(langchain_docs)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print(f"Created vector store with {len(splits)} document chunks")
    """)
    
    print_code_block("2. Build RAG Chain", """
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Set up the language model
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0
)

# Create custom prompt template
prompt_template = \"\"\"Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer: \"\"\"

PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
)

# Create retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# Example usage
def ask_question(question):
    result = qa_chain({"query": question})
    
    print(f"Question: {question}")
    print(f"Answer: {result['result']}")
    print("\\nSources:")
    for i, doc in enumerate(result['source_documents'], 1):
        print(f"{i}. {doc.metadata['title']} (Quality: {doc.metadata['quality_score']:.2f})")
    
    return result

# Ask questions about your processed documents
ask_question("What are the main applications of AI in healthcare?")
ask_question("How does climate change affect renewable energy adoption?")
    """)

def llamaindex_integration():
    """Demonstrate LlamaIndex integration."""
    print_integration(
        "LlamaIndex",
        "Create knowledge bases and query engines with LlamaIndex"
    )
    
    print_install_note(["llama-index", "llama-index-llms-openai"])
    
    print_code_block("1. Build Index from QuData Documents", """
from qudata import QuDataPipeline
from llama_index.core import Document as LlamaDocument
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.llms.openai import OpenAI
from llama_index.core.embeddings import resolve_embed_model

# Process documents with QuData
pipeline = QuDataPipeline()
results = pipeline.process_directory("raw_data", "processed_data")

# Convert to LlamaIndex documents
llama_docs = []
for result in results:
    if result.success:
        doc = result.document
        llama_doc = LlamaDocument(
            text=doc.content,
            metadata={
                "title": doc.metadata.title or "Untitled",
                "domain": doc.metadata.domain or "general",
                "quality_score": doc.quality_score,
                "language": doc.metadata.language,
                "source": doc.source_path,
                "word_count": len(doc.content.split())
            }
        )
        llama_docs.append(llama_doc)

# Set up service context
llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

# Create index
index = VectorStoreIndex.from_documents(
    llama_docs,
    embed_model=embed_model
)

# Save index
index.storage_context.persist("./llama_index_storage")

print(f"Created LlamaIndex with {len(llama_docs)} documents")
    """)
    
    print_code_block("2. Query the Knowledge Base", """
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor

# Create query engine
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5
)

# Add post-processor to filter by similarity
postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[postprocessor]
)

# Query the knowledge base
def query_knowledge_base(question):
    response = query_engine.query(question)
    
    print(f"Question: {question}")
    print(f"Answer: {response}")
    
    # Show source information
    if hasattr(response, 'source_nodes'):
        print("\\nSources:")
        for i, node in enumerate(response.source_nodes, 1):
            metadata = node.node.metadata
            print(f"{i}. {metadata.get('title', 'Unknown')} "
                  f"(Quality: {metadata.get('quality_score', 0):.2f}, "
                  f"Domain: {metadata.get('domain', 'general')})")
    
    return response

# Example queries
query_knowledge_base("What are the latest developments in AI?")
query_knowledge_base("How can renewable energy help with climate change?")
query_knowledge_base("What are the challenges in healthcare AI implementation?")
    """)

def pytorch_integration():
    """Demonstrate PyTorch integration."""
    print_integration(
        "PyTorch",
        "Create custom datasets and train models with PyTorch"
    )
    
    print_install_note(["torch", "torchtext", "scikit-learn"])
    
    print_code_block("1. Create PyTorch Dataset from QuData", """
from qudata import QuDataPipeline
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

class QuDataTextDataset(Dataset):
    def __init__(self, documents, labels, tokenizer, max_length=512):
        self.documents = documents
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        text = self.documents[idx]
        label = self.labels[idx]
        
        # Tokenize text (simplified - use proper tokenizer in practice)
        tokens = text.split()[:self.max_length]
        
        # Convert to tensor (simplified)
        token_ids = [hash(token) % 10000 for token in tokens]  # Simplified tokenization
        
        # Pad or truncate
        if len(token_ids) < self.max_length:
            token_ids.extend([0] * (self.max_length - len(token_ids)))
        else:
            token_ids = token_ids[:self.max_length]
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Process documents with QuData
pipeline = QuDataPipeline()
results = pipeline.process_directory("raw_data", "processed_data")

# Prepare data
texts = []
domains = []

for result in results:
    if result.success:
        texts.append(result.document.content)
        domains.append(result.document.metadata.domain or "general")

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(domains)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts, encoded_labels, test_size=0.2, random_state=42
)

# Create datasets
train_dataset = QuDataTextDataset(X_train, y_train, None)
test_dataset = QuDataTextDataset(X_test, y_test, None)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Number of classes: {len(label_encoder.classes_)}")
    """)

def streamlit_dashboard():
    """Demonstrate Streamlit dashboard integration."""
    print_integration(
        "Streamlit Dashboard",
        "Create interactive dashboards for QuData results"
    )
    
    print_install_note(["streamlit", "plotly", "pandas"])
    
    print_code_block("Streamlit Dashboard App", """
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from qudata import QuDataPipeline
import json

st.set_page_config(
    page_title="QuData Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 QuData Processing Dashboard")

# Sidebar for controls
st.sidebar.header("Controls")

# File upload
uploaded_files = st.sidebar.file_uploader(
    "Upload documents",
    accept_multiple_files=True,
    type=['txt', 'pdf', 'docx']
)

# Processing options
min_quality = st.sidebar.slider("Minimum Quality Score", 0.0, 1.0, 0.6)
enable_dedup = st.sidebar.checkbox("Enable Deduplication", True)

if st.sidebar.button("Process Documents"):
    if uploaded_files:
        # Save uploaded files temporarily
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        for uploaded_file in uploaded_files:
            with open(f"{temp_dir}/{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        # Process with QuData
        with st.spinner("Processing documents..."):
            pipeline = QuDataPipeline()
            results = pipeline.process_directory(temp_dir, "processed_temp")
        
        # Store results in session state
        st.session_state.results = results
        st.success(f"Processed {len(results)} documents!")

# Display results if available
if 'results' in st.session_state:
    results = st.session_state.results
    successful = [r for r in results if r.success]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", len(results))
    
    with col2:
        st.metric("Successful", len(successful))
    
    with col3:
        if successful:
            avg_quality = sum(r.document.quality_score for r in successful) / len(successful)
            st.metric("Avg Quality", f"{avg_quality:.2f}")
    
    with col4:
        total_words = sum(len(r.document.content.split()) for r in successful)
        st.metric("Total Words", f"{total_words:,}")
    
    # Quality distribution chart
    if successful:
        quality_scores = [r.document.quality_score for r in successful]
        
        fig = px.histogram(
            x=quality_scores,
            nbins=20,
            title="Quality Score Distribution",
            labels={'x': 'Quality Score', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Language distribution
        languages = [r.document.metadata.language for r in successful]
        lang_counts = pd.Series(languages).value_counts()
        
        fig_pie = px.pie(
            values=lang_counts.values,
            names=lang_counts.index,
            title="Language Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Document details table
        st.subheader("Document Details")
        
        doc_data = []
        for result in successful:
            doc = result.document
            doc_data.append({
                'Title': doc.metadata.title or 'Untitled',
                'Quality Score': f"{doc.quality_score:.2f}",
                'Language': doc.metadata.language,
                'Word Count': len(doc.content.split()),
                'Domain': doc.metadata.domain or 'General',
                'Processing Time': f"{result.processing_time:.2f}s"
            })
        
        df = pd.DataFrame(doc_data)
        st.dataframe(df, use_container_width=True)
        
        # Export options
        st.subheader("Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export JSONL"):
                # Export logic here
                st.success("JSONL export completed!")
        
        with col2:
            if st.button("Export CSV"):
                # Export logic here
                st.success("CSV export completed!")
        
        with col3:
            if st.button("Export ChatML"):
                # Export logic here
                st.success("ChatML export completed!")

# Run with: streamlit run qudata_dashboard.py
    """)

def main():
    """Run all integration examples."""
    print("🔗 QuData AI Framework Integration Examples")
    print("Learn how to integrate QuData with popular AI frameworks and tools")
    print("=" * 70)
    
    try:
        # Run integration examples
        hugging_face_integration()
        openai_integration()
        langchain_integration()
        llamaindex_integration()
        pytorch_integration()
        streamlit_dashboard()
        
        print("\n" + "🎉" + "=" * 69)
        print("INTEGRATION EXAMPLES COMPLETE!")
        print("=" * 70)
        
        print("\n🔗 Integration Summary:")
        print("✅ Hugging Face Transformers: Fine-tuning and datasets")
        print("✅ OpenAI API: Fine-tuning and embeddings")
        print("✅ LangChain: RAG systems and vector stores")
        print("✅ LlamaIndex: Knowledge bases and query engines")
        print("✅ PyTorch: Custom datasets and model training")
        print("✅ Streamlit: Interactive dashboards")
        
        print("\n💡 Key Benefits:")
        print("• QuData provides clean, high-quality training data")
        print("• Seamless integration with popular AI frameworks")
        print("• Consistent data format across different tools")
        print("• Quality scores help filter training data")
        print("• Rich metadata enables advanced filtering and analysis")
        
        print("\n🚀 Next Steps:")
        print("• Choose the framework that matches your project")
        print("• Install the required dependencies")
        print("• Adapt the code examples to your specific use case")
        print("• Experiment with different QuData configurations")
        print("• Build end-to-end AI applications with clean data")
        
        print("\n📚 Additional Resources:")
        print("• Hugging Face Documentation: https://huggingface.co/docs")
        print("• OpenAI API Reference: https://platform.openai.com/docs")
        print("• LangChain Documentation: https://python.langchain.com/")
        print("• LlamaIndex Documentation: https://docs.llamaindex.ai/")
        print("• PyTorch Tutorials: https://pytorch.org/tutorials/")
        print("• Streamlit Documentation: https://docs.streamlit.io/")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Integration examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Integration examples failed: {e}")

if __name__ == "__main__":
    main()