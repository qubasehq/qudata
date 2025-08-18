"""
GraphQL Endpoint for flexible data queries.

Provides GraphQL interface for complex queries and mutations
on datasets, documents, and processing operations.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

import strawberry
from strawberry.fastapi import GraphQLRouter
from strawberry.types import Info

from ..models import (
    Document, Dataset, ProcessingResult, ProcessingError,
    ErrorSeverity, ProcessingStage, QualityMetrics, DatasetMetadata
)
from ..pipeline import QuDataPipeline


# GraphQL Types
@strawberry.type
class DocumentType:
    """GraphQL type for Document."""
    id: str
    source_path: str
    content: str
    word_count: int
    character_count: int
    language: str
    domain: str
    quality_score: float
    processing_timestamp: datetime
    version: str
    
    @classmethod
    def from_document(cls, doc: Document) -> "DocumentType":
        """Create GraphQL type from Document model."""
        return cls(
            id=doc.id,
            source_path=doc.source_path,
            content=doc.content,
            word_count=doc.get_word_count(),
            character_count=doc.get_character_count(),
            language=doc.metadata.language,
            domain=doc.metadata.domain,
            quality_score=doc.metadata.quality_score,
            processing_timestamp=doc.processing_timestamp,
            version=doc.version
        )


@strawberry.type
class EntityType:
    """GraphQL type for Entity."""
    text: str
    label: str
    start: int
    end: int
    confidence: float


@strawberry.type
class QualityMetricsType:
    """GraphQL type for QualityMetrics."""
    overall_score: float
    length_score: float
    language_score: float
    coherence_score: float
    uniqueness_score: float
    completeness_score: float


@strawberry.type
class DatasetType:
    """GraphQL type for Dataset."""
    id: str
    name: str
    version: str
    document_count: int
    total_words: int
    total_characters: int
    languages: List[str]
    domains: List[str]
    quality_metrics: QualityMetricsType
    created_at: datetime
    last_modified: datetime
    
    @classmethod
    def from_dataset(cls, dataset: Dataset) -> "DatasetType":
        """Create GraphQL type from Dataset model."""
        return cls(
            id=dataset.id,
            name=dataset.name,
            version=dataset.version,
            document_count=dataset.get_document_count(),
            total_words=dataset.get_total_word_count(),
            total_characters=dataset.get_total_character_count(),
            languages=dataset.get_languages(),
            domains=dataset.get_domains(),
            quality_metrics=QualityMetricsType(
                overall_score=dataset.quality_metrics.overall_score,
                length_score=dataset.quality_metrics.length_score,
                language_score=dataset.quality_metrics.language_score,
                coherence_score=dataset.quality_metrics.coherence_score,
                uniqueness_score=dataset.quality_metrics.uniqueness_score,
                completeness_score=dataset.quality_metrics.completeness_score
            ),
            created_at=dataset.metadata.creation_date,
            last_modified=dataset.metadata.last_modified
        )


@strawberry.type
class ProcessingJobType:
    """GraphQL type for processing jobs."""
    job_id: str
    status: str
    progress: float
    message: str
    started_at: datetime
    completed_at: Optional[datetime]
    result: Optional[str]  # JSON string
    errors: List[str]


@strawberry.type
class SystemStatusType:
    """GraphQL type for system status."""
    status: str
    version: str
    uptime: float
    active_jobs: int
    total_jobs_processed: int
    memory_usage_percent: float
    disk_usage_percent: float


# Input Types
@strawberry.input
class ProcessingInput:
    """Input type for processing operations."""
    input_path: str
    output_path: str
    config_path: Optional[str] = None
    format: str = "jsonl"


@strawberry.input
class DocumentFilter:
    """Input type for filtering documents."""
    language: Optional[str] = None
    domain: Optional[str] = None
    min_quality_score: Optional[float] = None
    max_quality_score: Optional[float] = None
    min_word_count: Optional[int] = None
    max_word_count: Optional[int] = None


@strawberry.input
class DatasetFilter:
    """Input type for filtering datasets."""
    name_contains: Optional[str] = None
    version: Optional[str] = None
    min_document_count: Optional[int] = None
    max_document_count: Optional[int] = None
    language: Optional[str] = None
    domain: Optional[str] = None


class GraphQLContext:
    """Context for GraphQL operations."""
    
    def __init__(self):
        """Initialize GraphQL context."""
        self.pipeline: Optional[QuDataPipeline] = None
        self.active_jobs: Dict[str, ProcessingJobType] = {}
        self.job_history: List[ProcessingJobType] = []
        self.datasets: Dict[str, Dataset] = {}  # In-memory storage for demo
        self.start_time = datetime.now()
    
    def get_pipeline(self, config_path: str = None) -> QuDataPipeline:
        """Get or create pipeline instance."""
        if not self.pipeline or config_path:
            self.pipeline = QuDataPipeline(config_path=config_path)
        return self.pipeline
    
    def add_job(self, job: ProcessingJobType):
        """Add a job to tracking."""
        self.active_jobs[job.job_id] = job
    
    def complete_job(self, job_id: str):
        """Move job from active to history."""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            self.job_history.append(job)
            del self.active_jobs[job_id]
    
    def get_system_status(self) -> SystemStatusType:
        """Get current system status."""
        import psutil
        
        uptime = (datetime.now() - self.start_time).total_seconds()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return SystemStatusType(
            status="running",
            version="1.0.0",
            uptime=uptime,
            active_jobs=len(self.active_jobs),
            total_jobs_processed=len(self.job_history),
            memory_usage_percent=memory.percent,
            disk_usage_percent=(disk.used / disk.total) * 100
        )


# Global context instance
graphql_context = GraphQLContext()


# Query resolvers
@strawberry.type
class Query:
    """GraphQL Query root."""
    
    @strawberry.field
    def system_status(self) -> SystemStatusType:
        """Get system status and metrics."""
        return graphql_context.get_system_status()
    
    @strawberry.field
    def datasets(
        self,
        filter: Optional[DatasetFilter] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[DatasetType]:
        """Get list of datasets with optional filtering."""
        datasets = list(graphql_context.datasets.values())
        
        # Apply filters
        if filter:
            if filter.name_contains:
                datasets = [d for d in datasets if filter.name_contains.lower() in d.name.lower()]
            if filter.version:
                datasets = [d for d in datasets if d.version == filter.version]
            if filter.min_document_count:
                datasets = [d for d in datasets if d.get_document_count() >= filter.min_document_count]
            if filter.max_document_count:
                datasets = [d for d in datasets if d.get_document_count() <= filter.max_document_count]
            if filter.language:
                datasets = [d for d in datasets if filter.language in d.get_languages()]
            if filter.domain:
                datasets = [d for d in datasets if filter.domain in d.get_domains()]
        
        # Apply pagination
        return [DatasetType.from_dataset(d) for d in datasets[offset:offset + limit]]
    
    @strawberry.field
    def dataset(self, id: str) -> Optional[DatasetType]:
        """Get a specific dataset by ID."""
        dataset = graphql_context.datasets.get(id)
        return DatasetType.from_dataset(dataset) if dataset else None
    
    @strawberry.field
    def documents(
        self,
        dataset_id: str,
        filter: Optional[DocumentFilter] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[DocumentType]:
        """Get documents from a dataset with optional filtering."""
        dataset = graphql_context.datasets.get(dataset_id)
        if not dataset:
            return []
        
        documents = dataset.documents
        
        # Apply filters
        if filter:
            if filter.language:
                documents = [d for d in documents if d.metadata.language == filter.language]
            if filter.domain:
                documents = [d for d in documents if d.metadata.domain == filter.domain]
            if filter.min_quality_score:
                documents = [d for d in documents if d.metadata.quality_score >= filter.min_quality_score]
            if filter.max_quality_score:
                documents = [d for d in documents if d.metadata.quality_score <= filter.max_quality_score]
            if filter.min_word_count:
                documents = [d for d in documents if d.get_word_count() >= filter.min_word_count]
            if filter.max_word_count:
                documents = [d for d in documents if d.get_word_count() <= filter.max_word_count]
        
        # Apply pagination
        return [DocumentType.from_document(d) for d in documents[offset:offset + limit]]
    
    @strawberry.field
    def document(self, dataset_id: str, document_id: str) -> Optional[DocumentType]:
        """Get a specific document by ID."""
        dataset = graphql_context.datasets.get(dataset_id)
        if not dataset:
            return None
        
        document = dataset.get_document_by_id(document_id)
        return DocumentType.from_document(document) if document else None
    
    @strawberry.field
    def jobs(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[ProcessingJobType]:
        """Get list of processing jobs with optional filtering."""
        all_jobs = list(graphql_context.active_jobs.values()) + graphql_context.job_history
        
        if status:
            all_jobs = [job for job in all_jobs if job.status == status]
        
        # Sort by start time (newest first)
        all_jobs.sort(key=lambda x: x.started_at, reverse=True)
        
        return all_jobs[offset:offset + limit]
    
    @strawberry.field
    def job(self, job_id: str) -> Optional[ProcessingJobType]:
        """Get a specific job by ID."""
        # Check active jobs first
        if job_id in graphql_context.active_jobs:
            return graphql_context.active_jobs[job_id]
        
        # Check job history
        for job in graphql_context.job_history:
            if job.job_id == job_id:
                return job
        
        return None


# Mutation resolvers
@strawberry.type
class Mutation:
    """GraphQL Mutation root."""
    
    @strawberry.mutation
    async def start_processing(self, input: ProcessingInput) -> ProcessingJobType:
        """Start a new processing job."""
        job_id = str(uuid.uuid4())
        
        job = ProcessingJobType(
            job_id=job_id,
            status="pending",
            progress=0.0,
            message="Job queued for processing",
            started_at=datetime.now(),
            completed_at=None,
            result=None,
            errors=[]
        )
        
        graphql_context.add_job(job)
        
        # Start background processing
        asyncio.create_task(
            _process_data_background(job_id, input.input_path, input.output_path, input.config_path, input.format)
        )
        
        return job
    
    @strawberry.mutation
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        if job_id not in graphql_context.active_jobs:
            return False
        
        job = graphql_context.active_jobs[job_id]
        if job.status in ["completed", "failed"]:
            return False
        
        job.status = "cancelled"
        job.message = "Job cancelled by user"
        job.completed_at = datetime.now()
        
        graphql_context.complete_job(job_id)
        return True
    
    @strawberry.mutation
    async def analyze_dataset(self, dataset_id: str) -> ProcessingJobType:
        """Start dataset analysis job."""
        job_id = str(uuid.uuid4())
        
        job = ProcessingJobType(
            job_id=job_id,
            status="pending",
            progress=0.0,
            message="Dataset analysis queued",
            started_at=datetime.now(),
            completed_at=None,
            result=None,
            errors=[]
        )
        
        graphql_context.add_job(job)
        
        # Start background analysis
        asyncio.create_task(_analyze_dataset_background(job_id, dataset_id))
        
        return job
    
    @strawberry.mutation
    def reload_configuration(self, config_path: Optional[str] = None) -> bool:
        """Reload system configuration."""
        try:
            graphql_context.pipeline = None  # Force reload
            graphql_context.get_pipeline(config_path)
            return True
        except Exception:
            return False


# Background task functions
async def _process_data_background(job_id: str, input_path: str, output_path: str, 
                                 config_path: str = None, format: str = "jsonl"):
    """Background task for data processing."""
    job = graphql_context.active_jobs.get(job_id)
    if not job:
        return
    
    try:
        job.status = "running"
        job.message = "Processing data..."
        job.progress = 0.1
        
        # Get pipeline
        pipeline = graphql_context.get_pipeline(config_path)
        
        # Process data
        result = pipeline.process_directory(input_path, output_path)
        
        job.status = "completed" if result.success else "failed"
        job.message = "Processing completed successfully" if result.success else "Processing failed"
        job.progress = 1.0
        job.completed_at = datetime.now()
        
        # Store result as JSON
        import json
        job.result = json.dumps({
            "success": result.success,
            "documents_processed": getattr(result, 'documents_processed', 0),
            "processing_time": getattr(result, 'processing_time', 0),
            "output_paths": getattr(result, 'output_paths', {})
        })
        
        if hasattr(result, 'errors') and result.errors:
            job.errors = [str(error) for error in result.errors]
        
    except Exception as e:
        job.status = "failed"
        job.message = f"Processing failed: {str(e)}"
        job.progress = 0.0
        job.completed_at = datetime.now()
        job.errors = [str(e)]
    
    finally:
        graphql_context.complete_job(job_id)


async def _analyze_dataset_background(job_id: str, dataset_id: str):
    """Background task for dataset analysis."""
    job = graphql_context.active_jobs.get(job_id)
    if not job:
        return
    
    try:
        job.status = "running"
        job.message = "Analyzing dataset..."
        job.progress = 0.1
        
        # Simulate analysis progress
        for i in range(1, 10):
            await asyncio.sleep(0.5)
            job.progress = i * 0.1
            job.message = f"Analyzing dataset... {int(job.progress * 100)}%"
        
        job.status = "completed"
        job.message = "Analysis completed successfully"
        job.progress = 1.0
        job.completed_at = datetime.now()
        
        # Store result as JSON
        import json
        job.result = json.dumps({
            "dataset_id": dataset_id,
            "analysis_type": "comprehensive",
            "metrics": {
                "quality_score": 0.85,
                "language_distribution": {"en": 0.8, "es": 0.2},
                "domain_distribution": {"technical": 0.6, "general": 0.4}
            }
        })
        
    except Exception as e:
        job.status = "failed"
        job.message = f"Analysis failed: {str(e)}"
        job.progress = 0.0
        job.completed_at = datetime.now()
        job.errors = [str(e)]
    
    finally:
        graphql_context.complete_job(job_id)


# Create GraphQL schema
schema = strawberry.Schema(query=Query, mutation=Mutation)


def create_graphql_router() -> GraphQLRouter:
    """Create GraphQL router for FastAPI integration."""
    return GraphQLRouter(schema, path="/graphql")


def get_graphql_context() -> GraphQLContext:
    """Get the global GraphQL context."""
    return graphql_context


# Standalone GraphQL server
def create_graphql_server(host: str = "localhost", port: int = 8001):
    """Create standalone GraphQL server."""
    from fastapi import FastAPI
    import uvicorn
    
    app = FastAPI(title="QuData GraphQL API")
    
    # Add GraphQL router
    graphql_router = create_graphql_router()
    app.include_router(graphql_router, prefix="/graphql")
    
    @app.get("/")
    def root():
        return {"message": "QuData GraphQL API", "graphql": "/graphql"}
    
    def run():
        uvicorn.run(app, host=host, port=port)
    
    return app, run


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="QuData GraphQL Server")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    
    args = parser.parse_args()
    
    app, run = create_graphql_server(host=args.host, port=args.port)
    
    print(f"Starting QuData GraphQL server on {args.host}:{args.port}")
    print(f"GraphQL endpoint: http://{args.host}:{args.port}/graphql")
    
    run()