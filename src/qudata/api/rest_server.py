"""
REST API Server for external system integration.

Provides RESTful endpoints for pipeline management, dataset operations,
and system monitoring.
"""

import os
import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status, Query, Path as PathParam

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

from ..models import (
    Document, Dataset, ProcessingResult, ProcessingError, 
    ErrorSeverity, ProcessingStage, QualityMetrics, DatasetMetadata
)
from ..pipeline import QuDataPipeline
from ..config import ConfigManager

# Pydantic models for API requests/responses
class ProcessingRequest(BaseModel):
    """Request model for processing operations."""
    input_path: str = Field(..., description="Path to input data")
    output_path: str = Field(..., description="Path for output data")
    config_path: Optional[str] = Field(None, description="Path to configuration file")
    format: Optional[str] = Field("jsonl", description="Output format")
    
class ProcessingResponse(BaseModel):
    """Response model for processing operations."""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")
    started_at: datetime = Field(..., description="Job start time")
    
class JobStatus(BaseModel):
    """Job status response model."""
    job_id: str
    status: str  # pending, running, completed, failed
    progress: float = Field(0.0, ge=0.0, le=1.0)
    message: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    errors: List[Dict[str, Any]] = []

class DatasetInfo(BaseModel):
    """Dataset information response model."""
    id: str
    name: str
    version: str
    document_count: int
    total_words: int
    languages: List[str]
    domains: List[str]
    quality_score: float
    created_at: datetime
    last_modified: datetime

class DocumentInfo(BaseModel):
    """Document information response model."""
    id: str
    source_path: str
    word_count: int
    character_count: int
    language: str
    domain: str
    quality_score: float
    processing_timestamp: datetime

class SystemStatus(BaseModel):
    """System status response model."""
    status: str
    version: str
    uptime: float
    active_jobs: int
    total_jobs_processed: int
    memory_usage: Dict[str, Any]
    disk_usage: Dict[str, Any]


class RESTAPIServer:
    """REST API Server for QuData system integration."""
    
    def __init__(self, config_path: str = None, host: str = "localhost", port: int = 8000):
        """
        Initialize REST API server.
        
        Args:
            config_path: Path to configuration file
            host: Server host address
            port: Server port number
        """
        self.config_path = config_path
        self.host = host
        self.port = port
        enable_docs = str(os.getenv("ENABLE_DOCS", "true")).lower() == "true"
        # Configure FastAPI with optional docs/openapi based on environment
        self.app = FastAPI(
            title="QuData API",
            description="REST API for QuData LLM Data Processing System",
            version="1.0.0",
            docs_url="/docs" if enable_docs else None,
            redoc_url=None,
            openapi_url="/openapi.json" if enable_docs else None,
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Job tracking
        self.active_jobs: Dict[str, JobStatus] = {}
        self.job_history: List[JobStatus] = []
        self.start_time = datetime.now()
        
        # Initialize pipeline
        self.pipeline = None
        if config_path:
            self.pipeline = QuDataPipeline(config_path=config_path)
        
        # Optionally mount GraphQL router
        if str(os.getenv("ENABLE_GRAPHQL", "true")).lower() == "true":
            try:
                from .graphql_endpoint import create_graphql_router
                self.app.include_router(create_graphql_router())
            except Exception as e:
                logging.getLogger(__name__).warning(f"GraphQL not mounted: {e}")
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint with API information."""
            return {
                "name": "QuData API",
                "version": "1.0.0",
                "description": "REST API for QuData LLM Data Processing System",
                "docs": "/docs"
            }
        
        @self.app.get("/health", response_model=Dict[str, str])
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @self.app.get("/status", response_model=SystemStatus)
        async def system_status():
            """Get system status and metrics."""
            import psutil
            
            uptime = (datetime.now() - self.start_time).total_seconds()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return SystemStatus(
                status="running",
                version="1.0.0",
                uptime=uptime,
                active_jobs=len(self.active_jobs),
                total_jobs_processed=len(self.job_history),
                memory_usage={
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent
                },
                disk_usage={
                    "total": disk.total,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100
                }
            )
        
        @self.app.post("/process", response_model=ProcessingResponse)
        async def start_processing(request: ProcessingRequest, background_tasks: BackgroundTasks):
            """Start a new processing job."""
            job_id = str(uuid.uuid4())
            
            # Create job status
            job_status = JobStatus(
                job_id=job_id,
                status="pending",
                message="Job queued for processing",
                started_at=datetime.now()
            )
            
            self.active_jobs[job_id] = job_status
            
            # Start background processing
            background_tasks.add_task(
                self._process_data_background,
                job_id,
                request.input_path,
                request.output_path,
                request.config_path,
                request.format
            )
            
            return ProcessingResponse(
                job_id=job_id,
                status="pending",
                message="Processing job started",
                started_at=job_status.started_at
            )
        
        @self.app.get("/jobs/{job_id}", response_model=JobStatus)
        async def get_job_status(job_id: str = PathParam(..., description="Job ID")):
            """Get status of a specific job."""
            if job_id not in self.active_jobs:
                # Check job history
                for job in self.job_history:
                    if job.job_id == job_id:
                        return job
                raise HTTPException(status_code=404, detail="Job not found")
            
            return self.active_jobs[job_id]
        
        @self.app.get("/jobs", response_model=List[JobStatus])
        async def list_jobs(
            status: Optional[str] = Query(None, description="Filter by status"),
            limit: int = Query(50, ge=1, le=1000, description="Maximum number of jobs to return")
        ):
            """List all jobs with optional filtering."""
            all_jobs = list(self.active_jobs.values()) + self.job_history
            
            if status:
                all_jobs = [job for job in all_jobs if job.status == status]
            
            # Sort by start time (newest first)
            all_jobs.sort(key=lambda x: x.started_at, reverse=True)
            
            return all_jobs[:limit]
        
        @self.app.delete("/jobs/{job_id}")
        async def cancel_job(job_id: str = PathParam(..., description="Job ID")):
            """Cancel a running job."""
            if job_id not in self.active_jobs:
                raise HTTPException(status_code=404, detail="Job not found")
            
            job = self.active_jobs[job_id]
            if job.status in ["completed", "failed"]:
                raise HTTPException(status_code=400, detail="Cannot cancel completed job")
            
            job.status = "cancelled"
            job.message = "Job cancelled by user"
            job.completed_at = datetime.now()
            
            # Move to history
            self.job_history.append(job)
            del self.active_jobs[job_id]
            
            return {"message": "Job cancelled successfully"}
        
        @self.app.get("/datasets", response_model=List[DatasetInfo])
        async def list_datasets():
            """List available datasets."""
            # This would typically query a database or file system
            # For now, return empty list as placeholder
            return []
        
        @self.app.get("/datasets/{dataset_id}", response_model=DatasetInfo)
        async def get_dataset(dataset_id: str = PathParam(..., description="Dataset ID")):
            """Get information about a specific dataset."""
            # Placeholder implementation
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        @self.app.get("/datasets/{dataset_id}/documents", response_model=List[DocumentInfo])
        async def list_dataset_documents(
            dataset_id: str = PathParam(..., description="Dataset ID"),
            limit: int = Query(100, ge=1, le=1000, description="Maximum number of documents to return"),
            offset: int = Query(0, ge=0, description="Number of documents to skip")
        ):
            """List documents in a dataset."""
            # Placeholder implementation
            return []
        
        @self.app.get("/datasets/{dataset_id}/export")
        async def export_dataset(
            dataset_id: str = PathParam(..., description="Dataset ID"),
            format: str = Query("jsonl", description="Export format"),
            split: Optional[str] = Query(None, description="Dataset split (train/val/test)")
        ):
            """Export a dataset in the specified format."""
            # Placeholder implementation
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        @self.app.post("/datasets/{dataset_id}/analyze")
        async def analyze_dataset(
            background_tasks: BackgroundTasks,
            dataset_id: str = PathParam(..., description="Dataset ID")
        ):
            """Start dataset analysis job."""
            job_id = str(uuid.uuid4())
            
            job_status = JobStatus(
                job_id=job_id,
                status="pending",
                message="Dataset analysis queued",
                started_at=datetime.now()
            )
            
            self.active_jobs[job_id] = job_status
            
            # Start background analysis
            background_tasks.add_task(self._analyze_dataset_background, job_id, dataset_id)
            
            return {"job_id": job_id, "message": "Analysis started"}
        
        @self.app.get("/config")
        async def get_configuration():
            """Get current system configuration."""
            if self.pipeline and self.pipeline.config:
                return self.pipeline.config.to_dict()
            return {"message": "No configuration loaded"}
        
        @self.app.post("/config/reload")
        async def reload_configuration(config_path: Optional[str] = None):
            """Reload system configuration."""
            try:
                path = config_path or self.config_path
                if not path:
                    raise HTTPException(status_code=400, detail="No configuration path provided")
                
                self.pipeline = QuDataPipeline(config_path=path)
                return {"message": "Configuration reloaded successfully"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to reload configuration: {str(e)}")
    
    async def _process_data_background(self, job_id: str, input_path: str, 
                                     output_path: str, config_path: str = None, 
                                     format: str = "jsonl"):
        """Background task for data processing."""
        job = self.active_jobs[job_id]
        
        try:
            job.status = "running"
            job.message = "Processing data..."
            
            # Initialize pipeline if needed
            if not self.pipeline or config_path:
                self.pipeline = QuDataPipeline(config_path=config_path or self.config_path)
            
            # Process data
            result = self.pipeline.process_directory(input_path, output_path)
            
            job.status = "completed" if result.success else "failed"
            job.message = "Processing completed successfully" if result.success else "Processing failed"
            job.completed_at = datetime.now()
            job.result = {
                "success": result.success,
                "documents_processed": getattr(result, 'documents_processed', 0),
                "processing_time": getattr(result, 'processing_time', 0),
                "output_paths": getattr(result, 'output_paths', {})
            }
            
            if hasattr(result, 'errors') and result.errors:
                job.errors = [error.to_dict() if hasattr(error, 'to_dict') else str(error) for error in result.errors]
            
        except Exception as e:
            job.status = "failed"
            job.message = f"Processing failed: {str(e)}"
            job.completed_at = datetime.now()
            job.errors = [{"error": str(e), "type": type(e).__name__}]
        
        finally:
            # Move completed job to history
            self.job_history.append(job)
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
    
    async def _analyze_dataset_background(self, job_id: str, dataset_id: str):
        """Background task for dataset analysis."""
        job = self.active_jobs[job_id]
        
        try:
            job.status = "running"
            job.message = "Analyzing dataset..."
            
            # Placeholder for actual analysis
            await asyncio.sleep(2)  # Simulate processing time
            
            job.status = "completed"
            job.message = "Analysis completed successfully"
            job.completed_at = datetime.now()
            job.result = {
                "dataset_id": dataset_id,
                "analysis_type": "comprehensive",
                "metrics": {
                    "quality_score": 0.85,
                    "language_distribution": {"en": 0.8, "es": 0.2},
                    "domain_distribution": {"technical": 0.6, "general": 0.4}
                }
            }
            
        except Exception as e:
            job.status = "failed"
            job.message = f"Analysis failed: {str(e)}"
            job.completed_at = datetime.now()
            job.errors = [{"error": str(e), "type": type(e).__name__}]
        
        finally:
            # Move completed job to history
            self.job_history.append(job)
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
    
    def run(self, **kwargs):
        """Run the API server."""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            **kwargs
        )
    
    def get_app(self):
        """Get the FastAPI application instance."""
        return self.app


# Factory function for creating server instance
def create_api_server(config_path: str = None, host: str = "localhost", port: int = 8000) -> RESTAPIServer:
    """
    Create and configure a REST API server instance.
    
    Args:
        config_path: Path to configuration file
        host: Server host address
        port: Server port number
        
    Returns:
        Configured RESTAPIServer instance
    """
    return RESTAPIServer(config_path=config_path, host=host, port=port)


# CLI entry point for running the server
def main():
    """Main entry point for running the API server from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="QuData REST API Server")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    server = create_api_server(
        config_path=args.config,
        host=args.host,
        port=args.port
    )
    
    print(f"Starting QuData API server on {args.host}:{args.port}")
    if args.config:
        print(f"Using configuration: {args.config}")
    
    server.run(reload=args.reload)


if __name__ == "__main__":
    main()

# Expose module-level FastAPI app for Uvicorn/Gunicorn
# Allows running with: `uvicorn qudata.api.rest_server:app` or gunicorn with UvicornWorker
try:
    _server_instance = RESTAPIServer(
        config_path=os.getenv("QUDATA_CONFIG_FILE"),
        host=os.getenv("API_HOST", "localhost"),
        port=int(os.getenv("API_PORT", "8000")),
    )
    app = _server_instance.get_app()
except Exception as e:
    # Fallback minimal app to surface initialization errors via /health
    from fastapi import FastAPI
    app = FastAPI(title="QuData API (init failed)")
    @app.get("/health")
    def _health():
        return {"status": "error", "detail": str(e)}