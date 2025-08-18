"""
Unit tests for API Gateway and External Interfaces.

Tests for REST API Server, GraphQL Endpoint, Webhook Manager, 
CLI Interface, and SDK Generator components.
"""

import asyncio
import json
import pytest
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from fastapi.testclient import TestClient
import aiohttp

# Import with error handling for optional dependencies
try:
    from src.qudata.api.rest_server import RESTAPIServer, create_api_server
    REST_AVAILABLE = True
except ImportError:
    REST_AVAILABLE = False

try:
    from src.qudata.api.graphql_endpoint import GraphQLContext, create_graphql_router
    GRAPHQL_AVAILABLE = True
except ImportError:
    GRAPHQL_AVAILABLE = False

try:
    from src.qudata.api.webhook_manager import (
        WebhookManager, WebhookEndpoint, WebhookEvent, WebhookStatus,
        WebhookEndpointCreate, WebhookEndpointUpdate
    )
    WEBHOOKS_AVAILABLE = True
except ImportError:
    WEBHOOKS_AVAILABLE = False

try:
    from src.qudata.api.sdk_generator import SDKGenerator, SDKConfig, SDKLanguage
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

try:
    from src.qudata.cli import main as cli_main
    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False

from src.qudata.models import ProcessingResult, Dataset, Document


@pytest.mark.skipif(not REST_AVAILABLE, reason="REST API dependencies not available")
class TestRESTAPIServer:
    """Test cases for REST API Server."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.server = RESTAPIServer(host="localhost", port=8000)
        self.client = TestClient(self.server.app)
    
    def test_root_endpoint(self):
        """Test root endpoint returns API information."""
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "QuData API"
        assert data["version"] == "1.0.0"
        assert "docs" in data
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_system_status(self):
        """Test system status endpoint."""
        response = self.client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["version"] == "1.0.0"
        assert "uptime" in data
        assert "active_jobs" in data
        assert "memory_usage" in data
        assert "disk_usage" in data
    
    def test_start_processing(self):
        """Test start processing endpoint."""
        request_data = {
            "input_path": "test/input",
            "output_path": "test/output",
            "format": "jsonl"
        }
        
        response = self.client.post("/process", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        assert data["message"] == "Processing job started"
        assert "started_at" in data
    
    def test_get_job_status(self):
        """Test get job status endpoint."""
        # First start a job
        request_data = {
            "input_path": "test/input",
            "output_path": "test/output"
        }
        
        start_response = self.client.post("/process", json=request_data)
        job_id = start_response.json()["job_id"]
        
        # Then get its status
        response = self.client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert "status" in data
        assert "progress" in data
        assert "message" in data
    
    def test_list_jobs(self):
        """Test list jobs endpoint."""
        response = self.client.get("/jobs")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_list_jobs_with_filters(self):
        """Test list jobs with status filter."""
        response = self.client.get("/jobs?status=completed&limit=10")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_cancel_job(self):
        """Test cancel job endpoint."""
        # First start a job
        request_data = {
            "input_path": "test/input",
            "output_path": "test/output"
        }
        
        start_response = self.client.post("/process", json=request_data)
        job_id = start_response.json()["job_id"]
        
        # Then cancel it
        response = self.client.delete(f"/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Job cancelled successfully"
    
    def test_cancel_nonexistent_job(self):
        """Test canceling non-existent job returns 404."""
        fake_job_id = str(uuid.uuid4())
        response = self.client.delete(f"/jobs/{fake_job_id}")
        assert response.status_code == 404
    
    def test_list_datasets(self):
        """Test list datasets endpoint."""
        response = self.client.get("/datasets")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_configuration(self):
        """Test get configuration endpoint."""
        response = self.client.get("/config")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data  # No config loaded by default
    
    @patch('src.qudata.pipeline.QuDataPipeline')
    def test_reload_configuration(self, mock_pipeline):
        """Test reload configuration endpoint."""
        mock_pipeline.return_value = Mock()
        
        response = self.client.post("/config/reload", params={"config_path": "test_config.yaml"})
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Configuration reloaded successfully"


@pytest.mark.skipif(not GRAPHQL_AVAILABLE, reason="GraphQL dependencies not available")
class TestGraphQLEndpoint:
    """Test cases for GraphQL Endpoint."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from fastapi import FastAPI
        self.app = FastAPI()
        self.graphql_router = create_graphql_router()
        self.app.include_router(self.graphql_router)
        self.client = TestClient(self.app)
    
    def test_graphql_endpoint_exists(self):
        """Test GraphQL endpoint is accessible."""
        # Test GraphQL playground/introspection
        response = self.client.get("/graphql")
        # GraphQL endpoint should return 405 for GET without query
        assert response.status_code in [200, 405]
    
    def test_system_status_query(self):
        """Test GraphQL system status query."""
        query = """
        query {
            systemStatus {
                status
                version
                uptime
                activeJobs
            }
        }
        """
        
        response = self.client.post("/graphql", json={"query": query})
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "systemStatus" in data["data"]
        
        status = data["data"]["systemStatus"]
        assert status["status"] == "running"
        assert status["version"] == "1.0.0"
    
    def test_datasets_query(self):
        """Test GraphQL datasets query."""
        query = """
        query {
            datasets {
                id
                name
                version
                documentCount
            }
        }
        """
        
        response = self.client.post("/graphql", json={"query": query})
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "datasets" in data["data"]
        assert isinstance(data["data"]["datasets"], list)
    
    def test_jobs_query(self):
        """Test GraphQL jobs query."""
        query = """
        query {
            jobs {
                jobId
                status
                progress
                message
            }
        }
        """
        
        response = self.client.post("/graphql", json={"query": query})
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "jobs" in data["data"]
        assert isinstance(data["data"]["jobs"], list)
    
    def test_start_processing_mutation(self):
        """Test GraphQL start processing mutation."""
        mutation = """
        mutation {
            startProcessing(input: {
                inputPath: "test/input"
                outputPath: "test/output"
                format: "jsonl"
            }) {
                jobId
                status
                message
            }
        }
        """
        
        response = self.client.post("/graphql", json={"query": mutation})
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "startProcessing" in data["data"]
        
        result = data["data"]["startProcessing"]
        assert "jobId" in result
        assert result["status"] == "pending"


@pytest.mark.skipif(not WEBHOOKS_AVAILABLE, reason="Webhook dependencies not available")
@pytest.mark.asyncio
class TestWebhookManager:
    """Test cases for Webhook Manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.webhook_manager = WebhookManager()
    
    async def test_add_endpoint(self):
        """Test adding webhook endpoint."""
        endpoint_data = WebhookEndpointCreate(
            url="https://example.com/webhook",
            events=["processing.completed", "dataset.created"],
            secret="test-secret"
        )
        
        endpoint = self.webhook_manager.add_endpoint(endpoint_data)
        
        assert endpoint.id is not None
        assert endpoint.url == "https://example.com/webhook"
        assert len(endpoint.events) == 2
        assert WebhookEvent.PROCESSING_COMPLETED in endpoint.events
        assert WebhookEvent.DATASET_CREATED in endpoint.events
        assert endpoint.secret == "test-secret"
        assert endpoint.active is True
    
    def test_add_endpoint_invalid_event(self):
        """Test adding endpoint with invalid event type."""
        endpoint_data = WebhookEndpointCreate(
            url="https://example.com/webhook",
            events=["invalid.event"],
        )
        
        with pytest.raises(ValueError, match="Invalid event type"):
            self.webhook_manager.add_endpoint(endpoint_data)
    
    def test_update_endpoint(self):
        """Test updating webhook endpoint."""
        # First add an endpoint
        endpoint_data = WebhookEndpointCreate(
            url="https://example.com/webhook",
            events=["processing.completed"]
        )
        endpoint = self.webhook_manager.add_endpoint(endpoint_data)
        
        # Then update it
        update_data = WebhookEndpointUpdate(
            url="https://updated.example.com/webhook",
            active=False
        )
        
        updated_endpoint = self.webhook_manager.update_endpoint(endpoint.id, update_data)
        
        assert updated_endpoint.url == "https://updated.example.com/webhook"
        assert updated_endpoint.active is False
    
    def test_remove_endpoint(self):
        """Test removing webhook endpoint."""
        # First add an endpoint
        endpoint_data = WebhookEndpointCreate(
            url="https://example.com/webhook",
            events=["processing.completed"]
        )
        endpoint = self.webhook_manager.add_endpoint(endpoint_data)
        
        # Then remove it
        result = self.webhook_manager.remove_endpoint(endpoint.id)
        assert result is True
        
        # Verify it's gone
        assert self.webhook_manager.get_endpoint(endpoint.id) is None
    
    def test_list_endpoints(self):
        """Test listing webhook endpoints."""
        # Add some endpoints
        for i in range(3):
            endpoint_data = WebhookEndpointCreate(
                url=f"https://example{i}.com/webhook",
                events=["processing.completed"]
            )
            self.webhook_manager.add_endpoint(endpoint_data)
        
        endpoints = self.webhook_manager.list_endpoints()
        assert len(endpoints) == 3
    
    async def test_emit_event(self):
        """Test emitting webhook event."""
        # Add an endpoint
        endpoint_data = WebhookEndpointCreate(
            url="https://httpbin.org/post",
            events=["processing.completed"]
        )
        endpoint = self.webhook_manager.add_endpoint(endpoint_data)
        
        # Start webhook manager
        await self.webhook_manager.start()
        
        try:
            # Emit event
            await self.webhook_manager.emit_event(
                WebhookEvent.PROCESSING_COMPLETED,
                {"job_id": "test-123", "success": True}
            )
            
            # Wait a bit for processing
            await asyncio.sleep(0.1)
            
            # Check deliveries
            deliveries = self.webhook_manager.get_deliveries()
            assert len(deliveries) >= 1
            
            delivery = deliveries[0]
            assert delivery.endpoint_id == endpoint.id
            assert delivery.event_type == WebhookEvent.PROCESSING_COMPLETED
            
        finally:
            await self.webhook_manager.stop()
    
    def test_generate_signature(self):
        """Test webhook signature generation."""
        payload = '{"test": "data"}'
        secret = "test-secret"
        
        signature = self.webhook_manager._generate_signature(payload, secret)
        
        assert signature.startswith("sha256=")
        assert len(signature) > 10
    
    def test_get_stats(self):
        """Test getting webhook statistics."""
        stats = self.webhook_manager.get_stats()
        
        assert "total_events" in stats
        assert "total_deliveries" in stats
        assert "successful_deliveries" in stats
        assert "failed_deliveries" in stats
        assert "active_endpoints" in stats
    
    async def test_helper_methods(self):
        """Test webhook helper methods."""
        # Add an endpoint
        endpoint_data = WebhookEndpointCreate(
            url="https://httpbin.org/post",
            events=["processing.started", "processing.completed", "dataset.created"]
        )
        self.webhook_manager.add_endpoint(endpoint_data)
        
        await self.webhook_manager.start()
        
        try:
            # Test processing started
            await self.webhook_manager.emit_processing_started("job-1", "input/path")
            
            # Test processing completed
            result = ProcessingResult(success=True, processing_time=10.5)
            await self.webhook_manager.emit_processing_completed("job-1", result)
            
            # Test dataset created
            dataset = Mock()
            dataset.id = "dataset-1"
            dataset.name = "test-dataset"
            dataset.version = "1.0"
            dataset.get_document_count.return_value = 100
            
            await self.webhook_manager.emit_dataset_created(dataset)
            
            # Wait for processing
            await asyncio.sleep(0.1)
            
            # Check that events were emitted
            stats = self.webhook_manager.get_stats()
            assert stats["total_events"] >= 3
            
        finally:
            await self.webhook_manager.stop()


@pytest.mark.skipif(not SDK_AVAILABLE, reason="SDK Generator dependencies not available")
class TestSDKGenerator:
    """Test cases for SDK Generator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = SDKGenerator()
        self.temp_dir = tempfile.mkdtemp()
    
    def test_initialization(self):
        """Test SDK generator initialization."""
        assert len(self.generator.endpoints) > 0
        assert len(self.generator.models) > 0
        
        # Check some expected endpoints
        endpoint_names = [ep.name for ep in self.generator.endpoints]
        assert "health_check" in endpoint_names
        assert "start_processing" in endpoint_names
        assert "get_job_status" in endpoint_names
    
    def test_generate_python_sdk(self):
        """Test generating Python SDK."""
        config = SDKConfig(
            package_name="qudata_client",
            version="1.0.0",
            author="Test Author",
            description="Test SDK",
            base_url="http://localhost:8000",
            language=SDKLanguage.PYTHON,
            output_dir=self.temp_dir
        )
        
        result = self.generator.generate_sdk(config)
        assert result is True
        
        # Check generated files
        package_dir = Path(self.temp_dir) / "qudata_client"
        assert package_dir.exists()
        assert (package_dir / "__init__.py").exists()
        assert (package_dir / "client.py").exists()
        assert (package_dir / "models.py").exists()
        assert (package_dir / "exceptions.py").exists()
        
        # Check setup.py
        assert (Path(self.temp_dir) / "setup.py").exists()
        assert (Path(self.temp_dir) / "README.md").exists()
        
        # Check examples and tests
        assert (Path(self.temp_dir) / "examples" / "basic_usage.py").exists()
        assert (Path(self.temp_dir) / "tests" / "test_client.py").exists()
    
    def test_generate_javascript_sdk(self):
        """Test generating JavaScript SDK."""
        config = SDKConfig(
            package_name="qudata-client",
            version="1.0.0",
            author="Test Author",
            description="Test SDK",
            base_url="http://localhost:8000",
            language=SDKLanguage.JAVASCRIPT,
            output_dir=self.temp_dir
        )
        
        result = self.generator.generate_sdk(config)
        assert result is True
        
        # Check generated files
        assert (Path(self.temp_dir) / "package.json").exists()
        assert (Path(self.temp_dir) / "index.js").exists()
        
        # Check package.json content
        with open(Path(self.temp_dir) / "package.json") as f:
            package_data = json.load(f)
            assert package_data["name"] == "qudata-client"
            assert package_data["version"] == "1.0.0"
    
    def test_generate_unsupported_language(self):
        """Test generating SDK for unsupported language."""
        config = SDKConfig(
            package_name="qudata_client",
            version="1.0.0",
            author="Test Author",
            description="Test SDK",
            base_url="http://localhost:8000",
            language=SDKLanguage.RUBY,  # Not implemented
            output_dir=self.temp_dir
        )
        
        result = self.generator.generate_sdk(config)
        assert result is False
    
    def test_python_type_conversion(self):
        """Test Python type conversion from JSON schema."""
        # Test basic types
        assert self.generator._python_type_from_schema({"type": "string"}) == "str"
        assert self.generator._python_type_from_schema({"type": "integer"}) == "int"
        assert self.generator._python_type_from_schema({"type": "number"}) == "float"
        assert self.generator._python_type_from_schema({"type": "boolean"}) == "bool"
        
        # Test array type
        array_schema = {"type": "array", "items": {"type": "string"}}
        assert self.generator._python_type_from_schema(array_schema) == "List[str]"
        
        # Test object type
        assert self.generator._python_type_from_schema({"type": "object"}) == "Dict[str, Any]"


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI dependencies not available")
class TestCLIInterface:
    """Test cases for CLI Interface."""
    
    def test_cli_help(self):
        """Test CLI help output."""
        with patch('sys.argv', ['qudata', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                cli_main()
            assert exc_info.value.code == 0
    
    def test_cli_no_command(self):
        """Test CLI with no command shows help."""
        with patch('sys.argv', ['qudata']):
            result = cli_main()
            assert result == 1
    
    @patch('src.qudata.pipeline.QuDataPipeline')
    def test_process_command(self, mock_pipeline):
        """Test process command."""
        mock_result = Mock()
        mock_result.success = True
        mock_result.documents_processed = 10
        mock_result.processing_time = 5.0
        mock_result.output_paths = {"jsonl": "output.jsonl"}
        mock_result.errors = []
        
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.process_directory.return_value = mock_result
        mock_pipeline.return_value = mock_pipeline_instance
        
        with patch('sys.argv', ['qudata', 'process', '--input', 'test_input', '--output', 'test_output']):
            result = cli_main()
            assert result == 0
            mock_pipeline_instance.process_directory.assert_called_once()
    
    @patch('src.qudata.api.rest_server.create_api_server')
    def test_server_command(self, mock_create_server):
        """Test server command."""
        mock_server = Mock()
        mock_server.app = Mock()
        mock_server.run = Mock()
        mock_create_server.return_value = mock_server
        
        with patch('sys.argv', ['qudata', 'server', '--host', '0.0.0.0', '--port', '9000']):
            result = cli_main()
            assert result == 0
            mock_create_server.assert_called_once()
            mock_server.run.assert_called_once()
    
    def test_webhook_add_command(self):
        """Test webhook add command."""
        with patch('src.qudata.api.webhook_manager.get_webhook_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_endpoint = Mock()
            mock_endpoint.id = "test-id"
            mock_endpoint.url = "https://example.com/webhook"
            mock_endpoint.events = [Mock(value="processing.completed")]
            mock_manager.add_endpoint.return_value = mock_endpoint
            mock_get_manager.return_value = mock_manager
            
            with patch('sys.argv', [
                'qudata', 'webhook', 'add', 
                '--url', 'https://example.com/webhook',
                '--events', 'processing.completed'
            ]):
                result = cli_main()
                assert result == 0
                mock_manager.add_endpoint.assert_called_once()
    
    def test_webhook_list_command(self):
        """Test webhook list command."""
        with patch('src.qudata.api.webhook_manager.get_webhook_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_endpoint = Mock()
            mock_endpoint.id = "test-id"
            mock_endpoint.url = "https://example.com/webhook"
            mock_endpoint.active = True
            mock_endpoint.events = [Mock(value="processing.completed")]
            mock_endpoint.created_at = datetime.now()
            mock_endpoint.to_dict.return_value = {"id": "test-id"}
            mock_manager.list_endpoints.return_value = [mock_endpoint]
            mock_get_manager.return_value = mock_manager
            
            with patch('sys.argv', ['qudata', 'webhook', 'list']):
                result = cli_main()
                assert result == 0
                mock_manager.list_endpoints.assert_called_once()
    
    def test_config_validate_command(self):
        """Test config validate command."""
        with patch('src.qudata.config.ConfigManager') as mock_config_manager:
            mock_manager = Mock()
            mock_manager.load_config.return_value = Mock()
            mock_config_manager.return_value = mock_manager
            
            with patch('sys.argv', ['qudata', 'config', 'validate', '--file', 'test_config.yaml']):
                result = cli_main()
                assert result == 0
                mock_manager.load_config.assert_called_once_with('test_config.yaml')
    
    def test_config_template_command(self):
        """Test config template generation command."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_file = f.name
        
        try:
            with patch('sys.argv', ['qudata', 'config', 'template', '--output', temp_file, '--type', 'pipeline']):
                result = cli_main()
                assert result == 0
                
                # Check that file was created
                assert Path(temp_file).exists()
                
                # Check content
                with open(temp_file) as f:
                    import yaml
                    config = yaml.safe_load(f)
                    assert 'version' in config
                    assert 'stages' in config
        finally:
            Path(temp_file).unlink(missing_ok=True)


class TestIntegrationScenarios:
    """Integration test scenarios for API components."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.server = RESTAPIServer(host="localhost", port=8000)
        self.client = TestClient(self.server.app)
        self.webhook_manager = WebhookManager()
    
    def test_full_processing_workflow(self):
        """Test complete processing workflow through API."""
        # 1. Check system health
        health_response = self.client.get("/health")
        assert health_response.status_code == 200
        
        # 2. Start processing job
        process_request = {
            "input_path": "test/input",
            "output_path": "test/output",
            "format": "jsonl"
        }
        
        process_response = self.client.post("/process", json=process_request)
        assert process_response.status_code == 200
        job_id = process_response.json()["job_id"]
        
        # 3. Check job status
        status_response = self.client.get(f"/jobs/{job_id}")
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["job_id"] == job_id
        
        # 4. List all jobs
        jobs_response = self.client.get("/jobs")
        assert jobs_response.status_code == 200
        jobs_data = jobs_response.json()
        job_ids = [job["job_id"] for job in jobs_data]
        assert job_id in job_ids
    
    @pytest.mark.asyncio
    async def test_webhook_integration(self):
        """Test webhook integration with processing events."""
        await self.webhook_manager.start()
        
        try:
            # Add webhook endpoint
            endpoint_data = WebhookEndpointCreate(
                url="https://httpbin.org/post",
                events=["processing.started", "processing.completed"]
            )
            endpoint = self.webhook_manager.add_endpoint(endpoint_data)
            
            # Emit processing events
            await self.webhook_manager.emit_processing_started("test-job", "input/path")
            
            result = ProcessingResult(success=True, processing_time=5.0)
            await self.webhook_manager.emit_processing_completed("test-job", result)
            
            # Wait for delivery attempts
            await asyncio.sleep(1.0)
            
            # Check deliveries
            deliveries = self.webhook_manager.get_deliveries(endpoint.id)
            assert len(deliveries) >= 2
            
            # Check event types
            event_types = [d.event_type for d in deliveries]
            assert WebhookEvent.PROCESSING_STARTED in event_types
            assert WebhookEvent.PROCESSING_COMPLETED in event_types
            
        finally:
            await self.webhook_manager.stop()
    
    def test_sdk_generation_integration(self):
        """Test SDK generation for multiple languages."""
        generator = SDKGenerator()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate Python SDK
            python_config = SDKConfig(
                package_name="qudata_client",
                version="1.0.0",
                author="Test Author",
                description="QuData Python SDK",
                base_url="http://localhost:8000",
                language=SDKLanguage.PYTHON,
                output_dir=str(Path(temp_dir) / "python")
            )
            
            python_result = generator.generate_sdk(python_config)
            assert python_result is True
            
            # Generate JavaScript SDK
            js_config = SDKConfig(
                package_name="qudata-client",
                version="1.0.0",
                author="Test Author",
                description="QuData JavaScript SDK",
                base_url="http://localhost:8000",
                language=SDKLanguage.JAVASCRIPT,
                output_dir=str(Path(temp_dir) / "javascript")
            )
            
            js_result = generator.generate_sdk(js_config)
            assert js_result is True
            
            # Verify both SDKs were generated
            python_dir = Path(temp_dir) / "python"
            js_dir = Path(temp_dir) / "javascript"
            
            assert (python_dir / "qudata_client" / "__init__.py").exists()
            assert (js_dir / "package.json").exists()


if __name__ == "__main__":
    pytest.main([__file__])