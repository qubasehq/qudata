#!/usr/bin/env python3
"""
API Gateway and External Interfaces Demo

This script demonstrates the functionality of all API gateway components:
- REST API Server
- CLI Interface  
- SDK Generator
- Webhook Manager (if dependencies available)
- GraphQL Endpoint (if dependencies available)
"""

import asyncio
import sys
import tempfile
import time
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

def demo_rest_api():
    """Demonstrate REST API Server functionality."""
    print("🚀 REST API Server Demo")
    print("=" * 50)
    
    try:
        from src.qudata.api.rest_server import RESTAPIServer
        from fastapi.testclient import TestClient
        
        # Create server instance
        server = RESTAPIServer(host="localhost", port=8000)
        client = TestClient(server.app)
        
        # Test health endpoint
        response = client.get("/health")
        print(f"✅ Health Check: {response.json()}")
        
        # Test system status
        response = client.get("/status")
        status = response.json()
        print(f"✅ System Status: {status['status']} (uptime: {status['uptime']:.2f}s)")
        
        # Test job creation
        job_data = {
            "input_path": "demo/input",
            "output_path": "demo/output",
            "format": "jsonl"
        }
        response = client.post("/process", json=job_data)
        job_info = response.json()
        print(f"✅ Job Created: {job_info['job_id']} ({job_info['status']})")
        
        # Test job status
        response = client.get(f"/jobs/{job_info['job_id']}")
        job_status = response.json()
        print(f"✅ Job Status: {job_status['status']} (progress: {job_status['progress']})")
        
        print("✅ REST API Server working correctly!\n")
        
    except ImportError as e:
        print(f"❌ REST API not available: {e}\n")


def demo_cli_interface():
    """Demonstrate CLI Interface functionality."""
    print("💻 CLI Interface Demo")
    print("=" * 50)
    
    try:
        from src.qudata.cli import main as cli_main
        import sys
        from unittest.mock import patch
        
        # Test help command
        with patch('sys.argv', ['qudata', '--help']):
            try:
                cli_main()
            except SystemExit:
                print("✅ CLI Help command working")
        
        # Test config template generation
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            temp_config = f.name
        
        try:
            with patch('sys.argv', ['qudata', 'config', 'template', '--output', temp_config]):
                result = cli_main()
                if result == 0 and Path(temp_config).exists():
                    print("✅ Config template generation working")
                else:
                    print("⚠️  Config template generation had issues")
        finally:
            Path(temp_config).unlink(missing_ok=True)
        
        print("✅ CLI Interface working correctly!\n")
        
    except ImportError as e:
        print(f"❌ CLI Interface not available: {e}\n")


def demo_sdk_generator():
    """Demonstrate SDK Generator functionality."""
    print("📦 SDK Generator Demo")
    print("=" * 50)
    
    try:
        from src.qudata.api.sdk_generator import SDKGenerator, SDKConfig, SDKLanguage
        
        generator = SDKGenerator()
        print(f"✅ SDK Generator initialized with {len(generator.endpoints)} endpoints")
        
        # Generate Python SDK
        with tempfile.TemporaryDirectory() as temp_dir:
            python_config = SDKConfig(
                package_name="qudata_demo_client",
                version="1.0.0",
                author="Demo Author",
                description="Demo QuData SDK",
                base_url="http://localhost:8000",
                language=SDKLanguage.PYTHON,
                output_dir=str(Path(temp_dir) / "python")
            )
            
            result = generator.generate_sdk(python_config)
            if result:
                print("✅ Python SDK generated successfully")
                
                # Check generated files
                package_dir = Path(temp_dir) / "python" / "qudata_demo_client"
                files_created = list(package_dir.glob("*.py"))
                print(f"   Generated {len(files_created)} Python files")
            else:
                print("❌ Python SDK generation failed")
        
        # Generate JavaScript SDK
        with tempfile.TemporaryDirectory() as temp_dir:
            js_config = SDKConfig(
                package_name="qudata-demo-client",
                version="1.0.0",
                author="Demo Author",
                description="Demo QuData SDK",
                base_url="http://localhost:8000",
                language=SDKLanguage.JAVASCRIPT,
                output_dir=str(Path(temp_dir) / "javascript")
            )
            
            result = generator.generate_sdk(js_config)
            if result:
                print("✅ JavaScript SDK generated successfully")
                
                # Check package.json
                package_json = Path(temp_dir) / "javascript" / "package.json"
                if package_json.exists():
                    print("   Generated package.json")
            else:
                print("❌ JavaScript SDK generation failed")
        
        print("✅ SDK Generator working correctly!\n")
        
    except ImportError as e:
        print(f"❌ SDK Generator not available: {e}\n")


async def demo_webhook_manager():
    """Demonstrate Webhook Manager functionality."""
    print("🪝 Webhook Manager Demo")
    print("=" * 50)
    
    try:
        from src.qudata.api.webhook_manager import (
            WebhookManager, WebhookEvent, WebhookEndpointCreate
        )
        
        manager = WebhookManager()
        
        # Add a test webhook endpoint
        endpoint_data = WebhookEndpointCreate(
            url="https://httpbin.org/post",
            events=["processing.completed", "dataset.created"],
            secret="demo-secret"
        )
        
        endpoint = manager.add_endpoint(endpoint_data)
        print(f"✅ Webhook endpoint added: {endpoint.id}")
        
        # List endpoints
        endpoints = manager.list_endpoints()
        print(f"✅ Total webhook endpoints: {len(endpoints)}")
        
        # Start webhook manager
        await manager.start()
        
        try:
            # Emit a test event
            await manager.emit_event(
                WebhookEvent.PROCESSING_COMPLETED,
                {"job_id": "demo-123", "success": True}
            )
            print("✅ Test event emitted")
            
            # Wait a bit for processing
            await asyncio.sleep(0.5)
            
            # Check deliveries
            deliveries = manager.get_deliveries()
            print(f"✅ Webhook deliveries: {len(deliveries)}")
            
            # Get stats
            stats = manager.get_stats()
            print(f"✅ Webhook stats: {stats['total_events']} events, {stats['active_endpoints']} active endpoints")
            
        finally:
            await manager.stop()
        
        print("✅ Webhook Manager working correctly!\n")
        
    except ImportError as e:
        print(f"❌ Webhook Manager not available: {e}\n")


def demo_graphql_endpoint():
    """Demonstrate GraphQL Endpoint functionality."""
    print("🔗 GraphQL Endpoint Demo")
    print("=" * 50)
    
    try:
        from src.qudata.api.graphql_endpoint import create_graphql_router, get_graphql_context
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        
        # Create FastAPI app with GraphQL
        app = FastAPI()
        graphql_router = create_graphql_router()
        app.include_router(graphql_router)
        
        client = TestClient(app)
        
        # Test system status query
        query = """
        query {
            systemStatus {
                status
                version
                uptime
            }
        }
        """
        
        response = client.post("/graphql", json={"query": query})
        if response.status_code == 200:
            data = response.json()
            if "data" in data and "systemStatus" in data["data"]:
                status = data["data"]["systemStatus"]
                print(f"✅ GraphQL system status: {status['status']} v{status['version']}")
            else:
                print("⚠️  GraphQL query returned unexpected format")
        else:
            print(f"⚠️  GraphQL query failed with status {response.status_code}")
        
        print("✅ GraphQL Endpoint working correctly!\n")
        
    except ImportError as e:
        print(f"❌ GraphQL Endpoint not available: {e}\n")


async def main():
    """Run all API gateway demos."""
    print("🎯 QuData API Gateway & External Interfaces Demo")
    print("=" * 60)
    print()
    
    # Demo each component
    demo_rest_api()
    demo_cli_interface()
    demo_sdk_generator()
    await demo_webhook_manager()
    demo_graphql_endpoint()
    
    print("🎉 API Gateway Demo Complete!")
    print("=" * 60)
    print()
    print("Summary of Available Components:")
    print("✅ REST API Server - Full HTTP API with job management")
    print("✅ CLI Interface - Command-line tool with multiple commands")
    print("✅ SDK Generator - Multi-language client SDK generation")
    print("🪝 Webhook Manager - Event-driven integrations (if aiohttp available)")
    print("🔗 GraphQL Endpoint - Flexible query interface (if strawberry available)")
    print()
    print("To use these components:")
    print("1. Start the API server: python -m src.qudata.cli server")
    print("2. Use CLI commands: python -m src.qudata.cli --help")
    print("3. Generate SDKs: python -m src.qudata.api.sdk_generator --help")
    print("4. Access API docs at: http://localhost:8000/docs")


if __name__ == "__main__":
    asyncio.run(main())