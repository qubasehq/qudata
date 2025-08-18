# API Gateway and External Interfaces

The API module provides comprehensive external access to the QuData system through multiple interfaces:

## Components

### REST API Server (`rest_server.py`)
- HTTP-based API for external system integration
- RESTful endpoints for all major pipeline operations
- Authentication and rate limiting support
- OpenAPI/Swagger documentation generation

### GraphQL Endpoint (`graphql_endpoint.py`)
- Flexible data queries with GraphQL
- Schema introspection and type safety
- Real-time subscriptions for pipeline status
- Efficient data fetching with field selection

### Webhook Manager (`webhook_manager.py`)
- Event-driven integrations with external systems
- Configurable webhook endpoints and payloads
- Retry logic and failure handling
- Webhook security with signature verification

### SDK Generator (`sdk_generator.py`)
- Automatic client library generation
- Support for multiple programming languages
- Type-safe client interfaces
- Documentation and examples generation

## Usage Examples

### Starting the REST API Server
```python
from qudata.api import RESTAPIServer

server = RESTAPIServer(host="0.0.0.0", port=8000)
server.start()
```

### Setting up GraphQL
```python
from qudata.api import create_graphql_router

router = create_graphql_router()
# Add to your FastAPI app
```

### Managing Webhooks
```python
from qudata.api import WebhookManager

webhook_manager = WebhookManager()
webhook_manager.register_endpoint("pipeline_complete", "https://example.com/webhook")
```

### Generating SDKs
```python
from qudata.api import SDKGenerator, SDKLanguage

generator = SDKGenerator()
generator.generate_sdk(SDKLanguage.PYTHON, output_dir="./sdk/python")
```

## Configuration

API components can be configured through the main pipeline configuration:

```yaml
api:
  rest:
    host: "0.0.0.0"
    port: 8000
    enable_docs: true
  graphql:
    enable_playground: true
  webhooks:
    max_retries: 3
    timeout: 30
  sdk:
    languages: ["python", "javascript", "go"]
```

## Security

- API key authentication for REST endpoints
- JWT token support for user sessions
- Rate limiting and request throttling
- CORS configuration for web clients
- Webhook signature verification

## Dependencies

- FastAPI for REST API framework
- Strawberry GraphQL for GraphQL implementation
- Pydantic for data validation
- Jinja2 for SDK template generation