"""
API Gateway and External Interfaces

This module provides comprehensive API access to the QuData system including:
- REST API server for HTTP-based integration
- GraphQL endpoint for flexible data queries
- Webhook manager for event-driven integrations
- Enhanced CLI interface for command-line operations
- SDK generator for client library generation
"""

# Import with error handling for optional dependencies
try:
    from .rest_server import RESTAPIServer, create_api_server
    REST_AVAILABLE = True
except ImportError as e:
    REST_AVAILABLE = False
    print(f"Warning: REST API not available: {e}")

try:
    from .graphql_endpoint import create_graphql_router, get_graphql_context
    GRAPHQL_AVAILABLE = True
except ImportError as e:
    GRAPHQL_AVAILABLE = False
    print(f"Warning: GraphQL not available: {e}")

try:
    from .webhook_manager import (
        WebhookManager, WebhookEvent, WebhookStatus, WebhookEndpoint,
        get_webhook_manager, create_webhook_routes
    )
    WEBHOOKS_AVAILABLE = True
except ImportError as e:
    WEBHOOKS_AVAILABLE = False
    print(f"Warning: Webhooks not available: {e}")

try:
    from .sdk_generator import SDKGenerator, SDKLanguage, SDKConfig
    SDK_AVAILABLE = True
except ImportError as e:
    SDK_AVAILABLE = False
    print(f"Warning: SDK Generator not available: {e}")

# Build __all__ based on available components
__all__ = []

if REST_AVAILABLE:
    __all__.extend(["RESTAPIServer", "create_api_server"])

if GRAPHQL_AVAILABLE:
    __all__.extend(["create_graphql_router", "get_graphql_context"])

if WEBHOOKS_AVAILABLE:
    __all__.extend([
        "WebhookManager", "WebhookEvent", "WebhookStatus", "WebhookEndpoint",
        "get_webhook_manager", "create_webhook_routes"
    ])

if SDK_AVAILABLE:
    __all__.extend(["SDKGenerator", "SDKLanguage", "SDKConfig"])