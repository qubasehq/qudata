"""
Webhook Manager for event-driven integrations.

Manages webhook endpoints, event dispatching, and external system notifications
for pipeline events and data processing updates.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import hmac

import aiohttp
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field, HttpUrl

from ..models import ProcessingResult, Dataset, Document, ProcessingError


class WebhookEvent(Enum):
    """Types of webhook events."""
    PROCESSING_STARTED = "processing.started"
    PROCESSING_COMPLETED = "processing.completed"
    PROCESSING_FAILED = "processing.failed"
    DATASET_CREATED = "dataset.created"
    DATASET_UPDATED = "dataset.updated"
    DATASET_DELETED = "dataset.deleted"
    DOCUMENT_PROCESSED = "document.processed"
    QUALITY_ALERT = "quality.alert"
    SYSTEM_ERROR = "system.error"
    PIPELINE_STATUS_CHANGED = "pipeline.status_changed"


class WebhookStatus(Enum):
    """Webhook delivery status."""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    DISABLED = "disabled"


@dataclass
class WebhookEndpoint:
    """Webhook endpoint configuration."""
    id: str
    url: str
    events: List[WebhookEvent]
    secret: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 60  # seconds
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "url": self.url,
            "events": [event.value for event in self.events],
            "secret": "***" if self.secret else None,
            "headers": self.headers,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "active": self.active,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None
        }


@dataclass
class WebhookDelivery:
    """Webhook delivery attempt record."""
    id: str
    endpoint_id: str
    event_type: WebhookEvent
    payload: Dict[str, Any]
    status: WebhookStatus = WebhookStatus.PENDING
    attempts: int = 0
    last_attempt: Optional[datetime] = None
    next_retry: Optional[datetime] = None
    response_status: Optional[int] = None
    response_body: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "endpoint_id": self.endpoint_id,
            "event_type": self.event_type.value,
            "payload": self.payload,
            "status": self.status.value,
            "attempts": self.attempts,
            "last_attempt": self.last_attempt.isoformat() if self.last_attempt else None,
            "next_retry": self.next_retry.isoformat() if self.next_retry else None,
            "response_status": self.response_status,
            "response_body": self.response_body,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat()
        }


# Pydantic models for API
class WebhookEndpointCreate(BaseModel):
    """Model for creating webhook endpoints."""
    url: HttpUrl = Field(..., description="Webhook URL")
    events: List[str] = Field(..., description="List of event types to subscribe to")
    secret: Optional[str] = Field(None, description="Secret for signature verification")
    headers: Dict[str, str] = Field(default_factory=dict, description="Additional headers")
    timeout: int = Field(30, ge=1, le=300, description="Request timeout in seconds")
    max_retries: int = Field(3, ge=0, le=10, description="Maximum retry attempts")
    retry_delay: int = Field(60, ge=1, le=3600, description="Retry delay in seconds")


class WebhookEndpointUpdate(BaseModel):
    """Model for updating webhook endpoints."""
    url: Optional[HttpUrl] = None
    events: Optional[List[str]] = None
    secret: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    timeout: Optional[int] = Field(None, ge=1, le=300)
    max_retries: Optional[int] = Field(None, ge=0, le=10)
    retry_delay: Optional[int] = Field(None, ge=1, le=3600)
    active: Optional[bool] = None


class WebhookEventPayload(BaseModel):
    """Base webhook event payload."""
    event_id: str = Field(..., description="Unique event identifier")
    event_type: str = Field(..., description="Type of event")
    timestamp: datetime = Field(..., description="Event timestamp")
    data: Dict[str, Any] = Field(..., description="Event data")
    source: str = Field("qudata", description="Event source system")


class WebhookManager:
    """Manager for webhook endpoints and event delivery."""
    
    def __init__(self):
        """Initialize webhook manager."""
        self.endpoints: Dict[str, WebhookEndpoint] = {}
        self.deliveries: Dict[str, WebhookDelivery] = {}
        self.event_handlers: Dict[WebhookEvent, List[Callable]] = {}
        self.delivery_queue: asyncio.Queue = asyncio.Queue()
        self.retry_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        self.delivery_task: Optional[asyncio.Task] = None
        self.retry_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            "total_events": 0,
            "total_deliveries": 0,
            "successful_deliveries": 0,
            "failed_deliveries": 0,
            "active_endpoints": 0
        }
        
        self.logger = logging.getLogger(__name__)
    
    def add_endpoint(self, endpoint_data: WebhookEndpointCreate) -> WebhookEndpoint:
        """Add a new webhook endpoint."""
        endpoint_id = str(uuid.uuid4())
        
        # Validate events
        events = []
        for event_str in endpoint_data.events:
            try:
                events.append(WebhookEvent(event_str))
            except ValueError:
                raise ValueError(f"Invalid event type: {event_str}")
        
        endpoint = WebhookEndpoint(
            id=endpoint_id,
            url=str(endpoint_data.url),
            events=events,
            secret=endpoint_data.secret,
            headers=endpoint_data.headers,
            timeout=endpoint_data.timeout,
            max_retries=endpoint_data.max_retries,
            retry_delay=endpoint_data.retry_delay
        )
        
        self.endpoints[endpoint_id] = endpoint
        self._update_stats()
        
        self.logger.info(f"Added webhook endpoint: {endpoint_id} -> {endpoint.url}")
        return endpoint
    
    def update_endpoint(self, endpoint_id: str, update_data: WebhookEndpointUpdate) -> WebhookEndpoint:
        """Update an existing webhook endpoint."""
        if endpoint_id not in self.endpoints:
            raise ValueError(f"Endpoint not found: {endpoint_id}")
        
        endpoint = self.endpoints[endpoint_id]
        
        # Update fields
        if update_data.url is not None:
            endpoint.url = str(update_data.url)
        if update_data.events is not None:
            events = []
            for event_str in update_data.events:
                try:
                    events.append(WebhookEvent(event_str))
                except ValueError:
                    raise ValueError(f"Invalid event type: {event_str}")
            endpoint.events = events
        if update_data.secret is not None:
            endpoint.secret = update_data.secret
        if update_data.headers is not None:
            endpoint.headers = update_data.headers
        if update_data.timeout is not None:
            endpoint.timeout = update_data.timeout
        if update_data.max_retries is not None:
            endpoint.max_retries = update_data.max_retries
        if update_data.retry_delay is not None:
            endpoint.retry_delay = update_data.retry_delay
        if update_data.active is not None:
            endpoint.active = update_data.active
        
        self._update_stats()
        
        self.logger.info(f"Updated webhook endpoint: {endpoint_id}")
        return endpoint
    
    def remove_endpoint(self, endpoint_id: str) -> bool:
        """Remove a webhook endpoint."""
        if endpoint_id not in self.endpoints:
            return False
        
        del self.endpoints[endpoint_id]
        self._update_stats()
        
        self.logger.info(f"Removed webhook endpoint: {endpoint_id}")
        return True
    
    def get_endpoint(self, endpoint_id: str) -> Optional[WebhookEndpoint]:
        """Get a webhook endpoint by ID."""
        return self.endpoints.get(endpoint_id)
    
    def list_endpoints(self) -> List[WebhookEndpoint]:
        """List all webhook endpoints."""
        return list(self.endpoints.values())
    
    def get_deliveries(self, endpoint_id: str = None, limit: int = 100) -> List[WebhookDelivery]:
        """Get webhook deliveries, optionally filtered by endpoint."""
        deliveries = list(self.deliveries.values())
        
        if endpoint_id:
            deliveries = [d for d in deliveries if d.endpoint_id == endpoint_id]
        
        # Sort by creation time (newest first)
        deliveries.sort(key=lambda x: x.created_at, reverse=True)
        
        return deliveries[:limit]
    
    async def emit_event(self, event_type: WebhookEvent, data: Dict[str, Any]):
        """Emit an event to all subscribed webhooks."""
        event_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        payload = WebhookEventPayload(
            event_id=event_id,
            event_type=event_type.value,
            timestamp=timestamp,
            data=data
        )
        
        # Find matching endpoints
        matching_endpoints = [
            endpoint for endpoint in self.endpoints.values()
            if endpoint.active and event_type in endpoint.events
        ]
        
        if not matching_endpoints:
            self.logger.debug(f"No endpoints subscribed to event: {event_type.value}")
            return
        
        # Create deliveries
        for endpoint in matching_endpoints:
            delivery_id = str(uuid.uuid4())
            delivery = WebhookDelivery(
                id=delivery_id,
                endpoint_id=endpoint.id,
                event_type=event_type,
                payload=payload.dict()
            )
            
            self.deliveries[delivery_id] = delivery
            await self.delivery_queue.put(delivery)
        
        self.stats["total_events"] += 1
        self.logger.info(f"Emitted event {event_type.value} to {len(matching_endpoints)} endpoints")
    
    async def start(self):
        """Start the webhook delivery service."""
        if self.running:
            return
        
        self.running = True
        self.delivery_task = asyncio.create_task(self._delivery_worker())
        self.retry_task = asyncio.create_task(self._retry_worker())
        
        self.logger.info("Webhook manager started")
    
    async def stop(self):
        """Stop the webhook delivery service."""
        if not self.running:
            return
        
        self.running = False
        
        if self.delivery_task:
            self.delivery_task.cancel()
            try:
                await self.delivery_task
            except asyncio.CancelledError:
                pass
        
        if self.retry_task:
            self.retry_task.cancel()
            try:
                await self.retry_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Webhook manager stopped")
    
    async def _delivery_worker(self):
        """Background worker for webhook delivery."""
        while self.running:
            try:
                delivery = await asyncio.wait_for(self.delivery_queue.get(), timeout=1.0)
                await self._deliver_webhook(delivery)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error in delivery worker: {e}")
    
    async def _retry_worker(self):
        """Background worker for webhook retries."""
        while self.running:
            try:
                # Check for deliveries that need retry
                now = datetime.now()
                for delivery in self.deliveries.values():
                    if (delivery.status == WebhookStatus.RETRYING and 
                        delivery.next_retry and 
                        delivery.next_retry <= now):
                        await self.retry_queue.put(delivery)
                
                # Process retry queue
                try:
                    delivery = await asyncio.wait_for(self.retry_queue.get(), timeout=1.0)
                    await self._deliver_webhook(delivery)
                except asyncio.TimeoutError:
                    pass
                
                # Sleep before next check
                await asyncio.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Error in retry worker: {e}")
    
    async def _deliver_webhook(self, delivery: WebhookDelivery):
        """Deliver a webhook to its endpoint."""
        endpoint = self.endpoints.get(delivery.endpoint_id)
        if not endpoint or not endpoint.active:
            delivery.status = WebhookStatus.FAILED
            delivery.error_message = "Endpoint not found or inactive"
            return
        
        delivery.attempts += 1
        delivery.last_attempt = datetime.now()
        delivery.status = WebhookStatus.PENDING
        
        try:
            # Prepare payload
            payload_json = json.dumps(delivery.payload)
            
            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "QuData-Webhook/1.0",
                **endpoint.headers
            }
            
            # Add signature if secret is configured
            if endpoint.secret:
                signature = self._generate_signature(payload_json, endpoint.secret)
                headers["X-QuData-Signature"] = signature
            
            # Make HTTP request
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=endpoint.timeout)) as session:
                async with session.post(
                    endpoint.url,
                    data=payload_json,
                    headers=headers
                ) as response:
                    delivery.response_status = response.status
                    delivery.response_body = await response.text()
                    
                    if 200 <= response.status < 300:
                        delivery.status = WebhookStatus.DELIVERED
                        endpoint.last_used = datetime.now()
                        self.stats["successful_deliveries"] += 1
                        self.logger.info(f"Webhook delivered successfully: {delivery.id}")
                    else:
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=f"HTTP {response.status}"
                        )
        
        except Exception as e:
            delivery.error_message = str(e)
            
            # Determine if we should retry
            if delivery.attempts < endpoint.max_retries:
                delivery.status = WebhookStatus.RETRYING
                delivery.next_retry = datetime.now() + timedelta(seconds=endpoint.retry_delay)
                self.logger.warning(f"Webhook delivery failed, will retry: {delivery.id} - {e}")
            else:
                delivery.status = WebhookStatus.FAILED
                self.stats["failed_deliveries"] += 1
                self.logger.error(f"Webhook delivery failed permanently: {delivery.id} - {e}")
        
        self.stats["total_deliveries"] += 1
    
    def _generate_signature(self, payload: str, secret: str) -> str:
        """Generate HMAC signature for webhook payload."""
        signature = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"
    
    def _update_stats(self):
        """Update statistics."""
        self.stats["active_endpoints"] = sum(1 for ep in self.endpoints.values() if ep.active)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get webhook statistics."""
        return self.stats.copy()
    
    # Event helper methods
    async def emit_processing_started(self, job_id: str, input_path: str, config: Dict[str, Any] = None):
        """Emit processing started event."""
        await self.emit_event(WebhookEvent.PROCESSING_STARTED, {
            "job_id": job_id,
            "input_path": input_path,
            "config": config or {}
        })
    
    async def emit_processing_completed(self, job_id: str, result: ProcessingResult):
        """Emit processing completed event."""
        await self.emit_event(WebhookEvent.PROCESSING_COMPLETED, {
            "job_id": job_id,
            "success": result.success,
            "documents_processed": getattr(result, 'documents_processed', 0),
            "processing_time": result.processing_time,
            "errors": [error.to_dict() if hasattr(error, 'to_dict') else str(error) for error in result.errors]
        })
    
    async def emit_processing_failed(self, job_id: str, error: str):
        """Emit processing failed event."""
        await self.emit_event(WebhookEvent.PROCESSING_FAILED, {
            "job_id": job_id,
            "error": error
        })
    
    async def emit_dataset_created(self, dataset: Dataset):
        """Emit dataset created event."""
        await self.emit_event(WebhookEvent.DATASET_CREATED, {
            "dataset_id": dataset.id,
            "name": dataset.name,
            "version": dataset.version,
            "document_count": dataset.get_document_count()
        })
    
    async def emit_dataset_updated(self, dataset: Dataset):
        """Emit dataset updated event."""
        await self.emit_event(WebhookEvent.DATASET_UPDATED, {
            "dataset_id": dataset.id,
            "name": dataset.name,
            "version": dataset.version,
            "document_count": dataset.get_document_count()
        })
    
    async def emit_quality_alert(self, dataset_id: str, quality_score: float, threshold: float):
        """Emit quality alert event."""
        await self.emit_event(WebhookEvent.QUALITY_ALERT, {
            "dataset_id": dataset_id,
            "quality_score": quality_score,
            "threshold": threshold,
            "severity": "high" if quality_score < threshold * 0.5 else "medium"
        })


# Global webhook manager instance
webhook_manager = WebhookManager()


def create_webhook_routes(app: FastAPI):
    """Add webhook management routes to FastAPI app."""
    
    @app.post("/webhooks", response_model=Dict[str, Any])
    async def create_webhook(endpoint_data: WebhookEndpointCreate):
        """Create a new webhook endpoint."""
        try:
            endpoint = webhook_manager.add_endpoint(endpoint_data)
            return {"id": endpoint.id, "message": "Webhook endpoint created successfully"}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/webhooks", response_model=List[Dict[str, Any]])
    async def list_webhooks():
        """List all webhook endpoints."""
        return [endpoint.to_dict() for endpoint in webhook_manager.list_endpoints()]
    
    @app.get("/webhooks/{endpoint_id}", response_model=Dict[str, Any])
    async def get_webhook(endpoint_id: str):
        """Get a specific webhook endpoint."""
        endpoint = webhook_manager.get_endpoint(endpoint_id)
        if not endpoint:
            raise HTTPException(status_code=404, detail="Webhook endpoint not found")
        return endpoint.to_dict()
    
    @app.put("/webhooks/{endpoint_id}", response_model=Dict[str, Any])
    async def update_webhook(endpoint_id: str, update_data: WebhookEndpointUpdate):
        """Update a webhook endpoint."""
        try:
            endpoint = webhook_manager.update_endpoint(endpoint_id, update_data)
            return {"message": "Webhook endpoint updated successfully"}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.delete("/webhooks/{endpoint_id}")
    async def delete_webhook(endpoint_id: str):
        """Delete a webhook endpoint."""
        if not webhook_manager.remove_endpoint(endpoint_id):
            raise HTTPException(status_code=404, detail="Webhook endpoint not found")
        return {"message": "Webhook endpoint deleted successfully"}
    
    @app.get("/webhooks/{endpoint_id}/deliveries", response_model=List[Dict[str, Any]])
    async def get_webhook_deliveries(endpoint_id: str, limit: int = 100):
        """Get delivery history for a webhook endpoint."""
        deliveries = webhook_manager.get_deliveries(endpoint_id, limit)
        return [delivery.to_dict() for delivery in deliveries]
    
    @app.get("/webhook-stats", response_model=Dict[str, Any])
    async def get_webhook_stats():
        """Get webhook system statistics."""
        return webhook_manager.get_stats()
    
    @app.post("/webhook-test/{endpoint_id}")
    async def test_webhook(endpoint_id: str):
        """Send a test event to a webhook endpoint."""
        endpoint = webhook_manager.get_endpoint(endpoint_id)
        if not endpoint:
            raise HTTPException(status_code=404, detail="Webhook endpoint not found")
        
        # Send test event
        await webhook_manager.emit_event(WebhookEvent.SYSTEM_ERROR, {
            "test": True,
            "message": "This is a test webhook event",
            "timestamp": datetime.now().isoformat()
        })
        
        return {"message": "Test webhook sent"}


def get_webhook_manager() -> WebhookManager:
    """Get the global webhook manager instance."""
    return webhook_manager


# Startup/shutdown handlers
async def startup_webhook_manager():
    """Start the webhook manager."""
    await webhook_manager.start()


async def shutdown_webhook_manager():
    """Stop the webhook manager."""
    await webhook_manager.stop()


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Start webhook manager
        await webhook_manager.start()
        
        # Add a test endpoint
        endpoint_data = WebhookEndpointCreate(
            url="https://httpbin.org/post",
            events=["processing.completed", "dataset.created"]
        )
        endpoint = webhook_manager.add_endpoint(endpoint_data)
        print(f"Added webhook endpoint: {endpoint.id}")
        
        # Emit a test event
        await webhook_manager.emit_processing_completed("test-job", ProcessingResult(success=True))
        
        # Wait a bit for delivery
        await asyncio.sleep(5)
        
        # Check deliveries
        deliveries = webhook_manager.get_deliveries()
        print(f"Deliveries: {len(deliveries)}")
        for delivery in deliveries:
            print(f"  {delivery.id}: {delivery.status.value}")
        
        # Stop webhook manager
        await webhook_manager.stop()
    
    asyncio.run(main())