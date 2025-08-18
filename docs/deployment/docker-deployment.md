# Docker Deployment Guide

This comprehensive guide covers deploying QuData using Docker for consistent, scalable, and portable deployments across different environments. QuData provides production-ready Docker configurations with security, monitoring, and scalability features built-in.

## Quick Start

### Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- At least 4GB RAM and 10GB disk space
- Basic understanding of Docker and containerization

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/qubasehq/qudata.git
cd qudata

# Copy environment template
cp .env.example .env

# Edit configuration (see Configuration section below)
nano .env
```

### 2. Development Deployment

```bash
# Start development environment
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f qudata

# Access services
# API: http://localhost:8000/docs
# Dashboard: http://localhost:8001
# Database: localhost:5432
```

### 3. Production Deployment

```bash
# Create secrets directory
mkdir -p secrets

# Generate secure passwords
openssl rand -base64 32 > secrets/db_password.txt
openssl rand -base64 32 > secrets/redis_password.txt
openssl rand -base64 32 > secrets/secret_key.txt
openssl rand -base64 32 > secrets/grafana_password.txt

# Set proper permissions
chmod 600 secrets/*

# Deploy production environment
docker-compose -f docker-compose.prod.yml up -d

# Check deployment status
docker-compose -f docker-compose.prod.yml ps
```

## Architecture Overview

QuData's Docker architecture provides:

- **Multi-stage builds** for optimized production images
- **Security-first design** with non-root users and secrets management
- **Horizontal scalability** with load balancing
- **Comprehensive monitoring** with Prometheus and Grafana
- **Centralized logging** with ELK stack
- **Health checks** and automatic recovery
- **Development and production** configurations

## Production Deployment

### Multi-Stage Dockerfile

```dockerfile
# Multi-stage Dockerfile for production
FROM python:3.9-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    tesseract-ocr \
    libtesseract4 \
    libxml2 \
    libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN groupadd -r qudata && useradd -r -g qudata qudata

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=qudata:qudata . .

# Install QuData
RUN pip install -e .

# Create directories with proper permissions
RUN mkdir -p data/{raw,processed,exports} logs && \
    chown -R qudata:qudata data logs

# Switch to non-root user
USER qudata

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "qudata.api.main:app"]
```

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  qudata:
    build:
      context: .
      target: production
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data:rw
      - ./configs:/app/configs:ro
      - ./logs:/app/logs:rw
    environment:
      - ENVIRONMENT=production
      - DB_HOST=postgres
      - DB_USER=qudata
      - DB_PASSWORD_FILE=/run/secrets/db_password
      - DB_NAME=qudata
      - REDIS_URL=redis://redis:6379/0
      - SECRET_KEY_FILE=/run/secrets/secret_key
    secrets:
      - db_password
      - secret_key
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G

  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=qudata
      - POSTGRES_USER=qudata
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    secrets:
      - db_password
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 2G

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - ./logs/nginx:/var/log/nginx
    depends_on:
      - qudata
    restart: unless-stopped

secrets:
  db_password:
    file: ./secrets/db_password.txt
  secret_key:
    file: ./secrets/secret_key.txt

volumes:
  postgres_data:
  redis_data:
```

## Configuration for Docker

### Environment-Specific Configuration

```yaml
# configs/docker.yaml
pipeline:
  input_directory: "/app/data/raw"
  output_directory: "/app/data/processed"
  export_directory: "/app/data/exports"
  temp_directory: "/tmp/qudata"
  
  # Docker-optimized settings
  parallel_processing: true
  max_workers: ${MAX_WORKERS:-4}
  batch_size: ${BATCH_SIZE:-100}
  max_memory_usage: "${MAX_MEMORY:-4GB}"

database:
  type: "postgresql"
  host: "${DB_HOST:-localhost}"
  port: ${DB_PORT:-5432}
  database: "${DB_NAME:-qudata}"
  username: "${DB_USER:-qudata}"
  password: "${DB_PASSWORD}"
  pool_size: ${DB_POOL_SIZE:-10}

cache:
  type: "redis"
  url: "${REDIS_URL:-redis://localhost:6379/0}"
  ttl: ${CACHE_TTL:-3600}

logging:
  level: "${LOG_LEVEL:-INFO}"
  file: "/app/logs/qudata.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### Docker Environment Variables

```bash
# .env file for docker-compose
COMPOSE_PROJECT_NAME=qudata

# Application settings
MAX_WORKERS=4
BATCH_SIZE=100
MAX_MEMORY=4GB
LOG_LEVEL=INFO

# Database settings
DB_HOST=postgres
DB_PORT=5432
DB_NAME=qudata
DB_USER=qudata
DB_PASSWORD=secure_password_here
DB_POOL_SIZE=10

# Redis settings
REDIS_URL=redis://redis:6379/0
CACHE_TTL=3600

# Security
SECRET_KEY=your_secret_key_here
ALLOWED_HOSTS=localhost,127.0.0.1,your-domain.com
```

## Scaling and Load Balancing

### Horizontal Scaling

```yaml
# docker-compose.scale.yml
version: '3.8'

services:
  qudata-worker:
    build: .
    command: ["qudata", "worker", "--concurrency", "4"]
    volumes:
      - ./data:/app/data
      - ./configs:/app/configs
    environment:
      - WORKER_TYPE=processing
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/1
    depends_on:
      - redis
    deploy:
      replicas: 3
    restart: unless-stopped

  qudata-api:
    build: .
    command: ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "qudata.api.main:app"]
    ports:
      - "8000-8002:8000"
    volumes:
      - ./configs:/app/configs:ro
    environment:
      - API_WORKERS=4
    depends_on:
      - postgres
      - redis
    deploy:
      replicas: 3
    restart: unless-stopped

  nginx-lb:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx/load-balancer.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - qudata-api
    restart: unless-stopped
```

### Load Balancer Configuration

```nginx
# nginx/load-balancer.conf
upstream qudata_backend {
    least_conn;
    server qudata-api:8000 max_fails=3 fail_timeout=30s;
    server qudata-api:8001 max_fails=3 fail_timeout=30s;
    server qudata-api:8002 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://qudata_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }

    location /health {
        access_log off;
        proxy_pass http://qudata_backend/health;
    }
}
```

## Monitoring and Logging

### Logging Configuration

```yaml
# docker-compose.logging.yml
version: '3.8'

services:
  qudata:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
        labels: "service=qudata"

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.15.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"

  logstash:
    image: docker.elastic.co/logstash/logstash:7.15.0
    volumes:
      - ./logstash/pipeline:/usr/share/logstash/pipeline:ro
      - ./logs:/logs:ro
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:7.15.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch

volumes:
  elasticsearch_data:
```

### Health Checks

```dockerfile
# Add to Dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "
import requests
import sys
try:
    response = requests.get('http://localhost:8000/health', timeout=5)
    sys.exit(0 if response.status_code == 200 else 1)
except:
    sys.exit(1)
"
```

### Monitoring with Prometheus

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./grafana/datasources:/etc/grafana/provisioning/datasources:ro
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

volumes:
  prometheus_data:
  grafana_data:
```

## Security Best Practices

### Secrets Management

```bash
# Create secrets directory
mkdir -p secrets

# Generate secure passwords
openssl rand -base64 32 > secrets/db_password.txt
openssl rand -base64 32 > secrets/secret_key.txt
openssl rand -base64 32 > secrets/redis_password.txt

# Set proper permissions
chmod 600 secrets/*
```

### Network Security

```yaml
# docker-compose.secure.yml
version: '3.8'

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true

services:
  qudata:
    networks:
      - frontend
      - backend
    # Remove direct port exposure in production

  postgres:
    networks:
      - backend
    # Only accessible from backend network

  nginx:
    networks:
      - frontend
    ports:
      - "80:80"
      - "443:443"
```

### SSL/TLS Configuration

```nginx
# nginx/ssl.conf
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;

    location / {
        proxy_pass http://qudata:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

## Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="qudata_backup_${DATE}.sql"

# Create backup
docker-compose exec postgres pg_dump -U qudata qudata > "${BACKUP_DIR}/${BACKUP_FILE}"

# Compress backup
gzip "${BACKUP_DIR}/${BACKUP_FILE}"

# Clean old backups (keep last 7 days)
find "${BACKUP_DIR}" -name "qudata_backup_*.sql.gz" -mtime +7 -delete

echo "Backup completed: ${BACKUP_FILE}.gz"
```

### Data Volume Backup

```bash
#!/bin/bash
# backup-volumes.sh

DATE=$(date +%Y%m%d_%H%M%S)

# Backup data volumes
docker run --rm \
  -v qudata_postgres_data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/postgres_data_${DATE}.tar.gz -C /data .

docker run --rm \
  -v qudata_redis_data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/redis_data_${DATE}.tar.gz -C /data .

echo "Volume backups completed"
```

## Deployment Commands

### Development Deployment

```bash
# Start development environment
docker-compose -f docker-compose.yml up -d

# View logs
docker-compose logs -f

# Execute commands in container
docker-compose exec qudata qudata process --help

# Stop services
docker-compose down
```

### Production Deployment

```bash
# Deploy to production
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose -f docker-compose.prod.yml up -d --scale qudata=3

# Rolling update
docker-compose -f docker-compose.prod.yml up -d --no-deps qudata

# Health check
docker-compose -f docker-compose.prod.yml ps
```

### Maintenance Commands

```bash
# Update images
docker-compose pull
docker-compose up -d

# Clean up
docker system prune -f
docker volume prune -f

# Backup before maintenance
./backup.sh

# Database migration
docker-compose exec qudata alembic upgrade head
```

This Docker deployment guide provides a comprehensive setup for running QuData in containerized environments, from development to production, with proper scaling, monitoring, and security considerations.

## Deployment Configurations

### Development Environment

The development configuration (`docker-compose.yml`) includes:

- **Hot reload** for code changes
- **Debug logging** and development tools
- **Exposed ports** for direct access
- **Volume mounts** for live code editing
- **Optional services** via profiles

**Services included:**
- QuData API server with auto-reload
- PostgreSQL database
- Redis cache
- Nginx reverse proxy (optional)
- Elasticsearch + Kibana (optional, `--profile logging`)
- Prometheus + Grafana (optional, `--profile monitoring`)

### Production Environment

The production configuration (`docker-compose.prod.yml`) includes:

- **Multi-replica deployment** with load balancing
- **Secrets management** for sensitive data
- **Resource limits** and health checks
- **Network isolation** for security
- **Comprehensive monitoring** and logging
- **Backup and recovery** mechanisms

**Services included:**
- QuData API (3 replicas by default)
- QuData Worker (2 replicas for background processing)
- PostgreSQL with optimized settings
- Redis with persistence
- Nginx load balancer with SSL
- Elasticsearch for centralized logging
- Logstash for log processing
- Prometheus for metrics collection
- Grafana for visualization

## Configuration

### Environment Variables

Key environment variables in `.env`:

```bash
# Application
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
VERSION=1.0.1

# Database
DB_HOST=postgres
DB_NAME=qudata
DB_USER=qudata
# DB_PASSWORD loaded from secrets/db_password.txt

# Performance
WORKERS=4
MAX_WORKERS=8
BATCH_SIZE=200
MAX_MEMORY=8GB

# Security
# SECRET_KEY loaded from secrets/secret_key.txt
ALLOWED_HOSTS=your-domain.com,api.your-domain.com

# Features
ENABLE_GRAPHQL=true
ENABLE_WEBHOOKS=true
ENABLE_MONITORING=true
```

### Secrets Management

Production deployments use Docker secrets for sensitive data:

```bash
# Create secrets directory
mkdir -p secrets

# Database password
echo "your-secure-db-password" > secrets/db_password.txt

# Redis password
echo "your-secure-redis-password" > secrets/redis_password.txt

# Application secret key
openssl rand -base64 32 > secrets/secret_key.txt

# Grafana admin password
echo "your-grafana-password" > secrets/grafana_password.txt

# Set proper permissions
chmod 600 secrets/*
```

### SSL/TLS Configuration

For production HTTPS deployment:

1. **Obtain SSL certificates** (Let's Encrypt recommended):

```bash
# Using certbot
certbot certonly --webroot -w /var/www/certbot -d your-domain.com

# Copy certificates
cp /etc/letsencrypt/live/your-domain.com/fullchain.pem docker/nginx/ssl/cert.pem
cp /etc/letsencrypt/live/your-domain.com/privkey.pem docker/nginx/ssl/key.pem
```

2. **Update Nginx configuration** in `docker/nginx/nginx-prod.conf`:

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    # ... rest of configuration
}
```

## Deployment Scripts

### Automated Deployment

Use the provided deployment script for automated deployments:

```bash
# Development deployment
./scripts/deploy.sh -e development

# Production deployment with backup
./scripts/deploy.sh -e production -b -p -c

# Deploy specific services
./scripts/deploy.sh -e production -s api,worker

# Scale services
./scripts/deploy.sh -e production --scale api=3 --scale worker=2
```

### Manual Deployment Steps

For manual deployment control:

```bash
# 1. Build images
docker-compose -f docker-compose.prod.yml build

# 2. Create networks and volumes
docker-compose -f docker-compose.prod.yml up --no-start

# 3. Start infrastructure services
docker-compose -f docker-compose.prod.yml up -d postgres redis

# 4. Wait for services to be ready
docker-compose -f docker-compose.prod.yml exec postgres pg_isready -U qudata

# 5. Start application services
docker-compose -f docker-compose.prod.yml up -d qudata-api qudata-worker

# 6. Start monitoring and proxy
docker-compose -f docker-compose.prod.yml up -d nginx prometheus grafana
```

## Scaling and Load Balancing

### Horizontal Scaling

Scale services based on load:

```bash
# Scale API servers
docker-compose -f docker-compose.prod.yml up -d --scale qudata-api=5

# Scale workers
docker-compose -f docker-compose.prod.yml up -d --scale qudata-worker=3

# Check scaled services
docker-compose -f docker-compose.prod.yml ps
```

### Load Balancer Configuration

Nginx automatically load balances across API replicas:

```nginx
upstream qudata_api {
    least_conn;
    server qudata-api:8000 max_fails=3 fail_timeout=30s weight=1;
    # Additional servers added automatically by Docker Compose
    keepalive 32;
}
```

### Resource Management

Configure resource limits in production:

```yaml
services:
  qudata-api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## Monitoring and Logging

### Health Checks

All services include comprehensive health checks:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 60s
```

### Metrics Collection

Prometheus collects metrics from:
- Application performance metrics
- System resource usage
- Database performance
- Request/response metrics
- Custom business metrics

Access Grafana dashboard at `http://localhost:3000` (admin/admin)

### Centralized Logging

ELK stack processes logs from all services:
- **Elasticsearch**: Log storage and indexing
- **Logstash**: Log processing and transformation
- **Kibana**: Log visualization and analysis

Access Kibana at `http://localhost:5601`

### Log Aggregation

Application logs are structured and include:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "service": "qudata-api",
  "message": "Document processed successfully",
  "document_id": "uuid-here",
  "processing_time": 1.23,
  "quality_score": 0.85
}
```

## Backup and Recovery

### Automated Backups

Production deployment includes automated backup:

```bash
# Database backup script
#!/bin/bash
BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup database
docker-compose -f docker-compose.prod.yml exec -T postgres \
  pg_dump -U qudata qudata > "$BACKUP_DIR/database.sql"

# Backup data volumes
docker run --rm \
  -v qudata_postgres_data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf "/backup/postgres_data_$(date +%Y%m%d_%H%M%S).tar.gz" -C /data .
```

### Recovery Procedures

To restore from backup:

```bash
# Stop services
docker-compose -f docker-compose.prod.yml down

# Restore database
docker-compose -f docker-compose.prod.yml up -d postgres
docker-compose -f docker-compose.prod.yml exec -T postgres \
  psql -U qudata -d qudata < backups/20240115_103000/database.sql

# Restore data volumes
docker run --rm \
  -v qudata_postgres_data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar xzf /backup/postgres_data_20240115_103000.tar.gz -C /data

# Restart services
docker-compose -f docker-compose.prod.yml up -d
```

## Security Best Practices

### Container Security

- **Non-root user**: All containers run as non-root user `qudata`
- **Read-only filesystems**: Where possible, containers use read-only root filesystems
- **Minimal base images**: Using Alpine Linux for smaller attack surface
- **Security scanning**: Regular vulnerability scanning of images

### Network Security

- **Network isolation**: Backend services isolated from external access
- **TLS encryption**: All external communication encrypted
- **Rate limiting**: API endpoints protected with rate limiting
- **Firewall rules**: Only necessary ports exposed

### Secrets Management

- **Docker secrets**: Sensitive data stored as Docker secrets
- **Environment separation**: Different secrets for each environment
- **Rotation policy**: Regular rotation of passwords and keys
- **Access control**: Restricted access to secrets files

## Troubleshooting

### Common Issues

1. **Service won't start**:
```bash
# Check logs
docker-compose logs service-name

# Check health status
docker-compose ps

# Restart service
docker-compose restart service-name
```

2. **Database connection issues**:
```bash
# Check database status
docker-compose exec postgres pg_isready -U qudata

# Check network connectivity
docker-compose exec qudata-api nc -z postgres 5432

# Reset database
docker-compose down
docker volume rm qudata_postgres_data
docker-compose up -d
```

3. **Performance issues**:
```bash
# Check resource usage
docker stats

# Scale services
docker-compose up -d --scale qudata-api=3

# Check logs for bottlenecks
docker-compose logs --tail=100 qudata-api
```

### Debug Mode

Enable debug mode for troubleshooting:

```bash
# Set debug environment
export DEBUG=true
export LOG_LEVEL=DEBUG

# Restart with debug logging
docker-compose restart qudata-api

# Follow debug logs
docker-compose logs -f qudata-api
```

### Health Check Endpoints

Monitor service health:

```bash
# Application health
curl http://localhost:8000/health

# Database health
curl http://localhost:8000/health/db

# Redis health
curl http://localhost:8000/health/redis

# Detailed system status
curl http://localhost:8000/status
```

## Maintenance

### Updates and Upgrades

```bash
# Pull latest images
docker-compose pull

# Backup before update
./scripts/backup.sh

# Update services with zero downtime
docker-compose up -d --no-deps qudata-api

# Verify update
docker-compose ps
curl http://localhost:8000/health
```

### Log Rotation

Configure log rotation to prevent disk space issues:

```bash
# Add to crontab
0 2 * * * docker system prune -f --filter "until=24h"
0 3 * * * find /var/lib/docker/containers -name "*.log" -exec truncate -s 0 {} \;
```

### Performance Tuning

Optimize for your workload:

```yaml
# Increase worker processes
environment:
  - WORKERS=8
  - MAX_WORKERS=16

# Adjust batch sizes
  - BATCH_SIZE=500
  - MAX_MEMORY=16GB

# Database tuning
postgres:
  command: >
    postgres
    -c shared_buffers=256MB
    -c effective_cache_size=1GB
    -c maintenance_work_mem=64MB
```

## Production Checklist

Before deploying to production:

- [ ] SSL certificates configured
- [ ] Secrets properly generated and secured
- [ ] Environment variables configured
- [ ] Resource limits set appropriately
- [ ] Monitoring and alerting configured
- [ ] Backup procedures tested
- [ ] Security scanning completed
- [ ] Load testing performed
- [ ] Documentation updated
- [ ] Team trained on operations

## Support and Resources

- **Documentation**: [docs/](../docs/)
- **API Reference**: http://localhost:8000/docs
- **Monitoring**: http://localhost:3000 (Grafana)
- **Logs**: http://localhost:5601 (Kibana)
- **Issues**: [GitHub Issues](https://github.com/qubasehq/qudata/issues)
- **Community**: [Discussions](https://github.com/qubasehq/qudata/discussions)