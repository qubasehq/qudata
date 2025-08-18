# Manual Deployment Guide

This guide covers manual deployment of QuData without Docker, suitable for environments where containerization is not available or preferred. This includes installation on bare metal servers, VMs, or cloud instances.

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 2 cores
- RAM: 4GB
- Storage: 20GB free space
- OS: Ubuntu 20.04+, CentOS 8+, or similar Linux distribution

**Recommended for Production:**
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 100GB+ SSD
- OS: Ubuntu 22.04 LTS

### Software Dependencies

- Python 3.8+ (3.11 recommended)
- PostgreSQL 13+ or MySQL 8.0+
- Redis 6.0+
- Nginx (for production)
- Supervisor (for process management)

## Installation Steps

### 1. System Preparation

#### Ubuntu/Debian

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
    python3 python3-pip python3-venv python3-dev \
    postgresql postgresql-contrib \
    redis-server \
    nginx \
    supervisor \
    build-essential \
    libpq-dev \
    tesseract-ocr \
    libtesseract-dev \
    libxml2-dev \
    libxslt1-dev \
    libffi-dev \
    libssl-dev \
    curl \
    git \
    htop \
    tree

# Install additional OCR languages (optional)
sudo apt install -y \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    tesseract-ocr-spa
```

#### CentOS/RHEL/Rocky Linux

```bash
# Update system packages
sudo dnf update -y

# Install EPEL repository
sudo dnf install -y epel-release

# Install system dependencies
sudo dnf install -y \
    python3 python3-pip python3-devel \
    postgresql postgresql-server postgresql-contrib \
    redis \
    nginx \
    supervisor \
    gcc gcc-c++ make \
    postgresql-devel \
    tesseract \
    tesseract-devel \
    libxml2-devel \
    libxslt-devel \
    libffi-devel \
    openssl-devel \
    curl \
    git

# Initialize PostgreSQL
sudo postgresql-setup --initdb
sudo systemctl enable postgresql redis nginx supervisor
sudo systemctl start postgresql redis
```

### 2. Database Setup

#### PostgreSQL Configuration

```bash
# Switch to postgres user
sudo -u postgres psql

# Create database and user
CREATE DATABASE qudata;
CREATE USER qudata WITH PASSWORD 'secure_password_here';
GRANT ALL PRIVILEGES ON DATABASE qudata TO qudata;
ALTER USER qudata CREATEDB;
\q

# Configure PostgreSQL for QuData
sudo nano /etc/postgresql/13/main/postgresql.conf
```

Add/modify these settings:

```ini
# Connection settings
listen_addresses = 'localhost'
port = 5432
max_connections = 100

# Memory settings
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100

# Logging
log_statement = 'mod'
log_min_duration_statement = 1000
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
```

Configure authentication:

```bash
sudo nano /etc/postgresql/13/main/pg_hba.conf
```

Add this line:

```
local   qudata          qudata                                  md5
host    qudata          qudata          127.0.0.1/32            md5
```

Restart PostgreSQL:

```bash
sudo systemctl restart postgresql
```

#### Redis Configuration

```bash
# Configure Redis
sudo nano /etc/redis/redis.conf
```

Key settings:

```ini
# Network
bind 127.0.0.1
port 6379

# Memory
maxmemory 1gb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000
appendonly yes
appendfsync everysec

# Security (optional)
# requirepass your_redis_password_here
```

Restart Redis:

```bash
sudo systemctl restart redis
```

### 3. Application Installation

#### Create Application User

```bash
# Create dedicated user for QuData
sudo useradd -r -m -s /bin/bash qudata
sudo usermod -aG sudo qudata

# Switch to qudata user
sudo -u qudata -i
```

#### Download and Install QuData

```bash
# Create application directory
mkdir -p /home/qudata/app
cd /home/qudata/app

# Clone repository
git clone https://github.com/qubasehq/qudata.git .

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip setuptools wheel

# Install QuData with all dependencies
pip install -e ".[ml,web,dev]"

# Install additional production dependencies
pip install gunicorn uvicorn[standard] supervisor
```

#### Create Directory Structure

```bash
# Create necessary directories
mkdir -p \
    data/{raw,staging,processed,exports} \
    logs \
    tmp \
    backups \
    configs/custom

# Set permissions
chmod 755 data logs tmp backups
chmod 644 configs/*.yaml
```

### 4. Configuration

#### Environment Configuration

```bash
# Create environment file
cp .env.example .env
nano .env
```

Configure for manual deployment:

```bash
# Application settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Database settings
DB_HOST=localhost
DB_PORT=5432
DB_NAME=qudata
DB_USER=qudata
DB_PASSWORD=secure_password_here

# Redis settings
REDIS_URL=redis://localhost:6379/0

# File paths
QUDATA_DATA_PATH=/home/qudata/app/data
QUDATA_CONFIG_PATH=/home/qudata/app/configs
QUDATA_LOG_PATH=/home/qudata/app/logs
QUDATA_TEMP_PATH=/home/qudata/app/tmp

# Performance settings
WORKERS=4
MAX_WORKERS=8
BATCH_SIZE=200
MAX_MEMORY=8GB

# API settings
API_HOST=127.0.0.1
API_PORT=8000

# Security
SECRET_KEY=generate_secure_key_here
ALLOWED_HOSTS=localhost,127.0.0.1,your-domain.com
```

#### Application Configuration

```bash
# Copy and customize pipeline configuration
cp configs/pipeline.yaml configs/custom/production.yaml
nano configs/custom/production.yaml
```

Customize for your environment:

```yaml
pipeline:
  name: "production"
  version: "1.0.0"
  
  paths:
    raw_data: "/home/qudata/app/data/raw"
    staging: "/home/qudata/app/data/staging"
    processed: "/home/qudata/app/data/processed"
    exports: "/home/qudata/app/data/exports"
  
  performance:
    parallel_processing: true
    max_workers: 8
    batch_size: 200
    streaming_mode: true
    max_memory_usage: "8GB"
  
  stages:
    ingest:
      enabled: true
      file_types: ["pdf", "docx", "txt", "html", "csv", "json", "xml", "md"]
      max_file_size: "500MB"
      
    clean:
      enabled: true
      normalize_unicode: true
      remove_boilerplate: true
      deduplicate: true
      similarity_threshold: 0.85
      
    annotate:
      enabled: true
      taxonomy_classification: true
      named_entity_recognition: true
      
    score:
      enabled: true
      min_quality_score: 0.7
      
    export:
      enabled: true
      formats: ["jsonl", "chatml", "parquet"]
```

### 5. Database Initialization

```bash
# Activate virtual environment
source /home/qudata/app/venv/bin/activate

# Initialize database schema
python -c "
from qudata.database import DatabaseConnector
connector = DatabaseConnector()
connector.initialize_schema()
print('Database initialized successfully')
"

# Verify installation
qudata --version
qudata config validate --file configs/custom/production.yaml
```

### 6. Process Management with Supervisor

#### Create Supervisor Configuration

```bash
# Create supervisor config directory
sudo mkdir -p /etc/supervisor/conf.d

# Create QuData supervisor configuration
sudo nano /etc/supervisor/conf.d/qudata.conf
```

Supervisor configuration:

```ini
[group:qudata]
programs=qudata-api,qudata-worker

[program:qudata-api]
command=/home/qudata/app/venv/bin/gunicorn qudata.api.rest_server:app --bind 127.0.0.1:8000 --workers 4 --worker-class uvicorn.workers.UvicornWorker --timeout 300 --keep-alive 2 --max-requests 1000 --max-requests-jitter 100
directory=/home/qudata/app
user=qudata
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/home/qudata/app/logs/api.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=5
environment=PATH="/home/qudata/app/venv/bin",PYTHONPATH="/home/qudata/app",QUDATA_CONFIG_PATH="/home/qudata/app/configs/custom/production.yaml"

[program:qudata-worker]
command=/home/qudata/app/venv/bin/celery -A qudata.orchestrate.celery_app worker --loglevel=info --concurrency=4
directory=/home/qudata/app
user=qudata
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/home/qudata/app/logs/worker.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=5
environment=PATH="/home/qudata/app/venv/bin",PYTHONPATH="/home/qudata/app",QUDATA_CONFIG_PATH="/home/qudata/app/configs/custom/production.yaml"

[program:qudata-scheduler]
command=/home/qudata/app/venv/bin/celery -A qudata.orchestrate.celery_app beat --loglevel=info
directory=/home/qudata/app
user=qudata
autostart=false
autorestart=true
redirect_stderr=true
stdout_logfile=/home/qudata/app/logs/scheduler.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=5
environment=PATH="/home/qudata/app/venv/bin",PYTHONPATH="/home/qudata/app",QUDATA_CONFIG_PATH="/home/qudata/app/configs/custom/production.yaml"
```

#### Start Services

```bash
# Reload supervisor configuration
sudo supervisorctl reread
sudo supervisorctl update

# Start QuData services
sudo supervisorctl start qudata:*

# Check status
sudo supervisorctl status
```

### 7. Nginx Configuration

#### Create Nginx Site Configuration

```bash
sudo nano /etc/nginx/sites-available/qudata
```

Nginx configuration:

```nginx
upstream qudata_backend {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name your-domain.com www.your-domain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com www.your-domain.com;
    
    # SSL configuration
    ssl_certificate /etc/ssl/certs/your-domain.com.crt;
    ssl_certificate_key /etc/ssl/private/your-domain.com.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # Client settings
    client_max_body_size 500M;
    client_body_timeout 300s;
    client_header_timeout 300s;
    
    # Logging
    access_log /var/log/nginx/qudata_access.log;
    error_log /var/log/nginx/qudata_error.log;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=30r/s;
    limit_req_zone $binary_remote_addr zone=upload:10m rate=5r/s;
    
    # API endpoints
    location /api/ {
        limit_req zone=api burst=50 nodelay;
        
        proxy_pass http://qudata_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;
        
        proxy_buffering on;
        proxy_buffer_size 8k;
        proxy_buffers 16 8k;
    }
    
    # File upload endpoints
    location /api/upload {
        limit_req zone=upload burst=10 nodelay;
        
        proxy_pass http://qudata_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 600s;
        
        proxy_request_buffering off;
        proxy_buffering off;
    }
    
    # GraphQL endpoint
    location /graphql {
        proxy_pass http://qudata_backend/graphql;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    # Health check
    location /health {
        access_log off;
        proxy_pass http://qudata_backend/health;
        proxy_set_header Host $host;
    }
    
    # Documentation
    location /docs {
        proxy_pass http://qudata_backend/docs;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Static files (if any)
    location /static/ {
        alias /home/qudata/app/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # Root redirect
    location = / {
        return 302 /docs;
    }
}
```

#### Enable Site and Start Nginx

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/qudata /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Start and enable Nginx
sudo systemctl enable nginx
sudo systemctl start nginx
```

### 8. SSL Certificate Setup

#### Using Let's Encrypt (Recommended)

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d your-domain.com -d www.your-domain.com

# Test automatic renewal
sudo certbot renew --dry-run

# Set up automatic renewal
echo "0 12 * * * /usr/bin/certbot renew --quiet" | sudo crontab -
```

#### Using Self-Signed Certificate (Development)

```bash
# Create SSL directory
sudo mkdir -p /etc/ssl/private

# Generate self-signed certificate
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/ssl/private/your-domain.com.key \
    -out /etc/ssl/certs/your-domain.com.crt

# Set permissions
sudo chmod 600 /etc/ssl/private/your-domain.com.key
sudo chmod 644 /etc/ssl/certs/your-domain.com.crt
```

### 9. Monitoring and Logging

#### Log Rotation

```bash
# Create logrotate configuration
sudo nano /etc/logrotate.d/qudata
```

Logrotate configuration:

```
/home/qudata/app/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 qudata qudata
    postrotate
        supervisorctl restart qudata:*
    endscript
}

/var/log/nginx/qudata_*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 www-data www-data
    postrotate
        systemctl reload nginx
    endscript
}
```

#### System Monitoring

Install and configure monitoring tools:

```bash
# Install monitoring tools
sudo apt install htop iotop nethogs

# Create monitoring script
nano /home/qudata/app/scripts/monitor.sh
```

Monitoring script:

```bash
#!/bin/bash

# QuData System Monitoring Script

LOG_FILE="/home/qudata/app/logs/system_monitor.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

# Function to log with timestamp
log() {
    echo "[$DATE] $1" >> "$LOG_FILE"
}

# Check disk space
DISK_USAGE=$(df -h /home/qudata/app | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 80 ]; then
    log "WARNING: Disk usage is ${DISK_USAGE}%"
fi

# Check memory usage
MEMORY_USAGE=$(free | awk 'NR==2{printf "%.2f", $3*100/$2}')
if (( $(echo "$MEMORY_USAGE > 80" | bc -l) )); then
    log "WARNING: Memory usage is ${MEMORY_USAGE}%"
fi

# Check QuData services
if ! supervisorctl status qudata:qudata-api | grep -q RUNNING; then
    log "ERROR: QuData API service is not running"
fi

if ! supervisorctl status qudata:qudata-worker | grep -q RUNNING; then
    log "ERROR: QuData Worker service is not running"
fi

# Check database connection
if ! pg_isready -h localhost -p 5432 -U qudata > /dev/null 2>&1; then
    log "ERROR: PostgreSQL is not accessible"
fi

# Check Redis connection
if ! redis-cli ping > /dev/null 2>&1; then
    log "ERROR: Redis is not accessible"
fi

log "System monitoring check completed"
```

Make script executable and add to cron:

```bash
chmod +x /home/qudata/app/scripts/monitor.sh

# Add to crontab (run every 5 minutes)
(crontab -l 2>/dev/null; echo "*/5 * * * * /home/qudata/app/scripts/monitor.sh") | crontab -
```

### 10. Backup Configuration

#### Database Backup Script

```bash
nano /home/qudata/app/scripts/backup_db.sh
```

Database backup script:

```bash
#!/bin/bash

# QuData Database Backup Script

BACKUP_DIR="/home/qudata/app/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="qudata_backup_${DATE}.sql"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Create database backup
pg_dump -h localhost -U qudata -d qudata > "$BACKUP_DIR/$BACKUP_FILE"

# Compress backup
gzip "$BACKUP_DIR/$BACKUP_FILE"

# Remove backups older than 30 days
find "$BACKUP_DIR" -name "qudata_backup_*.sql.gz" -mtime +30 -delete

echo "Database backup completed: ${BACKUP_FILE}.gz"
```

#### Data Backup Script

```bash
nano /home/qudata/app/scripts/backup_data.sh
```

Data backup script:

```bash
#!/bin/bash

# QuData Data Backup Script

BACKUP_DIR="/home/qudata/app/backups"
DATE=$(date +%Y%m%d_%H%M%S)
DATA_DIR="/home/qudata/app/data"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Backup processed data
tar -czf "$BACKUP_DIR/data_backup_${DATE}.tar.gz" -C "$DATA_DIR" processed/

# Backup configuration
tar -czf "$BACKUP_DIR/config_backup_${DATE}.tar.gz" -C "/home/qudata/app" configs/

# Remove backups older than 7 days
find "$BACKUP_DIR" -name "*_backup_*.tar.gz" -mtime +7 -delete

echo "Data backup completed: data_backup_${DATE}.tar.gz"
```

Make scripts executable and schedule:

```bash
chmod +x /home/qudata/app/scripts/backup_*.sh

# Add to crontab
(crontab -l 2>/dev/null; echo "0 2 * * * /home/qudata/app/scripts/backup_db.sh") | crontab -
(crontab -l 2>/dev/null; echo "0 3 * * * /home/qudata/app/scripts/backup_data.sh") | crontab -
```

## Service Management

### Starting Services

```bash
# Start all QuData services
sudo supervisorctl start qudata:*

# Start specific service
sudo supervisorctl start qudata:qudata-api

# Check status
sudo supervisorctl status
```

### Stopping Services

```bash
# Stop all QuData services
sudo supervisorctl stop qudata:*

# Stop specific service
sudo supervisorctl stop qudata:qudata-api
```

### Restarting Services

```bash
# Restart all QuData services
sudo supervisorctl restart qudata:*

# Restart specific service
sudo supervisorctl restart qudata:qudata-api
```

### Viewing Logs

```bash
# View API logs
sudo supervisorctl tail -f qudata:qudata-api

# View worker logs
sudo supervisorctl tail -f qudata:qudata-worker

# View all logs
tail -f /home/qudata/app/logs/*.log

# View Nginx logs
sudo tail -f /var/log/nginx/qudata_*.log
```

## Troubleshooting

### Common Issues

1. **Service won't start**:
```bash
# Check supervisor logs
sudo supervisorctl tail qudata:qudata-api stderr

# Check application logs
tail -f /home/qudata/app/logs/api.log

# Check system resources
htop
df -h
```

2. **Database connection issues**:
```bash
# Test database connection
pg_isready -h localhost -p 5432 -U qudata

# Check PostgreSQL status
sudo systemctl status postgresql

# Check PostgreSQL logs
sudo tail -f /var/log/postgresql/postgresql-13-main.log
```

3. **Permission issues**:
```bash
# Fix file permissions
sudo chown -R qudata:qudata /home/qudata/app
sudo chmod -R 755 /home/qudata/app/data
sudo chmod -R 644 /home/qudata/app/configs/*.yaml
```

4. **Performance issues**:
```bash
# Check system resources
htop
iotop
nethogs

# Check application metrics
curl http://localhost:8000/metrics

# Adjust worker processes
sudo nano /etc/supervisor/conf.d/qudata.conf
# Modify --workers parameter
sudo supervisorctl restart qudata:qudata-api
```

### Health Checks

```bash
# Application health
curl http://localhost:8000/health

# Database health
curl http://localhost:8000/health/db

# Redis health
curl http://localhost:8000/health/redis

# System status
curl http://localhost:8000/status
```

## Updates and Maintenance

### Updating QuData

```bash
# Switch to qudata user
sudo -u qudata -i
cd /home/qudata/app

# Backup current installation
cp -r /home/qudata/app /home/qudata/app.backup.$(date +%Y%m%d)

# Activate virtual environment
source venv/bin/activate

# Pull latest changes
git pull origin main

# Update dependencies
pip install -e ".[ml,web,dev]" --upgrade

# Run database migrations if needed
python -c "
from qudata.database import DatabaseConnector
connector = DatabaseConnector()
connector.migrate_schema()
print('Database migration completed')
"

# Restart services
sudo supervisorctl restart qudata:*

# Verify update
qudata --version
curl http://localhost:8000/health
```

### System Maintenance

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Clean up logs
sudo logrotate -f /etc/logrotate.d/qudata

# Clean up temporary files
sudo find /tmp -type f -atime +7 -delete

# Check disk space
df -h

# Check system performance
htop
```

## Security Hardening

### Firewall Configuration

```bash
# Install and configure UFW
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH
sudo ufw allow ssh

# Allow HTTP and HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow specific IPs for database access (if needed)
# sudo ufw allow from YOUR_IP_ADDRESS to any port 5432

# Check status
sudo ufw status verbose
```

### System Security

```bash
# Disable root login
sudo passwd -l root

# Configure automatic security updates
sudo apt install unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades

# Install fail2ban for intrusion prevention
sudo apt install fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

### Application Security

```bash
# Set secure file permissions
sudo chmod 600 /home/qudata/app/.env
sudo chmod 600 /home/qudata/app/configs/custom/*.yaml

# Regular security updates
pip list --outdated
pip install --upgrade pip setuptools wheel

# Monitor security advisories
pip-audit
```

## Performance Optimization

### Database Optimization

```bash
# Analyze database performance
sudo -u postgres psql -d qudata -c "
SELECT schemaname,tablename,attname,n_distinct,correlation 
FROM pg_stats 
WHERE schemaname = 'public' 
ORDER BY n_distinct DESC;
"

# Update database statistics
sudo -u postgres psql -d qudata -c "ANALYZE;"

# Check slow queries
sudo -u postgres psql -d qudata -c "
SELECT query, mean_time, calls, total_time 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;
"
```

### Application Optimization

```bash
# Monitor application performance
curl http://localhost:8000/metrics

# Adjust worker processes based on CPU cores
CORES=$(nproc)
WORKERS=$((CORES * 2 + 1))

# Update supervisor configuration
sudo nano /etc/supervisor/conf.d/qudata.conf
# Modify --workers parameter to $WORKERS

# Restart services
sudo supervisorctl restart qudata:qudata-api
```

### System Optimization

```bash
# Optimize system parameters
sudo nano /etc/sysctl.conf
```

Add these optimizations:

```ini
# Network optimizations
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216

# File system optimizations
fs.file-max = 65536
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
```

Apply changes:

```bash
sudo sysctl -p
```

This manual deployment guide provides a comprehensive approach to installing and managing QuData without Docker, suitable for production environments that require fine-grained control over the deployment process.