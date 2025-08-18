#!/bin/bash
set -e

# QuData Docker Entrypoint Script
# Handles initialization and service startup

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Function to wait for service
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local timeout=${4:-30}
    
    log "Waiting for $service_name at $host:$port..."
    
    for i in $(seq 1 $timeout); do
        if nc -z "$host" "$port" 2>/dev/null; then
            log "$service_name is ready!"
            return 0
        fi
        sleep 1
    done
    
    error "$service_name is not available after ${timeout}s"
    return 1
}

# Function to load secrets from files
load_secrets() {
    if [ -f "/run/secrets/db_password" ]; then
        export DB_PASSWORD=$(cat /run/secrets/db_password)
        log "Database password loaded from secret"
    fi
    
    if [ -f "/run/secrets/redis_password" ]; then
        export REDIS_PASSWORD=$(cat /run/secrets/redis_password)
        log "Redis password loaded from secret"
    fi
    
    if [ -f "/run/secrets/secret_key" ]; then
        export SECRET_KEY=$(cat /run/secrets/secret_key)
        log "Secret key loaded from secret"
    fi
}

# Function to initialize database
init_database() {
    log "Initializing database..."
    
    # Wait for database to be ready
    if ! wait_for_service "$DB_HOST" "$DB_PORT" "PostgreSQL" 60; then
        error "Database is not available"
        exit 1
    fi
    
    # Run database migrations if needed
    if command -v alembic >/dev/null 2>&1; then
        log "Running database migrations..."
        alembic upgrade head || warn "Database migration failed"
    fi
    
    # Initialize database schema if needed
    python -c "
from qudata.database import DatabaseConnector
try:
    connector = DatabaseConnector()
    connector.initialize_schema()
    print('Database schema initialized')
except Exception as e:
    print(f'Database initialization failed: {e}')
" || warn "Database schema initialization failed"
}

# Function to initialize Redis
init_redis() {
    log "Checking Redis connection..."
    
    # Extract Redis host and port from URL
    if [[ $REDIS_URL =~ redis://([^:]+):([0-9]+) ]]; then
        REDIS_HOST=${BASH_REMATCH[1]}
        REDIS_PORT=${BASH_REMATCH[2]}
    else
        REDIS_HOST="redis"
        REDIS_PORT="6379"
    fi
    
    if ! wait_for_service "$REDIS_HOST" "$REDIS_PORT" "Redis" 30; then
        warn "Redis is not available, some features may not work"
    fi
}

# Function to setup directories
setup_directories() {
    log "Setting up directories..."
    
    # Create necessary directories
    mkdir -p \
        "$QUDATA_DATA_PATH"/{raw,staging,processed,exports} \
        "$QUDATA_LOG_PATH" \
        "$QUDATA_TEMP_PATH" \
        /app/.cache
    
    # Set permissions (may fail on Windows bind mounts; ignore and warn)
    chmod -R 755 "$QUDATA_DATA_PATH" "$QUDATA_LOG_PATH" "$QUDATA_TEMP_PATH" 2>/dev/null || \
        warn "Skipping chmod on bind-mounted paths (permission change not supported)"
    
    log "Directories setup completed"
}

# Function to validate configuration
validate_config() {
    log "Validating configuration..."
    
    # Check if config file exists
    if [ ! -f "$QUDATA_CONFIG_PATH/pipeline.yaml" ]; then
        warn "Pipeline configuration not found, using defaults"
        cp /app/configs/pipeline.yaml "$QUDATA_CONFIG_PATH/pipeline.yaml"
    fi
    
    # Validate configuration
    python -c "
from qudata.config import ConfigManager
try:
    config_manager = ConfigManager()
    config = config_manager.load_config('$QUDATA_CONFIG_PATH/pipeline.yaml')
    print('Configuration is valid')
except Exception as e:
    print(f'Configuration validation failed: {e}')
    exit(1)
" || {
        error "Configuration validation failed"
        exit 1
    }
    
    log "Configuration validation completed"
}

# Function to start API server
start_api() {
    log "Starting QuData API server..."
    
    # Only enable reload if explicitly requested
    if [ "${RELOAD}" = "true" ]; then
        log "Starting with auto-reload (RELOAD=true)"
        exec uvicorn qudata.api.rest_server:app \
            --host "$API_HOST" \
            --port "$API_PORT" \
            --reload \
            --log-level debug
    fi

    log "Starting in production mode"
    exec gunicorn qudata.api.rest_server:app \
        --bind "$API_HOST:$API_PORT" \
        --workers "$WORKERS" \
        --worker-class uvicorn.workers.UvicornWorker \
        --worker-connections 1000 \
        --max-requests 1000 \
        --max-requests-jitter 100 \
        --timeout 300 \
        --keep-alive 2 \
        --log-level info \
        --access-logfile - \
        --error-logfile -
}

# Function to start worker
start_worker() {
    log "Starting QuData worker..."
    
    exec celery -A qudata.orchestrate.celery_app worker \
        --loglevel=info \
        --concurrency="$CELERY_CONCURRENCY" \
        --max-tasks-per-child="$CELERY_MAX_TASKS_PER_CHILD" \
        --prefetch-multiplier="$CELERY_PREFETCH_MULTIPLIER"
}

# Function to start scheduler
start_scheduler() {
    log "Starting QuData scheduler..."
    
    exec celery -A qudata.orchestrate.celery_app beat \
        --loglevel=info \
        --schedule=/tmp/celerybeat-schedule \
        --pidfile=/tmp/celerybeat.pid
}

# Function to start dashboard
start_dashboard() {
    log "Starting QuData dashboard..."
    
    exec streamlit run qudata/visualize/dashboard.py \
        --server.port 8001 \
        --server.address 0.0.0.0 \
        --server.headless true \
        --server.enableCORS false \
        --server.enableXsrfProtection false
}

# Function to start monitoring
start_monitoring() {
    log "Starting QuData monitoring..."
    
    exec python -m qudata.visualize.metrics \
        --host 0.0.0.0 \
        --port 8080
}

# Function to run CLI command
run_cli() {
    log "Running QuData CLI command: $*"
    exec qudata "$@"
}

# Function to run supervisor (multiple services)
start_supervisor() {
    log "Starting QuData with supervisor..."
    exec supervisord -c /etc/supervisor/conf.d/supervisord.conf
}

# Main execution
main() {
    log "Starting QuData container..."
    log "Environment: ${ENVIRONMENT:-development}"
    log "Version: ${VERSION:-unknown}"
    
    # Load secrets
    load_secrets
    
    # Setup directories
    setup_directories
    
    # Validate configuration
    validate_config
    
    # Initialize services based on command
    case "${1:-api}" in
        "api")
            init_database
            init_redis
            start_api
            ;;
        "worker")
            init_database
            init_redis
            start_worker
            ;;
        "scheduler")
            init_database
            init_redis
            start_scheduler
            ;;
        "dashboard")
            init_database
            start_dashboard
            ;;
        "monitoring")
            start_monitoring
            ;;
        "supervisor")
            init_database
            init_redis
            start_supervisor
            ;;
        "cli")
            shift
            run_cli "$@"
            ;;
        "bash"|"sh")
            log "Starting interactive shell..."
            exec /bin/bash
            ;;
        *)
            log "Running custom command: $*"
            exec "$@"
            ;;
    esac
}

# Handle signals
trap 'log "Received SIGTERM, shutting down gracefully..."; exit 0' TERM
trap 'log "Received SIGINT, shutting down gracefully..."; exit 0' INT

# Run main function
main "$@"