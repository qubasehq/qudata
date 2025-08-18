#!/bin/bash
set -e

# QuData Deployment Script
# Supports development, staging, and production deployments

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="development"
COMPOSE_FILE="docker-compose.yml"
BUILD_ARGS=""
SERVICES=""
SCALE_ARGS=""
BACKUP_BEFORE_DEPLOY=false
PULL_IMAGES=false
PRUNE_AFTER_DEPLOY=false

# Function to print colored output
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Function to show usage
usage() {
    cat << EOF
QuData Deployment Script

Usage: $0 [OPTIONS]

OPTIONS:
    -e, --environment ENV    Deployment environment (development, staging, production)
    -f, --file FILE         Docker Compose file to use
    -s, --services SERVICES Specific services to deploy (comma-separated)
    -b, --backup            Create backup before deployment
    -p, --pull              Pull latest images before deployment
    -c, --clean             Clean up unused containers and images after deployment
    --scale SERVICE=NUM     Scale specific services (e.g., --scale api=3)
    --build-arg ARG=VALUE   Pass build arguments
    -h, --help              Show this help message

EXAMPLES:
    # Development deployment
    $0 -e development

    # Production deployment with backup
    $0 -e production -b -p -c

    # Deploy specific services
    $0 -e production -s api,worker

    # Scale services
    $0 -e production --scale api=3 --scale worker=2

    # Custom compose file
    $0 -f docker-compose.custom.yml

EOF
}

# Function to validate environment
validate_environment() {
    case $ENVIRONMENT in
        development|staging|production)
            log "Deploying to $ENVIRONMENT environment"
            ;;
        *)
            error "Invalid environment: $ENVIRONMENT"
            error "Valid environments: development, staging, production"
            exit 1
            ;;
    esac
}

# Function to set compose file based on environment
set_compose_file() {
    if [[ -z "$COMPOSE_FILE" || "$COMPOSE_FILE" == "docker-compose.yml" ]]; then
        case $ENVIRONMENT in
            development)
                COMPOSE_FILE="docker-compose.yml"
                ;;
            staging)
                COMPOSE_FILE="docker-compose.staging.yml"
                if [[ ! -f "$COMPOSE_FILE" ]]; then
                    warn "Staging compose file not found, using production file"
                    COMPOSE_FILE="docker-compose.prod.yml"
                fi
                ;;
            production)
                COMPOSE_FILE="docker-compose.prod.yml"
                ;;
        esac
    fi

    if [[ ! -f "$COMPOSE_FILE" ]]; then
        error "Compose file not found: $COMPOSE_FILE"
        exit 1
    fi

    log "Using compose file: $COMPOSE_FILE"
}

# Function to check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        error "Docker is not running"
        exit 1
    fi

    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
        exit 1
    fi

    # Check if .env file exists for production
    if [[ "$ENVIRONMENT" == "production" && ! -f ".env" ]]; then
        warn ".env file not found for production deployment"
        warn "Please create .env file from .env.example"
    fi

    # Check if secrets directory exists for production
    if [[ "$ENVIRONMENT" == "production" && ! -d "secrets" ]]; then
        warn "Secrets directory not found for production deployment"
        warn "Please create secrets directory with required secret files"
    fi

    log "Prerequisites check completed"
}

# Function to create necessary directories
create_directories() {
    log "Creating necessary directories..."

    # Create data directories
    mkdir -p data/{raw,staging,processed,exports}
    mkdir -p logs
    mkdir -p backups
    mkdir -p tmp

    # Create secrets directory for production
    if [[ "$ENVIRONMENT" == "production" ]]; then
        mkdir -p secrets
        
        # Create placeholder secret files if they don't exist
        if [[ ! -f "secrets/db_password.txt" ]]; then
            warn "Creating placeholder database password file"
            echo "change-this-password" > secrets/db_password.txt
            chmod 600 secrets/db_password.txt
        fi
        
        if [[ ! -f "secrets/redis_password.txt" ]]; then
            warn "Creating placeholder Redis password file"
            echo "change-this-password" > secrets/redis_password.txt
            chmod 600 secrets/redis_password.txt
        fi
        
        if [[ ! -f "secrets/secret_key.txt" ]]; then
            warn "Creating placeholder secret key file"
            openssl rand -base64 32 > secrets/secret_key.txt
            chmod 600 secrets/secret_key.txt
        fi
        
        if [[ ! -f "secrets/grafana_password.txt" ]]; then
            warn "Creating placeholder Grafana password file"
            echo "admin" > secrets/grafana_password.txt
            chmod 600 secrets/grafana_password.txt
        fi
    fi

    log "Directories created successfully"
}

# Function to create backup
create_backup() {
    if [[ "$BACKUP_BEFORE_DEPLOY" == "true" ]]; then
        log "Creating backup before deployment..."
        
        BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$BACKUP_DIR"
        
        # Backup database if running
        if docker-compose -f "$COMPOSE_FILE" ps postgres | grep -q "Up"; then
            log "Backing up database..."
            docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_dump -U qudata qudata > "$BACKUP_DIR/database.sql"
        fi
        
        # Backup data volumes
        if [[ -d "data" ]]; then
            log "Backing up data directory..."
            tar -czf "$BACKUP_DIR/data.tar.gz" data/
        fi
        
        # Backup configuration
        if [[ -f ".env" ]]; then
            cp .env "$BACKUP_DIR/"
        fi
        
        log "Backup created at: $BACKUP_DIR"
    fi
}

# Function to pull images
pull_images() {
    if [[ "$PULL_IMAGES" == "true" ]]; then
        log "Pulling latest images..."
        docker-compose -f "$COMPOSE_FILE" pull
    fi
}

# Function to build images
build_images() {
    log "Building images..."
    
    BUILD_CMD="docker-compose -f $COMPOSE_FILE build"
    
    # Add build arguments
    if [[ -n "$BUILD_ARGS" ]]; then
        BUILD_CMD="$BUILD_CMD $BUILD_ARGS"
    fi
    
    # Add services if specified
    if [[ -n "$SERVICES" ]]; then
        BUILD_CMD="$BUILD_CMD $SERVICES"
    fi
    
    eval "$BUILD_CMD"
}

# Function to deploy services
deploy_services() {
    log "Deploying services..."
    
    DEPLOY_CMD="docker-compose -f $COMPOSE_FILE up -d"
    
    # Add services if specified
    if [[ -n "$SERVICES" ]]; then
        DEPLOY_CMD="$DEPLOY_CMD $SERVICES"
    fi
    
    eval "$DEPLOY_CMD"
    
    # Scale services if specified
    if [[ -n "$SCALE_ARGS" ]]; then
        log "Scaling services..."
        docker-compose -f "$COMPOSE_FILE" up -d --scale $SCALE_ARGS
    fi
}

# Function to wait for services
wait_for_services() {
    log "Waiting for services to be ready..."
    
    # Wait for database
    if docker-compose -f "$COMPOSE_FILE" ps postgres | grep -q "Up"; then
        log "Waiting for PostgreSQL..."
        timeout 60 bash -c 'until docker-compose -f "'$COMPOSE_FILE'" exec -T postgres pg_isready -U qudata; do sleep 2; done'
    fi
    
    # Wait for Redis
    if docker-compose -f "$COMPOSE_FILE" ps redis | grep -q "Up"; then
        log "Waiting for Redis..."
        timeout 30 bash -c 'until docker-compose -f "'$COMPOSE_FILE'" exec -T redis redis-cli ping; do sleep 2; done'
    fi
    
    # Wait for API
    if docker-compose -f "$COMPOSE_FILE" ps qudata | grep -q "Up" || docker-compose -f "$COMPOSE_FILE" ps qudata-api | grep -q "Up"; then
        log "Waiting for QuData API..."
        timeout 120 bash -c 'until curl -f http://localhost:8000/health; do sleep 5; done'
    fi
    
    log "Services are ready"
}

# Function to run post-deployment tasks
post_deployment_tasks() {
    log "Running post-deployment tasks..."
    
    # Run database migrations
    if docker-compose -f "$COMPOSE_FILE" ps postgres | grep -q "Up"; then
        log "Running database migrations..."
        docker-compose -f "$COMPOSE_FILE" exec -T qudata-api python -c "
from qudata.database import DatabaseConnector
try:
    connector = DatabaseConnector()
    connector.initialize_schema()
    print('Database schema updated successfully')
except Exception as e:
    print(f'Database migration failed: {e}')
" || warn "Database migration failed"
    fi
    
    # Warm up cache
    log "Warming up application cache..."
    curl -s http://localhost:8000/health > /dev/null || warn "Failed to warm up cache"
    
    log "Post-deployment tasks completed"
}

# Function to show deployment status
show_status() {
    log "Deployment status:"
    docker-compose -f "$COMPOSE_FILE" ps
    
    echo ""
    log "Service URLs:"
   
    if docker-compose -f "$COMPOSE_FILE" ps grafana | grep -q "Up"; then
        echo "  Monitoring Dashboard: http://localhost:3000"
    fi
    
    if docker-compose -f "$COMPOSE_FILE" ps nginx | grep -q "Up"; then
        echo "  Load Balancer: http://localhost"
    fi
}

# Function to clean up
cleanup() {
    if [[ "$PRUNE_AFTER_DEPLOY" == "true" ]]; then
        log "Cleaning up unused containers and images..."
        docker system prune -f
        docker volume prune -f
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -f|--file)
            COMPOSE_FILE="$2"
            shift 2
            ;;
        -s|--services)
            SERVICES="$2"
            shift 2
            ;;
        -b|--backup)
            BACKUP_BEFORE_DEPLOY=true
            shift
            ;;
        -p|--pull)
            PULL_IMAGES=true
            shift
            ;;
        -c|--clean)
            PRUNE_AFTER_DEPLOY=true
            shift
            ;;
        --scale)
            if [[ -n "$SCALE_ARGS" ]]; then
                SCALE_ARGS="$SCALE_ARGS $2"
            else
                SCALE_ARGS="$2"
            fi
            shift 2
            ;;
        --build-arg)
            if [[ -n "$BUILD_ARGS" ]]; then
                BUILD_ARGS="$BUILD_ARGS --build-arg $2"
            else
                BUILD_ARGS="--build-arg $2"
            fi
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main deployment process
main() {
    log "Starting QuData deployment..."
    
    validate_environment
    set_compose_file
    check_prerequisites
    create_directories
    create_backup
    pull_images
    build_images
    deploy_services
    wait_for_services
    post_deployment_tasks
    show_status
    cleanup
    
    log "Deployment completed successfully!"
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        warn "Production deployment completed. Please:"
        warn "1. Update secret files in the secrets/ directory"
        warn "2. Configure SSL certificates"
        warn "3. Set up monitoring and alerting"
        warn "4. Configure backup schedules"
    fi
}

# Handle script interruption
trap 'error "Deployment interrupted"; exit 1' INT TERM

# Run main function
main