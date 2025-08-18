-- QuData Database Initialization Script
-- This script sets up the initial database schema and configuration

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS qudata;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Set default schema
SET search_path TO qudata, public;

-- Create enum types
CREATE TYPE processing_status AS ENUM ('pending', 'processing', 'completed', 'failed', 'cancelled');
CREATE TYPE document_type AS ENUM ('pdf', 'docx', 'txt', 'html', 'csv', 'json', 'xml', 'md', 'epub');
CREATE TYPE quality_level AS ENUM ('low', 'medium', 'high', 'excellent');

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_type document_type NOT NULL,
    file_size BIGINT NOT NULL,
    content_hash TEXT NOT NULL,
    raw_content TEXT,
    cleaned_content TEXT,
    metadata JSONB DEFAULT '{}',
    quality_score DECIMAL(3,2) DEFAULT 0.0,
    quality_level quality_level,
    language VARCHAR(10),
    word_count INTEGER DEFAULT 0,
    character_count INTEGER DEFAULT 0,
    processing_status processing_status DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for documents
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(processing_status);
CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(file_type);
CREATE INDEX IF NOT EXISTS idx_documents_quality ON documents(quality_score);
CREATE INDEX IF NOT EXISTS idx_documents_language ON documents(language);
CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING GIN(metadata);

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_documents_content_search ON documents USING GIN(to_tsvector('english', cleaned_content));

-- Datasets table
CREATE TABLE IF NOT EXISTS datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    version VARCHAR(50) DEFAULT '1.0.0',
    configuration JSONB DEFAULT '{}',
    statistics JSONB DEFAULT '{}',
    quality_metrics JSONB DEFAULT '{}',
    document_count INTEGER DEFAULT 0,
    total_size BIGINT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Dataset documents relationship
CREATE TABLE IF NOT EXISTS dataset_documents (
    dataset_id UUID REFERENCES datasets(id) ON DELETE CASCADE,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    added_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (dataset_id, document_id)
);

-- Processing jobs table
CREATE TABLE IF NOT EXISTS processing_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_type VARCHAR(50) NOT NULL,
    status processing_status DEFAULT 'pending',
    configuration JSONB DEFAULT '{}',
    progress INTEGER DEFAULT 0,
    total_items INTEGER DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Job documents relationship
CREATE TABLE IF NOT EXISTS job_documents (
    job_id UUID REFERENCES processing_jobs(id) ON DELETE CASCADE,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    status processing_status DEFAULT 'pending',
    error_message TEXT,
    processing_time INTERVAL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (job_id, document_id)
);

-- Quality metrics table
CREATE TABLE IF NOT EXISTS quality_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,6) NOT NULL,
    metric_metadata JSONB DEFAULT '{}',
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_quality_metrics_document ON quality_metrics(document_id);
CREATE INDEX IF NOT EXISTS idx_quality_metrics_name ON quality_metrics(metric_name);

-- Annotations table
CREATE TABLE IF NOT EXISTS annotations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    annotation_type VARCHAR(50) NOT NULL,
    annotation_data JSONB NOT NULL,
    confidence_score DECIMAL(3,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_annotations_document ON annotations(document_id);
CREATE INDEX IF NOT EXISTS idx_annotations_type ON annotations(annotation_type);
CREATE INDEX IF NOT EXISTS idx_annotations_data ON annotations USING GIN(annotation_data);

-- Entities table (for NER results)
CREATE TABLE IF NOT EXISTS entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    entity_text TEXT NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    start_position INTEGER NOT NULL,
    end_position INTEGER NOT NULL,
    confidence_score DECIMAL(3,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_entities_document ON entities(document_id);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_text ON entities(entity_text);

-- Export jobs table
CREATE TABLE IF NOT EXISTS export_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id UUID REFERENCES datasets(id) ON DELETE CASCADE,
    export_format VARCHAR(50) NOT NULL,
    export_path TEXT NOT NULL,
    configuration JSONB DEFAULT '{}',
    status processing_status DEFAULT 'pending',
    file_count INTEGER DEFAULT 0,
    total_size BIGINT DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Webhook endpoints table
CREATE TABLE IF NOT EXISTS webhook_endpoints (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    url TEXT NOT NULL,
    events TEXT[] NOT NULL,
    secret_key TEXT,
    active BOOLEAN DEFAULT true,
    timeout_seconds INTEGER DEFAULT 30,
    retry_count INTEGER DEFAULT 3,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Webhook deliveries table
CREATE TABLE IF NOT EXISTS webhook_deliveries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    endpoint_id UUID REFERENCES webhook_endpoints(id) ON DELETE CASCADE,
    event_type VARCHAR(100) NOT NULL,
    payload JSONB NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    response_status INTEGER,
    response_body TEXT,
    attempt_count INTEGER DEFAULT 0,
    next_retry_at TIMESTAMP WITH TIME ZONE,
    delivered_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Analytics schema tables
SET search_path TO analytics, public;

-- Daily processing statistics
CREATE TABLE IF NOT EXISTS daily_stats (
    date DATE PRIMARY KEY,
    documents_processed INTEGER DEFAULT 0,
    documents_failed INTEGER DEFAULT 0,
    total_processing_time INTERVAL DEFAULT '0 seconds',
    average_quality_score DECIMAL(3,2),
    total_data_size BIGINT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Language distribution
CREATE TABLE IF NOT EXISTS language_stats (
    date DATE NOT NULL,
    language VARCHAR(10) NOT NULL,
    document_count INTEGER DEFAULT 0,
    total_words BIGINT DEFAULT 0,
    PRIMARY KEY (date, language)
);

-- Quality distribution
CREATE TABLE IF NOT EXISTS quality_stats (
    date DATE NOT NULL,
    quality_range VARCHAR(20) NOT NULL, -- '0.0-0.2', '0.2-0.4', etc.
    document_count INTEGER DEFAULT 0,
    PRIMARY KEY (date, quality_range)
);

-- Monitoring schema tables
SET search_path TO monitoring, public;

-- System metrics
CREATE TABLE IF NOT EXISTS system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,6) NOT NULL,
    metric_unit VARCHAR(20),
    tags JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_system_metrics_name_time ON system_metrics(metric_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_system_metrics_tags ON system_metrics USING GIN(tags);

-- Application logs
CREATE TABLE IF NOT EXISTS application_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    level VARCHAR(20) NOT NULL,
    logger_name VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    exception_info TEXT,
    extra_data JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_application_logs_level_time ON application_logs(level, timestamp);
CREATE INDEX IF NOT EXISTS idx_application_logs_logger_time ON application_logs(logger_name, timestamp);

-- Reset search path
SET search_path TO qudata, public;

-- Create functions for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at columns
CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_datasets_updated_at BEFORE UPDATE ON datasets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_processing_jobs_updated_at BEFORE UPDATE ON processing_jobs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_webhook_endpoints_updated_at BEFORE UPDATE ON webhook_endpoints
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create function to calculate document statistics
CREATE OR REPLACE FUNCTION calculate_document_stats(doc_id UUID)
RETURNS VOID AS $$
BEGIN
    UPDATE documents 
    SET 
        word_count = array_length(string_to_array(cleaned_content, ' '), 1),
        character_count = length(cleaned_content),
        quality_level = CASE 
            WHEN quality_score >= 0.8 THEN 'excellent'::quality_level
            WHEN quality_score >= 0.6 THEN 'high'::quality_level
            WHEN quality_score >= 0.4 THEN 'medium'::quality_level
            ELSE 'low'::quality_level
        END
    WHERE id = doc_id;
END;
$$ LANGUAGE plpgsql;

-- Create function to update dataset statistics
CREATE OR REPLACE FUNCTION update_dataset_stats(dataset_id UUID)
RETURNS VOID AS $$
BEGIN
    UPDATE datasets 
    SET 
        document_count = (
            SELECT COUNT(*) 
            FROM dataset_documents 
            WHERE dataset_documents.dataset_id = datasets.id
        ),
        total_size = (
            SELECT COALESCE(SUM(d.file_size), 0)
            FROM dataset_documents dd
            JOIN documents d ON dd.document_id = d.id
            WHERE dd.dataset_id = datasets.id
        ),
        statistics = jsonb_build_object(
            'avg_quality_score', (
                SELECT COALESCE(AVG(d.quality_score), 0)
                FROM dataset_documents dd
                JOIN documents d ON dd.document_id = d.id
                WHERE dd.dataset_id = datasets.id
            ),
            'language_distribution', (
                SELECT jsonb_object_agg(d.language, lang_count)
                FROM (
                    SELECT d.language, COUNT(*) as lang_count
                    FROM dataset_documents dd
                    JOIN documents d ON dd.document_id = d.id
                    WHERE dd.dataset_id = datasets.id AND d.language IS NOT NULL
                    GROUP BY d.language
                ) d
            ),
            'file_type_distribution', (
                SELECT jsonb_object_agg(d.file_type, type_count)
                FROM (
                    SELECT d.file_type, COUNT(*) as type_count
                    FROM dataset_documents dd
                    JOIN documents d ON dd.document_id = d.id
                    WHERE dd.dataset_id = datasets.id
                    GROUP BY d.file_type
                ) d
            )
        )
    WHERE id = dataset_id;
END;
$$ LANGUAGE plpgsql;

-- Create views for common queries
CREATE OR REPLACE VIEW document_summary AS
SELECT 
    d.id,
    d.file_name,
    d.file_type,
    d.language,
    d.quality_score,
    d.quality_level,
    d.word_count,
    d.processing_status,
    d.created_at,
    COUNT(a.id) as annotation_count,
    COUNT(e.id) as entity_count
FROM documents d
LEFT JOIN annotations a ON d.id = a.document_id
LEFT JOIN entities e ON d.id = e.document_id
GROUP BY d.id, d.file_name, d.file_type, d.language, d.quality_score, 
         d.quality_level, d.word_count, d.processing_status, d.created_at;

-- Create view for dataset summary
CREATE OR REPLACE VIEW dataset_summary AS
SELECT 
    ds.id,
    ds.name,
    ds.version,
    ds.document_count,
    ds.total_size,
    ds.created_at,
    ds.statistics,
    COALESCE(AVG(d.quality_score), 0) as avg_quality_score,
    COUNT(DISTINCT d.language) as language_count,
    COUNT(DISTINCT d.file_type) as file_type_count
FROM datasets ds
LEFT JOIN dataset_documents dd ON ds.id = dd.dataset_id
LEFT JOIN documents d ON dd.document_id = d.id
GROUP BY ds.id, ds.name, ds.version, ds.document_count, ds.total_size, 
         ds.created_at, ds.statistics;

-- Insert default configuration data
INSERT INTO datasets (name, description, version, configuration) VALUES 
('default', 'Default dataset for processed documents', '1.0.0', '{"auto_add": true}')
ON CONFLICT (name) DO NOTHING;

-- Grant permissions
GRANT USAGE ON SCHEMA qudata TO qudata;
GRANT USAGE ON SCHEMA analytics TO qudata;
GRANT USAGE ON SCHEMA monitoring TO qudata;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA qudata TO qudata;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO qudata;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA monitoring TO qudata;

GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA qudata TO qudata;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA analytics TO qudata;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA monitoring TO qudata;

-- Create indexes for performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_composite 
ON documents(processing_status, quality_score, language, created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_processing_jobs_status_created 
ON processing_jobs(status, created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_webhook_deliveries_status_retry 
ON webhook_deliveries(status, next_retry_at);

-- Analyze tables for query optimization
ANALYZE documents;
ANALYZE datasets;
ANALYZE processing_jobs;
ANALYZE annotations;
ANALYZE entities;

-- Log initialization completion
INSERT INTO monitoring.application_logs (level, logger_name, message, extra_data)
VALUES ('INFO', 'database.init', 'Database initialization completed successfully', 
        jsonb_build_object('timestamp', NOW(), 'version', '1.0.0'));