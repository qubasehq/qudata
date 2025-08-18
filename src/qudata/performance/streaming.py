"""
Streaming processor for large file handling.

Provides streaming processing capabilities for handling large files
and datasets that don't fit in memory, with configurable chunk sizes
and processing pipelines.
"""

import io
import mmap
import threading
import time
from typing import Iterator, Callable, Any, Optional, Dict, List, Union, TextIO, BinaryIO
from dataclasses import dataclass
from enum import Enum
import logging
import os
import gzip
import bz2
import lzma
from pathlib import Path
import json
import csv

logger = logging.getLogger(__name__)


class StreamingMode(Enum):
    """Streaming processing modes."""
    LINE_BY_LINE = "line_by_line"
    CHUNK_BY_SIZE = "chunk_by_size"
    CHUNK_BY_COUNT = "chunk_by_count"
    MEMORY_MAPPED = "memory_mapped"
    BUFFERED = "buffered"


class CompressionType(Enum):
    """Supported compression types."""
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZMA = "lzma"


@dataclass
class StreamingConfig:
    """Streaming processor configuration."""
    mode: StreamingMode = StreamingMode.CHUNK_BY_SIZE
    chunk_size: int = 1024 * 1024  # 1MB chunks
    buffer_size: int = 8192        # 8KB buffer
    max_memory_usage: int = 100 * 1024 * 1024  # 100MB
    enable_compression: bool = True
    compression_type: CompressionType = CompressionType.GZIP
    encoding: str = 'utf-8'
    error_handling: str = 'ignore'  # 'strict', 'ignore', 'replace'
    enable_progress_tracking: bool = True
    enable_parallel_processing: bool = False
    num_workers: int = 4


@dataclass
class StreamingStats:
    """Streaming processing statistics."""
    total_bytes_processed: int = 0
    total_chunks_processed: int = 0
    processing_time: float = 0.0
    throughput_mbps: float = 0.0
    current_position: int = 0
    total_size: Optional[int] = None
    progress_percent: float = 0.0


class StreamingProcessor:
    """
    Streaming processor for handling large files and datasets.
    
    Provides memory-efficient processing of large files using various
    streaming strategies including line-by-line, chunked, and memory-mapped processing.
    """
    
    def __init__(self, config: Optional[StreamingConfig] = None):
        """Initialize streaming processor."""
        self.config = config or StreamingConfig()
        self._stats = StreamingStats()
        self._lock = threading.Lock()
        self._stop_processing = threading.Event()
        self._progress_callbacks = []
    
    def process_file(self, 
                    file_path: Union[str, Path],
                    processor: Callable[[Any], Any],
                    output_handler: Optional[Callable[[Any], None]] = None) -> StreamingStats:
        """
        Process a file using streaming.
        
        Args:
            file_path: Path to the file to process
            processor: Function to process each chunk/line
            output_handler: Optional function to handle processed results
            
        Returns:
            Processing statistics
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Reset stats
        self._stats = StreamingStats()
        self._stats.total_size = file_path.stat().st_size
        
        start_time = time.time()
        
        try:
            # Detect compression
            compression_type = self._detect_compression(file_path)
            
            # Open file with appropriate handler
            with self._open_file(file_path, compression_type) as file_handle:
                if self.config.mode == StreamingMode.LINE_BY_LINE:
                    self._process_line_by_line(file_handle, processor, output_handler)
                elif self.config.mode == StreamingMode.CHUNK_BY_SIZE:
                    self._process_chunk_by_size(file_handle, processor, output_handler)
                elif self.config.mode == StreamingMode.CHUNK_BY_COUNT:
                    self._process_chunk_by_count(file_handle, processor, output_handler)
                elif self.config.mode == StreamingMode.MEMORY_MAPPED:
                    self._process_memory_mapped(file_path, processor, output_handler)
                else:
                    self._process_buffered(file_handle, processor, output_handler)
            
            # Calculate final stats
            self._stats.processing_time = time.time() - start_time
            if self._stats.processing_time > 0:
                self._stats.throughput_mbps = (self._stats.total_bytes_processed / 1024 / 1024) / self._stats.processing_time
            
            logger.info(f"Streaming processing complete: {self._stats.total_bytes_processed} bytes in {self._stats.processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Streaming processing error: {e}")
            raise
        
        return self._stats
    
    def process_text_stream(self,
                           text_stream: Union[str, TextIO],
                           processor: Callable[[str], Any],
                           output_handler: Optional[Callable[[Any], None]] = None) -> StreamingStats:
        """
        Process a text stream.
        
        Args:
            text_stream: Text stream or string to process
            processor: Function to process each chunk/line
            output_handler: Optional function to handle processed results
            
        Returns:
            Processing statistics
        """
        self._stats = StreamingStats()
        start_time = time.time()
        
        try:
            if isinstance(text_stream, str):
                stream = io.StringIO(text_stream)
            else:
                stream = text_stream
            
            if self.config.mode == StreamingMode.LINE_BY_LINE:
                self._process_line_by_line(stream, processor, output_handler)
            else:
                self._process_chunk_by_size(stream, processor, output_handler)
            
            self._stats.processing_time = time.time() - start_time
            if self._stats.processing_time > 0:
                self._stats.throughput_mbps = (self._stats.total_bytes_processed / 1024 / 1024) / self._stats.processing_time
            
        except Exception as e:
            logger.error(f"Text stream processing error: {e}")
            raise
        
        return self._stats
    
    def process_json_lines(self,
                          file_path: Union[str, Path],
                          processor: Callable[[Dict], Any],
                          output_handler: Optional[Callable[[Any], None]] = None) -> StreamingStats:
        """
        Process JSONL (JSON Lines) file.
        
        Args:
            file_path: Path to JSONL file
            processor: Function to process each JSON object
            output_handler: Optional function to handle processed results
            
        Returns:
            Processing statistics
        """
        # Force line-by-line processing for JSON Lines
        original_mode = self.config.mode
        self.config.mode = StreamingMode.LINE_BY_LINE
        
        try:
            def json_processor(line: str) -> Any:
                try:
                    json_obj = json.loads(line.strip())
                    return processor(json_obj)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON line: {line[:100]}... Error: {e}")
                    return None
            
            return self.process_file(file_path, json_processor, output_handler)
        finally:
            # Restore original mode
            self.config.mode = original_mode
    
    def process_csv_stream(self,
                          file_path: Union[str, Path],
                          processor: Callable[[Dict], Any],
                          output_handler: Optional[Callable[[Any], None]] = None,
                          **csv_kwargs) -> StreamingStats:
        """
        Process CSV file in streaming mode.
        
        Args:
            file_path: Path to CSV file
            processor: Function to process each row
            output_handler: Optional function to handle processed results
            **csv_kwargs: Additional CSV reader arguments
            
        Returns:
            Processing statistics
        """
        file_path = Path(file_path)
        self._stats = StreamingStats()
        self._stats.total_size = file_path.stat().st_size
        
        start_time = time.time()
        
        try:
            with open(file_path, 'r', encoding=self.config.encoding, errors=self.config.error_handling) as f:
                reader = csv.DictReader(f, **csv_kwargs)
                
                for row_num, row in enumerate(reader):
                    if self._stop_processing.is_set():
                        break
                    
                    try:
                        result = processor(row)
                        if output_handler and result is not None:
                            output_handler(result)
                    except Exception as e:
                        logger.error(f"Error processing CSV row {row_num}: {e}")
                    
                    # Update stats
                    self._stats.total_chunks_processed += 1
                    # Note: f.tell() doesn't work with csv.DictReader due to buffering
                    # Estimate position based on processed rows
                    estimated_bytes_per_row = self._stats.total_size / max(1, self._stats.total_chunks_processed + 100)
                    self._stats.current_position = int(self._stats.total_chunks_processed * estimated_bytes_per_row)
                    self._update_progress()
            
            self._stats.processing_time = time.time() - start_time
            if self._stats.processing_time > 0:
                self._stats.throughput_mbps = (self._stats.total_bytes_processed / 1024 / 1024) / self._stats.processing_time
            
        except Exception as e:
            logger.error(f"CSV streaming processing error: {e}")
            raise
        
        return self._stats
    
    def _detect_compression(self, file_path: Path) -> CompressionType:
        """Detect file compression type."""
        suffix = file_path.suffix.lower()
        
        if suffix == '.gz':
            return CompressionType.GZIP
        elif suffix == '.bz2':
            return CompressionType.BZIP2
        elif suffix in ['.xz', '.lzma']:
            return CompressionType.LZMA
        else:
            return CompressionType.NONE
    
    def _open_file(self, file_path: Path, compression_type: CompressionType):
        """Open file with appropriate compression handler."""
        mode = 'rt' if self.config.mode == StreamingMode.LINE_BY_LINE else 'rb'
        
        if compression_type == CompressionType.GZIP:
            return gzip.open(file_path, mode, encoding=self.config.encoding, errors=self.config.error_handling)
        elif compression_type == CompressionType.BZIP2:
            return bz2.open(file_path, mode, encoding=self.config.encoding, errors=self.config.error_handling)
        elif compression_type == CompressionType.LZMA:
            return lzma.open(file_path, mode, encoding=self.config.encoding, errors=self.config.error_handling)
        else:
            if mode == 'rb':
                return open(file_path, mode)
            else:
                return open(file_path, mode, encoding=self.config.encoding, errors=self.config.error_handling)
    
    def _process_line_by_line(self, 
                             file_handle,
                             processor: Callable,
                             output_handler: Optional[Callable]):
        """Process file line by line."""
        for line_num, line in enumerate(file_handle):
            if self._stop_processing.is_set():
                break
            
            try:
                result = processor(line.rstrip('\n\r'))
                if output_handler and result is not None:
                    output_handler(result)
            except Exception as e:
                logger.error(f"Error processing line {line_num}: {e}")
            
            # Update stats
            self._stats.total_bytes_processed += len(line.encode(self.config.encoding))
            self._stats.total_chunks_processed += 1
            self._update_progress()
    
    def _process_chunk_by_size(self,
                              file_handle,
                              processor: Callable,
                              output_handler: Optional[Callable]):
        """Process file in fixed-size chunks."""
        while True:
            if self._stop_processing.is_set():
                break
            
            chunk = file_handle.read(self.config.chunk_size)
            if not chunk:
                break
            
            try:
                result = processor(chunk)
                if output_handler and result is not None:
                    output_handler(result)
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")
            
            # Update stats
            chunk_size = len(chunk) if isinstance(chunk, str) else len(chunk)
            self._stats.total_bytes_processed += chunk_size
            self._stats.total_chunks_processed += 1
            self._update_progress()
    
    def _process_chunk_by_count(self,
                               file_handle,
                               processor: Callable,
                               output_handler: Optional[Callable]):
        """Process file in chunks of fixed line count."""
        chunk_lines = []
        
        for line in file_handle:
            if self._stop_processing.is_set():
                break
            
            chunk_lines.append(line.rstrip('\n\r'))
            
            if len(chunk_lines) >= self.config.chunk_size:
                try:
                    result = processor(chunk_lines)
                    if output_handler and result is not None:
                        output_handler(result)
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")
                
                # Update stats
                chunk_bytes = sum(len(line.encode(self.config.encoding)) for line in chunk_lines)
                self._stats.total_bytes_processed += chunk_bytes
                self._stats.total_chunks_processed += 1
                self._update_progress()
                
                chunk_lines = []
        
        # Process remaining lines
        if chunk_lines:
            try:
                result = processor(chunk_lines)
                if output_handler and result is not None:
                    output_handler(result)
            except Exception as e:
                logger.error(f"Error processing final chunk: {e}")
            
            chunk_bytes = sum(len(line.encode(self.config.encoding)) for line in chunk_lines)
            self._stats.total_bytes_processed += chunk_bytes
            self._stats.total_chunks_processed += 1
            self._update_progress()
    
    def _process_memory_mapped(self,
                              file_path: Path,
                              processor: Callable,
                              output_handler: Optional[Callable]):
        """Process file using memory mapping."""
        with open(file_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                # Process in chunks
                for i in range(0, len(mm), self.config.chunk_size):
                    if self._stop_processing.is_set():
                        break
                    
                    chunk = mm[i:i + self.config.chunk_size]
                    
                    try:
                        # Decode chunk
                        text_chunk = chunk.decode(self.config.encoding, errors=self.config.error_handling)
                        result = processor(text_chunk)
                        if output_handler and result is not None:
                            output_handler(result)
                    except Exception as e:
                        logger.error(f"Error processing memory-mapped chunk: {e}")
                    
                    # Update stats
                    self._stats.total_bytes_processed += len(chunk)
                    self._stats.total_chunks_processed += 1
                    self._stats.current_position = i + len(chunk)
                    self._update_progress()
    
    def _process_buffered(self,
                         file_handle,
                         processor: Callable,
                         output_handler: Optional[Callable]):
        """Process file with buffered reading."""
        buffer = ""
        
        while True:
            if self._stop_processing.is_set():
                break
            
            chunk = file_handle.read(self.config.buffer_size)
            if not chunk:
                # Process remaining buffer
                if buffer:
                    try:
                        result = processor(buffer)
                        if output_handler and result is not None:
                            output_handler(result)
                    except Exception as e:
                        logger.error(f"Error processing final buffer: {e}")
                break
            
            buffer += chunk if isinstance(chunk, str) else chunk.decode(self.config.encoding, errors=self.config.error_handling)
            
            # Process complete lines
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                
                try:
                    result = processor(line)
                    if output_handler and result is not None:
                        output_handler(result)
                except Exception as e:
                    logger.error(f"Error processing buffered line: {e}")
                
                # Update stats
                self._stats.total_bytes_processed += len(line.encode(self.config.encoding))
                self._stats.total_chunks_processed += 1
                self._update_progress()
    
    def _update_progress(self):
        """Update processing progress."""
        if self._stats.total_size:
            self._stats.progress_percent = (self._stats.current_position / self._stats.total_size) * 100
        
        # Call progress callbacks
        if self.config.enable_progress_tracking:
            for callback in self._progress_callbacks:
                try:
                    callback(self._stats)
                except Exception as e:
                    logger.error(f"Progress callback error: {e}")
    
    def add_progress_callback(self, callback: Callable[[StreamingStats], None]):
        """Add a progress callback."""
        self._progress_callbacks.append(callback)
    
    def remove_progress_callback(self, callback: Callable[[StreamingStats], None]):
        """Remove a progress callback."""
        if callback in self._progress_callbacks:
            self._progress_callbacks.remove(callback)
    
    def stop_processing(self):
        """Stop the current processing operation."""
        self._stop_processing.set()
        logger.info("Streaming processing stop requested")
    
    def get_stats(self) -> StreamingStats:
        """Get current processing statistics."""
        return self._stats
    
    def estimate_processing_time(self, file_size: int, sample_processing_time: float) -> float:
        """
        Estimate total processing time based on sample.
        
        Args:
            file_size: Total file size in bytes
            sample_processing_time: Time taken to process a sample
            
        Returns:
            Estimated total processing time in seconds
        """
        if self._stats.total_bytes_processed == 0:
            return 0.0
        
        bytes_per_second = self._stats.total_bytes_processed / sample_processing_time
        return file_size / bytes_per_second
    
    def create_streaming_iterator(self, 
                                 file_path: Union[str, Path],
                                 chunk_processor: Optional[Callable] = None) -> Iterator[Any]:
        """
        Create a streaming iterator for processing large files.
        
        Args:
            file_path: Path to file
            chunk_processor: Optional processor for each chunk
            
        Yields:
            Processed chunks or raw chunks if no processor provided
        """
        file_path = Path(file_path)
        compression_type = self._detect_compression(file_path)
        
        with self._open_file(file_path, compression_type) as file_handle:
            if self.config.mode == StreamingMode.LINE_BY_LINE:
                for line in file_handle:
                    chunk = line.rstrip('\n\r')
                    if chunk_processor:
                        yield chunk_processor(chunk)
                    else:
                        yield chunk
            else:
                while True:
                    chunk = file_handle.read(self.config.chunk_size)
                    if not chunk:
                        break
                    
                    if chunk_processor:
                        yield chunk_processor(chunk)
                    else:
                        yield chunk


def create_streaming_processor(mode: str = "chunk_by_size",
                              chunk_size: int = 1024 * 1024,
                              enable_compression: bool = True) -> StreamingProcessor:
    """
    Factory function to create a streaming processor.
    
    Args:
        mode: Streaming mode
        chunk_size: Chunk size in bytes
        enable_compression: Enable compression support
        
    Returns:
        Configured StreamingProcessor instance
    """
    config = StreamingConfig(
        mode=StreamingMode(mode),
        chunk_size=chunk_size,
        enable_compression=enable_compression
    )
    return StreamingProcessor(config)