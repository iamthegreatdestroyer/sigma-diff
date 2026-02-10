package services

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

// StreamChunk represents a single chunk in a response stream
type StreamChunk struct {
	ID        string
	Data      []byte
	Timestamp time.Time
	Error     error
	Final     bool // True if this is the last chunk
}

// StreamConfig defines streaming configuration
type StreamConfig struct {
	ChunkSize           int
	FlushInterval       time.Duration
	BufferSize          int
	EnableCompression   bool
	EnableChunking      bool
	MaxConcurrentStream int
}

// DefaultStreamConfig returns default streaming configuration
func DefaultStreamConfig() *StreamConfig {
	return &StreamConfig{
		ChunkSize:           4096,
		FlushInterval:       10 * time.Millisecond,
		BufferSize:          8192,
		EnableCompression:   true,
		EnableChunking:      true,
		MaxConcurrentStream: 1000,
	}
}

// StreamMetrics tracks streaming statistics
type StreamMetrics struct {
	TotalStreams     int64
	TotalChunks      int64
	TotalBytes       int64
	AverageChunkSize int64
	MaxChunkSize     int64
	ActiveStreams    int32
	FailedStreams    int64
	Throughput       float64 // bytes per second
}

// ResponseStreamer handles streaming responses efficiently
type ResponseStreamer struct {
	config          *StreamConfig
	metrics         *StreamMetrics
	mu              sync.RWMutex
	stopCh          chan struct{}
	wg              sync.WaitGroup
	activeStreams   int32
	maxObservedSize int64
	flushTicker     *time.Ticker
}

// NewResponseStreamer creates a new response streamer
func NewResponseStreamer(config *StreamConfig) *ResponseStreamer {
	if config == nil {
		config = DefaultStreamConfig()
	}

	streamer := &ResponseStreamer{
		config:      config,
		metrics:     &StreamMetrics{},
		stopCh:      make(chan struct{}),
		flushTicker: time.NewTicker(config.FlushInterval),
	}

	return streamer
}

// StreamHTTPResponse streams an HTTP response in chunks
func (rs *ResponseStreamer) StreamHTTPResponse(
	ctx context.Context,
	writer http.ResponseWriter,
	reader io.Reader,
) error {
	// Check if we're at max concurrent streams
	if rs.config.MaxConcurrentStream > 0 {
		for atomic.LoadInt32(&rs.activeStreams) >= int32(rs.config.MaxConcurrentStream) {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(10 * time.Millisecond):
			}
		}
	}

	atomic.AddInt32(&rs.activeStreams, 1)
	atomic.AddInt64(&rs.metrics.TotalStreams, 1)
	defer func() {
		atomic.AddInt32(&rs.activeStreams, -1)
	}()

	// Set up response headers for streaming
	writer.Header().Set("Transfer-Encoding", "chunked")
	writer.Header().Set("Content-Type", "text/plain; charset=utf-8")
	writer.Header().Set("Cache-Control", "no-cache")

	// Create buffered writer for efficiency
	buffer := make([]byte, rs.config.BufferSize)
	var startTime time.Time
	var totalBytes int64

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-rs.stopCh:
			return fmt.Errorf("streamer closed")
		default:
		}

		// Read chunk from reader
		n, err := reader.Read(buffer)
		if err != nil && err != io.EOF {
			atomic.AddInt64(&rs.metrics.FailedStreams, 1)
			return err
		}

		if n > 0 {
			if startTime.IsZero() {
				startTime = time.Now()
			}

			// Track metrics
			atomic.AddInt64(&rs.metrics.TotalChunks, 1)
			atomic.AddInt64(&rs.metrics.TotalBytes, int64(n))
			totalBytes += int64(n)

			if int64(n) > atomic.LoadInt64(&rs.maxObservedSize) {
				atomic.StoreInt64(&rs.maxObservedSize, int64(n))
			}

			// Write chunk to response
			if _, writeErr := writer.Write(buffer[:n]); writeErr != nil {
				atomic.AddInt64(&rs.metrics.FailedStreams, 1)
				return writeErr
			}

			// Flush if supported by writer
			if flusher, ok := writer.(http.Flusher); ok {
				flusher.Flush()
			}
		}

		if err == io.EOF {
			break
		}
	}

	// Calculate throughput
	if !startTime.IsZero() {
		duration := time.Since(startTime).Seconds()
		if duration > 0 {
			throughput := float64(totalBytes) / duration
			rs.mu.Lock()
			rs.metrics.Throughput = throughput
			rs.mu.Unlock()
		}
	}

	return nil
}

// StreamReader creates a streaming reader that chunks data
func (rs *ResponseStreamer) StreamReader(
	ctx context.Context,
	reader io.Reader,
) chan *StreamChunk {
	chunks := make(chan *StreamChunk, 10)

	go func() {
		defer close(chunks)

		atomic.AddInt32(&rs.activeStreams, 1)
		atomic.AddInt64(&rs.metrics.TotalStreams, 1)
		defer func() {
			atomic.AddInt32(&rs.activeStreams, -1)
		}()

		buffer := make([]byte, rs.config.ChunkSize)

		for {
			select {
			case <-ctx.Done():
				chunks <- &StreamChunk{
					Error: ctx.Err(),
					Final: true,
				}
				return
			case <-rs.stopCh:
				chunks <- &StreamChunk{
					Error: fmt.Errorf("streamer closed"),
					Final: true,
				}
				return
			default:
			}

			n, err := reader.Read(buffer)
			if n > 0 {
				atomic.AddInt64(&rs.metrics.TotalChunks, 1)
				atomic.AddInt64(&rs.metrics.TotalBytes, int64(n))

				if int64(n) > atomic.LoadInt64(&rs.maxObservedSize) {
					atomic.StoreInt64(&rs.maxObservedSize, int64(n))
				}

				chunk := &StreamChunk{
					ID:        fmt.Sprintf("chunk-%d", atomic.LoadInt64(&rs.metrics.TotalChunks)),
					Data:      make([]byte, n),
					Timestamp: time.Now(),
					Final:     false,
				}
				copy(chunk.Data, buffer[:n])

				select {
				case chunks <- chunk:
				case <-ctx.Done():
					return
				}
			}

			if err != nil {
				if err == io.EOF {
					chunks <- &StreamChunk{
						Final: true,
					}
				} else {
					atomic.AddInt64(&rs.metrics.FailedStreams, 1)
					chunks <- &StreamChunk{
						Error: err,
						Final: true,
					}
				}
				return
			}
		}
	}()

	return chunks
}

// StreamWriter writes data from a stream channel
func (rs *ResponseStreamer) StreamWriter(
	ctx context.Context,
	writer io.Writer,
	chunks <-chan *StreamChunk,
) error {
	bufferedWriter := bufio.NewWriterSize(writer, rs.config.BufferSize)
	defer bufferedWriter.Flush()

	var totalBytes int64
	var startTime time.Time

	for chunk := range chunks {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-rs.stopCh:
			return fmt.Errorf("streamer closed")
		default:
		}

		if chunk == nil {
			continue
		}

		if chunk.Error != nil {
			return chunk.Error
		}

		if len(chunk.Data) > 0 {
			if startTime.IsZero() {
				startTime = time.Now()
			}

			totalBytes += int64(len(chunk.Data))

			if _, err := bufferedWriter.Write(chunk.Data); err != nil {
				atomic.AddInt64(&rs.metrics.FailedStreams, 1)
				return err
			}

			// Periodic flush
			if atomic.LoadInt64(&rs.metrics.TotalChunks)%10 == 0 {
				if err := bufferedWriter.Flush(); err != nil {
					atomic.AddInt64(&rs.metrics.FailedStreams, 1)
					return err
				}
			}
		}

		if chunk.Final {
			break
		}
	}

	// Final flush
	if err := bufferedWriter.Flush(); err != nil {
		atomic.AddInt64(&rs.metrics.FailedStreams, 1)
		return err
	}

	// Calculate throughput
	if !startTime.IsZero() {
		duration := time.Since(startTime).Seconds()
		if duration > 0 {
			throughput := float64(totalBytes) / duration
			rs.mu.Lock()
			rs.metrics.Throughput = throughput
			rs.mu.Unlock()
		}
	}

	return nil
}

// GetMetrics returns current streaming metrics
func (rs *ResponseStreamer) GetMetrics() *StreamMetrics {
	rs.mu.RLock()
	defer rs.mu.RUnlock()

	metrics := &StreamMetrics{
		TotalStreams:  atomic.LoadInt64(&rs.metrics.TotalStreams),
		TotalChunks:   atomic.LoadInt64(&rs.metrics.TotalChunks),
		TotalBytes:    atomic.LoadInt64(&rs.metrics.TotalBytes),
		ActiveStreams: atomic.LoadInt32(&rs.activeStreams),
		FailedStreams: atomic.LoadInt64(&rs.metrics.FailedStreams),
		MaxChunkSize:  atomic.LoadInt64(&rs.maxObservedSize),
		Throughput:    rs.metrics.Throughput,
	}

	// Calculate average chunk size
	if metrics.TotalChunks > 0 {
		metrics.AverageChunkSize = metrics.TotalBytes / metrics.TotalChunks
	}

	return metrics
}

// GetThroughput returns current streaming throughput in bytes/sec
func (rs *ResponseStreamer) GetThroughput() float64 {
	rs.mu.RLock()
	defer rs.mu.RUnlock()
	return rs.metrics.Throughput
}

// Close closes the response streamer
func (rs *ResponseStreamer) Close() error {
	close(rs.stopCh)
	rs.flushTicker.Stop()
	rs.wg.Wait()
	return nil
}

// String returns string representation of stream metrics
func (m *StreamMetrics) String() string {
	return fmt.Sprintf(
		"StreamMetrics{Streams:%d, Chunks:%d, Bytes:%d, AvgChunk:%d, MaxChunk:%d, Active:%d, Failed:%d, Throughput:%.1f B/s}",
		m.TotalStreams,
		m.TotalChunks,
		m.TotalBytes,
		m.AverageChunkSize,
		m.MaxChunkSize,
		m.ActiveStreams,
		m.FailedStreams,
		m.Throughput,
	)
}
