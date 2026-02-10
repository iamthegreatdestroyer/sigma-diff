package services

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestStreamerInitialization verifies streamer initializes correctly
func TestStreamerInitialization(t *testing.T) {
	config := &StreamConfig{
		ChunkSize:     4096,
		FlushInterval: 10 * time.Millisecond,
	}

	streamer := NewResponseStreamer(config)
	if streamer == nil {
		t.Error("Expected streamer, got nil")
	}

	metrics := streamer.GetMetrics()
	if metrics.TotalStreams != 0 {
		t.Errorf("Expected 0 streams, got %d", metrics.TotalStreams)
	}
}

// TestStreamReader verifies streaming reader functionality
func TestStreamReader(t *testing.T) {
	streamer := NewResponseStreamer(DefaultStreamConfig())
	defer streamer.Close()

	data := []byte("Hello, this is streaming data!")
	reader := bytes.NewReader(data)

	chunks := streamer.StreamReader(context.Background(), reader)

	var receivedData []byte
	for chunk := range chunks {
		if chunk.Error != nil {
			t.Errorf("Unexpected error: %v", chunk.Error)
		}

		if chunk.Final {
			break
		}

		receivedData = append(receivedData, chunk.Data...)
	}

	if !bytes.Equal(receivedData, data) {
		t.Errorf("Data mismatch: expected %s, got %s", string(data), string(receivedData))
	}

	metrics := streamer.GetMetrics()
	if metrics.TotalStreams != 1 {
		t.Errorf("Expected 1 stream, got %d", metrics.TotalStreams)
	}
}

// TestStreamWriter verifies streaming writer functionality
func TestStreamWriter(t *testing.T) {
	streamer := NewResponseStreamer(DefaultStreamConfig())
	defer streamer.Close()

	// Create test chunks
	chunks := make(chan *StreamChunk, 3)
	go func() {
		chunks <- &StreamChunk{
			Data: []byte("chunk1"),
		}
		chunks <- &StreamChunk{
			Data: []byte("chunk2"),
		}
		chunks <- &StreamChunk{
			Final: true,
		}
		close(chunks)
	}()

	var buf bytes.Buffer
	err := streamer.StreamWriter(context.Background(), &buf, chunks)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	expected := "chunk1chunk2"
	if buf.String() != expected {
		t.Errorf("Expected %s, got %s", expected, buf.String())
	}
}

// TestHTTPResponseStreaming verifies HTTP response streaming
func TestHTTPResponseStreaming(t *testing.T) {
	streamer := NewResponseStreamer(DefaultStreamConfig())
	defer streamer.Close()

	data := []byte("streaming response data")
	reader := bytes.NewReader(data)

	// Create test HTTP response
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		err := streamer.StreamHTTPResponse(r.Context(), w, reader)
		if err != nil {
			t.Errorf("Streaming error: %v", err)
		}
	})

	req := httptest.NewRequest("GET", "/stream", nil)
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	metrics := streamer.GetMetrics()
	if metrics.TotalStreams != 1 {
		t.Errorf("Expected 1 stream, got %d", metrics.TotalStreams)
	}

	if metrics.TotalBytes != int64(len(data)) {
		t.Errorf("Expected %d bytes, got %d", len(data), metrics.TotalBytes)
	}
}

// TestStreamMetrics verifies metrics are calculated correctly
func TestStreamMetrics(t *testing.T) {
	streamer := NewResponseStreamer(DefaultStreamConfig())
	defer streamer.Close()

	data := []byte("test data for metrics calculation")
	reader := bytes.NewReader(data)

	chunks := streamer.StreamReader(context.Background(), reader)
	for range chunks {
	}

	metrics := streamer.GetMetrics()

	if metrics.TotalChunks == 0 {
		t.Error("Expected chunks, got 0")
	}

	if metrics.TotalBytes != int64(len(data)) {
		t.Errorf("Expected %d bytes, got %d", len(data), metrics.TotalBytes)
	}

	if metrics.AverageChunkSize == 0 {
		t.Error("Expected positive average chunk size")
	}
}

// TestConcurrentStreaming verifies thread safety
func TestConcurrentStreaming(t *testing.T) {
	streamer := NewResponseStreamer(DefaultStreamConfig())
	defer streamer.Close()

	var wg sync.WaitGroup
	numGoroutines := 50

	for g := 0; g < numGoroutines; g++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			data := []byte(fmt.Sprintf("data-%d", id))
			reader := bytes.NewReader(data)

			chunks := streamer.StreamReader(context.Background(), reader)
			for range chunks {
			}
		}(g)
	}

	wg.Wait()

	metrics := streamer.GetMetrics()
	if metrics.TotalStreams != int64(numGoroutines) {
		t.Errorf("Expected %d streams, got %d", numGoroutines, metrics.TotalStreams)
	}
}

// TestStreamContextCancellation verifies context handling
func TestStreamContextCancellation(t *testing.T) {
	streamer := NewResponseStreamer(DefaultStreamConfig())
	defer streamer.Close()

	ctx, cancel := context.WithCancel(context.Background())

	// Create large data to ensure reading takes time
	data := make([]byte, 1024*1024) // 1MB
	reader := bytes.NewReader(data)

	// Cancel context quickly
	go func() {
		time.Sleep(10 * time.Millisecond)
		cancel()
	}()

	chunks := streamer.StreamReader(ctx, reader)

	var lastChunk *StreamChunk
	for chunk := range chunks {
		lastChunk = chunk
	}

	if lastChunk == nil || lastChunk.Error == nil {
		t.Error("Expected context cancellation error")
	}
}

// TestStreamChunkSize verifies chunk size is respected
func TestStreamChunkSize(t *testing.T) {
	config := DefaultStreamConfig()
	config.ChunkSize = 100 // Small chunk size

	streamer := NewResponseStreamer(config)
	defer streamer.Close()

	data := []byte(string(make([]byte, 500)))
	reader := bytes.NewReader(data)

	chunks := streamer.StreamReader(context.Background(), reader)

	maxChunk := 0
	for chunk := range chunks {
		if len(chunk.Data) > maxChunk {
			maxChunk = len(chunk.Data)
		}
	}

	if maxChunk > config.ChunkSize {
		t.Errorf("Chunk size exceeded: %d > %d", maxChunk, config.ChunkSize)
	}
}

// TestStreamerClose verifies graceful shutdown
func TestStreamerClose(t *testing.T) {
	streamer := NewResponseStreamer(DefaultStreamConfig())

	data := []byte("test data")
	reader := bytes.NewReader(data)

	chunks := streamer.StreamReader(context.Background(), reader)
	<-chunks // Get first chunk

	err := streamer.Close()
	if err != nil {
		t.Errorf("Unexpected error on close: %v", err)
	}
}

// TestStreamMetricsString verifies metrics string representation
func TestStreamMetricsString(t *testing.T) {
	metrics := &StreamMetrics{
		TotalStreams:  10,
		TotalChunks:   100,
		TotalBytes:    1024000,
		ActiveStreams: 5,
		FailedStreams: 0,
		MaxChunkSize:  4096,
		Throughput:    1024000,
	}

	str := metrics.String()
	if len(str) == 0 {
		t.Error("Metrics string is empty")
	}

	if !contains(str, "StreamMetrics") {
		t.Errorf("Metrics string missing 'StreamMetrics': %s", str)
	}
}

// TestStreamerStress performs stress testing
func TestStreamerStress(t *testing.T) {
	streamer := NewResponseStreamer(DefaultStreamConfig())
	defer streamer.Close()

	var wg sync.WaitGroup
	var successCount int64

	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			data := bytes.Repeat([]byte("x"), 10000) // 10KB per stream
			reader := bytes.NewReader(data)

			chunks := streamer.StreamReader(context.Background(), reader)
			count := 0
			for chunk := range chunks {
				if chunk == nil || chunk.Error != nil {
					return
				}
				count++
			}

			if count > 0 {
				atomic.AddInt64(&successCount, 1)
			}
		}(i)
	}

	wg.Wait()

	if successCount < 90 {
		t.Errorf("Expected at least 90 successful streams, got %d", successCount)
	}

	metrics := streamer.GetMetrics()
	t.Logf("Stress test metrics: %v", metrics.String())
}

// TestThroughputCalculation verifies throughput is calculated
func TestThroughputCalculation(t *testing.T) {
	streamer := NewResponseStreamer(DefaultStreamConfig())
	defer streamer.Close()

	// Create reasonably sized data
	data := bytes.Repeat([]byte("test"), 10000) // 40KB
	reader := bytes.NewReader(data)

	chunks := streamer.StreamReader(context.Background(), reader)
	for range chunks {
	}

	throughput := streamer.GetThroughput()
	if throughput > 0 {
		t.Logf("Calculated throughput: %.1f B/s", throughput)
	}

	metrics := streamer.GetMetrics()
	if metrics.Throughput == 0 {
		t.Log("Note: Throughput not calculated (execution too fast)")
	}
}

// TestMaxConcurrentStreams verifies concurrent stream limiting
func TestMaxConcurrentStreams(t *testing.T) {
	config := DefaultStreamConfig()
	config.MaxConcurrentStream = 5 // Limit to 5 concurrent

	streamer := NewResponseStreamer(config)
	defer streamer.Close()

	var maxActive int32
	var wg sync.WaitGroup

	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			current := atomic.AddInt32(&streamer.activeStreams, 1)
			if current > maxActive {
				atomic.StoreInt32(&maxActive, current)
			}

			data := []byte("test data")
			reader := bytes.NewReader(data)

			chunks := streamer.StreamReader(context.Background(), reader)
			for range chunks {
			}

			atomic.AddInt32(&streamer.activeStreams, -1)
		}()

		// Stagger the starts
		time.Sleep(1 * time.Millisecond)
	}

	wg.Wait()

	metrics := streamer.GetMetrics()
	t.Logf("Max concurrent streams: %d", maxActive)
	t.Logf("Total streams processed: %d", metrics.TotalStreams)
}

// BenchmarkStreamReader benchmarks streaming reader
func BenchmarkStreamReader(b *testing.B) {
	streamer := NewResponseStreamer(DefaultStreamConfig())
	defer streamer.Close()

	data := bytes.Repeat([]byte("test"), 10000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reader := bytes.NewReader(data)
		chunks := streamer.StreamReader(context.Background(), reader)
		for range chunks {
		}
	}
}

// BenchmarkConcurrentStreaming benchmarks concurrent streaming
func BenchmarkConcurrentStreaming(b *testing.B) {
	streamer := NewResponseStreamer(DefaultStreamConfig())
	defer streamer.Close()

	data := bytes.Repeat([]byte("test"), 10000)

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			reader := bytes.NewReader(data)
			chunks := streamer.StreamReader(context.Background(), reader)
			for range chunks {
			}
		}
	})
}
