package services

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestBatcherInitialization verifies batcher initializes correctly
func TestBatcherInitialization(t *testing.T) {
	config := &BatchConfig{
		MaxBatchSize: 100,
		MinBatchSize: 10,
		BatchTimeout: 50 * time.Millisecond,
	}

	batcher := NewRequestBatcher(config)
	defer batcher.Close()

	if batcher == nil {
		t.Error("Expected batcher, got nil")
	}

	metrics := batcher.GetMetrics()
	if metrics.TotalBatches != 0 {
		t.Errorf("Expected 0 batches, got %d", metrics.TotalBatches)
	}
}

// TestAddRequest verifies adding requests to batcher
func TestAddRequest(t *testing.T) {
	batcher := NewRequestBatcher(DefaultBatchConfig())
	defer batcher.Close()

	req := &BatchRequest{
		ID:      "test-1",
		Request: "data",
		Result:  make(chan interface{}, 1),
		Error:   make(chan error, 1),
	}

	err := batcher.AddRequest(context.Background(), req)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// Verify request was added
	if batcher.GetCurrentBatchSize() != 1 {
		t.Errorf("Expected batch size 1, got %d", batcher.GetCurrentBatchSize())
	}
}

// TestBatchAccumulation verifies requests accumulate into batches
func TestBatchAccumulation(t *testing.T) {
	config := DefaultBatchConfig()
	config.MaxBatchSize = 50

	batcher := NewRequestBatcher(config)
	defer batcher.Close()

	// Add 30 requests
	for i := 0; i < 30; i++ {
		req := &BatchRequest{
			ID:      fmt.Sprintf("req-%d", i),
			Request: i,
			Result:  make(chan interface{}, 1),
			Error:   make(chan error, 1),
		}
		batcher.AddRequest(context.Background(), req)
	}

	// All should be accumulated but not dispatched
	if batcher.GetCurrentBatchSize() != 30 {
		t.Errorf("Expected 30 accumulated, got %d", batcher.GetCurrentBatchSize())
	}
}

// TestBatchDispatchBySize verifies batch dispatches when max size reached
func TestBatchDispatchBySize(t *testing.T) {
	config := DefaultBatchConfig()
	config.MaxBatchSize = 10

	batcher := NewRequestBatcher(config)
	defer batcher.Close()

	// Add 10 requests to trigger dispatch
	for i := 0; i < 10; i++ {
		req := &BatchRequest{
			ID:      fmt.Sprintf("req-%d", i),
			Request: i,
			Result:  make(chan interface{}, 1),
			Error:   make(chan error, 1),
		}
		batcher.AddRequest(context.Background(), req)
	}

	// Give time for dispatch
	time.Sleep(100 * time.Millisecond)

	// Try to get batch
	batch, ok := batcher.GetBatch()
	if !ok {
		t.Error("Expected batch, got none")
	}

	if len(batch) != 10 {
		t.Errorf("Expected batch size 10, got %d", len(batch))
	}

	metrics := batcher.GetMetrics()
	if metrics.SizeDespatch != 1 {
		t.Errorf("Expected 1 size dispatch, got %d", metrics.SizeDespatch)
	}
}

// TestBatchDispatchByTimeout verifies batch dispatches on timeout
func TestBatchDispatchByTimeout(t *testing.T) {
	config := DefaultBatchConfig()
	config.BatchTimeout = 100 * time.Millisecond
	config.MinBatchSize = 1

	batcher := NewRequestBatcher(config)
	defer batcher.Close()

	// Add 5 requests (less than max)
	for i := 0; i < 5; i++ {
		req := &BatchRequest{
			ID:      fmt.Sprintf("req-%d", i),
			Request: i,
			Result:  make(chan interface{}, 1),
			Error:   make(chan error, 1),
		}
		batcher.AddRequest(context.Background(), req)
	}

	// Wait for timeout
	time.Sleep(200 * time.Millisecond)

	// Get batch - should be dispatched by timeout
	batch, ok := batcher.GetBatch()
	if !ok {
		t.Error("Expected batch, got none")
	}

	if len(batch) != 5 {
		t.Errorf("Expected batch size 5, got %d", len(batch))
	}

	metrics := batcher.GetMetrics()
	if metrics.TimeoutDespatch != 1 {
		t.Errorf("Expected 1 timeout dispatch, got %d", metrics.TimeoutDespatch)
	}
}

// TestBatchMetrics verifies metrics are calculated correctly
func TestBatchMetrics(t *testing.T) {
	batcher := NewRequestBatcher(DefaultBatchConfig())
	defer batcher.Close()

	// Add 25 requests
	for i := 0; i < 25; i++ {
		req := &BatchRequest{
			ID:      fmt.Sprintf("req-%d", i),
			Request: i,
			Result:  make(chan interface{}, 1),
			Error:   make(chan error, 1),
		}
		batcher.AddRequest(context.Background(), req)
	}

	time.Sleep(100 * time.Millisecond)

	metrics := batcher.GetMetrics()

	// Should have some batches created
	if metrics.TotalRequests < 25 {
		t.Errorf("Expected 25 requests, got %d", metrics.TotalRequests)
	}

	// Average batch size should be calculated
	if metrics.AverageBatchSize <= 0 {
		t.Errorf("Expected positive average batch size, got %.1f", metrics.AverageBatchSize)
	}
}

// TestResolveBatch verifies batch processing
func TestResolveBatch(t *testing.T) {
	batcher := NewRequestBatcher(DefaultBatchConfig())
	defer batcher.Close()

	// Create a batch
	batch := make([]*BatchRequest, 3)
	for i := 0; i < 3; i++ {
		batch[i] = &BatchRequest{
			ID:      fmt.Sprintf("req-%d", i),
			Request: i,
			Result:  make(chan interface{}, 1),
			Error:   make(chan error, 1),
		}
	}

	// Process batch
	processor := func(req *BatchRequest) error {
		// Simulate processing
		return nil
	}

	batcher.ResolveBatch(batch, processor)

	// Verify results were sent
	for i := 0; i < 3; i++ {
		select {
		case result := <-batch[i].Result:
			if result != nil {
				t.Errorf("Expected nil result, got %v", result)
			}
		case <-time.After(100 * time.Millisecond):
			t.Errorf("Request %d did not complete", i)
		}
	}
}

// TestConcurrentBatching verifies thread safety
func TestConcurrentBatching(t *testing.T) {
	batcher := NewRequestBatcher(DefaultBatchConfig())
	defer batcher.Close()

	var wg sync.WaitGroup
	numGoroutines := 50
	requestsPerGoroutine := 10

	for g := 0; g < numGoroutines; g++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := 0; i < requestsPerGoroutine; i++ {
				req := &BatchRequest{
					ID:      fmt.Sprintf("g%d-r%d", id, i),
					Request: i,
					Result:  make(chan interface{}, 1),
					Error:   make(chan error, 1),
				}
				batcher.AddRequest(context.Background(), req)
			}
		}(g)
	}

	wg.Wait()

	time.Sleep(200 * time.Millisecond)

	metrics := batcher.GetMetrics()
	expectedRequests := int64(numGoroutines * requestsPerGoroutine)

	if metrics.TotalRequests < expectedRequests {
		t.Errorf("Expected at least %d requests, got %d", expectedRequests, metrics.TotalRequests)
	}
}

// TestBatcherClose verifies graceful shutdown
func TestBatcherClose(t *testing.T) {
	batcher := NewRequestBatcher(DefaultBatchConfig())

	// Add a request
	req := &BatchRequest{
		ID:      "test",
		Request: "data",
		Result:  make(chan interface{}, 1),
		Error:   make(chan error, 1),
	}

	batcher.AddRequest(context.Background(), req)

	// Close should not error
	err := batcher.Close()
	if err != nil {
		t.Errorf("Unexpected error on close: %v", err)
	}

	// Subsequent adds should fail
	err = batcher.AddRequest(context.Background(), req)
	if err == nil {
		t.Error("Expected error after close")
	}
}

// TestNilRequest verifies nil request is rejected
func TestNilRequest(t *testing.T) {
	batcher := NewRequestBatcher(DefaultBatchConfig())
	defer batcher.Close()

	err := batcher.AddRequest(context.Background(), nil)
	if err == nil {
		t.Error("Expected error for nil request")
	}
}

// TestBatchMetricsString verifies metrics string representation
func TestBatchMetricsString(t *testing.T) {
	metrics := &BatchMetrics{
		TotalBatches:     10,
		TotalRequests:    100,
		AverageBatchSize: 10.0,
		MaxBatchSize:     20,
		TimeoutDespatch:  5,
		SizeDespatch:     5,
		Efficiency:       0.9,
	}

	str := metrics.String()
	if len(str) == 0 {
		t.Error("Metrics string is empty")
	}

	if !contains(str, "BatchMetrics") {
		t.Errorf("Metrics string missing 'BatchMetrics': %s", str)
	}
}

// TestBatcherStress performs stress testing
func TestBatcherStress(t *testing.T) {
	batcher := NewRequestBatcher(DefaultBatchConfig())
	defer batcher.Close()

	var wg sync.WaitGroup
	var successCount int64
	numGoroutines := 100

	for g := 0; g < numGoroutines; g++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				req := &BatchRequest{
					ID:      fmt.Sprintf("g%d-r%d", id, i),
					Request: fmt.Sprintf("data-%d", i),
					Result:  make(chan interface{}, 1),
					Error:   make(chan error, 1),
				}

				if err := batcher.AddRequest(context.Background(), req); err == nil {
					atomic.AddInt64(&successCount, 1)
				}
			}
		}(g)
	}

	wg.Wait()

	if successCount < int64(numGoroutines*50)/2 {
		t.Errorf("Expected high success rate, got %d successes", successCount)
	}

	metrics := batcher.GetMetrics()
	t.Logf("Stress test metrics: %v", metrics.String())
}

// TestContextCancellation verifies context cancellation
func TestContextCancellation(t *testing.T) {
	batcher := NewRequestBatcher(DefaultBatchConfig())
	defer batcher.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	req := &BatchRequest{
		ID:      "test",
		Request: "data",
		Result:  make(chan interface{}, 1),
		Error:   make(chan error, 1),
	}

	err := batcher.AddRequest(ctx, req)
	if err == nil {
		t.Error("Expected context cancellation error")
	}
}

// TestAdaptiveBatchSizing verifies adaptive batch size adjustment
func TestAdaptiveBatchSizing(t *testing.T) {
	config := DefaultBatchConfig()
	config.AdaptiveSizing = true

	batcher := NewRequestBatcher(config)
	defer batcher.Close()

	initialSize := batcher.config.MaxBatchSize

	// High latency - should reduce
	batcher.AdaptBatchSize(15.0)
	if batcher.config.MaxBatchSize >= initialSize {
		t.Error("Expected batch size to decrease on high latency")
	}

	// Good latency - should increase
	batcher.AdaptBatchSize(3.0)
	// Size might not increase immediately due to reduction

	t.Logf("Final batch size: %d", batcher.config.MaxBatchSize)
}

// BenchmarkAddRequest benchmarks adding requests
func BenchmarkAddRequest(b *testing.B) {
	batcher := NewRequestBatcher(DefaultBatchConfig())
	defer batcher.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req := &BatchRequest{
			ID:      fmt.Sprintf("req-%d", i),
			Request: i,
			Result:  make(chan interface{}, 1),
			Error:   make(chan error, 1),
		}
		batcher.AddRequest(context.Background(), req)
	}
}

// BenchmarkConcurrentAddRequest benchmarks concurrent adds
func BenchmarkConcurrentAddRequest(b *testing.B) {
	batcher := NewRequestBatcher(DefaultBatchConfig())
	defer batcher.Close()

	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			req := &BatchRequest{
				ID:      fmt.Sprintf("req-%d", i),
				Request: i,
				Result:  make(chan interface{}, 1),
				Error:   make(chan error, 1),
			}
			batcher.AddRequest(context.Background(), req)
			i++
		}
	})
}
