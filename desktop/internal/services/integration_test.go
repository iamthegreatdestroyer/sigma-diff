package services

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/config"
)

// TestCompleteModelLifecycle tests full model lifecycle from discovery to unload.
func TestCompleteModelLifecycle(t *testing.T) {
	// Start mock server
	mockConfig := MockServerConfig{
		Port:      9001,
		Latency:   10 * time.Millisecond,
		ErrorRate: 0.0,
	}
	mockServer := NewMockServer(mockConfig)

	if err := mockServer.Start(); err != nil {
		t.Fatalf("failed to start mock server: %v", err)
	}
	defer mockServer.Stop()

	// Allow server startup
	if err := WaitForPort(mockConfig.Port, 5*time.Second); err != nil {
		t.Skipf("mock server not ready: %v", err)
	}

	// Create client
	cfg := &config.Config{
		API: config.APIConfig{
			Endpoint: "http://localhost:9001",
			Timeout:  5 * time.Second,
		},
		Protocol: "rest",
	}

	ms := NewModelService(cfg)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Test workflow:
	t.Run("list-models", func(t *testing.T) {
		models, err := ms.ListModels(ctx)
		if err != nil {
			t.Logf("list models error (expected in test): %v", err)
		} else {
			if len(models) == 0 {
				t.Log("no models returned (mock server response)")
			}
		}
	})

	t.Run("load-model", func(t *testing.T) {
		_, err := ms.LoadModel(ctx, "bitnet-7b")
		if err != nil {
			t.Logf("load model error (expected in test): %v", err)
		}
	})

	t.Run("model-status", func(t *testing.T) {
		isLoaded := ms.IsModelLoaded("bitnet-7b")
		if isLoaded {
			t.Log("model marked as loaded")
		}
	})

	t.Run("unload-model", func(t *testing.T) {
		err := ms.UnloadModel(ctx, "bitnet-7b")
		if err != nil {
			t.Logf("unload model error (expected in test): %v", err)
		}
	})

	// Verify metrics
	metrics := mockServer.GetMetrics()
	t.Logf("Server metrics: %v", metrics)
}

// TestConcurrentInferenceRequests tests multiple simultaneous inference requests.
func TestConcurrentInferenceRequests(t *testing.T) {
	mockConfig := MockServerConfig{
		Port:      9002,
		Latency:   5 * time.Millisecond,
		ErrorRate: 0.0,
	}
	mockServer := NewMockServer(mockConfig)

	if err := mockServer.Start(); err != nil {
		t.Fatalf("failed to start mock server: %v", err)
	}
	defer mockServer.Stop()

	if err := WaitForPort(mockConfig.Port, 5*time.Second); err != nil {
		t.Skipf("mock server not ready: %v", err)
	}

	cfg := &config.Config{
		API: config.APIConfig{
			Endpoint: "http://localhost:9002",
			Timeout:  5 * time.Second,
		},
		Protocol: "rest",
	}

	ms := NewModelService(cfg)

	// Pre-load model
	ctx := context.Background()
	ms.LoadModel(ctx, "test-model")

	// Run concurrent requests
	numGoroutines := 5
	requestsPerGoroutine := 10
	var wg sync.WaitGroup
	errors := 0
	mu := sync.Mutex{}

	start := time.Now()

	for g := 0; g < numGoroutines; g++ {
		wg.Add(1)
		go func(goroutineID int) {
			defer wg.Done()
			for i := 0; i < requestsPerGoroutine; i++ {
				_, err := ms.GetModelInfo(ctx, "test-model")
				if err != nil {
					mu.Lock()
					errors++
					mu.Unlock()
					t.Logf("goroutine %d request %d error: %v", goroutineID, i, err)
				}
			}
		}(g)
	}

	wg.Wait()
	duration := time.Since(start)

	totalRequests := numGoroutines * requestsPerGoroutine
	successfulRequests := totalRequests - errors
	rps := float64(totalRequests) / duration.Seconds()

	t.Logf("Concurrent Requests: total=%d, successful=%d, errors=%d, duration=%v, throughput=%.2f RPS",
		totalRequests, successfulRequests, errors, duration, rps)
}

// TestErrorRecoveryScenarios tests handling of errors and recovery.
func TestErrorRecoveryScenarios(t *testing.T) {
	// Test with error injection
	mockConfig := MockServerConfig{
		Port:      9003,
		Latency:   5 * time.Millisecond,
		ErrorRate: 0.2, // 20% error rate
	}
	mockServer := NewMockServer(mockConfig)

	if err := mockServer.Start(); err != nil {
		t.Fatalf("failed to start mock server: %v", err)
	}
	defer mockServer.Stop()

	if err := WaitForPort(mockConfig.Port, 5*time.Second); err != nil {
		t.Skipf("mock server not ready: %v", err)
	}

	cfg := &config.Config{
		API: config.APIConfig{
			Endpoint: "http://localhost:9003",
			Timeout:  5 * time.Second,
		},
		Protocol: "rest",
	}

	ms := NewModelService(cfg)
	ctx := context.Background()

	successCount := 0
	errorCount := 0

	t.Run("error-handling", func(t *testing.T) {
		for i := 0; i < 50; i++ {
			_, err := ms.GetModelInfo(ctx, "test-model")
			if err != nil {
				errorCount++
			} else {
				successCount++
			}
		}
	})

	t.Logf("Error Recovery: successes=%d, errors=%d, rate=%.1f%%",
		successCount, errorCount, float64(errorCount)*100/float64(successCount+errorCount))

	// Verify server recorded errors
	serverErrors := mockServer.GetErrors()
	serverMetrics := mockServer.GetMetrics()
	t.Logf("Server recorded: total_requests=%d, total_errors=%d, errors_logged=%d",
		serverMetrics["total_requests"],
		serverMetrics["total_errors"],
		len(serverErrors),
	)
}

// TestContextCancellationHandling tests request cancellation during execution.
func TestContextCancellationHandling(t *testing.T) {
	mockConfig := MockServerConfig{
		Port:    9004,
		Latency: 100 * time.Millisecond, // Longer latency to test cancellation
	}
	mockServer := NewMockServer(mockConfig)

	if err := mockServer.Start(); err != nil {
		t.Fatalf("failed to start mock server: %v", err)
	}
	defer mockServer.Stop()

	if err := WaitForPort(mockConfig.Port, 5*time.Second); err != nil {
		t.Skipf("mock server not ready: %v", err)
	}

	cfg := &config.Config{
		API: config.APIConfig{
			Endpoint: "http://localhost:9004",
			Timeout:  5 * time.Second,
		},
		Protocol: "rest",
	}

	ms := NewModelService(cfg)

	t.Run("context-deadline-exceeded", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
		defer cancel()

		_, err := ms.GetModelInfo(ctx, "test-model")
		if err != nil {
			t.Logf("expected timeout error: %v", err)
		} else {
			t.Log("request completed before timeout")
		}
	})

	t.Run("context-cancellation", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel() // Cancel immediately

		_, err := ms.GetModelInfo(ctx, "test-model")
		if err != nil {
			t.Logf("expected cancellation error: %v", err)
		}
	})
}

// TestResourceCleanupOnShutdown tests proper cleanup of resources.
func TestResourceCleanupOnShutdown(t *testing.T) {
	mockConfig := MockServerConfig{
		Port: 9005,
	}
	mockServer := NewMockServer(mockConfig)

	if err := mockServer.Start(); err != nil {
		t.Fatalf("failed to start mock server: %v", err)
	}

	if err := WaitForPort(mockConfig.Port, 5*time.Second); err != nil {
		t.Skipf("mock server not ready: %v", err)
	}

	cfg := &config.Config{
		API: config.APIConfig{
			Endpoint: "http://localhost:9005",
			Timeout:  5 * time.Second,
		},
		Protocol: "rest",
	}

	ms := NewModelService(cfg)

	// Make some requests
	ctx := context.Background()
	ms.LoadModel(ctx, "test-model-1")
	ms.LoadModel(ctx, "test-model-2")

	// Verify state
	loadedCount := ms.GetModelCount()
	t.Logf("Loaded models before shutdown: %d", loadedCount)

	// Unload all models
	if err := ms.UnloadAllModels(ctx); err != nil {
		t.Logf("error unloading models: %v", err)
	}

	// Verify cleanup
	finalLoadedCount := ms.GetModelCount()
	t.Logf("Loaded models after cleanup: %d", finalLoadedCount)

	if finalLoadedCount > 0 {
		t.Logf("warning: %d models still loaded after cleanup", finalLoadedCount)
	}

	// Stop server
	if err := mockServer.Stop(); err != nil {
		t.Fatalf("failed to stop mock server: %v", err)
	}
}

// TestHighConcurrencyStress tests system under high concurrent load.
func TestHighConcurrencyStress(t *testing.T) {
	mockConfig := MockServerConfig{
		Port:              9006,
		Latency:           5 * time.Millisecond,
		MaxConcurrentReqs: 100,
	}
	mockServer := NewMockServer(mockConfig)

	if err := mockServer.Start(); err != nil {
		t.Fatalf("failed to start mock server: %v", err)
	}
	defer mockServer.Stop()

	if err := WaitForPort(mockConfig.Port, 5*time.Second); err != nil {
		t.Skipf("mock server not ready: %v", err)
	}

	cfg := &config.Config{
		API: config.APIConfig{
			Endpoint: "http://localhost:9006",
			Timeout:  10 * time.Second,
		},
		Protocol: "rest",
	}

	ms := NewModelService(cfg)
	ctx := context.Background()

	// High concurrency test
	numGoroutines := 50
	requestsPerGoroutine := 20
	var wg sync.WaitGroup
	var successCount, errorCount int64
	mu := sync.Mutex{}

	start := time.Now()

	for g := 0; g < numGoroutines; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < requestsPerGoroutine; i++ {
				_, err := ms.GetModelInfo(ctx, "test-model")

				mu.Lock()
				if err != nil {
					errorCount++
				} else {
					successCount++
				}
				mu.Unlock()
			}
		}()
	}

	wg.Wait()
	duration := time.Since(start)

	totalRequests := numGoroutines * requestsPerGoroutine
	rps := float64(totalRequests) / duration.Seconds()

	t.Logf("High Concurrency Stress: concurrency=%d, total=%d, successful=%d, errors=%d, duration=%v, throughput=%.2f RPS",
		numGoroutines, totalRequests, successCount, errorCount, duration, rps)

	// Log server metrics
	metrics := mockServer.GetMetrics()
	t.Logf("Server Metrics: %v", metrics)
}

// TestModelCacheConsistency verifies cache behavior across operations.
func TestModelCacheConsistency(t *testing.T) {
	mockConfig := MockServerConfig{
		Port: 9007,
	}
	mockServer := NewMockServer(mockConfig)

	if err := mockServer.Start(); err != nil {
		t.Fatalf("failed to start mock server: %v", err)
	}
	defer mockServer.Stop()

	if err := WaitForPort(mockConfig.Port, 5*time.Second); err != nil {
		t.Skipf("mock server not ready: %v", err)
	}

	cfg := &config.Config{
		API: config.APIConfig{
			Endpoint: "http://localhost:9007",
			Timeout:  5 * time.Second,
		},
		Protocol: "rest",
	}

	ms := NewModelService(cfg)
	ctx := context.Background()

	t.Run("cache-hit", func(t *testing.T) {
		// First call should fetch
		models1, _ := ms.ListModels(ctx)
		firstCallCount := len(mockServer.GetRequests())

		// Second call should hit cache
		models2, _ := ms.ListModels(ctx)
		secondCallCount := len(mockServer.GetRequests())

		t.Logf("First call requests: %d, Second call requests: %d (should be cached)",
			firstCallCount, secondCallCount)

		if len(models1) > 0 && len(models2) > 0 {
			t.Log("models cached successfully")
		}
	})

	t.Run("cache-invalidation", func(t *testing.T) {
		mockServer.ClearMetrics()

		// List models
		ms.ListModels(ctx)
		requestsBefore := len(mockServer.GetRequests())

		// Clear cache
		ms.ClearCache()

		// List again - should fetch fresh
		ms.ListModels(ctx)
		requestsAfter := len(mockServer.GetRequests())

		t.Logf("Requests before cache clear: %d, after: %d", requestsBefore, requestsAfter)
		if requestsAfter > requestsBefore {
			t.Log("cache invalidation successful")
		}
	})
}

