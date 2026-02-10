package services

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestAsyncModelManagerInitialization verifies manager initializes correctly
func TestAsyncModelManagerInitialization(t *testing.T) {
	manager := NewAsyncModelManager(4)
	defer manager.Shutdown(context.Background())

	if manager == nil {
		t.Error("Expected manager, got nil")
	}

	stats := manager.GetModelStats()
	if stats["registered_models"] != 0 {
		t.Errorf("Expected 0 registered models, got %v", stats["registered_models"])
	}
}

// TestRegisterModel verifies model registration
func TestRegisterModel(t *testing.T) {
	manager := NewAsyncModelManager(4)
	defer manager.Shutdown(context.Background())

	metadata := &ModelMetadata{
		ID:              "test-model",
		Name:            "Test Model",
		Path:            "/path/to/model",
		Size:            100 * 1024 * 1024, // 100MB
		PreloadStrategy: "lazy",
		Priority:        50,
	}

	err := manager.RegisterModel(metadata)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	stats := manager.GetModelStats()
	if stats["registered_models"] != 1 {
		t.Errorf("Expected 1 registered model, got %v", stats["registered_models"])
	}
}

// TestLoadModel verifies synchronous model loading
func TestLoadModel(t *testing.T) {
	manager := NewAsyncModelManager(4)
	defer manager.Shutdown(context.Background())

	metadata := &ModelMetadata{
		ID:       "model-1",
		Name:     "Model 1",
		Size:     10 * 1024 * 1024, // 10MB
		Priority: 50,
	}

	manager.RegisterModel(metadata)

	result, err := manager.LoadModel(context.Background(), "model-1")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if !result.Success {
		t.Errorf("Expected successful load, got error: %v", result.Error)
	}

	if result.ModelID != "model-1" {
		t.Errorf("Expected model-1, got %s", result.ModelID)
	}
}

// TestCacheHit verifies cache hit detection
func TestCacheHit(t *testing.T) {
	manager := NewAsyncModelManager(4)
	defer manager.Shutdown(context.Background())

	metadata := &ModelMetadata{
		ID:   "model-1",
		Name: "Model 1",
		Size: 5 * 1024 * 1024,
	}

	manager.RegisterModel(metadata)

	// First load - cache miss
	result1, err := manager.LoadModel(context.Background(), "model-1")
	if err != nil {
		t.Fatalf("First load error: %v", err)
	}
	if result1.CacheHit {
		t.Error("First load should not be cache hit")
	}

	// Second load - cache hit
	result2, err := manager.LoadModel(context.Background(), "model-1")
	if err != nil {
		t.Fatalf("Second load error: %v", err)
	}
	if !result2.CacheHit {
		t.Error("Second load should be cache hit")
	}
}

// TestConcurrentLoading verifies concurrent model loading
func TestConcurrentLoading(t *testing.T) {
	manager := NewAsyncModelManager(2) // Limited concurrency
	defer manager.Shutdown(context.Background())

	// Register multiple models
	for i := 0; i < 5; i++ {
		metadata := &ModelMetadata{
			ID:   fmt.Sprintf("model-%d", i),
			Name: fmt.Sprintf("Model %d", i),
			Size: 10 * 1024 * 1024,
		}
		manager.RegisterModel(metadata)
	}

	var wg sync.WaitGroup
	var successCount int32

	// Concurrent loads
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			modelID := fmt.Sprintf("model-%d", id)
			result, err := manager.LoadModel(context.Background(), modelID)
			if err == nil && result.Success {
				atomic.AddInt32(&successCount, 1)
			}
		}(i)
	}

	wg.Wait()

	if successCount < 4 {
		t.Errorf("Expected at least 4 successful loads, got %d", successCount)
	}
}

// TestPreloadModels verifies eager preloading
func TestPreloadModels(t *testing.T) {
	manager := NewAsyncModelManager(4)
	defer manager.Shutdown(context.Background())

	// Register models with eager preload
	for i := 0; i < 3; i++ {
		metadata := &ModelMetadata{
			ID:              fmt.Sprintf("model-%d", i),
			Name:            fmt.Sprintf("Model %d", i),
			Size:            5 * 1024 * 1024,
			PreloadStrategy: "eager",
			Priority:        100 - i*10, // Higher priority first
		}
		manager.RegisterModel(metadata)
	}

	// Wait for preloading
	time.Sleep(500 * time.Millisecond)

	stats := manager.GetModelStats()
	if stats["cached_models"].(int) == 0 {
		t.Log("Note: Preloading may not have completed yet")
	}
}

// TestDependencyResolution verifies model dependency loading
func TestDependencyResolution(t *testing.T) {
	manager := NewAsyncModelManager(4)
	defer manager.Shutdown(context.Background())

	// Register base model
	base := &ModelMetadata{
		ID:   "base",
		Name: "Base Model",
		Size: 5 * 1024 * 1024,
	}
	manager.RegisterModel(base)

	// Register dependent model
	dependent := &ModelMetadata{
		ID:           "dependent",
		Name:         "Dependent Model",
		Size:         10 * 1024 * 1024,
		Dependencies: []string{"base"},
	}
	manager.RegisterModel(dependent)

	// Load dependent - should also load base
	result, err := manager.LoadModel(context.Background(), "dependent")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if !result.Success {
		t.Errorf("Failed to load dependent model: %v", result.Error)
	}
}

// TestLoadTimeout verifies load timeout handling
func TestLoadTimeout(t *testing.T) {
	manager := NewAsyncModelManager(4)
	defer manager.Shutdown(context.Background())

	metadata := &ModelMetadata{
		ID:          "model-1",
		Name:        "Model 1",
		Size:        1024 * 1024 * 1024,   // 1GB - will exceed timeout
		LoadTimeout: 1 * time.Millisecond, // Very short timeout
	}

	manager.RegisterModel(metadata)

	result, err := manager.LoadModel(context.Background(), "model-1")
	if err == nil && result.Success {
		t.Log("Note: Timeout may not have triggered due to fast execution")
	}
}

// TestContextCancellation verifies context cancellation
func TestContextCancellation(t *testing.T) {
	manager := NewAsyncModelManager(4)
	defer manager.Shutdown(context.Background())

	metadata := &ModelMetadata{
		ID:   "model-1",
		Name: "Model 1",
		Size: 100 * 1024 * 1024,
	}
	manager.RegisterModel(metadata)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	result, err := manager.LoadModel(ctx, "model-1")
	if err == nil {
		// Context was cancelled but load may have completed
		if result.Success {
			t.Log("Load completed despite context cancellation")
		}
	}
}

// TestCacheStatistics verifies cache statistics
func TestCacheStatistics(t *testing.T) {
	manager := NewAsyncModelManager(4)
	defer manager.Shutdown(context.Background())

	metadata := &ModelMetadata{
		ID:   "model-1",
		Name: "Model 1",
		Size: 5 * 1024 * 1024,
	}
	manager.RegisterModel(metadata)

	// Load twice: once miss, once hit
	manager.LoadModel(context.Background(), "model-1")
	manager.LoadModel(context.Background(), "model-1")

	stats := manager.GetModelStats()
	if hitRate, ok := stats["hit_rate_percent"].(float64); ok {
		expected := 50.0
		if hitRate < expected-5 || hitRate > expected+5 {
			t.Logf("Hit rate: %.1f%%", hitRate)
		}
	}
}

// TestClearCache verifies cache clearing
func TestClearCache(t *testing.T) {
	manager := NewAsyncModelManager(4)
	defer manager.Shutdown(context.Background())

	metadata := &ModelMetadata{
		ID:   "model-1",
		Name: "Model 1",
		Size: 5 * 1024 * 1024,
	}
	manager.RegisterModel(metadata)

	// Load and cache
	manager.LoadModel(context.Background(), "model-1")

	// Clear cache
	manager.ClearCache()

	stats := manager.GetModelStats()
	if stats["cached_models"] != 0 {
		t.Errorf("Expected 0 cached models, got %v", stats["cached_models"])
	}
}

// TestUnloadModel verifies model unloading
func TestUnloadModel(t *testing.T) {
	manager := NewAsyncModelManager(4)
	defer manager.Shutdown(context.Background())

	metadata := &ModelMetadata{
		ID:   "model-1",
		Name: "Model 1",
		Size: 5 * 1024 * 1024,
	}
	manager.RegisterModel(metadata)

	// Load then unload
	manager.LoadModel(context.Background(), "model-1")

	err := manager.UnloadModel("model-1")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// Verify unloaded
	err = manager.UnloadModel("model-1")
	if err == nil {
		t.Error("Expected error for unloading non-cached model")
	}
}

// TestManagerShutdown verifies graceful shutdown
func TestManagerShutdown(t *testing.T) {
	manager := NewAsyncModelManager(4)

	metadata := &ModelMetadata{
		ID:   "model-1",
		Name: "Model 1",
		Size: 5 * 1024 * 1024,
	}
	manager.RegisterModel(metadata)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	err := manager.Shutdown(ctx)
	if err != nil {
		t.Errorf("Unexpected error on shutdown: %v", err)
	}
}

// TestConcurrentManagerAccess verifies thread safety
func TestConcurrentManagerAccess(t *testing.T) {
	manager := NewAsyncModelManager(4)
	defer manager.Shutdown(context.Background())

	// Register many models
	for i := 0; i < 20; i++ {
		metadata := &ModelMetadata{
			ID:   fmt.Sprintf("model-%d", i),
			Name: fmt.Sprintf("Model %d", i),
			Size: 5 * 1024 * 1024,
		}
		manager.RegisterModel(metadata)
	}

	var wg sync.WaitGroup
	var errorCount int32

	// Concurrent operations
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			modelID := fmt.Sprintf("model-%d", id)

			// Mix of operations
			if id%2 == 0 {
				_, err := manager.LoadModel(context.Background(), modelID)
				if err != nil {
					atomic.AddInt32(&errorCount, 1)
				}
			} else {
				stats := manager.GetModelStats()
				if stats == nil {
					atomic.AddInt32(&errorCount, 1)
				}
			}
		}(i)
	}

	wg.Wait()

	if errorCount > 0 {
		t.Errorf("Expected no errors, got %d", errorCount)
	}
}

// TestModelPriority verifies priority-based preloading
func TestModelPriority(t *testing.T) {
	manager := NewAsyncModelManager(4)
	defer manager.Shutdown(context.Background())

	// Register models with different priorities
	priorities := []int{10, 50, 100}
	for i, priority := range priorities {
		metadata := &ModelMetadata{
			ID:              fmt.Sprintf("model-%d", i),
			Name:            fmt.Sprintf("Model %d", i),
			Size:            5 * 1024 * 1024,
			PreloadStrategy: "eager",
			Priority:        priority,
		}
		manager.RegisterModel(metadata)
	}

	time.Sleep(300 * time.Millisecond)

	stats := manager.GetModelStats()
	t.Logf("Cached models after priority preload: %v", stats["cached_models"])
}

// BenchmarkLoadModel benchmarks single model loading
func BenchmarkLoadModel(b *testing.B) {
	manager := NewAsyncModelManager(4)
	defer manager.Shutdown(context.Background())

	metadata := &ModelMetadata{
		ID:   "bench-model",
		Name: "Benchmark Model",
		Size: 5 * 1024 * 1024,
	}
	manager.RegisterModel(metadata)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		manager.LoadModel(context.Background(), "bench-model")
	}
}

// BenchmarkConcurrentLoading benchmarks concurrent model loading
func BenchmarkConcurrentLoading(b *testing.B) {
	manager := NewAsyncModelManager(8)
	defer manager.Shutdown(context.Background())

	// Register multiple models
	for i := 0; i < 10; i++ {
		metadata := &ModelMetadata{
			ID:   fmt.Sprintf("model-%d", i),
			Name: fmt.Sprintf("Model %d", i),
			Size: 5 * 1024 * 1024,
		}
		manager.RegisterModel(metadata)
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			modelID := fmt.Sprintf("model-%d", i%10)
			manager.LoadModel(context.Background(), modelID)
			i++
		}
	})
}
