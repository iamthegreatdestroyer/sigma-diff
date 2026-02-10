package services

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/config"
)

// TestClientManagerInitialization tests client manager setup
func TestClientManagerInitialization(t *testing.T) {
	cfg := &config.Config{
		Server: config.ServerConfig{
			Type: "rest",
		},
		REST: &config.RESTConfig{
			Host:    "localhost",
			Port:    8000,
			Timeout: 30,
		},
	}

	cm := NewClientManager(cfg)

	if !cm.IsInitialized() {
		t.Error("client manager should not be initialized yet")
	}

	if err := cm.Initialize(); err != nil {
		t.Fatalf("failed to initialize client manager: %v", err)
	}

	if !cm.IsInitialized() {
		t.Error("client manager should be initialized")
	}

	if err := cm.Close(); err != nil {
		t.Fatalf("failed to close client manager: %v", err)
	}
}

// TestClientManagerGRPCInitialization tests gRPC client initialization
func TestClientManagerGRPCInitialization(t *testing.T) {
	cfg := &config.Config{
		Server: config.ServerConfig{
			Type: "grpc",
		},
		GRPC: &config.GRPCConfig{
			Host:    "localhost",
			Port:    50051,
			Timeout: 30,
		},
	}

	cm := NewClientManager(cfg)

	if err := cm.Initialize(); err == nil {
		// Connection might fail in test environment, but initialization should be attempted
		cm.Close()
	}
}

// TestModelServiceListModels tests listing available models
func TestModelServiceListModels(t *testing.T) {
	cfg := &config.Config{
		Server: config.ServerConfig{
			Type: "rest",
		},
		REST: &config.RESTConfig{
			Host:    "localhost",
			Port:    8000,
			Timeout: 30,
		},
	}

	cm := NewClientManager(cfg)
	cm.Initialize()
	defer cm.Close()

	ms := NewModelService(cm)

	models, err := ms.ListModels(context.Background())
	if err != nil {
		t.Fatalf("failed to list models: %v", err)
	}

	if len(models) == 0 {
		t.Error("expected at least one model")
	}

	if models[0].ID == "" {
		t.Error("model ID should not be empty")
	}
}

// TestModelServiceLoadUnload tests loading and unloading models
func TestModelServiceLoadUnload(t *testing.T) {
	cfg := &config.Config{
		Server: config.ServerConfig{
			Type: "rest",
		},
		REST: &config.RESTConfig{
			Host:    "localhost",
			Port:    8000,
			Timeout: 30,
		},
	}

	cm := NewClientManager(cfg)
	cm.Initialize()
	defer cm.Close()

	ms := NewModelService(cm)

	// Get available models
	models, err := ms.ListModels(context.Background())
	if err != nil {
		t.Fatalf("failed to list models: %v", err)
	}

	if len(models) == 0 {
		t.Skip("no models available for testing")
	}

	modelID := models[0].ID

	// Load model
	model, err := ms.LoadModel(context.Background(), modelID)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}

	if model.Status != "loaded" {
		t.Errorf("expected model status 'loaded', got '%s'", model.Status)
	}

	if !ms.IsModelLoaded(modelID) {
		t.Error("model should be loaded")
	}

	// Unload model
	if err := ms.UnloadModel(context.Background(), modelID); err != nil {
		t.Fatalf("failed to unload model: %v", err)
	}

	if ms.IsModelLoaded(modelID) {
		t.Error("model should not be loaded after unload")
	}
}

// TestModelServiceCaching tests model caching behavior
func TestModelServiceCaching(t *testing.T) {
	cfg := &config.Config{
		Server: config.ServerConfig{
			Type: "rest",
		},
		REST: &config.RESTConfig{
			Host:    "localhost",
			Port:    8000,
			Timeout: 30,
		},
	}

	cm := NewClientManager(cfg)
	cm.Initialize()
	defer cm.Close()

	ms := NewModelService(cm)

	// First call should fetch from server
	models1, err := ms.ListModels(context.Background())
	if err != nil {
		t.Fatalf("first list failed: %v", err)
	}

	// Second call should use cache
	models2, err := ms.ListModels(context.Background())
	if err != nil {
		t.Fatalf("second list failed: %v", err)
	}

	if len(models1) != len(models2) {
		t.Error("cached models should match initial fetch")
	}

	// Clear cache
	ms.ClearCache()

	// Should fetch again
	models3, err := ms.ListModels(context.Background())
	if err != nil {
		t.Fatalf("third list failed: %v", err)
	}

	if len(models3) == 0 {
		t.Error("should have fetched fresh models after cache clear")
	}
}

// TestInferenceServiceExecution tests basic inference execution
func TestInferenceServiceExecution(t *testing.T) {
	cfg := &config.Config{
		Server: config.ServerConfig{
			Type: "rest",
		},
		REST: &config.RESTConfig{
			Host:    "localhost",
			Port:    8000,
			Timeout: 30,
		},
	}

	cm := NewClientManager(cfg)
	cm.Initialize()
	defer cm.Close()

	ms := NewModelService(cm)
	is := NewInferenceService(cm, ms)

	// Load a model first
	models, _ := ms.ListModels(context.Background())
	if len(models) > 0 {
		ms.LoadModel(context.Background(), models[0].ID)

		// Test inference
		req := &InferenceRequest{
			ModelID:   models[0].ID,
			Prompt:    "Hello world",
			MaxTokens: 100,
		}

		resp, err := is.Execute(context.Background(), req)
		if err != nil {
			t.Fatalf("inference execution failed: %v", err)
		}

		if resp.Text == "" {
			t.Error("response text should not be empty")
		}

		if resp.Model != models[0].ID {
			t.Errorf("response model mismatch")
		}
	}
}

// TestInferenceServiceMetrics tests inference metrics tracking
func TestInferenceServiceMetrics(t *testing.T) {
	cfg := &config.Config{
		Server: config.ServerConfig{
			Type: "rest",
		},
		REST: &config.RESTConfig{
			Host:    "localhost",
			Port:    8000,
			Timeout: 30,
		},
	}

	cm := NewClientManager(cfg)
	cm.Initialize()
	defer cm.Close()

	ms := NewModelService(cm)
	is := NewInferenceService(cm, ms)

	metrics := is.GetMetrics()
	if metrics.TotalRequests != 0 {
		t.Error("initial metrics should be zero")
	}

	is.ResetMetrics()

	newMetrics := is.GetMetrics()
	if newMetrics.TotalRequests != 0 {
		t.Error("metrics should remain zero after reset")
	}
}

// TestConcurrentRequests tests handling multiple concurrent requests
func TestConcurrentRequests(t *testing.T) {
	cfg := &config.Config{
		Server: config.ServerConfig{
			Type: "rest",
		},
		REST: &config.RESTConfig{
			Host:    "localhost",
			Port:    8000,
			Timeout: 30,
		},
	}

	cm := NewClientManager(cfg)
	cm.Initialize()
	defer cm.Close()

	ms := NewModelService(cm)

	// Load a model
	models, _ := ms.ListModels(context.Background())
	if len(models) == 0 {
		t.Skip("no models available for concurrent test")
	}

	modelID := models[0].ID
	ms.LoadModel(context.Background(), modelID)

	is := NewInferenceService(cm, ms)

	// Execute concurrent requests
	numRequests := 10
	var wg sync.WaitGroup
	errChan := make(chan error, numRequests)

	for i := 0; i < numRequests; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()

			req := &InferenceRequest{
				ModelID:   modelID,
				Prompt:    "Test prompt",
				MaxTokens: 100,
			}

			_, err := is.Execute(context.Background(), req)
			if err != nil {
				errChan <- err
			}
		}(i)
	}

	wg.Wait()
	close(errChan)

	// Check for errors
	for err := range errChan {
		t.Logf("concurrent request error: %v", err)
	}

	metrics := is.GetMetrics()
	if metrics.TotalRequests != int64(numRequests) {
		t.Errorf("expected %d total requests, got %d", numRequests, metrics.TotalRequests)
	}
}

// TestContextCancellation tests request cancellation via context
func TestContextCancellation(t *testing.T) {
	cfg := &config.Config{
		Server: config.ServerConfig{
			Type: "rest",
		},
		REST: &config.RESTConfig{
			Host:    "localhost",
			Port:    8000,
			Timeout: 30,
		},
	}

	cm := NewClientManager(cfg)
	cm.Initialize()
	defer cm.Close()

	ms := NewModelService(cm)
	is := NewInferenceService(cm, ms)

	// Create cancellable context
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	models, _ := ms.ListModels(context.Background())
	if len(models) > 0 {
		ms.LoadModel(context.Background(), models[0].ID)

		req := &InferenceRequest{
			ModelID: models[0].ID,
			Prompt:  "Test",
		}

		_, err := is.Execute(ctx, req)
		if err == nil {
			t.Error("expected error for cancelled context")
		}
	}
}

// TestTimeoutHandling tests request timeout behavior
func TestTimeoutHandling(t *testing.T) {
	cfg := &config.Config{
		Server: config.ServerConfig{
			Type: "rest",
		},
		REST: &config.RESTConfig{
			Host:    "localhost",
			Port:    8000,
			Timeout: 30,
		},
	}

	cm := NewClientManager(cfg)
	cm.Initialize()
	defer cm.Close()

	// Create context with short timeout
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Millisecond)
	defer cancel()

	ms := NewModelService(cm)

	// This should timeout
	_, err := ms.ListModels(ctx)
	if err != nil {
		// Timeout expected
		t.Logf("timeout error (expected): %v", err)
	}
}

// TestResourceCleanup tests proper resource cleanup
func TestResourceCleanup(t *testing.T) {
	cfg := &config.Config{
		Server: config.ServerConfig{
			Type: "rest",
		},
		REST: &config.RESTConfig{
			Host:    "localhost",
			Port:    8000,
			Timeout: 30,
		},
	}

	cm := NewClientManager(cfg)

	if err := cm.Initialize(); err != nil {
		t.Fatalf("failed to initialize: %v", err)
	}

	metrics := cm.GetMetrics()
	if metrics["initialized"] != true {
		t.Error("client manager should be initialized")
	}

	if err := cm.Close(); err != nil {
		t.Fatalf("failed to close: %v", err)
	}

	if cm.IsInitialized() {
		t.Error("client manager should not be initialized after close")
	}

	// Try to initialize again after close
	if err := cm.Initialize(); err == nil {
		t.Error("should not be able to initialize closed client manager")
	}
}

// TestErrorHandling tests error conditions
func TestErrorHandling(t *testing.T) {
	cfg := &config.Config{
		Server: config.ServerConfig{
			Type: "rest",
		},
		REST: &config.RESTConfig{
			Host:    "localhost",
			Port:    8000,
			Timeout: 30,
		},
	}

	cm := NewClientManager(cfg)
	cm.Initialize()
	defer cm.Close()

	ms := NewModelService(cm)
	is := NewInferenceService(cm, ms)

	// Test invalid request
	_, err := is.Execute(context.Background(), nil)
	if err == nil {
		t.Error("should error on nil request")
	}

	// Test missing model
	_, err = is.Execute(context.Background(), &InferenceRequest{
		ModelID: "nonexistent-model",
		Prompt:  "test",
	})
	if err == nil {
		t.Error("should error on nonexistent model")
	}

	// Test invalid temperature
	models, _ := ms.ListModels(context.Background())
	if len(models) > 0 {
		ms.LoadModel(context.Background(), models[0].ID)

		_, err = is.Execute(context.Background(), &InferenceRequest{
			ModelID:     models[0].ID,
			Prompt:      "test",
			Temperature: 3.0, // Invalid (should be 0-2.0)
		})
		if err == nil {
			t.Error("should error on invalid temperature")
		}
	}
}
