package services

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

// MockServerConfig configures the mock server behavior.
type MockServerConfig struct {
	Port              int
	Latency           time.Duration
	ErrorRate         float64 // 0.0 to 1.0
	MaxConcurrentReqs int
}

// MockRequest captures details of a received request.
type MockRequest struct {
	Path      string
	Method    string
	Timestamp time.Time
	Duration  time.Duration
	Success   bool
	Error     string
}

// MockServer simulates a remote inference server for testing.
type MockServer struct {
	config   MockServerConfig
	server   *http.Server
	mu       sync.RWMutex
	requests []MockRequest
	errors   []string
	active   bool
	reqCount int64
	errCount int64
	limiter  chan struct{} // concurrency limiter
}

// NewMockServer creates a new mock server.
func NewMockServer(config MockServerConfig) *MockServer {
	if config.Port == 0 {
		config.Port = 8888
	}
	if config.ErrorRate < 0 {
		config.ErrorRate = 0
	}
	if config.ErrorRate > 1 {
		config.ErrorRate = 1
	}
	if config.MaxConcurrentReqs == 0 {
		config.MaxConcurrentReqs = 100
	}

	ms := &MockServer{
		config:   config,
		requests: make([]MockRequest, 0),
		errors:   make([]string, 0),
		limiter:  make(chan struct{}, config.MaxConcurrentReqs),
	}

	// Initialize semaphore
	for i := 0; i < config.MaxConcurrentReqs; i++ {
		ms.limiter <- struct{}{}
	}

	return ms
}

// Start starts the mock server.
func (ms *MockServer) Start() error {
	ms.mu.Lock()
	if ms.active {
		ms.mu.Unlock()
		return fmt.Errorf("mock server already running")
	}
	ms.mu.Unlock()

	mux := http.NewServeMux()

	// Register handlers
	mux.HandleFunc("/api/models", ms.handleListModels)
	mux.HandleFunc("/api/models/load", ms.handleLoadModel)
	mux.HandleFunc("/api/models/unload", ms.handleUnloadModel)
	mux.HandleFunc("/api/infer", ms.handleInference)
	mux.HandleFunc("/api/health", ms.handleHealth)

	addr := fmt.Sprintf(":%d", ms.config.Port)
	ms.server = &http.Server{
		Addr:    addr,
		Handler: mux,
	}

	go func() {
		if err := ms.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			ms.mu.Lock()
			ms.errors = append(ms.errors, fmt.Sprintf("server error: %v", err))
			ms.mu.Unlock()
		}
	}()

	// Wait for server to be ready
	time.Sleep(100 * time.Millisecond)

	ms.mu.Lock()
	ms.active = true
	ms.mu.Unlock()

	return nil
}

// Stop stops the mock server.
func (ms *MockServer) Stop() error {
	ms.mu.Lock()
	if !ms.active {
		ms.mu.Unlock()
		return fmt.Errorf("mock server not running")
	}
	ms.active = false
	ms.mu.Unlock()

	if ms.server != nil {
		ctx, cancel := time.WithTimeout(time.Background(), 5*time.Second)
		defer cancel()
		return ms.server.Shutdown(ctx)
	}
	return nil
}

// handleListModels handles model listing requests.
func (ms *MockServer) handleListModels(w http.ResponseWriter, r *http.Request) {
	ms.recordRequest(r, time.Now())
	<-ms.limiter
	defer func() { ms.limiter <- struct{}{} }()

	start := time.Now()
	time.Sleep(ms.config.Latency)

	if ms.shouldError() {
		ms.recordError("list_models", "simulated error")
		http.Error(w, "Internal Server Error", http.StatusInternalServerError)
		return
	}

	atomic.AddInt64(&ms.reqCount, 1)

	models := []map[string]interface{}{
		{
			"id":                "bitnet-7b",
			"name":              "BitNet b1.58 7B",
			"context_window":    4096,
			"max_output_tokens": 2048,
		},
		{
			"id":                "llama-13b",
			"name":              "Llama 13B",
			"context_window":    4096,
			"max_output_tokens": 2048,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(models)

	ms.recordLatency("list_models", time.Since(start), true)
}

// handleLoadModel handles model loading requests.
func (ms *MockServer) handleLoadModel(w http.ResponseWriter, r *http.Request) {
	ms.recordRequest(r, time.Now())
	<-ms.limiter
	defer func() { ms.limiter <- struct{}{} }()

	start := time.Now()
	time.Sleep(ms.config.Latency * 10) // Model loading takes longer

	if ms.shouldError() {
		ms.recordError("load_model", "simulated error")
		http.Error(w, "Failed to load model", http.StatusInternalServerError)
		return
	}

	atomic.AddInt64(&ms.reqCount, 1)

	response := map[string]interface{}{
		"status":      "loaded",
		"loaded_at":   time.Now(),
		"memory_used": "2.5GB",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)

	ms.recordLatency("load_model", time.Since(start), true)
}

// handleUnloadModel handles model unloading requests.
func (ms *MockServer) handleUnloadModel(w http.ResponseWriter, r *http.Request) {
	ms.recordRequest(r, time.Now())
	<-ms.limiter
	defer func() { ms.limiter <- struct{}{} }()

	start := time.Now()
	time.Sleep(ms.config.Latency)

	if ms.shouldError() {
		ms.recordError("unload_model", "simulated error")
		http.Error(w, "Failed to unload model", http.StatusInternalServerError)
		return
	}

	atomic.AddInt64(&ms.reqCount, 1)

	response := map[string]interface{}{
		"status": "unloaded",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)

	ms.recordLatency("unload_model", time.Since(start), true)
}

// handleInference handles inference requests.
func (ms *MockServer) handleInference(w http.ResponseWriter, r *http.Request) {
	ms.recordRequest(r, time.Now())
	<-ms.limiter
	defer func() { ms.limiter <- struct{}{} }()

	start := time.Now()
	time.Sleep(ms.config.Latency * 2)

	if ms.shouldError() {
		ms.recordError("infer", "simulated error")
		http.Error(w, "Inference failed", http.StatusInternalServerError)
		atomic.AddInt64(&ms.errCount, 1)
		return
	}

	atomic.AddInt64(&ms.reqCount, 1)

	response := map[string]interface{}{
		"text":          "Generated response from mock server",
		"tokens":        100,
		"finish_reason": "stop",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)

	ms.recordLatency("infer", time.Since(start), true)
}

// handleHealth handles health check requests.
func (ms *MockServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status":  "healthy",
		"version": "1.0.0",
	})
}

// shouldError determines if request should fail based on error rate.
func (ms *MockServer) shouldError() bool {
	if ms.config.ErrorRate <= 0 {
		return false
	}
	// Simulate errors based on error rate
	return ms.requestCount()%int(1.0/ms.config.ErrorRate) == 0
}

// requestCount returns current request count safely.
func (ms *MockServer) requestCount() int {
	return int(atomic.LoadInt64(&ms.reqCount))
}

// recordRequest records a request.
func (ms *MockServer) recordRequest(r *http.Request, timestamp time.Time) {
	ms.mu.Lock()
	defer ms.mu.Unlock()

	ms.requests = append(ms.requests, MockRequest{
		Path:      r.URL.Path,
		Method:    r.Method,
		Timestamp: timestamp,
	})
}

// recordLatency records request latency.
func (ms *MockServer) recordLatency(operation string, duration time.Duration, success bool) {
	ms.mu.Lock()
	if len(ms.requests) > 0 {
		ms.requests[len(ms.requests)-1].Duration = duration
		ms.requests[len(ms.requests)-1].Success = success
	}
	ms.mu.Unlock()
}

// recordError records an error.
func (ms *MockServer) recordError(operation string, errMsg string) {
	ms.mu.Lock()
	defer ms.mu.Unlock()

	ms.errors = append(ms.errors, fmt.Sprintf("%s: %s", operation, errMsg))
	atomic.AddInt64(&ms.errCount, 1)
}

// GetMetrics returns server metrics.
func (ms *MockServer) GetMetrics() map[string]interface{} {
	ms.mu.RLock()
	defer ms.mu.RUnlock()

	return map[string]interface{}{
		"total_requests":  atomic.LoadInt64(&ms.reqCount),
		"total_errors":    atomic.LoadInt64(&ms.errCount),
		"requests_logged": len(ms.requests),
		"errors_logged":   len(ms.errors),
		"active":          ms.active,
		"port":            ms.config.Port,
	}
}

// GetRequests returns all recorded requests.
func (ms *MockServer) GetRequests() []MockRequest {
	ms.mu.RLock()
	defer ms.mu.RUnlock()

	requests := make([]MockRequest, len(ms.requests))
	copy(requests, ms.requests)
	return requests
}

// GetErrors returns all recorded errors.
func (ms *MockServer) GetErrors() []string {
	ms.mu.RLock()
	defer ms.mu.RUnlock()

	errors := make([]string, len(ms.errors))
	copy(errors, ms.errors)
	return errors
}

// ClearMetrics clears all recorded metrics.
func (ms *MockServer) ClearMetrics() {
	ms.mu.Lock()
	defer ms.mu.Unlock()

	ms.requests = make([]MockRequest, 0)
	ms.errors = make([]string, 0)
	atomic.StoreInt64(&ms.reqCount, 0)
	atomic.StoreInt64(&ms.errCount, 0)
}

// WaitForPort waits for server to be listening on port.
func WaitForPort(port int, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)

	for time.Now().Before(deadline) {
		conn, err := net.DialTimeout("tcp", fmt.Sprintf("localhost:%d", port), 100*time.Millisecond)
		if err == nil {
			conn.Close()
			return nil
		}
		time.Sleep(50 * time.Millisecond)
	}

	return fmt.Errorf("port %d not ready after %v", port, timeout)
}
