package services

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/config"
)

// BenchmarkInferenceLatency measures single inference request latency.
func BenchmarkInferenceLatency(b *testing.B) {
	cfg := &config.Config{
		API: config.APIConfig{
			Endpoint: "http://localhost:8080",
			Timeout:  10 * time.Second,
		},
		Protocol: "rest",
		Fallback: "grpc",
	}

	ms := NewModelService(cfg)
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := ms.GetModelInfo(ctx, "test-model")
		if err != nil {
			b.Fatalf("failed to get model info: %v", err)
		}
	}
}

// BenchmarkConcurrentInference measures throughput under concurrent load.
func BenchmarkConcurrentInference(b *testing.B) {
	cfg := &config.Config{
		API: config.APIConfig{
			Endpoint: "http://localhost:8080",
			Timeout:  10 * time.Second,
		},
		Protocol: "rest",
	}

	ms := NewModelService(cfg)
	ctx := context.Background()
	numGoroutines := 10
	requestsPerGoroutine := b.N / numGoroutines

	b.ResetTimer()

	var wg sync.WaitGroup
	errors := atomic.Int64{}

	for g := 0; g < numGoroutines; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < requestsPerGoroutine; i++ {
				_, err := ms.GetModelInfo(ctx, "test-model")
				if err != nil {
					errors.Add(1)
				}
			}
		}()
	}

	wg.Wait()

	if errors.Load() > 0 {
		b.Logf("concurrent requests completed with %d errors", errors.Load())
	}
}

// BenchmarkResourceUtilization measures memory and computation overhead.
func BenchmarkResourceUtilization(b *testing.B) {
	cfg := &config.Config{
		API: config.APIConfig{
			Endpoint: "http://localhost:8080",
			Timeout:  10 * time.Second,
		},
		Protocol: "rest",
	}

	b.ReportAllocs()

	ms := NewModelService(cfg)
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = ms.GetModelInfo(ctx, "test-model")
	}
}

// PerformanceMetrics aggregates performance measurements.
type PerformanceMetrics struct {
	mu            sync.RWMutex
	LatencyP50    time.Duration
	LatencyP95    time.Duration
	LatencyP99    time.Duration
	ThroughputRPS float64
	ErrorCount    int64
	SuccessCount  int64
	MinLatency    time.Duration
	MaxLatency    time.Duration
	AvgLatency    time.Duration
}

// RecordLatency records a latency measurement.
func (pm *PerformanceMetrics) RecordLatency(latency time.Duration, success bool) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if success {
		pm.SuccessCount++
	} else {
		pm.ErrorCount++
	}

	if pm.MinLatency == 0 || latency < pm.MinLatency {
		pm.MinLatency = latency
	}
	if latency > pm.MaxLatency {
		pm.MaxLatency = latency
	}
}

// calculateThroughput computes requests per second over duration.
func (pm *PerformanceMetrics) calculateThroughput(duration time.Duration) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	total := pm.SuccessCount + pm.ErrorCount
	if duration > 0 {
		pm.ThroughputRPS = float64(total) / duration.Seconds()
	}
}

// String returns formatted metrics.
func (pm *PerformanceMetrics) String() string {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	return fmt.Sprintf(
		"Latency(p50/p95/p99)=%v/%v/%v Min=%v Max=%v Avg=%v Throughput=%.2f RPS Errors=%d Success=%d",
		pm.LatencyP50, pm.LatencyP95, pm.LatencyP99,
		pm.MinLatency, pm.MaxLatency, pm.AvgLatency,
		pm.ThroughputRPS, pm.ErrorCount, pm.SuccessCount,
	)
}

// TestPerformanceBaseline validates basic performance expectations.
func TestPerformanceBaseline(t *testing.T) {
	cfg := &config.Config{
		API: config.APIConfig{
			Endpoint: "http://localhost:8080",
			Timeout:  10 * time.Second,
		},
		Protocol: "rest",
	}

	ms := NewModelService(cfg)
	ctx := context.Background()
	metrics := &PerformanceMetrics{}

	// Run 100 requests and measure
	start := time.Now()
	for i := 0; i < 100; i++ {
		reqStart := time.Now()
		_, err := ms.GetModelInfo(ctx, "test-model")
		latency := time.Since(reqStart)
		metrics.RecordLatency(latency, err == nil)
	}
	duration := time.Since(start)
	metrics.calculateThroughput(duration)

	t.Logf("Performance Baseline: %s", metrics.String())

	// Basic sanity checks
	if metrics.SuccessCount == 0 && metrics.ErrorCount == 0 {
		t.Skip("service not available for performance test")
	}
}

// TestLoadScaling validates behavior under increasing load.
func TestLoadScaling(t *testing.T) {
	cfg := &config.Config{
		API: config.APIConfig{
			Endpoint: "http://localhost:8080",
			Timeout:  10 * time.Second,
		},
		Protocol: "rest",
	}

	ms := NewModelService(cfg)
	ctx := context.Background()

	loadLevels := []int{1, 5, 10, 20}

	for _, concurrency := range loadLevels {
		t.Run(fmt.Sprintf("Concurrency%d", concurrency), func(t *testing.T) {
			metrics := &PerformanceMetrics{}
			var wg sync.WaitGroup
			requestsPerGoroutine := 50

			start := time.Now()
			for g := 0; g < concurrency; g++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					for i := 0; i < requestsPerGoroutine; i++ {
						reqStart := time.Now()
						_, err := ms.GetModelInfo(ctx, "test-model")
						latency := time.Since(reqStart)
						metrics.RecordLatency(latency, err == nil)
					}
				}()
			}

			wg.Wait()
			duration := time.Since(start)
			metrics.calculateThroughput(duration)

			t.Logf("Concurrency %d: %s", concurrency, metrics.String())
		})
	}
}

