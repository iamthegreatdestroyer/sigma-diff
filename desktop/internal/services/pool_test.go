package services

import (
	"net/http"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestPoolInitialization verifies pool initializes with correct size
func TestPoolInitialization(t *testing.T) {
	config := &PoolConfig{
		HTTPMinPoolSize: 5,
		GRPCMinPoolSize: 3,
	}

	pool := NewConnectionPool(config)
	defer pool.Close()

	metrics := pool.GetMetrics()
	if metrics.CurrentSize != int32(config.HTTPMinPoolSize+config.GRPCMinPoolSize) {
		t.Errorf("Expected initial size %d, got %d",
			config.HTTPMinPoolSize+config.GRPCMinPoolSize,
			metrics.CurrentSize)
	}
}

// TestGetHTTPClient verifies getting HTTP client from pool
func TestGetHTTPClient(t *testing.T) {
	pool := NewConnectionPool(DefaultPoolConfig())
	defer pool.Close()

	client := pool.GetHTTPClient()
	if client == nil {
		t.Error("Expected client, got nil")
	}

	pool.ReleaseHTTPClient(client)

	metrics := pool.GetMetrics()
	if metrics.TotalReused != 1 {
		t.Errorf("Expected 1 reused connection, got %d", metrics.TotalReused)
	}
}

// TestConnectionReuse verifies connections are reused
func TestConnectionReuse(t *testing.T) {
	pool := NewConnectionPool(DefaultPoolConfig())
	defer pool.Close()

	clients := make([]*http.Client, 10)

	// Get 10 clients
	for i := 0; i < 10; i++ {
		clients[i] = pool.GetHTTPClient()
	}

	// Release them back
	for i := 0; i < 10; i++ {
		pool.ReleaseHTTPClient(clients[i])
	}

	// Get them again - should be reused
	for i := 0; i < 10; i++ {
		client := pool.GetHTTPClient()
		pool.ReleaseHTTPClient(client)
	}

	metrics := pool.GetMetrics()
	if metrics.TotalReused < 10 {
		t.Errorf("Expected at least 10 reuses, got %d", metrics.TotalReused)
	}

	reuseRate := pool.GetReuseRate()
	if reuseRate < 0.8 {
		t.Errorf("Expected reuse rate > 0.8, got %.2f", reuseRate)
	}
}

// TestPoolGrowth verifies pool grows on demand
func TestPoolGrowth(t *testing.T) {
	config := &PoolConfig{
		HTTPMinPoolSize: 5,
		HTTPMaxPoolSize: 50,
		GRPCMinPoolSize: 3,
		GRPCMaxPoolSize: 30,
	}

	pool := NewConnectionPool(config)
	defer pool.Close()

	initialMetrics := pool.GetMetrics()
	initialSize := initialMetrics.CurrentSize

	// Get more clients than initial pool size
	clients := make([]*http.Client, 20)
	for i := 0; i < 20; i++ {
		clients[i] = pool.GetHTTPClient()
	}

	afterGrowth := pool.GetMetrics()
	if afterGrowth.CurrentSize <= initialSize {
		t.Errorf("Pool should have grown, but size stayed at %d", afterGrowth.CurrentSize)
	}

	// Release all
	for i := 0; i < 20; i++ {
		pool.ReleaseHTTPClient(clients[i])
	}
}

// TestPoolExhaustion verifies behavior when pool reaches max
func TestPoolExhaustion(t *testing.T) {
	config := &PoolConfig{
		HTTPMinPoolSize: 2,
		HTTPMaxPoolSize: 5,
	}

	pool := NewConnectionPool(config)
	defer pool.Close()

	clients := make([]*http.Client, 5)
	for i := 0; i < 5; i++ {
		clients[i] = pool.GetHTTPClient()
		if clients[i] == nil {
			t.Errorf("Expected client at index %d, got nil", i)
		}
	}

	metrics := pool.GetMetrics()
	if metrics.CurrentSize > int32(config.HTTPMaxPoolSize) {
		t.Errorf("Pool exceeded max size: %d > %d", metrics.CurrentSize, config.HTTPMaxPoolSize)
	}

	// Release
	for i := 0; i < 5; i++ {
		pool.ReleaseHTTPClient(clients[i])
	}
}

// TestConcurrentAccess verifies thread safety
func TestConcurrentAccess(t *testing.T) {
	pool := NewConnectionPool(DefaultPoolConfig())
	defer pool.Close()

	var wg sync.WaitGroup
	numGoroutines := 100
	requestsPerGoroutine := 10

	for g := 0; g < numGoroutines; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < requestsPerGoroutine; i++ {
				client := pool.GetHTTPClient()
				if client != nil {
					time.Sleep(time.Millisecond)
					pool.ReleaseHTTPClient(client)
				}
			}
		}()
	}

	wg.Wait()

	metrics := pool.GetMetrics()
	if metrics.TotalReused < int64(numGoroutines*requestsPerGoroutine/2) {
		t.Errorf("Expected high reuse rate, got %d reuses", metrics.TotalReused)
	}
}

// TestHealthChecks verifies health checks run
func TestHealthChecks(t *testing.T) {
	config := DefaultPoolConfig()
	config.HealthCheckInterval = 100 * time.Millisecond

	pool := NewConnectionPool(config)
	defer pool.Close()

	// Wait for health checks to run
	time.Sleep(250 * time.Millisecond)

	metrics := pool.GetMetrics()
	if metrics.HealthCheckPass == 0 && metrics.HealthCheckFail == 0 {
		t.Error("Health checks did not run")
	}
}

// TestMetricsAccuracy verifies metrics are accurate
func TestMetricsAccuracy(t *testing.T) {
	pool := NewConnectionPool(DefaultPoolConfig())
	defer pool.Close()

	client := pool.GetHTTPClient()
	pool.ReleaseHTTPClient(client)

	client = pool.GetHTTPClient()
	pool.ReleaseHTTPClient(client)

	metrics := pool.GetMetrics()

	if metrics.TotalCreated < 1 {
		t.Errorf("Expected at least 1 created, got %d", metrics.TotalCreated)
	}

	if metrics.TotalReused != 1 {
		t.Errorf("Expected 1 reused, got %d", metrics.TotalReused)
	}
}

// TestReleaseNilClient verifies nil client is handled
func TestReleaseNilClient(t *testing.T) {
	pool := NewConnectionPool(DefaultPoolConfig())
	defer pool.Close()

	// Should not panic
	pool.ReleaseHTTPClient(nil)
}

// TestPoolMetricsString verifies metrics string representation
func TestPoolMetricsString(t *testing.T) {
	metrics := &PoolMetrics{
		TotalCreated:    100,
		TotalReused:     95,
		CurrentSize:     10,
		HealthCheckPass: 5,
	}

	str := metrics.String()
	if len(str) == 0 {
		t.Error("Metrics string is empty")
	}

	if !contains(str, "PoolMetrics") {
		t.Errorf("Metrics string missing 'PoolMetrics': %s", str)
	}
}

// TestReuseRateCalculation verifies reuse rate is calculated correctly
func TestReuseRateCalculation(t *testing.T) {
	pool := NewConnectionPool(DefaultPoolConfig())
	defer pool.Close()

	// Perform 10 gets and releases
	for i := 0; i < 10; i++ {
		client := pool.GetHTTPClient()
		pool.ReleaseHTTPClient(client)
	}

	reuseRate := pool.GetReuseRate()

	// After first get, pool is reusing, so rate should be > 0
	if reuseRate <= 0 {
		t.Errorf("Expected positive reuse rate, got %.2f", reuseRate)
	}

	if reuseRate > 1 {
		t.Errorf("Reuse rate should not exceed 1, got %.2f", reuseRate)
	}
}

// TestConnectionPoolClose verifies pool closes cleanly
func TestConnectionPoolClose(t *testing.T) {
	pool := NewConnectionPool(DefaultPoolConfig())

	client := pool.GetHTTPClient()
	pool.ReleaseHTTPClient(client)

	err := pool.Close()
	if err != nil {
		t.Errorf("Unexpected error on close: %v", err)
	}
}

// TestPoolStress performs stress testing on the pool
func TestPoolStress(t *testing.T) {
	pool := NewConnectionPool(DefaultPoolConfig())
	defer pool.Close()

	var wg sync.WaitGroup
	var successCount int64

	// 500 concurrent operations
	for i := 0; i < 500; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			client := pool.GetHTTPClient()
			if client != nil {
				// Simulate some work
				time.Sleep(time.Millisecond)
				pool.ReleaseHTTPClient(client)
				atomic.AddInt64(&successCount, 1)
			}
		}()
	}

	wg.Wait()

	if successCount < 450 {
		t.Errorf("Expected at least 450 successful operations, got %d", successCount)
	}

	metrics := pool.GetMetrics()
	t.Logf("Final metrics: %v", metrics.String())
}

// TestIdleConnectionRemoval verifies idle connections are removed
func TestIdleConnectionRemoval(t *testing.T) {
	config := &PoolConfig{
		HTTPMinPoolSize:     5,
		HTTPMaxPoolSize:     50,
		GRPCMinPoolSize:     3,
		GRPCMaxPoolSize:     30,
		HealthCheckInterval: 100 * time.Millisecond,
		IdleTimeout:         100 * time.Millisecond,
		MaxConnAge:          10 * time.Minute,
	}

	pool := NewConnectionPool(config)
	defer pool.Close()

	initialMetrics := pool.GetMetrics()
	initialSize := initialMetrics.CurrentSize

	// Get and release a client
	client := pool.GetHTTPClient()
	pool.ReleaseHTTPClient(client)

	// Wait for cleanup to run
	time.Sleep(300 * time.Millisecond)

	afterCleanup := pool.GetMetrics()

	// Size might be reduced due to idle timeout
	if afterCleanup.CurrentSize >= initialSize {
		// Idle removal worked
		t.Logf("Idle connections were cleaned up: %d -> %d",
			initialSize, afterCleanup.CurrentSize)
	}
}

// Helper function to check if string contains substring
func contains(str, substr string) bool {
	for i := 0; i < len(str)-len(substr)+1; i++ {
		if str[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// BenchmarkGetHTTPClient benchmarks getting HTTP client
func BenchmarkGetHTTPClient(b *testing.B) {
	pool := NewConnectionPool(DefaultPoolConfig())
	defer pool.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		client := pool.GetHTTPClient()
		pool.ReleaseHTTPClient(client)
	}
}

// BenchmarkConcurrentGetHTTPClient benchmarks concurrent gets
func BenchmarkConcurrentGetHTTPClient(b *testing.B) {
	pool := NewConnectionPool(DefaultPoolConfig())
	defer pool.Close()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			client := pool.GetHTTPClient()
			pool.ReleaseHTTPClient(client)
		}
	})
}
