package services

import (
	"fmt"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"google.golang.org/grpc"
)

// PoolConfig defines connection pool configuration
type PoolConfig struct {
	HTTPMinPoolSize     int
	HTTPMaxPoolSize     int
	GRPCMinPoolSize     int
	GRPCMaxPoolSize     int
	HealthCheckInterval time.Duration
	IdleTimeout         time.Duration
	MaxConnAge          time.Duration
}

// DefaultPoolConfig returns default connection pool configuration
func DefaultPoolConfig() *PoolConfig {
	return &PoolConfig{
		HTTPMinPoolSize:     10,
		HTTPMaxPoolSize:     100,
		GRPCMinPoolSize:     5,
		GRPCMaxPoolSize:     50,
		HealthCheckInterval: 30 * time.Second,
		IdleTimeout:         5 * time.Minute,
		MaxConnAge:          10 * time.Minute,
	}
}

// PoolMetrics tracks connection pool statistics
type PoolMetrics struct {
	TotalCreated    int64
	TotalReused     int64
	TotalFailed     int64
	CurrentSize     int32
	HealthCheckPass int64
	HealthCheckFail int64
}

// ConnectionPool manages reusable HTTP and gRPC connections
type ConnectionPool struct {
	httpPool *HTTPClientPool
	grpcPool *GRPCChannelPool
	metrics  *PoolMetrics
	config   *PoolConfig
	mu       sync.RWMutex
	stopCh   chan struct{}
	wg       sync.WaitGroup
}

// HTTPClientPool manages a pool of reusable HTTP clients
type HTTPClientPool struct {
	clients      chan *http.Client
	config       *PoolConfig
	metrics      *PoolMetrics
	createdCount int32
	lastUsedTime sync.Map // map[*http.Client]time.Time
	clientAgeMap sync.Map // map[*http.Client]time.Time
	mu           sync.RWMutex
}

// GRPCChannelPool manages a pool of reusable gRPC channels
type GRPCChannelPool struct {
	channels      chan *grpc.ClientConn
	config        *PoolConfig
	metrics       *PoolMetrics
	createdCount  int32
	lastUsedTime  sync.Map // map[*grpc.ClientConn]time.Time
	channelAgeMap sync.Map // map[*grpc.ClientConn]time.Time
	mu            sync.RWMutex
}

// NewConnectionPool creates a new connection pool
func NewConnectionPool(config *PoolConfig) *ConnectionPool {
	if config == nil {
		config = DefaultPoolConfig()
	}

	metrics := &PoolMetrics{}

	pool := &ConnectionPool{
		httpPool: &HTTPClientPool{
			clients: make(chan *http.Client, config.HTTPMaxPoolSize),
			config:  config,
			metrics: metrics,
		},
		grpcPool: &GRPCChannelPool{
			channels: make(chan *grpc.ClientConn, config.GRPCMaxPoolSize),
			config:   config,
			metrics:  metrics,
		},
		metrics: metrics,
		config:  config,
		stopCh:  make(chan struct{}),
	}

	// Initialize pools with minimum size
	pool.initializeHTTPPool()
	pool.initializeGRPCPool()

	// Start health check routine
	pool.wg.Add(1)
	go pool.healthCheckRoutine()

	// Start cleanup routine
	pool.wg.Add(1)
	go pool.cleanupRoutine()

	return pool
}

// initializeHTTPPool creates initial HTTP clients
func (cp *ConnectionPool) initializeHTTPPool() {
	for i := 0; i < cp.config.HTTPMinPoolSize; i++ {
		client := &http.Client{
			Timeout: 30 * time.Second,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     90 * time.Second,
			},
		}
		cp.httpPool.clients <- client
		atomic.AddInt64(&cp.metrics.TotalCreated, 1)
		atomic.AddInt32(&cp.metrics.CurrentSize, 1)
		atomic.AddInt32(&cp.httpPool.createdCount, 1)

		// Track creation time
		cp.httpPool.clientAgeMap.Store(client, time.Now())
		cp.httpPool.lastUsedTime.Store(client, time.Now())
	}
}

// initializeGRPCPool creates initial gRPC channels
func (cp *ConnectionPool) initializeGRPCPool() {
	for i := 0; i < cp.config.GRPCMinPoolSize; i++ {
		// Create dummy channel for initialization
		// In production, actual endpoint would be used
		conn, err := grpc.Dial(
			"localhost:50051",
			grpc.WithInsecure(),
			grpc.WithTimeout(5*time.Second),
		)
		if err != nil {
			// Skip on error, will be created on demand
			continue
		}

		cp.grpcPool.channels <- conn
		atomic.AddInt64(&cp.metrics.TotalCreated, 1)
		atomic.AddInt32(&cp.metrics.CurrentSize, 1)
		atomic.AddInt32(&cp.grpcPool.createdCount, 1)

		// Track creation time
		cp.grpcPool.channelAgeMap.Store(conn, time.Now())
		cp.grpcPool.lastUsedTime.Store(conn, time.Now())
	}
}

// GetHTTPClient retrieves an HTTP client from the pool
func (cp *ConnectionPool) GetHTTPClient() *http.Client {
	select {
	case client := <-cp.httpPool.clients:
		// Got existing client
		cp.httpPool.lastUsedTime.Store(client, time.Now())
		atomic.AddInt64(&cp.metrics.TotalReused, 1)
		return client
	default:
		// Pool empty, create new if under max
		currentSize := atomic.LoadInt32(&cp.metrics.CurrentSize)
		if currentSize < int32(cp.config.HTTPMaxPoolSize) {
			client := &http.Client{
				Timeout: 30 * time.Second,
				Transport: &http.Transport{
					MaxIdleConns:        100,
					MaxIdleConnsPerHost: 10,
					IdleConnTimeout:     90 * time.Second,
				},
			}
			atomic.AddInt64(&cp.metrics.TotalCreated, 1)
			atomic.AddInt32(&cp.metrics.CurrentSize, 1)
			atomic.AddInt32(&cp.httpPool.createdCount, 1)

			cp.httpPool.clientAgeMap.Store(client, time.Now())
			cp.httpPool.lastUsedTime.Store(client, time.Now())

			return client
		}

		// Pool full, wait for available client
		client := <-cp.httpPool.clients
		cp.httpPool.lastUsedTime.Store(client, time.Now())
		atomic.AddInt64(&cp.metrics.TotalReused, 1)
		return client
	}
}

// ReleaseHTTPClient returns an HTTP client to the pool
func (cp *ConnectionPool) ReleaseHTTPClient(client *http.Client) {
	if client == nil {
		return
	}

	select {
	case cp.httpPool.clients <- client:
		cp.httpPool.lastUsedTime.Store(client, time.Now())
	default:
		// Pool full, close client
		if transport, ok := client.Transport.(*http.Transport); ok {
			transport.CloseIdleConnections()
		}
		atomic.AddInt32(&cp.metrics.CurrentSize, -1)
	}
}

// GetGRPCChannel retrieves a gRPC channel from the pool
func (cp *ConnectionPool) GetGRPCChannel() *grpc.ClientConn {
	select {
	case conn := <-cp.grpcPool.channels:
		// Got existing channel
		cp.grpcPool.lastUsedTime.Store(conn, time.Now())
		atomic.AddInt64(&cp.metrics.TotalReused, 1)
		return conn
	default:
		// Pool empty, create new if under max
		currentSize := atomic.LoadInt32(&cp.metrics.CurrentSize)
		if currentSize < int32(cp.config.GRPCMaxPoolSize) {
			conn, err := grpc.Dial(
				"localhost:50051",
				grpc.WithInsecure(),
				grpc.WithTimeout(5*time.Second),
			)
			if err != nil {
				atomic.AddInt64(&cp.metrics.TotalFailed, 1)
				return nil
			}

			atomic.AddInt64(&cp.metrics.TotalCreated, 1)
			atomic.AddInt32(&cp.metrics.CurrentSize, 1)
			atomic.AddInt32(&cp.grpcPool.createdCount, 1)

			cp.grpcPool.channelAgeMap.Store(conn, time.Now())
			cp.grpcPool.lastUsedTime.Store(conn, time.Now())

			return conn
		}

		// Pool full, wait for available channel
		conn := <-cp.grpcPool.channels
		cp.grpcPool.lastUsedTime.Store(conn, time.Now())
		atomic.AddInt64(&cp.metrics.TotalReused, 1)
		return conn
	}
}

// ReleaseGRPCChannel returns a gRPC channel to the pool
func (cp *ConnectionPool) ReleaseGRPCChannel(conn *grpc.ClientConn) {
	if conn == nil {
		return
	}

	select {
	case cp.grpcPool.channels <- conn:
		cp.grpcPool.lastUsedTime.Store(conn, time.Now())
	default:
		// Pool full, close connection
		conn.Close()
		atomic.AddInt32(&cp.metrics.CurrentSize, -1)
	}
}

// healthCheckRoutine performs periodic health checks on connections
func (cp *ConnectionPool) healthCheckRoutine() {
	defer cp.wg.Done()

	ticker := time.NewTicker(cp.config.HealthCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-cp.stopCh:
			return
		case <-ticker.C:
			cp.performHealthChecks()
		}
	}
}

// performHealthChecks checks health of connections in pools
func (cp *ConnectionPool) performHealthChecks() {
	// Check HTTP clients
	deadClients := []int{}
	clientCount := 0

	for i := 0; i < len(cp.httpPool.clients); i++ {
		select {
		case client := <-cp.httpPool.clients:
			// Verify client is still usable
			resp, err := client.Head("http://localhost:8080/health")
			if err != nil {
				atomic.AddInt64(&cp.metrics.HealthCheckFail, 1)
				deadClients = append(deadClients, i)
				// Don't return dead client to pool
				continue
			}
			if resp != nil && resp.Body != nil {
				resp.Body.Close()
			}

			atomic.AddInt64(&cp.metrics.HealthCheckPass, 1)
			cp.httpPool.clients <- client
			clientCount++
		default:
			break
		}
	}

	// Check gRPC channels
	deadChannels := []int{}
	channelCount := 0

	for i := 0; i < len(cp.grpcPool.channels); i++ {
		select {
		case conn := <-cp.grpcPool.channels:
			// Check connection state
			if conn.GetState().String() == "READY" || conn.GetState().String() == "IDLE" {
				atomic.AddInt64(&cp.metrics.HealthCheckPass, 1)
				cp.grpcPool.channels <- conn
				channelCount++
			} else {
				atomic.AddInt64(&cp.metrics.HealthCheckFail, 1)
				deadChannels = append(deadChannels, i)
				conn.Close()
			}
		default:
			break
		}
	}
}

// cleanupRoutine removes idle and aged connections
func (cp *ConnectionPool) cleanupRoutine() {
	defer cp.wg.Done()

	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-cp.stopCh:
			return
		case <-ticker.C:
			cp.performCleanup()
		}
	}
}

// performCleanup removes idle and aged connections
func (cp *ConnectionPool) performCleanup() {
	now := time.Now()

	// Check HTTP clients for idle timeout and max age
	cp.httpPool.lastUsedTime.Range(func(key, value interface{}) bool {
		client := key.(*http.Client)
		lastUsed := value.(time.Time)

		if now.Sub(lastUsed) > cp.config.IdleTimeout {
			// Remove idle client
			if transport, ok := client.Transport.(*http.Transport); ok {
				transport.CloseIdleConnections()
			}
			atomic.AddInt32(&cp.metrics.CurrentSize, -1)
			cp.httpPool.lastUsedTime.Delete(client)
			cp.httpPool.clientAgeMap.Delete(client)
		}

		// Check max age
		if createdTime, ok := cp.httpPool.clientAgeMap.Load(client); ok {
			if now.Sub(createdTime.(time.Time)) > cp.config.MaxConnAge {
				if transport, ok := client.Transport.(*http.Transport); ok {
					transport.CloseIdleConnections()
				}
				atomic.AddInt32(&cp.metrics.CurrentSize, -1)
				cp.httpPool.lastUsedTime.Delete(client)
				cp.httpPool.clientAgeMap.Delete(client)
			}
		}

		return true
	})

	// Check gRPC channels for idle timeout and max age
	cp.grpcPool.lastUsedTime.Range(func(key, value interface{}) bool {
		conn := key.(*grpc.ClientConn)
		lastUsed := value.(time.Time)

		if now.Sub(lastUsed) > cp.config.IdleTimeout {
			// Remove idle channel
			conn.Close()
			atomic.AddInt32(&cp.metrics.CurrentSize, -1)
			cp.grpcPool.lastUsedTime.Delete(conn)
			cp.grpcPool.channelAgeMap.Delete(conn)
		}

		// Check max age
		if createdTime, ok := cp.grpcPool.channelAgeMap.Load(conn); ok {
			if now.Sub(createdTime.(time.Time)) > cp.config.MaxConnAge {
				conn.Close()
				atomic.AddInt32(&cp.metrics.CurrentSize, -1)
				cp.grpcPool.lastUsedTime.Delete(conn)
				cp.grpcPool.channelAgeMap.Delete(conn)
			}
		}

		return true
	})
}

// GetMetrics returns current pool metrics
func (cp *ConnectionPool) GetMetrics() *PoolMetrics {
	cp.mu.RLock()
	defer cp.mu.RUnlock()

	metrics := &PoolMetrics{
		TotalCreated:    atomic.LoadInt64(&cp.metrics.TotalCreated),
		TotalReused:     atomic.LoadInt64(&cp.metrics.TotalReused),
		TotalFailed:     atomic.LoadInt64(&cp.metrics.TotalFailed),
		CurrentSize:     atomic.LoadInt32(&cp.metrics.CurrentSize),
		HealthCheckPass: atomic.LoadInt64(&cp.metrics.HealthCheckPass),
		HealthCheckFail: atomic.LoadInt64(&cp.metrics.HealthCheckFail),
	}

	return metrics
}

// GetReuseRate returns the connection reuse rate
func (cp *ConnectionPool) GetReuseRate() float64 {
	total := atomic.LoadInt64(&cp.metrics.TotalCreated)
	if total == 0 {
		return 0
	}

	reused := atomic.LoadInt64(&cp.metrics.TotalReused)
	return float64(reused) / float64(total)
}

// Close closes the connection pool and all connections
func (cp *ConnectionPool) Close() error {
	close(cp.stopCh)

	// Drain HTTP client pool
	for {
		select {
		case client := <-cp.httpPool.clients:
			if transport, ok := client.Transport.(*http.Transport); ok {
				transport.CloseIdleConnections()
			}
		default:
			goto httpDone
		}
	}

httpDone:
	// Drain gRPC channel pool
	for {
		select {
		case conn := <-cp.grpcPool.channels:
			conn.Close()
		default:
			goto grpcDone
		}
	}

grpcDone:
	// Wait for routines to finish
	cp.wg.Wait()

	return nil
}

// String returns a string representation of pool metrics
func (m *PoolMetrics) String() string {
	reuseRate := float64(0)
	if m.TotalCreated > 0 {
		reuseRate = float64(m.TotalReused) / float64(m.TotalCreated)
	}

	return fmt.Sprintf(
		"PoolMetrics{Created:%d, Reused:%d, Failed:%d, Size:%d, ReuseRate:%.2f%%, HealthPass:%d, HealthFail:%d}",
		m.TotalCreated,
		m.TotalReused,
		m.TotalFailed,
		m.CurrentSize,
		reuseRate*100,
		m.HealthCheckPass,
		m.HealthCheckFail,
	)
}
