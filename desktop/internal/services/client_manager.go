package services

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/config"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// ClientManager handles REST and gRPC client initialization and lifecycle
type ClientManager struct {
	config *config.AppConfig

	// REST client
	restClient interface{} // Would be *http.Client in real implementation

	// gRPC client
	grpcConn interface{} // Would be *grpc.ClientConn in real implementation

	// Lifecycle management
	mu          sync.RWMutex
	initialized bool
	closed      bool
	ctx         context.Context
	cancel      context.CancelFunc

	// Metrics
	createdAt time.Time
	requests  int64
}

// NewClientManager creates a new client manager with given config
func NewClientManager(cfg *config.AppConfig) *ClientManager {
	ctx, cancel := context.WithCancel(context.Background())

	return &ClientManager{
		config:    cfg,
		ctx:       ctx,
		cancel:    cancel,
		createdAt: time.Now(),
		requests:  0,
	}
}

// Initialize sets up REST and gRPC clients based on configuration
func (cm *ClientManager) Initialize() error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	if cm.initialized {
		return fmt.Errorf("client manager already initialized")
	}

	if cm.closed {
		return fmt.Errorf("client manager is closed")
	}

	// Initialize based on server type configuration
	switch cm.config.Server.Type {
	case "rest":
		if err := cm.initializeRESTClient(); err != nil {
			return fmt.Errorf("failed to initialize REST client: %w", err)
		}
	case "grpc":
		if err := cm.initializeGRPCClient(); err != nil {
			return fmt.Errorf("failed to initialize gRPC client: %w", err)
		}
	case "hybrid":
		if err := cm.initializeRESTClient(); err != nil {
			return fmt.Errorf("failed to initialize REST client: %w", err)
		}
		if err := cm.initializeGRPCClient(); err != nil {
			// Log warning but don't fail on gRPC if REST works
			fmt.Printf("warning: gRPC initialization failed: %v\n", err)
		}
	default:
		return fmt.Errorf("unsupported server type: %s", cm.config.Server.Type)
	}

	cm.initialized = true
	return nil
}

// initializeRESTClient sets up REST client with proper configuration
func (cm *ClientManager) initializeRESTClient() error {
	// In real implementation, create http.Client with timeouts
	// For now, just validate configuration
	if cm.config.REST == nil {
		return fmt.Errorf("REST configuration missing but REST server type specified")
	}

	if cm.config.REST.Host == "" || cm.config.REST.Port == 0 {
		return fmt.Errorf("invalid REST configuration: host or port missing")
	}

	// Verify connectivity
	endpoint := fmt.Sprintf("http://%s:%d", cm.config.REST.Host, cm.config.REST.Port)
	timeout := time.Duration(cm.config.REST.Timeout) * time.Second

	// Test connection with timeout
	testCtx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	// Simulate connection attempt
	select {
	case <-testCtx.Done():
		return fmt.Errorf("REST server connection timeout: %s", endpoint)
	default:
		// Connection successful
	}

	// Store configured endpoint
	cm.restClient = endpoint

	return nil
}

// initializeGRPCClient sets up gRPC client with proper configuration
func (cm *ClientManager) initializeGRPCClient() error {
	if cm.config.GRPC == nil {
		return fmt.Errorf("gRPC configuration missing but gRPC server type specified")
	}

	if cm.config.GRPC.Host == "" || cm.config.GRPC.Port == 0 {
		return fmt.Errorf("invalid gRPC configuration: host or port missing")
	}

	// Create gRPC connection with timeouts
	endpoint := fmt.Sprintf("%s:%d", cm.config.GRPC.Host, cm.config.GRPC.Port)
	timeout := time.Duration(cm.config.GRPC.Timeout) * time.Second

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	// Attempt connection
	conn, err := grpc.DialContext(
		ctx,
		endpoint,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithReturnConnectionError(),
	)
	if err != nil {
		return fmt.Errorf("failed to connect to gRPC server %s: %w", endpoint, err)
	}

	cm.grpcConn = conn
	return nil
}

// GetRESTEndpoint returns the configured REST endpoint
func (cm *ClientManager) GetRESTEndpoint() (string, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	if !cm.initialized {
		return "", fmt.Errorf("client manager not initialized")
	}

	if cm.restClient == nil {
		return "", fmt.Errorf("REST client not available")
	}

	endpoint, ok := cm.restClient.(string)
	if !ok {
		return "", fmt.Errorf("invalid REST client state")
	}

	return endpoint, nil
}

// GetGRPCConnection returns the gRPC connection
func (cm *ClientManager) GetGRPCConnection() (*grpc.ClientConn, error) {
	cm.mu.RLock()
	defer cm.mu.Unlock()

	if !cm.initialized {
		return nil, fmt.Errorf("client manager not initialized")
	}

	if cm.grpcConn == nil {
		return nil, fmt.Errorf("gRPC client not available")
	}

	conn, ok := cm.grpcConn.(*grpc.ClientConn)
	if !ok {
		return nil, fmt.Errorf("invalid gRPC connection state")
	}

	return conn, nil
}

// ExecuteWithRouting routes request based on configuration
func (cm *ClientManager) ExecuteWithRouting(ctx context.Context, operation string, data interface{}) (interface{}, error) {
	cm.mu.Lock()
	cm.requests++
	cm.mu.Unlock()

	if !cm.IsInitialized() {
		return nil, fmt.Errorf("client manager not initialized")
	}

	switch cm.config.Server.Type {
	case "rest":
		return cm.executeREST(ctx, operation, data)
	case "grpc":
		return cm.executeGRPC(ctx, operation, data)
	case "hybrid":
		// Try gRPC first, fall back to REST
		result, err := cm.executeGRPC(ctx, operation, data)
		if err != nil {
			return cm.executeREST(ctx, operation, data)
		}
		return result, nil
	default:
		return nil, fmt.Errorf("unsupported server type: %s", cm.config.Server.Type)
	}
}

// executeREST handles REST-based requests
func (cm *ClientManager) executeREST(ctx context.Context, operation string, data interface{}) (interface{}, error) {
	endpoint, err := cm.GetRESTEndpoint()
	if err != nil {
		return nil, err
	}

	// Simulate REST request with context
	select {
	case <-ctx.Done():
		return nil, fmt.Errorf("context cancelled during REST execution")
	default:
	}

	// Return operation result (would use actual HTTP client in real implementation)
	return map[string]interface{}{
		"operation": operation,
		"protocol":  "REST",
		"endpoint":  endpoint,
		"data":      data,
	}, nil
}

// executeGRPC handles gRPC-based requests
func (cm *ClientManager) executeGRPC(ctx context.Context, operation string, data interface{}) (interface{}, error) {
	conn, err := cm.GetGRPCConnection()
	if err != nil {
		return nil, err
	}

	// Verify connection is still open
	if conn.GetState().String() == "SHUTDOWN" {
		return nil, fmt.Errorf("gRPC connection is shutdown")
	}

	// Simulate gRPC request with context
	select {
	case <-ctx.Done():
		return nil, fmt.Errorf("context cancelled during gRPC execution")
	default:
	}

	// Return operation result (would use actual gRPC client in real implementation)
	return map[string]interface{}{
		"operation": operation,
		"protocol":  "gRPC",
		"data":      data,
	}, nil
}

// IsInitialized returns whether client manager is ready
func (cm *ClientManager) IsInitialized() bool {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	return cm.initialized && !cm.closed
}

// Close closes all client connections and resources
func (cm *ClientManager) Close() error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	if cm.closed {
		return fmt.Errorf("client manager already closed")
	}

	// Close gRPC connection if exists
	if cm.grpcConn != nil {
		if conn, ok := cm.grpcConn.(*grpc.ClientConn); ok {
			conn.Close()
		}
	}

	// Cancel context
	cm.cancel()

	cm.closed = true
	return nil
}

// GetMetrics returns current metrics
func (cm *ClientManager) GetMetrics() map[string]interface{} {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	return map[string]interface{}{
		"initialized": cm.initialized,
		"requests":    cm.requests,
		"uptime":      time.Since(cm.createdAt).Seconds(),
		"server_type": cm.config.Server.Type,
	}
}
