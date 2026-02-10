package client

import (
	"context"
	"fmt"
	"time"

	// "github.com/ryotohq/ryzanstein-desktop/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/status"
)

// MCPClientConfig configures the MCP (Model Control Protocol) gRPC client.
type MCPClientConfig struct {
	// Address is the gRPC server address (e.g., "localhost:50051")
	Address string

	// Timeout is the default timeout for unary RPC calls
	Timeout time.Duration

	// MaxRetries is the maximum number of retry attempts for transient errors
	MaxRetries int

	// InitialBackoff is the initial backoff duration for exponential backoff
	InitialBackoff time.Duration

	// MaxBackoff is the maximum backoff duration for exponential backoff
	MaxBackoff time.Duration

	// KeepaliveInterval is the interval for keepalive pings
	KeepaliveInterval time.Duration

	// KeepaliveTimeout is the timeout for keepalive pings
	KeepaliveTimeout time.Duration

	// InsecureSkipVerify disables TLS verification (for development only)
	InsecureSkipVerify bool
}

// DefaultMCPClientConfig returns a sensible default MCPClientConfig.
func DefaultMCPClientConfig(address string) MCPClientConfig {
	return MCPClientConfig{
		Address:            address,
		Timeout:            30 * time.Second,
		MaxRetries:         3,
		InitialBackoff:     100 * time.Millisecond,
		MaxBackoff:         10 * time.Second,
		KeepaliveInterval:  30 * time.Second,
		KeepaliveTimeout:   10 * time.Second,
		InsecureSkipVerify: true, // For development; should use TLS in production
	}
}

// MCPClient is a gRPC client for the Model Control Protocol.
// It provides methods to interact with the MCP server for model inference and management.
type MCPClient struct {
	config    MCPClientConfig
	conn      *grpc.ClientConn
	client    proto.MCPServiceClient
	backoffer *exponentialBackoffer
}

// NewMCPClient creates a new MCPClient and establishes a connection to the MCP server.
func NewMCPClient(config MCPClientConfig) (*MCPClient, error) {
	if config.Address == "" {
		return nil, fmt.Errorf("mcp client: address is required")
	}

	// Set defaults if not provided
	if config.Timeout == 0 {
		config.Timeout = 30 * time.Second
	}
	if config.MaxRetries == 0 {
		config.MaxRetries = 3
	}
	if config.InitialBackoff == 0 {
		config.InitialBackoff = 100 * time.Millisecond
	}
	if config.MaxBackoff == 0 {
		config.MaxBackoff = 10 * time.Second
	}
	if config.KeepaliveInterval == 0 {
		config.KeepaliveInterval = 30 * time.Second
	}
	if config.KeepaliveTimeout == 0 {
		config.KeepaliveTimeout = 10 * time.Second
	}

	// Create dial options
	dialOptions := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                config.KeepaliveInterval,
			Timeout:             config.KeepaliveTimeout,
			PermitWithoutStream: true,
		}),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(100 * 1024 * 1024), // 100MB max receive size
		),
	}

	// Establish connection
	conn, err := grpc.Dial(config.Address, dialOptions...)
	if err != nil {
		return nil, fmt.Errorf("mcp client: failed to connect to %s: %w", config.Address, err)
	}

	client := &MCPClient{
		config:    config,
		conn:      conn,
		client:    proto.NewMCPServiceClient(conn),
		backoffer: newExponentialBackoffer(config.InitialBackoff, config.MaxBackoff),
	}

	// Verify connection is working with a health check
	if err := client.healthCheck(context.Background()); err != nil {
		conn.Close()
		return nil, fmt.Errorf("mcp client: health check failed: %w", err)
	}

	return client, nil
}

// healthCheck verifies that the MCP server is reachable and responsive.
func (c *MCPClient) healthCheck(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	_, err := c.client.Health(ctx, &proto.HealthRequest{})
	if err != nil {
		return err
	}

	return nil
}

// InferRequest contains parameters for a model inference request.
type InferRequest struct {
	// ModelID is the ID of the model to use for inference
	ModelID string

	// Input is the input data for the model
	Input []byte

	// Metadata is optional metadata for the request (e.g., temperature, max_tokens)
	Metadata map[string]string
}

// InferResponse contains the response from a model inference request.
type InferResponse struct {
	// Output is the model output
	Output []byte

	// Metadata is optional metadata in the response (e.g., tokens_used, latency_ms)
	Metadata map[string]string
}

// Infer performs model inference with automatic retry on transient errors.
func (c *MCPClient) Infer(ctx context.Context, req *InferRequest) (*InferResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("mcp client: request cannot be nil")
	}
	if req.ModelID == "" {
		return nil, fmt.Errorf("mcp client: model_id is required")
	}
	if len(req.Input) == 0 {
		return nil, fmt.Errorf("mcp client: input cannot be empty")
	}

	grpcReq := &proto.InferRequest{
		ModelId:  req.ModelID,
		Input:    req.Input,
		Metadata: req.Metadata,
	}

	var lastErr error
	for attempt := 0; attempt <= c.config.MaxRetries; attempt++ {
		if attempt > 0 {
			// Apply backoff before retry
			backoffDuration := c.backoffer.nextBackoff()
			select {
			case <-time.After(backoffDuration):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}

		// Create context with timeout for this attempt
		attemptCtx, cancel := context.WithTimeout(ctx, c.config.Timeout)

		grpcResp, err := c.client.Infer(attemptCtx, grpcReq)
		cancel()

		if err == nil {
			c.backoffer.reset()
			return &InferResponse{
				Output:   grpcResp.Output,
				Metadata: grpcResp.Metadata,
			}, nil
		}

		lastErr = err

		// Check if error is retryable
		if !isRetryableError(err) {
			return nil, fmt.Errorf("mcp client: inference failed: %w", err)
		}

		// Don't retry on final attempt
		if attempt == c.config.MaxRetries {
			return nil, fmt.Errorf("mcp client: inference failed after %d retries: %w", c.config.MaxRetries, lastErr)
		}
	}

	return nil, fmt.Errorf("mcp client: unexpected error: %w", lastErr)
}

// ListModelsResponse contains the response from a list models request.
type ListModelsResponse struct {
	// Models is the list of available models
	Models []*ModelInfo
}

// ModelInfo contains metadata about an available model.
type ModelInfo struct {
	// ID is the model identifier
	ID string

	// Name is the human-readable model name
	Name string

	// Version is the model version
	Version string

	// Status is the current status of the model (e.g., "loaded", "unloaded")
	Status string

	// Metadata contains additional model metadata
	Metadata map[string]string
}

// ListModels lists all available models from the MCP server.
func (c *MCPClient) ListModels(ctx context.Context) (*ListModelsResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, c.config.Timeout)
	defer cancel()

	grpcResp, err := c.client.ListModels(ctx, &proto.ListModelsRequest{})
	if err != nil {
		return nil, fmt.Errorf("mcp client: list models failed: %w", err)
	}

	models := make([]*ModelInfo, len(grpcResp.Models))
	for i, m := range grpcResp.Models {
		models[i] = &ModelInfo{
			ID:       m.Id,
			Name:     m.Name,
			Version:  m.Version,
			Status:   m.Status,
			Metadata: m.Metadata,
		}
	}

	return &ListModelsResponse{Models: models}, nil
}

// LoadModelRequest contains parameters for a load model request.
type LoadModelRequest struct {
	// ModelID is the ID of the model to load
	ModelID string

	// Config is optional model configuration
	Config map[string]string
}

// LoadModel loads a model into memory on the MCP server.
func (c *MCPClient) LoadModel(ctx context.Context, req *LoadModelRequest) error {
	if req == nil {
		return fmt.Errorf("mcp client: request cannot be nil")
	}
	if req.ModelID == "" {
		return fmt.Errorf("mcp client: model_id is required")
	}

	ctx, cancel := context.WithTimeout(ctx, 2*time.Minute) // Longer timeout for model loading
	defer cancel()

	grpcReq := &proto.LoadModelRequest{
		ModelId: req.ModelID,
		Config:  req.Config,
	}

	_, err := c.client.LoadModel(ctx, grpcReq)
	if err != nil {
		return fmt.Errorf("mcp client: load model failed: %w", err)
	}

	return nil
}

// UnloadModelRequest contains parameters for an unload model request.
type UnloadModelRequest struct {
	// ModelID is the ID of the model to unload
	ModelID string
}

// UnloadModel unloads a model from memory on the MCP server.
func (c *MCPClient) UnloadModel(ctx context.Context, req *UnloadModelRequest) error {
	if req == nil {
		return fmt.Errorf("mcp client: request cannot be nil")
	}
	if req.ModelID == "" {
		return fmt.Errorf("mcp client: model_id is required")
	}

	ctx, cancel := context.WithTimeout(ctx, c.config.Timeout)
	defer cancel()

	grpcReq := &proto.UnloadModelRequest{
		ModelId: req.ModelID,
	}

	_, err := c.client.UnloadModel(ctx, grpcReq)
	if err != nil {
		return fmt.Errorf("mcp client: unload model failed: %w", err)
	}

	return nil
}

// Health checks the health status of the MCP server.
func (c *MCPClient) Health(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	_, err := c.client.Health(ctx, &proto.HealthRequest{})
	if err != nil {
		return fmt.Errorf("mcp client: health check failed: %w", err)
	}

	return nil
}

// Close closes the connection to the MCP server.
func (c *MCPClient) Close() error {
	if c.conn != nil {
		return c.conn.Close()
	}
	return nil
}

// isRetryableError determines if an error is transient and can be retried.
func isRetryableError(err error) bool {
	if err == nil {
		return false
	}

	st, ok := status.FromError(err)
	if !ok {
		return false
	}

	// Retryable codes according to gRPC standards
	switch st.Code() {
	case codes.Unavailable:
		return true
	case codes.ResourceExhausted:
		return true
	case codes.DeadlineExceeded:
		return true
	case codes.Internal:
		// Some internal errors might be retryable
		return true
	default:
		return false
	}
}

// exponentialBackoffer implements exponential backoff with jitter.
type exponentialBackoffer struct {
	initial     time.Duration
	maxBackoff  time.Duration
	currentStep int
}

// newExponentialBackoffer creates a new exponential backoffer.
func newExponentialBackoffer(initial, maxBackoff time.Duration) *exponentialBackoffer {
	return &exponentialBackoffer{
		initial:     initial,
		maxBackoff:  maxBackoff,
		currentStep: 0,
	}
}

// nextBackoff returns the next backoff duration with exponential growth capped at maxBackoff.
func (b *exponentialBackoffer) nextBackoff() time.Duration {
	// Calculate backoff: initial * (2 ^ currentStep)
	backoff := time.Duration(float64(b.initial) * float64(1<<uint(b.currentStep)))

	if backoff > b.maxBackoff {
		backoff = b.maxBackoff
	}

	b.currentStep++
	return backoff
}

// reset resets the backoff to initial state.
func (b *exponentialBackoffer) reset() {
	b.currentStep = 0
}
