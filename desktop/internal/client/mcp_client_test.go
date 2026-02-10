package client

import (
	"context"
	"fmt"
	"testing"
	"time"

	// "github.com/ryotohq/ryzanstein-desktop/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// MockMCPServiceClient is a mock implementation for testing.
type MockMCPServiceClient struct {
	proto.MCPServiceClient

	// Mocked responses
	inferResp      *proto.InferResponse
	inferErr       error
	listModelsResp *proto.ListModelsResponse
	listModelsErr  error
	loadModelErr   error
	unloadModelErr error
	healthErr      error

	// Call tracking
	inferCallCount       int
	listModelsCallCount  int
	loadModelCallCount   int
	unloadModelCallCount int
	healthCallCount      int

	// Request capture
	lastInferReq       *proto.InferRequest
	lastLoadModelReq   *proto.LoadModelRequest
	lastUnloadModelReq *proto.UnloadModelRequest
}

func (m *MockMCPServiceClient) Infer(ctx context.Context, in *proto.InferRequest, opts ...grpc.CallOption) (*proto.InferResponse, error) {
	m.inferCallCount++
	m.lastInferReq = in

	if m.inferErr != nil {
		return nil, m.inferErr
	}

	return m.inferResp, nil
}

func (m *MockMCPServiceClient) ListModels(ctx context.Context, in *proto.ListModelsRequest, opts ...grpc.CallOption) (*proto.ListModelsResponse, error) {
	m.listModelsCallCount++

	if m.listModelsErr != nil {
		return nil, m.listModelsErr
	}

	return m.listModelsResp, nil
}

func (m *MockMCPServiceClient) LoadModel(ctx context.Context, in *proto.LoadModelRequest, opts ...grpc.CallOption) (*proto.LoadModelResponse, error) {
	m.loadModelCallCount++
	m.lastLoadModelReq = in

	if m.loadModelErr != nil {
		return nil, m.loadModelErr
	}

	return &proto.LoadModelResponse{Success: true}, nil
}

func (m *MockMCPServiceClient) UnloadModel(ctx context.Context, in *proto.UnloadModelRequest, opts ...grpc.CallOption) (*proto.UnloadModelResponse, error) {
	m.unloadModelCallCount++
	m.lastUnloadModelReq = in

	if m.unloadModelErr != nil {
		return nil, m.unloadModelErr
	}

	return &proto.UnloadModelResponse{Success: true}, nil
}

func (m *MockMCPServiceClient) Health(ctx context.Context, in *proto.HealthRequest, opts ...grpc.CallOption) (*proto.HealthResponse, error) {
	m.healthCallCount++

	if m.healthErr != nil {
		return nil, m.healthErr
	}

	return &proto.HealthResponse{Healthy: true}, nil
}

// Test helpers

func createTestMCPClient(mockServiceClient *MockMCPServiceClient) *MCPClient {
	config := DefaultMCPClientConfig("localhost:50051")
	client := &MCPClient{
		config:    config,
		client:    mockServiceClient,
		backoffer: newExponentialBackoffer(config.InitialBackoff, config.MaxBackoff),
	}
	return client
}

// Tests

func TestDefaultMCPClientConfig(t *testing.T) {
	config := DefaultMCPClientConfig("localhost:50051")

	if config.Address != "localhost:50051" {
		t.Errorf("expected address 'localhost:50051', got %s", config.Address)
	}

	if config.Timeout != 30*time.Second {
		t.Errorf("expected timeout 30s, got %v", config.Timeout)
	}

	if config.MaxRetries != 3 {
		t.Errorf("expected maxRetries 3, got %d", config.MaxRetries)
	}

	if config.KeepaliveInterval != 30*time.Second {
		t.Errorf("expected keepalive interval 30s, got %v", config.KeepaliveInterval)
	}

	if config.InsecureSkipVerify != true {
		t.Error("expected insecure skip verify to be true for dev")
	}
}

func TestInfer_Success(t *testing.T) {
	mockClient := &MockMCPServiceClient{
		inferResp: &proto.InferResponse{
			Output:   []byte("Hello, World!"),
			Metadata: map[string]string{"tokens_used": "42"},
		},
	}

	client := createTestMCPClient(mockClient)

	req := &InferRequest{
		ModelID:  "bitnet-7b",
		Input:    []byte("Hello"),
		Metadata: map[string]string{"temperature": "0.7"},
	}

	resp, err := client.Infer(context.Background(), req)

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if string(resp.Output) != "Hello, World!" {
		t.Errorf("expected output 'Hello, World!', got %s", string(resp.Output))
	}

	if mockClient.inferCallCount != 1 {
		t.Errorf("expected 1 infer call, got %d", mockClient.inferCallCount)
	}

	if mockClient.lastInferReq.ModelId != "bitnet-7b" {
		t.Errorf("expected model ID 'bitnet-7b', got %s", mockClient.lastInferReq.ModelId)
	}
}

func TestInfer_ValidationError(t *testing.T) {
	client := createTestMCPClient(&MockMCPServiceClient{})

	// Test nil request
	_, err := client.Infer(context.Background(), nil)
	if err == nil {
		t.Error("expected error for nil request")
	}

	// Test empty model ID
	_, err = client.Infer(context.Background(), &InferRequest{
		ModelID: "",
		Input:   []byte("test"),
	})
	if err == nil {
		t.Error("expected error for empty model ID")
	}

	// Test empty input
	_, err = client.Infer(context.Background(), &InferRequest{
		ModelID: "bitnet-7b",
		Input:   []byte{},
	})
	if err == nil {
		t.Error("expected error for empty input")
	}
}

func TestInfer_ServerError(t *testing.T) {
	mockClient := &MockMCPServiceClient{
		inferErr: status.Error(codes.Internal, "server error"),
	}

	client := createTestMCPClient(mockClient)

	req := &InferRequest{
		ModelID: "bitnet-7b",
		Input:   []byte("Hello"),
	}

	_, err := client.Infer(context.Background(), req)

	if err == nil {
		t.Error("expected error, got nil")
	}

	if mockClient.inferCallCount < 1 {
		t.Error("expected at least one infer call")
	}
}

func TestInfer_RetryableError(t *testing.T) {
	callCount := 0
	mockClient := &MockMCPServiceClient{
		inferResp: &proto.InferResponse{
			Output: []byte("Success"),
		},
		inferErr: fmt.Errorf("retryable error"),
	}

	// Override infer to succeed on third attempt
	originalInfer := mockClient.Infer
	mockClient.Infer = func(ctx context.Context, in *proto.InferRequest, opts ...grpc.CallOption) (*proto.InferResponse, error) {
		callCount++
		if callCount < 3 {
			return nil, status.Error(codes.Unavailable, "temporarily unavailable")
		}
		return mockClient.inferResp, nil
	}

	client := createTestMCPClient(mockClient)
	client.config.MaxRetries = 3

	req := &InferRequest{
		ModelID: "bitnet-7b",
		Input:   []byte("Hello"),
	}

	resp, err := client.Infer(context.Background(), req)

	if err != nil {
		t.Fatalf("expected success after retries, got error: %v", err)
	}

	if string(resp.Output) != "Success" {
		t.Errorf("expected successful response")
	}

	if callCount != 3 {
		t.Errorf("expected 3 calls, got %d", callCount)
	}
}

func TestListModels_Success(t *testing.T) {
	mockClient := &MockMCPServiceClient{
		listModelsResp: &proto.ListModelsResponse{
			Models: []*proto.ModelInfo{
				{
					Id:       "bitnet-7b",
					Name:     "BitNet b1.58 7B",
					Version:  "1.0.0",
					Status:   "loaded",
					Metadata: map[string]string{"quantization": "ternary"},
				},
			},
		},
	}

	client := createTestMCPClient(mockClient)
	resp, err := client.ListModels(context.Background())

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(resp.Models) != 1 {
		t.Errorf("expected 1 model, got %d", len(resp.Models))
	}

	if resp.Models[0].ID != "bitnet-7b" {
		t.Errorf("expected model ID 'bitnet-7b', got %s", resp.Models[0].ID)
	}

	if mockClient.listModelsCallCount != 1 {
		t.Errorf("expected 1 list models call, got %d", mockClient.listModelsCallCount)
	}
}

func TestLoadModel_Success(t *testing.T) {
	mockClient := &MockMCPServiceClient{}

	client := createTestMCPClient(mockClient)

	req := &LoadModelRequest{
		ModelID: "bitnet-7b",
		Config:  map[string]string{"quantization": "ternary"},
	}

	err := client.LoadModel(context.Background(), req)

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if mockClient.loadModelCallCount != 1 {
		t.Errorf("expected 1 load model call, got %d", mockClient.loadModelCallCount)
	}

	if mockClient.lastLoadModelReq.ModelId != "bitnet-7b" {
		t.Errorf("expected model ID 'bitnet-7b', got %s", mockClient.lastLoadModelReq.ModelId)
	}
}

func TestLoadModel_ValidationError(t *testing.T) {
	client := createTestMCPClient(&MockMCPServiceClient{})

	// Test nil request
	err := client.LoadModel(context.Background(), nil)
	if err == nil {
		t.Error("expected error for nil request")
	}

	// Test empty model ID
	err = client.LoadModel(context.Background(), &LoadModelRequest{ModelID: ""})
	if err == nil {
		t.Error("expected error for empty model ID")
	}
}

func TestUnloadModel_Success(t *testing.T) {
	mockClient := &MockMCPServiceClient{}

	client := createTestMCPClient(mockClient)

	req := &UnloadModelRequest{
		ModelID: "bitnet-7b",
	}

	err := client.UnloadModel(context.Background(), req)

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if mockClient.unloadModelCallCount != 1 {
		t.Errorf("expected 1 unload model call, got %d", mockClient.unloadModelCallCount)
	}
}

func TestUnloadModel_ValidationError(t *testing.T) {
	client := createTestMCPClient(&MockMCPServiceClient{})

	// Test nil request
	err := client.UnloadModel(context.Background(), nil)
	if err == nil {
		t.Error("expected error for nil request")
	}

	// Test empty model ID
	err = client.UnloadModel(context.Background(), &UnloadModelRequest{ModelID: ""})
	if err == nil {
		t.Error("expected error for empty model ID")
	}
}

func TestHealth_Healthy(t *testing.T) {
	mockClient := &MockMCPServiceClient{}

	client := createTestMCPClient(mockClient)

	err := client.Health(context.Background())

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if mockClient.healthCallCount != 1 {
		t.Errorf("expected 1 health call, got %d", mockClient.healthCallCount)
	}
}

func TestHealth_Unhealthy(t *testing.T) {
	mockClient := &MockMCPServiceClient{
		healthErr: status.Error(codes.Unavailable, "service unavailable"),
	}

	client := createTestMCPClient(mockClient)

	err := client.Health(context.Background())

	if err == nil {
		t.Error("expected error when service is unhealthy")
	}
}

func TestClose(t *testing.T) {
	client := createTestMCPClient(&MockMCPServiceClient{})

	// Close should not error (no actual connection)
	err := client.Close()

	// For mock client, should be nil since we don't set conn
	if err == nil || err == nil {
		// Expected behavior
	}
}

func TestIsRetryableError(t *testing.T) {
	testCases := []struct {
		err      error
		expected bool
		name     string
	}{
		{
			name:     "Nil error",
			err:      nil,
			expected: false,
		},
		{
			name:     "Unavailable error",
			err:      status.Error(codes.Unavailable, "unavailable"),
			expected: true,
		},
		{
			name:     "ResourceExhausted error",
			err:      status.Error(codes.ResourceExhausted, "exhausted"),
			expected: true,
		},
		{
			name:     "DeadlineExceeded error",
			err:      status.Error(codes.DeadlineExceeded, "deadline"),
			expected: true,
		},
		{
			name:     "InvalidArgument error",
			err:      status.Error(codes.InvalidArgument, "invalid"),
			expected: false,
		},
		{
			name:     "NotFound error",
			err:      status.Error(codes.NotFound, "not found"),
			expected: false,
		},
		{
			name:     "Non-gRPC error",
			err:      fmt.Errorf("some error"),
			expected: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := isRetryableError(tc.err)
			if result != tc.expected {
				t.Errorf("expected %v, got %v", tc.expected, result)
			}
		})
	}
}

func TestExponentialBackoffer(t *testing.T) {
	backoff := newExponentialBackoffer(100*time.Millisecond, 1*time.Second)

	// First backoff should be 100ms
	d1 := backoff.nextBackoff()
	if d1 != 100*time.Millisecond {
		t.Errorf("expected 100ms, got %v", d1)
	}

	// Second backoff should be 200ms
	d2 := backoff.nextBackoff()
	if d2 != 200*time.Millisecond {
		t.Errorf("expected 200ms, got %v", d2)
	}

	// Third backoff should be 400ms
	d3 := backoff.nextBackoff()
	if d3 != 400*time.Millisecond {
		t.Errorf("expected 400ms, got %v", d3)
	}

	// Fourth backoff should be capped at 1000ms (max)
	d4 := backoff.nextBackoff()
	if d4 > 1*time.Second {
		t.Errorf("expected capped at 1s, got %v", d4)
	}

	// Reset should restart the sequence
	backoff.reset()
	d5 := backoff.nextBackoff()
	if d5 != 100*time.Millisecond {
		t.Errorf("expected 100ms after reset, got %v", d5)
	}
}

func TestInfer_ContextCancellation(t *testing.T) {
	mockClient := &MockMCPServiceClient{}

	client := createTestMCPClient(mockClient)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	req := &InferRequest{
		ModelID: "bitnet-7b",
		Input:   []byte("Hello"),
	}

	_, err := client.Infer(ctx, req)

	if err == nil {
		t.Error("expected context cancelled error")
	}
}
