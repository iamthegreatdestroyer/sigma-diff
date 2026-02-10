package main

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"

	pb "github.com/iamthegreatdestroyer/Ryzanstein/mcp/proto"
)

// ============================================================================
// Test Utilities
// ============================================================================

type testContext struct {
	t                *testing.T
	inferenceConn    *grpc.ClientConn
	agentConn        *grpc.ClientConn
	memoryConn       *grpc.ClientConn
	optimizationConn *grpc.ClientConn
	debugConn        *grpc.ClientConn
}

func setupTest(t *testing.T) *testContext {
	ctx := &testContext{t: t}

	// Connect to inference service
	conn, err := grpc.Dial("localhost:8001", grpc.WithInsecure())
	require.NoError(t, err)
	ctx.inferenceConn = conn

	// Connect to pb.Agent service
	conn, err = grpc.Dial("localhost:8002", grpc.WithInsecure())
	require.NoError(t, err)
	ctx.agentConn = conn

	// Connect to memory service
	conn, err = grpc.Dial("localhost:8003", grpc.WithInsecure())
	require.NoError(t, err)
	ctx.memoryConn = conn

	// Connect to optimization service
	conn, err = grpc.Dial("localhost:8004", grpc.WithInsecure())
	require.NoError(t, err)
	ctx.optimizationConn = conn

	// Connect to debug service
	conn, err = grpc.Dial("localhost:8005", grpc.WithInsecure())
	require.NoError(t, err)
	ctx.debugConn = conn

	return ctx
}

func (ctx *testContext) cleanup() {
	if ctx.inferenceConn != nil {
		ctx.inferenceConn.Close()
	}
	if ctx.agentConn != nil {
		ctx.agentConn.Close()
	}
	if ctx.memoryConn != nil {
		ctx.memoryConn.Close()
	}
	if ctx.optimizationConn != nil {
		ctx.optimizationConn.Close()
	}
	if ctx.debugConn != nil {
		ctx.debugConn.Close()
	}
}

// ============================================================================
// Inference Service Tests
// ============================================================================

func TestInferenceServiceBasic(t *testing.T) {
	ctx := setupTest(t)
	defer ctx.cleanup()

	client := pb.NewInferenceServiceClient(ctx.inferenceConn)
	timeout, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req := &pb.InferenceRequest{
		Metadata: &pb.RequestMetadata{
			RequestId: "test_001",
			ClientId:  "test",
		},
		Model:       "ryzanstein-7b",
		Temperature: 0.7,
		MaxTokens:   100,
	}

	resp, err := client.Infer(timeout, req)
	require.NoError(t, err)

	assert.Equal(t, "test_001", resp.Metadata.RequestId)
	assert.Equal(t, int32(200), resp.Metadata.StatusCode)
	assert.True(t, len(resp.Content) > 0)
	assert.Greater(t, resp.TokensUsed, int32(0))
}

func TestInferenceServiceStreaming(t *testing.T) {
	ctx := setupTest(t)
	defer ctx.cleanup()

	client := pb.NewInferenceServiceClient(ctx.inferenceConn)
	timeout, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req := &pb.InferenceRequest{
		Metadata: &pb.RequestMetadata{
			RequestId: "test_stream_001",
			ClientId:  "test",
		},
		Model:  "ryzanstein-7b",
		Stream: true,
	}

	stream, err := client.InferStream(timeout, req)
	require.NoError(t, err)

	chunkCount := 0
	for {
		chunk, err := stream.Recv()
		if err != nil {
			break
		}
		chunkCount++
		assert.True(t, len(chunk.Content) > 0)
	}

	assert.Greater(t, chunkCount, 0)
}

func TestInferenceServiceHealth(t *testing.T) {
	ctx := setupTest(t)
	defer ctx.cleanup()

	client := pb.NewInferenceServiceClient(ctx.inferenceConn)
	timeout, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req := &pb.HealthRequest{
		Metadata: &pb.RequestMetadata{
			RequestId: "health_001",
			ClientId:  "test",
		},
	}

	resp, err := client.Health(timeout, req)
	require.NoError(t, err)

	assert.True(t, resp.Healthy)
	assert.NotEmpty(t, resp.Version)
}

func TestInferenceServiceModelInfo(t *testing.T) {
	ctx := setupTest(t)
	defer ctx.cleanup()

	client := pb.NewInferenceServiceClient(ctx.inferenceConn)
	timeout, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req := &pb.ModelInfoRequest{
		Metadata: &pb.RequestMetadata{
			RequestId: "info_001",
			ClientId:  "test",
		},
		Model: "ryzanstein-7b",
	}

	resp, err := client.GetModelInfo(timeout, req)
	require.NoError(t, err)

	assert.Equal(t, "ryzanstein-7b", resp.Model)
	assert.Greater(t, resp.ContextLength, int32(0))
	assert.Greater(t, resp.MaxTokens, int32(0))
}

// ============================================================================
// pb.Agent Service Tests
// ============================================================================

func TestAgentServiceRegister(t *testing.T) {
	ctx := setupTest(t)
	defer ctx.cleanup()

	client := pb.NewAgentServiceClient(ctx.agentConn)
	timeout, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	agent := &pb.Agent{
		Codename:     "@TEST_AGENT",
		Name:         "Test Agent",
		Tier:         1,
		Philosophy:   "Testing is important",
		Capabilities: []string{"testing", "validation"},
	}

	tools := []*pb.Tool{
		{
			Name:        "test_tool",
			Description: "A test tool",
		},
	}

	req := &pb.RegisterAgentRequest{
		Metadata: &pb.RequestMetadata{
			RequestId: "reg_001",
			ClientId:  "test",
		},
		Agent: agent,
		Tools: tools,
	}

	resp, err := client.RegisterAgent(timeout, req)
	require.NoError(t, err)

	assert.True(t, resp.Success)
	assert.NotEmpty(t, resp.AgentId)
}

func TestAgentServiceList(t *testing.T) {
	ctx := setupTest(t)
	defer ctx.cleanup()

	client := pb.NewAgentServiceClient(ctx.agentConn)
	timeout, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req := &pb.ListAgentsRequest{
		Metadata: &pb.RequestMetadata{
			RequestId: "list_001",
			ClientId:  "test",
		},
	}

	resp, err := client.ListAgents(timeout, req)
	require.NoError(t, err)

	assert.GreaterOrEqual(t, resp.TotalCount, int32(0))
}

func TestAgentServiceListTools(t *testing.T) {
	ctx := setupTest(t)
	defer ctx.cleanup()

	client := pb.NewAgentServiceClient(ctx.agentConn)
	timeout, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req := &pb.ListToolsRequest{
		Metadata: &pb.RequestMetadata{
			RequestId: "tools_001",
			ClientId:  "test",
		},
		AgentId: "@APEX",
	}

	resp, err := client.ListTools(timeout, req)
	require.NoError(t, err)

	// Should have tools or be empty initially
	assert.GreaterOrEqual(t, len(resp.Tools), 0)
}

// ============================================================================
// Memory Service Tests
// ============================================================================

func TestMemoryServiceStore(t *testing.T) {
	ctx := setupTest(t)
	defer ctx.cleanup()

	client := pb.NewMemoryServiceClient(ctx.memoryConn)
	timeout, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	experience := &pb.Experience{
		Agent:        "@APEX",
		Task:         "code_review",
		Input:        "Review this code",
		Output:       "Code looks good",
		Strategy:     "pattern_matching",
		Embedding:    []float32{0.1, 0.2, 0.3},
		FitnessScore: 0.85,
	}

	req := &pb.StoreExperienceRequest{
		Metadata: &pb.RequestMetadata{
			RequestId: "store_001",
			ClientId:  "test",
		},
		Experience: experience,
	}

	resp, err := client.StoreExperience(timeout, req)
	require.NoError(t, err)

	assert.NotEmpty(t, resp.ExperienceId)
}

func TestMemoryServiceRetrieve(t *testing.T) {
	ctx := setupTest(t)
	defer ctx.cleanup()

	client := pb.NewMemoryServiceClient(ctx.memoryConn)
	timeout, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req := &pb.RetrieveExperienceRequest{
		Metadata: &pb.RequestMetadata{
			RequestId: "retrieve_001",
			ClientId:  "test",
		},
		QueryEmbedding: []float32{0.1, 0.2, 0.3},
		Limit:          10,
	}

	resp, err := client.RetrieveExperience(timeout, req)
	require.NoError(t, err)

	assert.GreaterOrEqual(t, resp.AverageSimilarity, float32(0))
}

func TestMemoryServiceUpdateFitness(t *testing.T) {
	ctx := setupTest(t)
	defer ctx.cleanup()

	client := pb.NewMemoryServiceClient(ctx.memoryConn)
	timeout, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// First store an experience
	experience := &pb.Experience{
		Agent:        "@APEX",
		Task:         "test",
		Input:        "test",
		Output:       "test",
		FitnessScore: 0.5,
	}

	storeReq := &pb.StoreExperienceRequest{
		Metadata: &pb.RequestMetadata{
			RequestId: "store_for_update",
			ClientId:  "test",
		},
		Experience: experience,
	}

	storeResp, err := client.StoreExperience(timeout, storeReq)
	require.NoError(t, err)

	// Now update fitness
	updateReq := &pb.UpdateFitnessRequest{
		Metadata: &pb.RequestMetadata{
			RequestId: "update_001",
			ClientId:  "test",
		},
		ExperienceId:    storeResp.ExperienceId,
		NewFitnessScore: 0.95,
	}

	updateResp, err := client.UpdateFitness(timeout, updateReq)
	require.NoError(t, err)

	assert.True(t, updateResp.Success)
	assert.Equal(t, float32(0.95), updateResp.UpdatedScore)
}

func TestMemoryServiceStats(t *testing.T) {
	ctx := setupTest(t)
	defer ctx.cleanup()

	client := NewMemoryServiceClient(ctx.memoryConn)
	timeout, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req := &MemoryStatsRequest{
		Metadata: &pb.RequestMetadata{
			RequestId: "stats_001",
			ClientId:  "test",
		},
	}

	resp, err := client.GetMemoryStats(timeout, req)
	require.NoError(t, err)

	assert.GreaterOrEqual(t, resp.TotalExperiences, int32(0))
	assert.GreaterOrEqual(t, resp.pb.AgentCount, int32(0))
}

// ============================================================================
// Optimization Service Tests
// ============================================================================

func TestOptimizationServiceMetrics(t *testing.T) {
	ctx := setupTest(t)
	defer ctx.cleanup()

	client := NewOptimizationServiceClient(ctx.optimizationConn)
	timeout, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req := &MetricsRequest{
		Metadata: &pb.RequestMetadata{
			RequestId: "metrics_001",
			ClientId:  "test",
		},
		Component: "inference",
	}

	resp, err := client.CollectMetrics(timeout, req)
	require.NoError(t, err)

	assert.NotEmpty(t, resp.Metrics)
	assert.NotNil(t, resp.Timestamp)
}

func TestOptimizationServiceSuggestions(t *testing.T) {
	ctx := setupTest(t)
	defer ctx.cleanup()

	client := NewOptimizationServiceClient(ctx.optimizationConn)
	timeout, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req := &OptimizationRequest{
		Metadata: &pb.RequestMetadata{
			RequestId: "opt_001",
			ClientId:  "test",
		},
		Component: "inference",
		CurrentMetrics: map[string]double{
			"throughput": 1000.0,
			"latency":    200.0,
		},
	}

	resp, err := client.GetOptimizationSuggestions(timeout, req)
	require.NoError(t, err)

	assert.GreaterOrEqual(t, len(resp.Suggestions), 0)
}

func TestOptimizationServiceProfile(t *testing.T) {
	ctx := setupTest(t)
	defer ctx.cleanup()

	client := NewOptimizationServiceClient(ctx.optimizationConn)
	timeout, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req := &ProfileRequest{
		Metadata: &pb.RequestMetadata{
			RequestId: "profile_001",
			ClientId:  "test",
		},
		DurationSeconds: 2,
		Components:      []string{"inference", "memory"},
	}

	stream, err := client.ProfilePerformance(timeout, req)
	require.NoError(t, err)

	metricsCount := 0
	for {
		metric, err := stream.Recv()
		if err != nil {
			break
		}
		metricsCount++
		assert.NotEmpty(t, metric.Values)
	}

	assert.Greater(t, metricsCount, 0)
}

func TestOptimizationServiceHealth(t *testing.T) {
	ctx := setupTest(t)
	defer ctx.cleanup()

	client := NewOptimizationServiceClient(ctx.optimizationConn)
	timeout, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req := &pb.HealthRequest{
		Metadata: &pb.RequestMetadata{
			RequestId: "health_opt_001",
			ClientId:  "test",
		},
	}

	resp, err := client.GetSystemHealth(timeout, req)
	require.NoError(t, err)

	assert.Greater(t, resp.OverallHealth, float32(0))
	assert.Less(t, resp.OverallHealth, float32(1.1))
}

// ============================================================================
// Debug Service Tests
// ============================================================================

func TestDebugServiceInspect(t *testing.T) {
	ctx := setupTest(t)
	defer ctx.cleanup()

	client := NewDebugServiceClient(ctx.debugConn)
	timeout, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req := &InspectRequest{
		Metadata: &pb.RequestMetadata{
			RequestId: "inspect_001",
			ClientId:  "test",
		},
		Component:   "inference",
		InspectType: "state",
	}

	resp, err := client.InspectComponent(timeout, req)
	require.NoError(t, err)

	assert.Greater(t, len(resp.DebugInfo), 0)
}

func TestDebugServiceDiagnostics(t *testing.T) {
	ctx := setupTest(t)
	defer ctx.cleanup()

	client := NewDebugServiceClient(ctx.debugConn)
	timeout, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req := &DiagnosticsRequest{
		Metadata: &pb.RequestMetadata{
			RequestId: "diag_001",
			ClientId:  "test",
		},
		Scope: "system",
	}

	resp, err := client.GetDiagnostics(timeout, req)
	require.NoError(t, err)

	assert.GreaterOrEqual(t, len(resp.Diagnostics), 0)
}

func TestDebugServiceTracePath(t *testing.T) {
	ctx := setupTest(t)
	defer ctx.cleanup()

	client := NewDebugServiceClient(ctx.debugConn)
	timeout, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req := &TraceRequest{
		Metadata: &pb.RequestMetadata{
			RequestId: "trace_001",
			ClientId:  "test",
		},
		Operation: "inference",
	}

	stream, err := client.TracePath(timeout, req)
	require.NoError(t, err)

	eventCount := 0
	for {
		event, err := stream.Recv()
		if err != nil {
			break
		}
		eventCount++
		assert.NotEmpty(t, event.Component)
	}

	assert.Greater(t, eventCount, 0)
}

// ============================================================================
// Integration Tests
// ============================================================================

func TestConcurrentRequests(t *testing.T) {
	ctx := setupTest(t)
	defer ctx.cleanup()

	inferenceClient := pb.NewInferenceServiceClient(ctx.inferenceConn)
	agentClient := pb.NewAgentServiceClient(ctx.agentConn)
	memoryClient := NewMemoryServiceClient(ctx.memoryConn)

	timeout, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Run concurrent requests
	errChan := make(chan error, 30)
	for i := 0; i < 10; i++ {
		go func(idx int) {
			req := &pb.InferenceRequest{
				Metadata: &pb.RequestMetadata{
					RequestId: fmt.Sprintf("concurrent_%d", idx),
					ClientId:  "test",
				},
				Model: "ryzanstein-7b",
			}
			_, err := inferenceClient.Infer(timeout, req)
			errChan <- err
		}(i)

		go func(idx int) {
			req := &Listpb.AgentsRequest{
				Metadata: &pb.RequestMetadata{
					RequestId: fmt.Sprintf("pb.Agents_%d", idx),
					ClientId:  "test",
				},
			}
			_, err := pb.AgentClient.Listpb.Agents(timeout, req)
			errChan <- err
		}(i)

		go func(idx int) {
			req := &MemoryStatsRequest{
				Metadata: &pb.RequestMetadata{
					RequestId: fmt.Sprintf("stats_%d", idx),
					ClientId:  "test",
				},
			}
			_, err := memoryClient.GetMemoryStats(timeout, req)
			errChan <- err
		}(i)
	}

	// Check for errors
	errorCount := 0
	for i := 0; i < 30; i++ {
		if err := <-errChan; err != nil {
			errorCount++
			t.Logf("Request error: %v", err)
		}
	}

	assert.Equal(t, 0, errorCount, "Should have no errors")
}

// ============================================================================
// Benchmark Tests
// ============================================================================

func BenchmarkInferenceRequest(b *testing.B) {
	ctx := setupTest(&testing.T{})
	defer ctx.cleanup()

	client := pb.NewInferenceServiceClient(ctx.inferenceConn)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		timeout, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		req := &pb.InferenceRequest{
			Metadata: &pb.RequestMetadata{
				RequestId: fmt.Sprintf("bench_%d", i),
				ClientId:  "test",
			},
			Model: "ryzanstein-7b",
		}
		client.Infer(timeout, req)
		cancel()
	}
}

func BenchmarkAgentRegistration(b *testing.B) {
	ctx := setupTest(&testing.T{})
	defer ctx.cleanup()

	client := pb.NewAgentServiceClient(ctx.agentConn)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		timeout, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		req := &Listpb.AgentsRequest{
			Metadata: &pb.RequestMetadata{
				RequestId: fmt.Sprintf("bench_pb.Agent_%d", i),
				ClientId:  "test",
			},
		}
		client.Listpb.Agents(timeout, req)
		cancel()
	}
}
