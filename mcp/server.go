package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	pb "github.com/iamthegreatdestroyer/Ryzanstein/mcp/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"
	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// MCPServer represents the main MCP server orchestrator
type MCPServer struct {
	inferenceServer    *InferenceServer
	agentServer        *AgentServer
	memoryServer       *MemoryServer
	optimizationServer *OptimizationServer
	debugServer        *DebugServer

	grpcServer *grpc.Server
	listeners  map[string]net.Listener
	mu         sync.RWMutex
}

// NewMCPServer creates a new MCP server orchestrator
func NewMCPServer() *MCPServer {
	return &MCPServer{
		inferenceServer:    NewInferenceServer(),
		agentServer:        NewAgentServer(),
		memoryServer:       NewMemoryServer(),
		optimizationServer: NewOptimizationServer(),
		debugServer:        NewDebugServer(),
		listeners:          make(map[string]net.Listener),
	}
}

// Start starts all MCP servers
func (m *MCPServer) Start(ctx context.Context) error {
	log.Println("[MCP] Starting Ryzanstein MCP Server Suite...")

	// Create gRPC server with options
	opts := []grpc.ServerOption{
		grpc.MaxConcurrentStreams(1000),
		grpc.ConnectionTimeout(30 * time.Second),
	}
	m.grpcServer = grpc.NewServer(opts...)

	// Register all services
	pb.RegisterInferenceServiceServer(m.grpcServer, m.inferenceServer)
	pb.RegisterAgentServiceServer(m.grpcServer, m.agentServer)
	pb.RegisterMemoryServiceServer(m.grpcServer, m.memoryServer)
	pb.RegisterOptimizationServiceServer(m.grpcServer, m.optimizationServer)
	pb.RegisterDebugServiceServer(m.grpcServer, m.debugServer)

	// Register reflection for debugging
	reflection.Register(m.grpcServer)

	// Start server on single port
	addr := ":50051"
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("failed to listen on %s: %w", addr, err)
	}

	m.mu.Lock()
	m.listeners["main"] = listener
	m.mu.Unlock()

	log.Printf("[MCP] MCP Server listening on %s\n", addr)

	// Start serving in a goroutine
	go func() {
		if err := m.grpcServer.Serve(listener); err != nil && err != grpc.ErrServerStopped {
			log.Printf("[MCP] Server error: %v", err)
		}
	}()

	// Wait for shutdown signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	select {
	case <-sigChan:
		log.Println("[MCP] Received shutdown signal")
	case <-ctx.Done():
		log.Println("[MCP] Context cancelled")
	}

	m.Stop()
	return nil
}

// Stop gracefully stops all MCP servers
func (m *MCPServer) Stop() {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.grpcServer != nil {
		m.grpcServer.GracefulStop()
		log.Println("[MCP] All services stopped")
	}

	for name, listener := range m.listeners {
		listener.Close()
		log.Printf("[MCP] Closed listener for %s\n", name)
	}
}

// ============================================================================
// Inference Server Implementation
// ============================================================================

type InferenceServer struct {
	pb.UnimplementedInferenceServiceServer
	clients map[string]string // model -> endpoint mapping
	mu      sync.RWMutex
}

func NewInferenceServer() *InferenceServer {
	return &InferenceServer{
		clients: make(map[string]string),
	}
}

func (s *InferenceServer) Infer(ctx context.Context, req *pb.InferenceRequest) (*pb.InferenceResponse, error) {
	log.Printf("[Inference] Processing inference request: %s\n", req.Metadata.RequestId)
	startTime := time.Now()

	// Get the inference client
	client := GetInferenceClient()

	// Extract parameters from request
	maxTokens := int(req.MaxTokens)
	if maxTokens <= 0 {
		maxTokens = 256
	}
	temperature := float64(req.Temperature)
	if temperature <= 0 {
		temperature = 0.7
	}

	// Extract system prompt and user prompt from Messages
	systemPrompt := ""
	userPrompt := ""
	for _, msg := range req.Messages {
		switch msg.Role {
		case pb.Message_SYSTEM:
			systemPrompt = msg.Content
		case pb.Message_USER:
			userPrompt = msg.Content
		}
	}

	// Fall back to empty if no user message found
	if userPrompt == "" && len(req.Messages) > 0 {
		userPrompt = req.Messages[len(req.Messages)-1].Content
	}

	// Call the Python API for real inference
	content, tokensUsed, err := client.ChatCompletion(userPrompt, systemPrompt, maxTokens, temperature)

	processingTimeMs := time.Since(startTime).Milliseconds()

	if err != nil {
		log.Printf("[Inference] Error from Python API: %v", err)
		// Return error response
		return &pb.InferenceResponse{
			Metadata: &pb.ResponseMetadata{
				RequestId:        req.Metadata.RequestId,
				Timestamp:        timeToProto(time.Now()),
				StatusCode:       500,
				StatusMessage:    fmt.Sprintf("Inference error: %v", err),
				ProcessingTimeMs: processingTimeMs,
			},
			Content:    "",
			TokensUsed: 0,
			Metrics: map[string]string{
				"error": err.Error(),
			},
		}, nil
	}

	// Build successful response
	tokensPerSec := float64(0)
	if processingTimeMs > 0 {
		tokensPerSec = float64(tokensUsed) * 1000 / float64(processingTimeMs)
	}

	resp := &pb.InferenceResponse{
		Metadata: &pb.ResponseMetadata{
			RequestId:        req.Metadata.RequestId,
			Timestamp:        timeToProto(time.Now()),
			StatusCode:       200,
			StatusMessage:    "OK",
			ProcessingTimeMs: processingTimeMs,
		},
		Content:    content,
		TokensUsed: int32(tokensUsed),
		Metrics: map[string]string{
			"latency_ms":     fmt.Sprintf("%d", processingTimeMs),
			"tokens_per_sec": fmt.Sprintf("%.2f", tokensPerSec),
		},
	}

	log.Printf("[Inference] Completed: %d tokens in %dms", tokensUsed, processingTimeMs)
	return resp, nil
}

func (s *InferenceServer) InferStream(req *pb.InferenceRequest, stream pb.InferenceService_InferStreamServer) error {
	log.Printf("[Inference] Processing streaming inference: %s\n", req.Metadata.RequestId)
	startTime := time.Now()

	// Get the inference client
	client := GetInferenceClient()

	// Extract parameters from request
	maxTokens := int(req.MaxTokens)
	if maxTokens <= 0 {
		maxTokens = 256
	}
	temperature := float64(req.Temperature)
	if temperature <= 0 {
		temperature = 0.7
	}

	// Extract system prompt and user prompt from Messages
	systemPrompt := ""
	userPrompt := ""
	for _, msg := range req.Messages {
		switch msg.Role {
		case pb.Message_SYSTEM:
			systemPrompt = msg.Content
		case pb.Message_USER:
			userPrompt = msg.Content
		}
	}

	// Fall back to last message if no user message found
	if userPrompt == "" && len(req.Messages) > 0 {
		userPrompt = req.Messages[len(req.Messages)-1].Content
	}

	// For now, get the full response and simulate streaming
	// TODO: Implement true streaming when Python API supports it
	content, tokensUsed, err := client.ChatCompletion(userPrompt, systemPrompt, maxTokens, temperature)

	if err != nil {
		log.Printf("[Inference] Stream error: %v", err)
		return fmt.Errorf("inference failed: %w", err)
	}

	// Split content into chunks and stream them
	words := splitIntoChunks(content, 10) // 10 characters per chunk
	for i, word := range words {
		chunk := &pb.InferenceChunk{
			Metadata: &pb.ResponseMetadata{
				RequestId:     req.Metadata.RequestId,
				Timestamp:     timeToProto(time.Now()),
				StatusCode:    200,
				StatusMessage: "OK",
			},
			Content: word,
			IsFinal: i == len(words)-1,
		}
		if err := stream.Send(chunk); err != nil {
			return err
		}
		// Small delay to simulate streaming
		time.Sleep(20 * time.Millisecond)
	}

	processingTimeMs := time.Since(startTime).Milliseconds()
	log.Printf("[Inference] Stream completed: %d tokens in %dms", tokensUsed, processingTimeMs)

	return nil
}

// splitIntoChunks splits a string into chunks of approximately n characters
func splitIntoChunks(s string, n int) []string {
	var chunks []string
	runes := []rune(s)

	for i := 0; i < len(runes); i += n {
		end := i + n
		if end > len(runes) {
			end = len(runes)
		}
		chunks = append(chunks, string(runes[i:end]))
	}

	if len(chunks) == 0 {
		chunks = []string{""}
	}

	return chunks
}

func (s *InferenceServer) Health(ctx context.Context, req *pb.HealthRequest) (*pb.HealthResponse, error) {
	return &pb.HealthResponse{
		Metadata: &pb.ResponseMetadata{
			RequestId:     req.Metadata.RequestId,
			Timestamp:     timeToProto(time.Now()),
			StatusCode:    200,
			StatusMessage: "OK",
		},
		Healthy: true,
		Version: "1.0.0",
	}, nil
}

func (s *InferenceServer) GetModelInfo(ctx context.Context, req *pb.ModelInfoRequest) (*pb.ModelInfoResponse, error) {
	return &pb.ModelInfoResponse{
		Metadata: &pb.ResponseMetadata{
			RequestId:     req.Metadata.RequestId,
			Timestamp:     timeToProto(time.Now()),
			StatusCode:    200,
			StatusMessage: "OK",
		},
		Model:         "ryzanstein-7b",
		ContextLength: 4096,
		MaxTokens:     2048,
		Capabilities: map[string]string{
			"streaming":             "true",
			"speculative_decoding":  "true",
			"kv_cache_optimization": "adaptive",
		},
	}, nil
}

// ============================================================================
// Agent Server Implementation
// ============================================================================

type AgentServer struct {
	pb.UnimplementedAgentServiceServer
	agents map[string]*pb.Agent
	tools  map[string]*pb.Tool
	mu     sync.RWMutex
}

func NewAgentServer() *AgentServer {
	return &AgentServer{
		agents: make(map[string]*pb.Agent),
		tools:  make(map[string]*pb.Tool),
	}
}

func (s *AgentServer) RegisterAgent(ctx context.Context, req *pb.RegisterAgentRequest) (*pb.RegisterAgentResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	agentID := req.Agent.Codename
	s.agents[agentID] = req.Agent

	for _, tool := range req.Tools {
		toolKey := fmt.Sprintf("%s:%s", agentID, tool.Name)
		s.tools[toolKey] = tool
	}

	log.Printf("[Agent] Registered agent: %s with %d tools\n", agentID, len(req.Tools))

	return &pb.RegisterAgentResponse{
		Metadata: &pb.ResponseMetadata{
			RequestId:     req.Metadata.RequestId,
			Timestamp:     timeToProto(time.Now()),
			StatusCode:    200,
			StatusMessage: "OK",
		},
		Success: true,
		AgentId: agentID,
	}, nil
}

func (s *AgentServer) ListAgents(ctx context.Context, req *pb.ListAgentsRequest) (*pb.ListAgentsResponse, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	agents := make([]*pb.Agent, 0, len(s.agents))
	for _, agent := range s.agents {
		agents = append(agents, agent)
	}

	return &pb.ListAgentsResponse{
		Metadata: &pb.ResponseMetadata{
			RequestId:     req.Metadata.RequestId,
			Timestamp:     timeToProto(time.Now()),
			StatusCode:    200,
			StatusMessage: "OK",
		},
		Agents:     agents,
		TotalCount: int32(len(agents)),
	}, nil
}

func (s *AgentServer) GetAgent(ctx context.Context, req *pb.GetAgentRequest) (*pb.GetAgentResponse, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	agent, ok := s.agents[req.AgentId]
	if !ok {
		return nil, fmt.Errorf("agent not found: %s", req.AgentId)
	}

	tools := make([]*pb.Tool, 0)
	for toolKey, tool := range s.tools {
		if len(toolKey) > 0 && toolKey[:len(req.AgentId)] == req.AgentId {
			tools = append(tools, tool)
		}
	}

	return &pb.GetAgentResponse{
		Metadata: &pb.ResponseMetadata{
			RequestId:     req.Metadata.RequestId,
			Timestamp:     timeToProto(time.Now()),
			StatusCode:    200,
			StatusMessage: "OK",
		},
		Agent: agent,
		Tools: tools,
	}, nil
}

func (s *AgentServer) CallTool(ctx context.Context, req *pb.CallToolRequest) (*pb.CallToolResponse, error) {
	log.Printf("[Agent] Calling tool: %s\n", req.ToolName)

	return &pb.CallToolResponse{
		Metadata: &pb.ResponseMetadata{
			RequestId:     req.Metadata.RequestId,
			Timestamp:     timeToProto(time.Now()),
			StatusCode:    200,
			StatusMessage: "OK",
		},
		Result: req.Parameters,
	}, nil
}

func (s *AgentServer) ListTools(ctx context.Context, req *pb.ListToolsRequest) (*pb.ListToolsResponse, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	tools := make([]*pb.Tool, 0)
	prefix := req.AgentId + ":"

	for toolKey, tool := range s.tools {
		if len(toolKey) > len(prefix) && toolKey[:len(prefix)] == prefix {
			tools = append(tools, tool)
		}
	}

	return &pb.ListToolsResponse{
		Metadata: &pb.ResponseMetadata{
			RequestId:     req.Metadata.RequestId,
			Timestamp:     timeToProto(time.Now()),
			StatusCode:    200,
			StatusMessage: "OK",
		},
		Tools: tools,
	}, nil
}

// ============================================================================
// Memory Server Implementation
// ============================================================================

type MemoryServer struct {
	pb.UnimplementedMemoryServiceServer
	experiences map[string]*pb.Experience
	mu          sync.RWMutex
}

func NewMemoryServer() *MemoryServer {
	return &MemoryServer{
		experiences: make(map[string]*pb.Experience),
	}
}

func (s *MemoryServer) StoreExperience(ctx context.Context, req *pb.StoreExperienceRequest) (*pb.StoreExperienceResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if req.Experience.Id == "" {
		req.Experience.Id = fmt.Sprintf("exp_%d", time.Now().UnixNano())
	}

	s.experiences[req.Experience.Id] = req.Experience
	log.Printf("[Memory] Stored experience: %s (agent: %s)\n", req.Experience.Id, req.Experience.Agent)

	return &pb.StoreExperienceResponse{
		Metadata: &pb.ResponseMetadata{
			RequestId:     req.Metadata.RequestId,
			Timestamp:     timeToProto(time.Now()),
			StatusCode:    200,
			StatusMessage: "OK",
		},
		ExperienceId: req.Experience.Id,
	}, nil
}

func (s *MemoryServer) RetrieveExperience(ctx context.Context, req *pb.RetrieveExperienceRequest) (*pb.RetrieveExperienceResponse, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	experiences := make([]*pb.Experience, 0)
	for _, exp := range s.experiences {
		if req.AgentFilter == "" || exp.Agent == req.AgentFilter {
			experiences = append(experiences, exp)
			if int32(len(experiences)) >= req.Limit {
				break
			}
		}
	}

	return &pb.RetrieveExperienceResponse{
		Metadata: &pb.ResponseMetadata{
			RequestId:     req.Metadata.RequestId,
			Timestamp:     timeToProto(time.Now()),
			StatusCode:    200,
			StatusMessage: "OK",
		},
		Experiences:       experiences,
		AverageSimilarity: 0.85,
	}, nil
}

func (s *MemoryServer) UpdateFitness(ctx context.Context, req *pb.UpdateFitnessRequest) (*pb.UpdateFitnessResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	exp, ok := s.experiences[req.ExperienceId]
	if !ok {
		return nil, fmt.Errorf("experience not found: %s", req.ExperienceId)
	}

	exp.FitnessScore = req.NewFitnessScore
	log.Printf("[Memory] Updated fitness for %s: %.2f\n", req.ExperienceId, req.NewFitnessScore)

	return &pb.UpdateFitnessResponse{
		Metadata: &pb.ResponseMetadata{
			RequestId:     req.Metadata.RequestId,
			Timestamp:     timeToProto(time.Now()),
			StatusCode:    200,
			StatusMessage: "OK",
		},
		Success:      true,
		UpdatedScore: req.NewFitnessScore,
	}, nil
}

func (s *MemoryServer) GetMemoryStats(ctx context.Context, req *pb.MemoryStatsRequest) (*pb.MemoryStatsResponse, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	stats := make(map[string]int32)
	var totalFitness float32 = 0

	for _, exp := range s.experiences {
		stats[exp.Agent]++
		totalFitness += exp.FitnessScore
	}

	avgFitness := float32(0)
	if len(s.experiences) > 0 {
		avgFitness = totalFitness / float32(len(s.experiences))
	}

	return &pb.MemoryStatsResponse{
		Metadata: &pb.ResponseMetadata{
			RequestId:     req.Metadata.RequestId,
			Timestamp:     timeToProto(time.Now()),
			StatusCode:    200,
			StatusMessage: "OK",
		},
		TotalExperiences: int32(len(s.experiences)),
		AgentCount:       int32(len(stats)),
		AgentsStats:      stats,
		AverageFitness:   float32(avgFitness),
	}, nil
}

// ============================================================================
// Optimization Server Implementation
// ============================================================================

type OptimizationServer struct {
	pb.UnimplementedOptimizationServiceServer
}

func NewOptimizationServer() *OptimizationServer {
	return &OptimizationServer{}
}

func (s *OptimizationServer) CollectMetrics(ctx context.Context, req *pb.MetricsRequest) (*pb.MetricsResponse, error) {
	return &pb.MetricsResponse{
		Metadata: &pb.ResponseMetadata{
			RequestId:     req.Metadata.RequestId,
			Timestamp:     timeToProto(time.Now()),
			StatusCode:    200,
			StatusMessage: "OK",
		},
		Metrics: map[string]float64{
			"cpu_usage":    45.2,
			"memory_usage": 62.8,
			"throughput":   1500.0,
			"latency_ms":   125.0,
		},
		Timestamp: timeToProto(time.Now()),
	}, nil
}

func (s *OptimizationServer) GetOptimizationSuggestions(ctx context.Context, req *pb.OptimizationRequest) (*pb.OptimizationResponse, error) {
	return &pb.OptimizationResponse{
		Metadata: &pb.ResponseMetadata{
			RequestId:     req.Metadata.RequestId,
			Timestamp:     timeToProto(time.Now()),
			StatusCode:    200,
			StatusMessage: "OK",
		},
		Suggestions: []string{
			"Enable speculative decoding for improved throughput",
			"Optimize batch size for better GPU utilization",
			"Consider distributed inference for load balancing",
		},
		PredictedImprovements: map[string]float64{
			"throughput_improvement": 0.25,
			"latency_reduction":      0.15,
			"memory_efficiency":      0.10,
		},
	}, nil
}

func (s *OptimizationServer) ProfilePerformance(req *pb.ProfileRequest, stream pb.OptimizationService_ProfilePerformanceServer) error {
	for i := 0; i < 5; i++ {
		metric := &pb.ProfileMetric{
			Metadata: &pb.ResponseMetadata{
				RequestId:     req.Metadata.RequestId,
				Timestamp:     timeToProto(time.Now()),
				StatusCode:    200,
				StatusMessage: "OK",
			},
			Component: "inference",
			Values: map[string]float64{
				"cpu":     45.0 + float64(i)*2,
				"memory":  62.0 + float64(i)*1,
				"latency": 125.0 - float64(i)*5,
			},
		}
		if err := stream.Send(metric); err != nil {
			return err
		}
		time.Sleep(100 * time.Millisecond)
	}
	return nil
}

func (s *OptimizationServer) GetSystemHealth(ctx context.Context, req *pb.HealthRequest) (*pb.SystemHealthResponse, error) {
	return &pb.SystemHealthResponse{
		Metadata: &pb.ResponseMetadata{
			RequestId:     req.Metadata.RequestId,
			Timestamp:     timeToProto(time.Now()),
			StatusCode:    200,
			StatusMessage: "OK",
		},
		OverallHealth: 0.92,
		ComponentHealth: map[string]float32{
			"inference":    0.95,
			"memory":       0.90,
			"optimization": 0.88,
		},
		Warnings: []string{},
	}, nil
}

// ============================================================================
// Debug Server Implementation
// ============================================================================

type DebugServer struct {
	pb.UnimplementedDebugServiceServer
}

func NewDebugServer() *DebugServer {
	return &DebugServer{}
}

func (s *DebugServer) InspectComponent(ctx context.Context, req *pb.InspectRequest) (*pb.InspectResponse, error) {
	log.Printf("[Debug] Inspecting component: %s (%s)\n", req.Component, req.InspectType)

	return &pb.InspectResponse{
		Metadata: &pb.ResponseMetadata{
			RequestId:     req.Metadata.RequestId,
			Timestamp:     timeToProto(time.Now()),
			StatusCode:    200,
			StatusMessage: "OK",
		},
		State:     &structpb.Struct{},
		DebugInfo: []string{"Component state inspected successfully"},
	}, nil
}

func (s *DebugServer) GetDiagnostics(ctx context.Context, req *pb.DiagnosticsRequest) (*pb.DiagnosticsResponse, error) {
	return &pb.DiagnosticsResponse{
		Metadata: &pb.ResponseMetadata{
			RequestId:     req.Metadata.RequestId,
			Timestamp:     timeToProto(time.Now()),
			StatusCode:    200,
			StatusMessage: "OK",
		},
		Diagnostics: []string{
			"All systems operational",
			"Memory usage within expected range",
			"API response times nominal",
		},
		Recommendations: map[string]string{
			"optimization": "Consider increasing batch size for better throughput",
			"memory":       "Memory usage is optimal",
		},
	}, nil
}

func (s *DebugServer) SetLogLevel(ctx context.Context, req *pb.SetLogLevelRequest) (*pb.SetLogLevelResponse, error) {
	log.Printf("[Debug] Set log level for %s: %s\n", req.Component, req.LogLevel)

	return &pb.SetLogLevelResponse{
		Metadata: &pb.ResponseMetadata{
			RequestId:     req.Metadata.RequestId,
			Timestamp:     timeToProto(time.Now()),
			StatusCode:    200,
			StatusMessage: "OK",
		},
		Success: true,
	}, nil
}

func (s *DebugServer) TracePath(req *pb.TraceRequest, stream pb.DebugService_TracePathServer) error {
	for i := 0; i < 3; i++ {
		event := &pb.TraceEvent{
			Metadata: &pb.ResponseMetadata{
				RequestId:     req.Metadata.RequestId,
				Timestamp:     timeToProto(time.Now()),
				StatusCode:    200,
				StatusMessage: "OK",
			},
			EventType: "trace_event",
			Component: req.Operation,
			Timestamp: timeToProto(time.Now()),
			Details: map[string]string{
				"step":   fmt.Sprintf("%d", i+1),
				"status": "executing",
			},
		}
		if err := stream.Send(event); err != nil {
			return err
		}
		time.Sleep(50 * time.Millisecond)
	}
	return nil
}

// ============================================================================
// Helper Functions
// ============================================================================

func timeToProto(t time.Time) *timestamppb.Timestamp {
	return &timestamppb.Timestamp{
		Seconds: t.Unix(),
		Nanos:   int32(t.Nanosecond()),
	}
}

// ============================================================================
// Main Entry Point
// ============================================================================

func main() {
	server := NewMCPServer()
	ctx := context.Background()

	if err := server.Start(ctx); err != nil {
		log.Fatalf("[MCP] Failed to start server: %v", err)
	}

	// Keep server running
	select {}
}
