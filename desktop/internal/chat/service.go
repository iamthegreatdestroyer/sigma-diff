package chat

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/client"
)

// Service handles chat operations
type Service struct {
	history   []Message
	mu        sync.RWMutex
	mcpClient *client.MCPClient
}

// Message represents a chat message
type Message struct {
	ID        string
	Role      string // "user" or "assistant"
	Content   string
	Timestamp int64
	Metadata  map[string]interface{}
}

// NewService creates a new chat service
func NewService() *Service {
	// Initialize MCP client
	config := client.DefaultMCPClientConfig("localhost:50051") // Default MCP server address
	mcpClient, err := client.NewMCPClient(config)
	if err != nil {
		log.Printf("Warning: Failed to initialize MCP client: %v. Chat will use simulation mode.", err)
		mcpClient = nil
	}

	return &Service{
		history:   make([]Message, 0),
		mcpClient: mcpClient,
	}
}

// SendMessage sends a message and returns response
func (s *Service) SendMessage(ctx context.Context, message string, modelID string, agentCodename string) (string, error) {
	log.Printf("[ChatService] Processing message (model: %s, agent: %s)\n", modelID, agentCodename)

	s.mu.Lock()
	defer s.mu.Unlock()

	// Add user message to history
	userMsg := Message{
		ID:        fmt.Sprintf("msg_%d", time.Now().UnixNano()),
		Role:      "user",
		Content:   message,
		Timestamp: time.Now().Unix(),
		Metadata: map[string]interface{}{
			"model": modelID,
			"agent": agentCodename,
		},
	}
	s.history = append(s.history, userMsg)

	var response string
	var err error

	// Try to use MCP client for real inference
	if s.mcpClient != nil {
		response, err = s.callMCPInference(ctx, message, modelID, agentCodename)
		if err != nil {
			log.Printf("MCP inference failed, falling back to simulation: %v", err)
			response = s.simulateResponse(message, modelID, agentCodename)
		}
	} else {
		// Fallback to simulation if MCP client not available
		response = s.simulateResponse(message, modelID, agentCodename)
	}

	// Add assistant message to history
	assistantMsg := Message{
		ID:        fmt.Sprintf("msg_%d", time.Now().UnixNano()),
		Role:      "assistant",
		Content:   response,
		Timestamp: time.Now().Unix(),
		Metadata: map[string]interface{}{
			"agent": agentCodename,
			"model": modelID,
		},
	}
	s.history = append(s.history, assistantMsg)

	return response, nil
}

// callMCPInference calls the MCP server for inference
func (s *Service) callMCPInference(ctx context.Context, message string, modelID string, agentCodename string) (string, error) {
	// Convert chat history to MCP messages
	var messages []*Message
	for _, msg := range s.history {
		messages = append(messages, &Message{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	// Add current message if not already in history
	if len(messages) == 0 || messages[len(messages)-1].Content != message {
		messages = append(messages, &Message{
			Role:    "user",
			Content: message,
		})
	}

	// Create inference request
	req := &InferenceRequest{
		Messages:      messages,
		Model:         modelID,
		AgentCodename: agentCodename,
		MaxTokens:     1000,
		Temperature:   0.7,
	}

	// Call MCP inference
	resp, err := s.mcpClient.Infer(ctx, req)
	if err != nil {
		return "", fmt.Errorf("MCP inference failed: %w", err)
	}

	return resp.Content, nil
}

// simulateResponse provides fallback simulation when MCP is unavailable
func (s *Service) simulateResponse(message string, modelID string, agentCodename string) string {
	return fmt.Sprintf("Response from %s (model: %s): Processed your request about '%s'",
		agentCodename, modelID, message)
}

// AddMessage adds a message to the chat history
func (s *Service) AddMessage(ctx context.Context, role string, content string, modelID string, agentCodename string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	msg := Message{
		ID:        fmt.Sprintf("msg_%d", time.Now().UnixNano()),
		Role:      role,
		Content:   content,
		Timestamp: time.Now().Unix(),
		Metadata: map[string]interface{}{
			"model": modelID,
			"agent": agentCodename,
		},
	}

	s.history = append(s.history, msg)
}

// GetHistory returns chat history
func (s *Service) GetHistory(limit int) []Message {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if limit <= 0 || limit > len(s.history) {
		return s.history
	}

	return s.history[len(s.history)-limit:]
}

// ClearHistory clears chat history
func (s *Service) ClearHistory() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.history = make([]Message, 0)
}

// Close closes the service
func (s *Service) Close() {
	log.Println("[ChatService] Closing")
	s.ClearHistory()
}
