// inference_client.go
// HTTP client for calling the Python API server's OpenAI-compatible endpoints
//
// This bridges the MCP gRPC server with the actual Ryzanstein LLM inference engine.

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"
)

const (
	// Default Python API server URL
	DefaultAPIURL = "http://127.0.0.1:8000"

	// Request timeout
	DefaultTimeout = 60 * time.Second
)

// InferenceClient wraps HTTP calls to the Python API
type InferenceClient struct {
	baseURL    string
	httpClient *http.Client
}

// NewInferenceClient creates a new inference client
func NewInferenceClient(baseURL string) *InferenceClient {
	if baseURL == "" {
		baseURL = DefaultAPIURL
	}

	return &InferenceClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: DefaultTimeout,
		},
	}
}

// ChatRequest represents an OpenAI-compatible chat completion request
type ChatRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Temperature float64       `json:"temperature,omitempty"`
	TopP        float64       `json:"top_p,omitempty"`
	Stream      bool          `json:"stream,omitempty"`
}

// ChatMessage represents a single message in the conversation
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatResponse represents an OpenAI-compatible chat completion response
type ChatResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []ChatChoice `json:"choices"`
	Usage   ChatUsage    `json:"usage"`
}

// ChatChoice represents a single completion choice
type ChatChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

// ChatUsage represents token usage
type ChatUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// HealthResponse represents the health check response
type HealthResponse struct {
	Status       string `json:"status"`
	EngineLoaded bool   `json:"engine_loaded"`
	ModelPath    string `json:"model_path"`
	Timestamp    int64  `json:"timestamp"`
}

// ChatCompletion sends a chat completion request to the Python API
func (c *InferenceClient) ChatCompletion(prompt string, systemPrompt string, maxTokens int, temperature float64) (string, int, error) {
	// Build messages
	messages := make([]ChatMessage, 0, 2)

	if systemPrompt != "" {
		messages = append(messages, ChatMessage{
			Role:    "system",
			Content: systemPrompt,
		})
	}

	messages = append(messages, ChatMessage{
		Role:    "user",
		Content: prompt,
	})

	// Build request
	req := ChatRequest{
		Model:       "ryzanstein-bitnet-3b",
		Messages:    messages,
		MaxTokens:   maxTokens,
		Temperature: temperature,
		TopP:        0.9,
		Stream:      false,
	}

	// Default values
	if req.MaxTokens == 0 {
		req.MaxTokens = 256
	}
	if req.Temperature == 0 {
		req.Temperature = 0.7
	}

	// Marshal to JSON
	body, err := json.Marshal(req)
	if err != nil {
		return "", 0, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Send request
	url := fmt.Sprintf("%s/v1/chat/completions", c.baseURL)
	log.Printf("[InferenceClient] Sending request to %s", url)

	httpReq, err := http.NewRequest("POST", url, bytes.NewBuffer(body))
	if err != nil {
		return "", 0, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return "", 0, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	// Read response
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", 0, fmt.Errorf("failed to read response: %w", err)
	}

	// Check status code
	if resp.StatusCode != http.StatusOK {
		return "", 0, fmt.Errorf("API returned status %d: %s", resp.StatusCode, string(respBody))
	}

	// Parse response
	var chatResp ChatResponse
	if err := json.Unmarshal(respBody, &chatResp); err != nil {
		return "", 0, fmt.Errorf("failed to parse response: %w", err)
	}

	// Extract content
	if len(chatResp.Choices) == 0 {
		return "", 0, fmt.Errorf("no choices in response")
	}

	content := chatResp.Choices[0].Message.Content
	tokensUsed := chatResp.Usage.TotalTokens

	log.Printf("[InferenceClient] Received response: %d tokens", tokensUsed)

	return content, tokensUsed, nil
}

// CheckHealth checks if the Python API is healthy
func (c *InferenceClient) CheckHealth() (*HealthResponse, error) {
	url := fmt.Sprintf("%s/health", c.baseURL)

	resp, err := c.httpClient.Get(url)
	if err != nil {
		return nil, fmt.Errorf("health check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("health check returned status %d", resp.StatusCode)
	}

	var health HealthResponse
	if err := json.NewDecoder(resp.Body).Decode(&health); err != nil {
		return nil, fmt.Errorf("failed to parse health response: %w", err)
	}

	return &health, nil
}

// IsAvailable checks if the Python API is reachable
func (c *InferenceClient) IsAvailable() bool {
	health, err := c.CheckHealth()
	if err != nil {
		log.Printf("[InferenceClient] API not available: %v", err)
		return false
	}

	return health.Status == "healthy"
}

// Global client instance
var globalInferenceClient *InferenceClient

// GetInferenceClient returns the global inference client
func GetInferenceClient() *InferenceClient {
	if globalInferenceClient == nil {
		globalInferenceClient = NewInferenceClient(DefaultAPIURL)
	}
	return globalInferenceClient
}

// InitInferenceClient initializes the global inference client with a custom URL
func InitInferenceClient(baseURL string) {
	globalInferenceClient = NewInferenceClient(baseURL)
}
