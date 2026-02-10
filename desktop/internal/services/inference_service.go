package services

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"
)

// InferenceRequest represents a request for inference
type InferenceRequest struct {
	ModelID     string                 `json:"model_id"`
	Prompt      string                 `json:"prompt"`
	MaxTokens   int                    `json:"max_tokens,omitempty"`
	Temperature float32                `json:"temperature,omitempty"`
	TopP        float32                `json:"top_p,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// InferenceResponse represents a response from inference
type InferenceResponse struct {
	Text     string                 `json:"text"`
	Tokens   int                    `json:"tokens"`
	Duration time.Duration          `json:"duration"`
	Model    string                 `json:"model"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// InferenceMetrics tracks inference statistics
type InferenceMetrics struct {
	TotalRequests      int64
	SuccessfulRequests int64
	FailedRequests     int64
	TotalTokens        int64
	AverageDuration    float64
	LastRequestTime    time.Time
}

// InferenceService handles inference operations
type InferenceService struct {
	cm *ClientManager
	ms *ModelService

	mu      sync.RWMutex
	metrics InferenceMetrics
}

// NewInferenceService creates a new inference service
func NewInferenceService(cm *ClientManager, ms *ModelService) *InferenceService {
	return &InferenceService{
		cm:      cm,
		ms:      ms,
		metrics: InferenceMetrics{},
	}
}

// Execute performs inference on the specified model
func (is *InferenceService) Execute(ctx context.Context, req *InferenceRequest) (*InferenceResponse, error) {
	is.mu.Lock()
	is.metrics.TotalRequests++
	is.mu.Unlock()

	// Validate request
	if req == nil {
		return nil, fmt.Errorf("inference request is nil")
	}

	if req.ModelID == "" {
		return nil, fmt.Errorf("model_id is required")
	}

	if req.Prompt == "" {
		return nil, fmt.Errorf("prompt is required")
	}

	// Check if model is loaded
	if !is.ms.IsModelLoaded(req.ModelID) {
		return nil, fmt.Errorf("model not loaded: %s", req.ModelID)
	}

	// Validate parameters
	if req.Temperature < 0 || req.Temperature > 2.0 {
		return nil, fmt.Errorf("invalid temperature: %f (must be 0-2.0)", req.Temperature)
	}

	if req.TopP < 0 || req.TopP > 1.0 {
		return nil, fmt.Errorf("invalid top_p: %f (must be 0-1.0)", req.TopP)
	}

	if req.MaxTokens <= 0 {
		req.MaxTokens = 2048 // Default
	}

	startTime := time.Now()

	// Execute inference via client manager
	result, err := is.cm.ExecuteWithRouting(ctx, "infer", req)
	if err != nil {
		is.mu.Lock()
		is.metrics.FailedRequests++
		is.mu.Unlock()
		return nil, fmt.Errorf("inference execution failed: %w", err)
	}

	duration := time.Since(startTime)

	// Parse response from client manager
	var responseText string
	if resultMap, ok := result.(map[string]interface{}); ok {
		if data, exists := resultMap["data"]; exists {
			if reqData, ok := data.(*InferenceRequest); ok {
				// Generate a more intelligent response based on the agent
				agent := "unknown"
				if reqData.Metadata != nil {
					if agentVal, ok := reqData.Metadata["agent"].(string); ok {
						agent = agentVal
					}
				}

				responseText = fmt.Sprintf("🤖 %s here! I've analyzed your request: '%s'. This is currently a simulated response - real LLM inference would be connected here.",
					agent, reqData.Prompt)
			}
		}
	}

	if responseText == "" {
		responseText = fmt.Sprintf("Generated response for: %s", req.Prompt)
	}

	response := &InferenceResponse{
		Text:     responseText,
		Tokens:   len(strings.Split(responseText, " ")), // Rough token count
		Duration: duration,
		Model:    req.ModelID,
		Metadata: map[string]interface{}{
			"timestamp": startTime,
			"agent":     req.Metadata["agent"],
		},
	}

	// Update metrics
	is.mu.Lock()
	is.metrics.SuccessfulRequests++
	is.metrics.TotalTokens += int64(response.Tokens)
	is.metrics.LastRequestTime = time.Now()

	if is.metrics.SuccessfulRequests > 0 {
		avgDuration := time.Duration(int64(is.metrics.TotalTokens) / is.metrics.SuccessfulRequests)
		is.metrics.AverageDuration = avgDuration.Seconds()
	}
	is.mu.Unlock()

	return response, nil
}

// ExecuteStream performs streaming inference
func (is *InferenceService) ExecuteStream(ctx context.Context, req *InferenceRequest, resultChan chan string, errChan chan error) {
	is.mu.Lock()
	is.metrics.TotalRequests++
	is.mu.Unlock()

	// Validate request
	if req == nil {
		errChan <- fmt.Errorf("inference request is nil")
		return
	}

	if req.ModelID == "" {
		errChan <- fmt.Errorf("model_id is required")
		return
	}

	// Check if model is loaded
	if !is.ms.IsModelLoaded(req.ModelID) {
		is.mu.Lock()
		is.metrics.FailedRequests++
		is.mu.Unlock()
		errChan <- fmt.Errorf("model not loaded: %s", req.ModelID)
		return
	}

	// Simulate streaming (in real implementation, would use actual streaming)
	go func() {
		defer close(resultChan)

		startTime := time.Now()

		// Execute inference
		_, err := is.cm.ExecuteWithRouting(ctx, "infer_stream", req)
		if err != nil {
			is.mu.Lock()
			is.metrics.FailedRequests++
			is.mu.Unlock()
			errChan <- fmt.Errorf("streaming inference failed: %w", err)
			return
		}

		// Simulate token streaming
		tokens := []string{"Hello", " world", " from", " the", " model"}
		totalTokens := int64(0)

		for _, token := range tokens {
			select {
			case <-ctx.Done():
				errChan <- ctx.Err()
				return
			case resultChan <- token:
				totalTokens++
			}
		}

		// Update metrics
		is.mu.Lock()
		is.metrics.SuccessfulRequests++
		is.metrics.TotalTokens += totalTokens
		is.metrics.LastRequestTime = time.Now()

		if is.metrics.SuccessfulRequests > 0 {
			avgDuration := time.Since(startTime)
			is.metrics.AverageDuration = avgDuration.Seconds()
		}
		is.mu.Unlock()
	}()
}

// ValidateModel checks if a model can accept requests
func (is *InferenceService) ValidateModel(modelID string) error {
	if modelID == "" {
		return fmt.Errorf("model_id is required")
	}

	if !is.ms.IsModelLoaded(modelID) {
		return fmt.Errorf("model not loaded: %s", modelID)
	}

	return nil
}

// GetMetrics returns current inference metrics
func (is *InferenceService) GetMetrics() *InferenceMetrics {
	is.mu.RLock()
	defer is.mu.RUnlock()

	// Return a copy
	metricsCopy := is.metrics
	return &metricsCopy
}

// ResetMetrics resets all metrics
func (is *InferenceService) ResetMetrics() {
	is.mu.Lock()
	defer is.mu.Unlock()

	is.metrics = InferenceMetrics{}
}

// GetSuccessRate returns the success rate as a percentage
func (is *InferenceService) GetSuccessRate() float64 {
	is.mu.RLock()
	defer is.mu.RUnlock()

	if is.metrics.TotalRequests == 0 {
		return 0
	}

	return float64(is.metrics.SuccessfulRequests) / float64(is.metrics.TotalRequests) * 100
}

// GetAverageTokensPerSecond returns throughput
func (is *InferenceService) GetAverageTokensPerSecond() float64 {
	is.mu.RLock()
	defer is.mu.RUnlock()

	if is.metrics.AverageDuration == 0 {
		return 0
	}

	return float64(is.metrics.TotalTokens) / is.metrics.AverageDuration
}

// GetLastRequestInfo returns info about last request
func (is *InferenceService) GetLastRequestInfo() map[string]interface{} {
	is.mu.RLock()
	defer is.mu.RUnlock()

	return map[string]interface{}{
		"total_requests":    is.metrics.TotalRequests,
		"successful":        is.metrics.SuccessfulRequests,
		"failed":            is.metrics.FailedRequests,
		"total_tokens":      is.metrics.TotalTokens,
		"average_duration":  is.metrics.AverageDuration,
		"success_rate":      float64(is.metrics.SuccessfulRequests) / float64(is.metrics.TotalRequests) * 100,
		"last_request_time": is.metrics.LastRequestTime,
	}
}
