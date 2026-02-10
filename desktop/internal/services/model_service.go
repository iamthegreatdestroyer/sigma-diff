package services

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// ModelInfo represents information about a model
type ModelInfo struct {
	ID              string            `json:"id"`
	Name            string            `json:"name"`
	Type            string            `json:"type"`
	Quantization    string            `json:"quantization"`
	ContextWindow   int               `json:"context_window"`
	MaxOutputTokens int               `json:"max_output_tokens"`
	Status          string            `json:"status"` // loaded, unloaded, loading, error
	LoadedAt        *time.Time        `json:"loaded_at,omitempty"`
	Metadata        map[string]string `json:"metadata,omitempty"`
}

// ModelService handles model operations
type ModelService struct {
	cm *ClientManager

	// Model cache
	mu             sync.RWMutex
	loadedModels   map[string]*ModelInfo
	availableModel []ModelInfo

	// Tracking
	lastRefresh time.Time
	cacheExpiry time.Duration
}

// NewModelService creates a new model service
func NewModelService(cm *ClientManager) *ModelService {
	return &ModelService{
		cm:             cm,
		loadedModels:   make(map[string]*ModelInfo),
		availableModel: make([]ModelInfo, 0),
		cacheExpiry:    5 * time.Minute,
	}
}

// ListModels returns all available models
func (ms *ModelService) ListModels(ctx context.Context) ([]ModelInfo, error) {
	ms.mu.Lock()
	defer ms.mu.Unlock()

	// Return cached if not expired
	if ms.lastRefresh.Add(ms.cacheExpiry).After(time.Now()) && len(ms.availableModel) > 0 {
		return ms.availableModel, nil
	}

	// Call server to get models
	result, err := ms.cm.ExecuteWithRouting(ctx, "list_models", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to list models: %w", err)
	}

	// Parse result and cache
	// In real implementation, would parse actual response
	models := []ModelInfo{
		{
			ID:              "bitnet-7b",
			Name:            "BitNet b1.58 7B",
			Type:            "bitnet",
			Quantization:    "ternary",
			ContextWindow:   4096,
			MaxOutputTokens: 2048,
			Status:          "unloaded",
			Metadata:        map[string]string{"version": "1.0"},
		},
		{
			ID:              "llama-13b",
			Name:            "Llama 13B",
			Type:            "transformer",
			Quantization:    "int4",
			ContextWindow:   4096,
			MaxOutputTokens: 2048,
			Status:          "unloaded",
			Metadata:        map[string]string{"version": "2.0"},
		},
	}

	_ = result // Use result in real implementation

	ms.availableModel = models
	ms.lastRefresh = time.Now()

	return models, nil
}

// LoadModel loads a model into memory
func (ms *ModelService) LoadModel(ctx context.Context, modelID string) (*ModelInfo, error) {
	ms.mu.Lock()

	// Check if already loaded
	if model, exists := ms.loadedModels[modelID]; exists {
		ms.mu.Unlock()
		return model, nil
	}

	ms.mu.Unlock()

	// Call server to load model
	result, err := ms.cm.ExecuteWithRouting(ctx, "load_model", map[string]string{
		"model_id": modelID,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to load model: %w", err)
	}

	_ = result // Use result in real implementation

	// Find model in available list
	ms.mu.Lock()
	defer ms.mu.Unlock()

	for _, model := range ms.availableModel {
		if model.ID == modelID {
			now := time.Now()
			model.Status = "loaded"
			model.LoadedAt = &now

			ms.loadedModels[modelID] = &model
			return &model, nil
		}
	}

	return nil, fmt.Errorf("model not found: %s", modelID)
}

// UnloadModel unloads a model from memory
func (ms *ModelService) UnloadModel(ctx context.Context, modelID string) error {
	ms.mu.Lock()

	// Check if model is loaded
	model, exists := ms.loadedModels[modelID]
	if !exists {
		ms.mu.Unlock()
		return fmt.Errorf("model not loaded: %s", modelID)
	}

	ms.mu.Unlock()

	// Call server to unload model
	_, err := ms.cm.ExecuteWithRouting(ctx, "unload_model", map[string]string{
		"model_id": modelID,
	})
	if err != nil {
		return fmt.Errorf("failed to unload model: %w", err)
	}

	// Update local state
	ms.mu.Lock()
	defer ms.mu.Unlock()

	if model, exists := ms.loadedModels[modelID]; exists {
		model.Status = "unloaded"
		model.LoadedAt = nil
		delete(ms.loadedModels, modelID)
	}

	return nil
}

// GetModelInfo returns information about a specific model
func (ms *ModelService) GetModelInfo(ctx context.Context, modelID string) (*ModelInfo, error) {
	ms.mu.RLock()

	// Check if loaded
	if model, exists := ms.loadedModels[modelID]; exists {
		defer ms.mu.RUnlock()
		return model, nil
	}

	ms.mu.RUnlock()

	// Check available models
	models, err := ms.ListModels(ctx)
	if err != nil {
		return nil, err
	}

	for _, model := range models {
		if model.ID == modelID {
			return &model, nil
		}
	}

	return nil, fmt.Errorf("model not found: %s", modelID)
}

// GetLoadedModels returns all currently loaded models
func (ms *ModelService) GetLoadedModels() []*ModelInfo {
	ms.mu.RLock()
	defer ms.mu.RUnlock()

	models := make([]*ModelInfo, 0, len(ms.loadedModels))
	for _, model := range ms.loadedModels {
		models = append(models, model)
	}

	return models
}

// IsModelLoaded checks if a model is loaded
func (ms *ModelService) IsModelLoaded(modelID string) bool {
	ms.mu.RLock()
	defer ms.mu.RUnlock()

	_, exists := ms.loadedModels[modelID]
	return exists
}

// ClearCache clears the model cache
func (ms *ModelService) ClearCache() {
	ms.mu.Lock()
	defer ms.mu.Unlock()

	ms.availableModel = make([]ModelInfo, 0)
	ms.lastRefresh = time.Time{}
}

// RefreshModels forces a refresh of available models
func (ms *ModelService) RefreshModels(ctx context.Context) error {
	ms.mu.Lock()
	ms.availableModel = make([]ModelInfo, 0)
	ms.lastRefresh = time.Time{}
	ms.mu.Unlock()

	_, err := ms.ListModels(ctx)
	return err
}

// GetModelCount returns count of loaded models
func (ms *ModelService) GetModelCount() int {
	ms.mu.RLock()
	defer ms.mu.RUnlock()

	return len(ms.loadedModels)
}

// UnloadAllModels unloads all loaded models
func (ms *ModelService) UnloadAllModels(ctx context.Context) error {
	ms.mu.RLock()
	modelIDs := make([]string, 0, len(ms.loadedModels))
	for id := range ms.loadedModels {
		modelIDs = append(modelIDs, id)
	}
	ms.mu.RUnlock()

	// Unload each model
	var lastErr error
	for _, modelID := range modelIDs {
		if err := ms.UnloadModel(ctx, modelID); err != nil {
			lastErr = err
		}
	}

	return lastErr
}
