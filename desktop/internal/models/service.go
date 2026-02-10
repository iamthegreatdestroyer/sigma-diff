package models

import (
	"fmt"
	"log"
	"sync"

	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/config"
)

// Service handles model operations
type Service struct {
	config   *config.Manager
	models   map[string]*ModelInfo
	mu       sync.RWMutex
	loadedID string
}

// ModelInfo represents model metadata
type ModelInfo struct {
	ID            string
	Name          string
	Size          string
	ContextLength int
	Loaded        bool
	Status        string
	Path          string
}

// NewService creates a new models service
func NewService(cfg *config.Manager) *Service {
	return &Service{
		config: cfg,
		models: make(map[string]*ModelInfo),
	}
}

// LoadInstalledModels loads models from disk
func (s *Service) LoadInstalledModels() {
	log.Println("[ModelsService] Loading installed models")

	s.mu.Lock()
	defer s.mu.Unlock()

	// Initialize with placeholder models
	// In real implementation, scan directories for models
	s.models["ryzanstein-7b"] = &ModelInfo{
		ID:            "ryzanstein-7b",
		Name:          "Ryzanstein 7B",
		Size:          "4.2GB",
		ContextLength: 4096,
		Loaded:        false,
		Status:        "ready",
		Path:          "/models/ryzanstein-7b",
	}

	s.models["ryzanstein-13b"] = &ModelInfo{
		ID:            "ryzanstein-13b",
		Name:          "Ryzanstein 13B",
		Size:          "7.8GB",
		ContextLength: 4096,
		Loaded:        false,
		Status:        "ready",
		Path:          "/models/ryzanstein-13b",
	}

	log.Printf("[ModelsService] Loaded %d models\n", len(s.models))
}

// ListModels returns available models
func (s *Service) ListModels() []ModelInfo {
	s.mu.RLock()
	defer s.mu.RUnlock()

	models := make([]ModelInfo, 0, len(s.models))
	for _, m := range s.models {
		models = append(models, *m)
	}

	return models
}

// LoadModel loads a model
func (s *Service) LoadModel(modelID string) error {
	log.Printf("[ModelsService] Loading model: %s\n", modelID)

	s.mu.Lock()
	defer s.mu.Unlock()

	model, ok := s.models[modelID]
	if !ok {
		return fmt.Errorf("model not found: %s", modelID)
	}

	// Unload previous model
	if s.loadedID != "" {
		if prev, ok := s.models[s.loadedID]; ok {
			prev.Loaded = false
		}
	}

	// Load new model
	model.Loaded = true
	model.Status = "loaded"
	s.loadedID = modelID

	log.Printf("[ModelsService] Model loaded: %s\n", modelID)
	return nil
}

// UnloadModel unloads a model
func (s *Service) UnloadModel(modelID string) error {
	log.Printf("[ModelsService] Unloading model: %s\n", modelID)

	s.mu.Lock()
	defer s.mu.Unlock()

	if model, ok := s.models[modelID]; ok {
		model.Loaded = false
		model.Status = "ready"
		if s.loadedID == modelID {
			s.loadedID = ""
		}
	}

	return nil
}

// GetLoadedModel returns the currently loaded model
func (s *Service) GetLoadedModel() *ModelInfo {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.loadedID == "" {
		return nil
	}

	return s.models[s.loadedID]
}
