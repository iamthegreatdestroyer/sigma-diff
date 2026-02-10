package services

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// ModelMetadata represents model information and status
type ModelMetadata struct {
	ID              string
	Name            string
	Path            string
	Size            int64
	Dependencies    []string
	Priority        int    // Higher priority loads first
	PreloadStrategy string // "eager", "lazy", "none"
	MaxConcurrency  int
	LoadTimeout     time.Duration
}

// ModelLoadResult contains model loading outcome
type ModelLoadResult struct {
	ModelID    string
	Success    bool
	Error      error
	LoadTime   time.Duration
	LoadedAt   time.Time
	CacheHit   bool
	Throughput float64 // models/sec
}

// AsyncModelManager handles concurrent model loading with caching
type AsyncModelManager struct {
	mu                    sync.RWMutex
	models                map[string]*ModelMetadata
	modelCache            map[string]interface{}
	loadingStates         map[string]*loadingState
	preloadQueue          chan *ModelMetadata
	resultChan            chan *ModelLoadResult
	maxConcurrentLoads    int
	activeSemaphore       chan struct{}
	cacheHitCount         atomic.Uint64
	cacheMissCount        atomic.Uint64
	totalLoadTime         atomic.Int64
	totalModelsLoaded     atomic.Uint32
	shutdownOnce          sync.Once
	ctx                   context.Context
	cancel                context.CancelFunc
	wg                    sync.WaitGroup
	metricsUpdateInterval time.Duration
	lastMetricsUpdate     time.Time
}

// loadingState tracks concurrent loading progress
type loadingState struct {
	startTime   time.Time
	inProgress  bool
	completions int
	errors      []error
	mu          sync.RWMutex
}

// NewAsyncModelManager creates a new async model manager with concurrency control
func NewAsyncModelManager(maxConcurrent int) *AsyncModelManager {
	if maxConcurrent < 1 {
		maxConcurrent = 4
	}

	ctx, cancel := context.WithCancel(context.Background())

	manager := &AsyncModelManager{
		models:                make(map[string]*ModelMetadata),
		modelCache:            make(map[string]interface{}),
		loadingStates:         make(map[string]*loadingState),
		preloadQueue:          make(chan *ModelMetadata, 100),
		resultChan:            make(chan *ModelLoadResult, 50),
		maxConcurrentLoads:    maxConcurrent,
		activeSemaphore:       make(chan struct{}, maxConcurrent),
		ctx:                   ctx,
		cancel:                cancel,
		metricsUpdateInterval: 1 * time.Second,
		lastMetricsUpdate:     time.Now(),
	}

	// Start worker goroutines for model loading
	for i := 0; i < maxConcurrent; i++ {
		manager.wg.Add(1)
		go manager.loadWorker()
	}

	// Start preload coordinator
	manager.wg.Add(1)
	go manager.preloadCoordinator()

	return manager
}

// RegisterModel registers a model for potential async loading
func (m *AsyncModelManager) RegisterModel(metadata *ModelMetadata) error {
	if metadata == nil || metadata.ID == "" {
		return fmt.Errorf("invalid model metadata: empty ID")
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.models[metadata.ID]; exists {
		return fmt.Errorf("model already registered: %s", metadata.ID)
	}

	// Set defaults
	if metadata.MaxConcurrency == 0 {
		metadata.MaxConcurrency = m.maxConcurrentLoads
	}
	if metadata.LoadTimeout == 0 {
		metadata.LoadTimeout = 30 * time.Second
	}
	if metadata.PreloadStrategy == "" {
		metadata.PreloadStrategy = "lazy"
	}
	if metadata.Priority == 0 {
		metadata.Priority = 50
	}

	m.models[metadata.ID] = metadata
	m.loadingStates[metadata.ID] = &loadingState{
		startTime:  time.Now(),
		inProgress: false,
	}

	// Queue for preloading if eager
	if metadata.PreloadStrategy == "eager" {
		select {
		case m.preloadQueue <- metadata:
		case <-m.ctx.Done():
			return fmt.Errorf("manager shutdown in progress")
		default:
			// Queue full, will be loaded on demand
		}
	}

	return nil
}

// LoadModel asynchronously loads a model with timeout and error handling
func (m *AsyncModelManager) LoadModel(ctx context.Context, modelID string) (*ModelLoadResult, error) {
	m.mu.RLock()
	metadata, exists := m.models[modelID]
	m.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("model not registered: %s", modelID)
	}

	// Check cache first
	m.mu.RLock()
	cachedModel, isCached := m.modelCache[modelID]
	m.mu.RUnlock()

	if isCached {
		m.cacheHitCount.Add(1)
		return &ModelLoadResult{
			ModelID:    modelID,
			Success:    true,
			CacheHit:   true,
			LoadedAt:   time.Now(),
			LoadTime:   0,
			Throughput: 0,
		}, nil
	}

	m.cacheMissCount.Add(1)

	// Create load context with timeout
	loadCtx, cancel := context.WithTimeout(ctx, metadata.LoadTimeout)
	defer cancel()

	// Acquire semaphore slot
	select {
	case m.activeSemaphore <- struct{}{}:
		defer func() { <-m.activeSemaphore }()
	case <-loadCtx.Done():
		return nil, fmt.Errorf("timeout acquiring load slot for model: %s", modelID)
	}

	// Perform actual load
	startTime := time.Now()
	result := &ModelLoadResult{
		ModelID:  modelID,
		LoadedAt: startTime,
		Success:  true,
	}

	// Simulate model loading with dependency resolution
	if err := m.loadWithDependencies(loadCtx, metadata); err != nil {
		result.Success = false
		result.Error = err
		m.recordLoadingState(modelID, err)
		return result, err
	}

	loadDuration := time.Since(startTime)
	result.LoadTime = loadDuration
	result.Throughput = m.calculateThroughput(loadDuration)

	// Cache the loaded model
	m.mu.Lock()
	m.modelCache[modelID] = fmt.Sprintf("model_%s_data", modelID)
	m.mu.Unlock()

	m.totalLoadTime.Add(loadDuration.Milliseconds())
	m.totalModelsLoaded.Add(1)

	return result, nil
}

// loadWithDependencies resolves and loads model dependencies first
func (m *AsyncModelManager) loadWithDependencies(ctx context.Context, metadata *ModelMetadata) error {
	// Load dependencies first (depth-first)
	for _, depID := range metadata.Dependencies {
		m.mu.RLock()
		depMetadata, exists := m.models[depID]
		m.mu.RUnlock()

		if !exists {
			continue // Skip missing dependencies
		}

		// Check if dependency is cached
		m.mu.RLock()
		_, isCached := m.modelCache[depID]
		m.mu.RUnlock()

		if !isCached {
			// Load dependency
			if err := m.loadWithDependencies(ctx, depMetadata); err != nil {
				return fmt.Errorf("failed to load dependency %s: %w", depID, err)
			}
		}
	}

	// Simulate model loading with delay proportional to size
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(time.Millisecond * time.Duration(metadata.Size/1000000+1)):
		// Simulate load time based on model size
	}

	return nil
}

// PreloadModels queues multiple models for eager preloading
func (m *AsyncModelManager) PreloadModels(modelIDs ...string) error {
	for _, modelID := range modelIDs {
		m.mu.RLock()
		metadata, exists := m.models[modelID]
		m.mu.RUnlock()

		if !exists {
			continue
		}

		select {
		case m.preloadQueue <- metadata:
		case <-m.ctx.Done():
			return fmt.Errorf("manager shutdown in progress")
		default:
			// Queue full, skip
		}
	}

	return nil
}

// loadWorker processes models from queue with concurrency control
func (m *AsyncModelManager) loadWorker() {
	defer m.wg.Done()

	for {
		select {
		case <-m.ctx.Done():
			return
		case metadata := <-m.preloadQueue:
			if metadata == nil {
				continue
			}

			// Acquire semaphore slot
			select {
			case m.activeSemaphore <- struct{}{}:
			case <-m.ctx.Done():
				return
			}

			// Load model
			startTime := time.Now()
			loadCtx, cancel := context.WithTimeout(m.ctx, metadata.LoadTimeout)
			err := m.loadWithDependencies(loadCtx, metadata)
			cancel()

			loadDuration := time.Since(startTime)

			result := &ModelLoadResult{
				ModelID:    metadata.ID,
				Success:    err == nil,
				Error:      err,
				LoadTime:   loadDuration,
				LoadedAt:   startTime,
				Throughput: m.calculateThroughput(loadDuration),
			}

			if err == nil {
				m.mu.Lock()
				m.modelCache[metadata.ID] = fmt.Sprintf("model_%s_data", metadata.ID)
				m.mu.Unlock()

				m.totalLoadTime.Add(loadDuration.Milliseconds())
				m.totalModelsLoaded.Add(1)
			}

			m.recordLoadingState(metadata.ID, err)

			// Send result (non-blocking)
			select {
			case m.resultChan <- result:
			default:
			}

			// Release semaphore
			<-m.activeSemaphore
		}
	}
}

// preloadCoordinator manages priority-based preloading order
func (m *AsyncModelManager) preloadCoordinator() {
	defer m.wg.Done()

	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-m.ctx.Done():
			return
		case <-ticker.C:
			m.coordinatePreloads()
		}
	}
}

// coordinatePreloads sorts and queues models by priority
func (m *AsyncModelManager) coordinatePreloads() {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Find models waiting to be preloaded (not in cache, not loaded)
	var waitingModels []*ModelMetadata
	for _, metadata := range m.models {
		if metadata.PreloadStrategy == "eager" {
			_, isCached := m.modelCache[metadata.ID]
			if !isCached {
				waitingModels = append(waitingModels, metadata)
			}
		}
	}

	// Sort by priority (higher first)
	for i := 0; i < len(waitingModels)-1; i++ {
		for j := i + 1; j < len(waitingModels); j++ {
			if waitingModels[j].Priority > waitingModels[i].Priority {
				waitingModels[i], waitingModels[j] = waitingModels[j], waitingModels[i]
			}
		}
	}

	// Queue high-priority models first
	for _, metadata := range waitingModels[:minInt(3, len(waitingModels))] {
		select {
		case m.preloadQueue <- metadata:
		case <-m.ctx.Done():
			return
		default:
			break
		}
	}
}

// GetModelStats returns current cache statistics
func (m *AsyncModelManager) GetModelStats() map[string]interface{} {
	cacheHits := m.cacheHitCount.Load()
	cacheMisses := m.cacheMissCount.Load()
	totalLoads := m.totalModelsLoaded.Load()

	hitRate := 0.0
	if cacheHits+uint64(cacheMisses) > 0 {
		hitRate = float64(cacheHits) / float64(cacheHits+uint64(cacheMisses)) * 100
	}

	avgLoadTime := 0.0
	if totalLoads > 0 {
		avgLoadTime = float64(m.totalLoadTime.Load()) / float64(totalLoads)
	}

	m.mu.RLock()
	cachedCount := len(m.modelCache)
	registeredCount := len(m.models)
	m.mu.RUnlock()

	return map[string]interface{}{
		"cache_hits":          cacheHits,
		"cache_misses":        cacheMisses,
		"hit_rate_percent":    hitRate,
		"total_models_loaded": totalLoads,
		"avg_load_time_ms":    avgLoadTime,
		"cached_models":       cachedCount,
		"registered_models":   registeredCount,
		"max_concurrent":      m.maxConcurrentLoads,
		"active_loads":        len(m.activeSemaphore),
	}
}

// ClearCache removes all cached models
func (m *AsyncModelManager) ClearCache() {
	m.mu.Lock()
	m.modelCache = make(map[string]interface{})
	m.mu.Unlock()
}

// UnloadModel removes a model from cache
func (m *AsyncModelManager) UnloadModel(modelID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.modelCache[modelID]; !exists {
		return fmt.Errorf("model not in cache: %s", modelID)
	}

	delete(m.modelCache, modelID)
	return nil
}

// Shutdown gracefully shuts down the manager
func (m *AsyncModelManager) Shutdown(ctx context.Context) error {
	var shutdownErr error

	m.shutdownOnce.Do(func() {
		m.cancel()

		// Wait for all goroutines with timeout
		done := make(chan struct{})
		go func() {
			m.wg.Wait()
			close(done)
		}()

		select {
		case <-done:
			// Graceful shutdown complete
		case <-ctx.Done():
			shutdownErr = fmt.Errorf("shutdown timeout: %w", ctx.Err())
		}

		close(m.preloadQueue)
		close(m.resultChan)
	})

	return shutdownErr
}

// recordLoadingState records model loading outcomes for metrics
func (m *AsyncModelManager) recordLoadingState(modelID string, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	state, exists := m.loadingStates[modelID]
	if !exists {
		return
	}

	state.mu.Lock()
	defer state.mu.Unlock()

	state.inProgress = false
	state.completions++

	if err != nil {
		state.errors = append(state.errors, err)
	}
}

// calculateThroughput calculates models loaded per second
func (m *AsyncModelManager) calculateThroughput(duration time.Duration) float64 {
	if duration <= 0 {
		return 0
	}
	return 1.0 / duration.Seconds()
}

// minInt returns minimum of two integers
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
