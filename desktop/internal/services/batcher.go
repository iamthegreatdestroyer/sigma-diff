package services

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// BatchRequest represents a single request in a batch
type BatchRequest struct {
	ID        string
	Request   interface{}
	Result    chan interface{}
	Error     chan error
	Timestamp time.Time
}

// BatchConfig defines batching configuration
type BatchConfig struct {
	MaxBatchSize   int
	MinBatchSize   int
	BatchTimeout   time.Duration
	AdaptiveSizing bool
	PreserveOrder  bool
}

// DefaultBatchConfig returns default batching configuration
func DefaultBatchConfig() *BatchConfig {
	return &BatchConfig{
		MaxBatchSize:   200,
		MinBatchSize:   10,
		BatchTimeout:   50 * time.Millisecond,
		AdaptiveSizing: true,
		PreserveOrder:  true,
	}
}

// BatchMetrics tracks batching statistics
type BatchMetrics struct {
	TotalBatches     int64
	TotalRequests    int64
	AverageBatchSize float64
	MaxBatchSize     int
	TimeoutDespatch  int64
	SizeDespatch     int64
	Efficiency       float64
}

// RequestBatcher accumulates and dispatches requests in batches
type RequestBatcher struct {
	accumulator      chan *BatchRequest
	dispatcher       chan []*BatchRequest
	config           *BatchConfig
	metrics          *BatchMetrics
	state            sync.RWMutex
	stopCh           chan struct{}
	wg               sync.WaitGroup
	currentBatchSize int32
	maxObservedSize  int32
}

// NewRequestBatcher creates a new request batcher
func NewRequestBatcher(config *BatchConfig) *RequestBatcher {
	if config == nil {
		config = DefaultBatchConfig()
	}

	batcher := &RequestBatcher{
		accumulator: make(chan *BatchRequest, config.MaxBatchSize*2),
		dispatcher:  make(chan []*BatchRequest, 10),
		config:      config,
		metrics:     &BatchMetrics{},
		stopCh:      make(chan struct{}),
	}

	// Start batch accumulator routine
	batcher.wg.Add(1)
	go batcher.batchAccumulatorRoutine()

	return batcher
}

// batchAccumulatorRoutine accumulates requests and dispatches batches
func (rb *RequestBatcher) batchAccumulatorRoutine() {
	defer rb.wg.Done()

	var currentBatch []*BatchRequest
	ticker := time.NewTicker(rb.config.BatchTimeout)
	defer ticker.Stop()

	for {
		select {
		case <-rb.stopCh:
			// Flush remaining batch
			if len(currentBatch) > 0 {
				rb.dispatchBatch(currentBatch)
			}
			return

		case req := <-rb.accumulator:
			currentBatch = append(currentBatch, req)
			atomic.AddInt32(&rb.currentBatchSize, 1)

			// Check if batch should be dispatched by size
			if len(currentBatch) >= rb.config.MaxBatchSize {
				rb.dispatchBatch(currentBatch)
				atomic.AddInt64(&rb.metrics.SizeDespatch, 1)
				currentBatch = nil
				atomic.StoreInt32(&rb.currentBatchSize, 0)
			}

		case <-ticker.C:
			// Timeout dispatch
			if len(currentBatch) >= rb.config.MinBatchSize {
				rb.dispatchBatch(currentBatch)
				atomic.AddInt64(&rb.metrics.TimeoutDespatch, 1)
				currentBatch = nil
				atomic.StoreInt32(&rb.currentBatchSize, 0)
				ticker.Reset(rb.config.BatchTimeout)
			} else if len(currentBatch) > 0 {
				// Small batch - dispatch anyway to avoid stalling
				rb.dispatchBatch(currentBatch)
				atomic.AddInt64(&rb.metrics.TimeoutDespatch, 1)
				currentBatch = nil
				atomic.StoreInt32(&rb.currentBatchSize, 0)
				ticker.Reset(rb.config.BatchTimeout)
			}
		}
	}
}

// dispatchBatch sends a batch for processing
func (rb *RequestBatcher) dispatchBatch(batch []*BatchRequest) {
	if len(batch) == 0 {
		return
	}

	// Update metrics
	atomic.AddInt64(&rb.metrics.TotalBatches, 1)
	atomic.AddInt64(&rb.metrics.TotalRequests, int64(len(batch)))

	if int32(len(batch)) > atomic.LoadInt32(&rb.maxObservedSize) {
		atomic.StoreInt32(&rb.maxObservedSize, int32(len(batch)))
	}

	// Calculate average batch size
	totalReqs := atomic.LoadInt64(&rb.metrics.TotalRequests)
	totalBatches := atomic.LoadInt64(&rb.metrics.TotalBatches)
	if totalBatches > 0 {
		avgSize := float64(totalReqs) / float64(totalBatches)
		rb.state.Lock()
		rb.metrics.AverageBatchSize = avgSize
		rb.metrics.MaxBatchSize = int(atomic.LoadInt32(&rb.maxObservedSize))
		rb.state.Unlock()
	}

	// Send batch for processing
	select {
	case rb.dispatcher <- batch:
		// Batch queued for processing
	case <-rb.stopCh:
		// Closed, send errors to all requests
		for _, req := range batch {
			select {
			case req.Error <- fmt.Errorf("batcher closed"):
			default:
			}
		}
	}
}

// AddRequest adds a request to the batcher
func (rb *RequestBatcher) AddRequest(ctx context.Context, req *BatchRequest) error {
	if req == nil {
		return fmt.Errorf("request cannot be nil")
	}

	select {
	case rb.accumulator <- req:
		return nil
	case <-rb.stopCh:
		return fmt.Errorf("batcher is closed")
	case <-ctx.Done():
		return ctx.Err()
	}
}

// GetBatch retrieves the next batch for processing
func (rb *RequestBatcher) GetBatch() ([]*BatchRequest, bool) {
	select {
	case batch := <-rb.dispatcher:
		return batch, true
	case <-rb.stopCh:
		return nil, false
	}
}

// GetBatchContext retrieves the next batch with context
func (rb *RequestBatcher) GetBatchContext(ctx context.Context) ([]*BatchRequest, bool) {
	select {
	case batch := <-rb.dispatcher:
		return batch, true
	case <-rb.stopCh:
		return nil, false
	case <-ctx.Done():
		return nil, false
	}
}

// ResolveBatch marks batch requests as completed
func (rb *RequestBatcher) ResolveBatch(batch []*BatchRequest, processor func(*BatchRequest) error) {
	for _, req := range batch {
		if req == nil {
			continue
		}

		// Process request
		err := processor(req)

		// Send result
		if err != nil {
			select {
			case req.Error <- err:
			default:
			}
		} else {
			select {
			case req.Result <- nil:
			default:
			}
		}
	}
}

// GetMetrics returns current batching metrics
func (rb *RequestBatcher) GetMetrics() *BatchMetrics {
	rb.state.RLock()
	defer rb.state.RUnlock()

	metrics := &BatchMetrics{
		TotalBatches:     atomic.LoadInt64(&rb.metrics.TotalBatches),
		TotalRequests:    atomic.LoadInt64(&rb.metrics.TotalRequests),
		AverageBatchSize: rb.metrics.AverageBatchSize,
		MaxBatchSize:     rb.metrics.MaxBatchSize,
		TimeoutDespatch:  atomic.LoadInt64(&rb.metrics.TimeoutDespatch),
		SizeDespatch:     atomic.LoadInt64(&rb.metrics.SizeDespatch),
	}

	// Calculate efficiency
	if metrics.TotalBatches > 0 {
		// Efficiency = (requests - batches) / requests
		// Higher = more requests per batch = more efficient
		overhead := float64(metrics.TotalBatches)
		efficiency := (float64(metrics.TotalRequests) - overhead) / float64(metrics.TotalRequests)
		metrics.Efficiency = efficiency
	}

	return metrics
}

// GetCurrentBatchSize returns current batch size
func (rb *RequestBatcher) GetCurrentBatchSize() int32 {
	return atomic.LoadInt32(&rb.currentBatchSize)
}

// AdaptBatchSize adapts batch size based on performance
func (rb *RequestBatcher) AdaptBatchSize(latencyMs float64) {
	if !rb.config.AdaptiveSizing {
		return
	}

	// If latency is too high, reduce batch size
	if latencyMs > 10 {
		if rb.config.MaxBatchSize > 50 {
			rb.config.MaxBatchSize -= 10
		}
	} else if latencyMs < 5 {
		// If latency is good, increase batch size
		if rb.config.MaxBatchSize < 300 {
			rb.config.MaxBatchSize += 10
		}
	}
}

// Close closes the request batcher
func (rb *RequestBatcher) Close() error {
	close(rb.stopCh)
	rb.wg.Wait()

	// Drain any remaining batches
	for {
		select {
		case batch := <-rb.dispatcher:
			for _, req := range batch {
				if req != nil {
					select {
					case req.Error <- fmt.Errorf("batcher closed"):
					default:
					}
				}
			}
		default:
			return nil
		}
	}
}

// String returns string representation of batch metrics
func (m *BatchMetrics) String() string {
	return fmt.Sprintf(
		"BatchMetrics{Batches:%d, Requests:%d, AvgSize:%.1f, MaxSize:%d, Efficiency:%.2f%%, TimeoutDispatch:%d, SizeDispatch:%d}",
		m.TotalBatches,
		m.TotalRequests,
		m.AverageBatchSize,
		m.MaxBatchSize,
		m.Efficiency*100,
		m.TimeoutDespatch,
		m.SizeDespatch,
	)
}
