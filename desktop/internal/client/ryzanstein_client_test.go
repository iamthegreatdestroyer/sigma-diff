package client

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestNewRyzansteinClient(t *testing.T) {
	client := NewRyzansteinClient("http://localhost:8000")

	if client.baseURL != "http://localhost:8000" {
		t.Errorf("expected baseURL to be 'http://localhost:8000', got %s", client.baseURL)
	}

	if client.maxRetries != 3 {
		t.Errorf("expected maxRetries to be 3, got %d", client.maxRetries)
	}

	if client.httpClient == nil {
		t.Error("expected httpClient to be initialized")
	}
}

func TestSetTimeout(t *testing.T) {
	client := NewRyzansteinClient("http://localhost:8000")

	newTimeout := 60 * time.Second
	client.SetTimeout(newTimeout)

	if client.timeout != newTimeout {
		t.Errorf("expected timeout to be %v, got %v", newTimeout, client.timeout)
	}
}

func TestSetMaxRetries(t *testing.T) {
	client := NewRyzansteinClient("http://localhost:8000")

	client.SetMaxRetries(5)

	if client.maxRetries != 5 {
		t.Errorf("expected maxRetries to be 5, got %d", client.maxRetries)
	}
}

func TestInfer_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("expected POST, got %s", r.Method)
		}

		resp := InferenceResponse{
			ID:    "test-123",
			Model: "bitnet-7b",
		}
		resp.Choices = make([]struct {
			Text         string
			FinishReason string
		}, 1)
		resp.Choices[0].Text = "Hello, World!"
		resp.Choices[0].FinishReason = "stop"

		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewRyzansteinClient(server.URL)
	req := &InferenceRequest{
		Prompt: "Hello",
		Model:  "bitnet-7b",
	}

	resp, err := client.Infer(context.Background(), req)

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if resp.ID != "test-123" {
		t.Errorf("expected ID to be 'test-123', got %s", resp.ID)
	}

	if len(resp.Choices) == 0 || resp.Choices[0].Text != "Hello, World!" {
		t.Error("expected response text to be 'Hello, World!'")
	}
}

func TestInfer_APIError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)

		apiErr := RyzansteinError{
			Code:    "invalid_prompt",
			Message: "Prompt is empty",
		}
		json.NewEncoder(w).Encode(apiErr)
	}))
	defer server.Close()

	client := NewRyzansteinClient(server.URL)
	req := &InferenceRequest{
		Prompt: "",
		Model:  "bitnet-7b",
	}

	_, err := client.Infer(context.Background(), req)

	if err == nil {
		t.Error("expected error, got nil")
	}

	if !strings.Contains(err.Error(), "invalid_prompt") {
		t.Errorf("expected error to contain 'invalid_prompt', got %v", err)
	}
}

func TestInfer_Timeout(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(2 * time.Second) // Simulate slow response
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	client := NewRyzansteinClient(server.URL)
	client.SetTimeout(100 * time.Millisecond) // Short timeout

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	req := &InferenceRequest{
		Prompt: "Hello",
		Model:  "bitnet-7b",
	}

	_, err := client.Infer(ctx, req)

	if err == nil {
		t.Error("expected timeout error, got nil")
	}
}

func TestInfer_RetryLogic(t *testing.T) {
	attemptCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attemptCount++

		if attemptCount < 3 {
			// Fail first 2 attempts
			w.WriteHeader(http.StatusInternalServerError)
			apiErr := RyzansteinError{
				Code:    "server_error",
				Message: "Temporary server error",
			}
			json.NewEncoder(w).Encode(apiErr)
		} else {
			// Succeed on 3rd attempt
			resp := InferenceResponse{
				ID:    "test-123",
				Model: "bitnet-7b",
			}
			resp.Choices = make([]struct {
				Text         string
				FinishReason string
			}, 1)
			resp.Choices[0].Text = "Hello!"
			resp.Choices[0].FinishReason = "stop"

			json.NewEncoder(w).Encode(resp)
		}
	}))
	defer server.Close()

	client := NewRyzansteinClient(server.URL)
	client.SetMaxRetries(3)

	req := &InferenceRequest{
		Prompt: "Hello",
		Model:  "bitnet-7b",
	}

	resp, err := client.Infer(context.Background(), req)

	if err != nil {
		t.Fatalf("expected no error after retries, got %v", err)
	}

	if attemptCount != 3 {
		t.Errorf("expected 3 attempts, got %d", attemptCount)
	}

	if resp.ID != "test-123" {
		t.Errorf("expected successful response")
	}
}

func TestListModels_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" {
			t.Errorf("expected GET, got %s", r.Method)
		}

		models := struct {
			Data []ModelInfo `json:"data"`
		}{
			Data: []ModelInfo{
				{
					ID:              "bitnet-7b",
					Name:            "BitNet b1.58 7B",
					ContextWindow:   4096,
					MaxOutputTokens: 2048,
					Type:            "bitnet",
					Quantization:    "ternary",
				},
			},
		}

		json.NewEncoder(w).Encode(models)
	}))
	defer server.Close()

	client := NewRyzansteinClient(server.URL)
	models, err := client.ListModels(context.Background())

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(models) != 1 {
		t.Errorf("expected 1 model, got %d", len(models))
	}

	if models[0].ID != "bitnet-7b" {
		t.Errorf("expected model ID 'bitnet-7b', got %s", models[0].ID)
	}
}

func TestLoadModel_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("expected POST, got %s", r.Method)
		}

		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{"status": "loaded"})
	}))
	defer server.Close()

	client := NewRyzansteinClient(server.URL)
	err := client.LoadModel(context.Background(), "bitnet-7b")

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
}

func TestLoadModel_NotFound(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)

		apiErr := RyzansteinError{
			Code:    "model_not_found",
			Message: "Model not found",
		}
		json.NewEncoder(w).Encode(apiErr)
	}))
	defer server.Close()

	client := NewRyzansteinClient(server.URL)
	err := client.LoadModel(context.Background(), "nonexistent-model")

	if err == nil {
		t.Error("expected error, got nil")
	}

	if !strings.Contains(err.Error(), "model_not_found") {
		t.Errorf("expected error to contain 'model_not_found', got %v", err)
	}
}

func TestUnloadModel_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("expected POST, got %s", r.Method)
		}

		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{"status": "unloaded"})
	}))
	defer server.Close()

	client := NewRyzansteinClient(server.URL)
	err := client.UnloadModel(context.Background(), "bitnet-7b")

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
}

func TestHealth_Healthy(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	client := NewRyzansteinClient(server.URL)
	healthy, err := client.Health(context.Background())

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if !healthy {
		t.Error("expected server to be healthy")
	}
}

func TestHealth_Unhealthy(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	client := NewRyzansteinClient(server.URL)
	healthy, err := client.Health(context.Background())

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if healthy {
		t.Error("expected server to be unhealthy")
	}
}

func TestRyzansteinError_Error(t *testing.T) {
	err := &RyzansteinError{
		Code:    "test_error",
		Message: "This is a test error",
	}

	errMsg := err.Error()
	if !strings.Contains(errMsg, "test_error") || !strings.Contains(errMsg, "This is a test error") {
		t.Errorf("expected error message to contain error code and message, got %s", errMsg)
	}
}

func TestInfer_ContextCancelled(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(2 * time.Second)
	}))
	defer server.Close()

	client := NewRyzansteinClient(server.URL)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	req := &InferenceRequest{
		Prompt: "Hello",
		Model:  "bitnet-7b",
	}

	_, err := client.Infer(ctx, req)

	if err == nil {
		t.Error("expected context cancelled error, got nil")
	}
}
