package config

import (
	"os"
	"testing"
	"time"
)

func TestDefaultAppConfig(t *testing.T) {
	config := DefaultAppConfig()

	if config.Server.Host != "localhost" {
		t.Errorf("expected host 'localhost', got %s", config.Server.Host)
	}

	if config.Server.Port != 8000 {
		t.Errorf("expected port 8000, got %d", config.Server.Port)
	}

	if config.Inference.Timeout != 30*time.Second {
		t.Errorf("expected timeout 30s, got %v", config.Inference.Timeout)
	}

	if len(config.Models) == 0 {
		t.Error("expected default models to be configured")
	}

	if err := config.Validate(); err != nil {
		t.Fatalf("default config should be valid, got error: %v", err)
	}
}

func TestModelConfigValidation(t *testing.T) {
	testCases := []struct {
		name      string
		model     ModelConfig
		expectErr bool
	}{
		{
			name: "Valid model",
			model: ModelConfig{
				ID:              "test-model",
				Name:            "Test Model",
				ContextWindow:   4096,
				MaxOutputTokens: 2048,
			},
			expectErr: false,
		},
		{
			name: "Missing ID",
			model: ModelConfig{
				Name:            "Test Model",
				ContextWindow:   4096,
				MaxOutputTokens: 2048,
			},
			expectErr: true,
		},
		{
			name: "Invalid context window",
			model: ModelConfig{
				ID:              "test-model",
				ContextWindow:   0,
				MaxOutputTokens: 2048,
			},
			expectErr: true,
		},
		{
			name: "Invalid max output tokens",
			model: ModelConfig{
				ID:              "test-model",
				ContextWindow:   4096,
				MaxOutputTokens: -1,
			},
			expectErr: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.model.Validate()
			if (err != nil) != tc.expectErr {
				t.Errorf("expected error=%v, got error=%v (%v)", tc.expectErr, err != nil, err)
			}
		})
	}
}

func TestInferenceConfigValidation(t *testing.T) {
	testCases := []struct {
		name      string
		config    InferenceConfig
		expectErr bool
	}{
		{
			name: "Valid config",
			config: InferenceConfig{
				Timeout:      30 * time.Second,
				MaxRetries:   3,
				RetryBackoff: 100 * time.Millisecond,
				MaxBackoff:   10 * time.Second,
				Temperature:  0.7,
				TopP:         0.9,
				MaxTokens:    2048,
			},
			expectErr: false,
		},
		{
			name: "Invalid timeout",
			config: InferenceConfig{
				Timeout:      0,
				MaxRetries:   3,
				RetryBackoff: 100 * time.Millisecond,
				MaxBackoff:   10 * time.Second,
				MaxTokens:    2048,
			},
			expectErr: true,
		},
		{
			name: "Invalid temperature",
			config: InferenceConfig{
				Timeout:     30 * time.Second,
				Temperature: 3.0,
				MaxTokens:   2048,
			},
			expectErr: true,
		},
		{
			name: "Invalid top_p",
			config: InferenceConfig{
				Timeout:   30 * time.Second,
				TopP:      1.5,
				MaxTokens: 2048,
			},
			expectErr: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.config.Validate()
			if (err != nil) != tc.expectErr {
				t.Errorf("expected error=%v, got error=%v (%v)", tc.expectErr, err != nil, err)
			}
		})
	}
}

func TestServerConfigValidation(t *testing.T) {
	testCases := []struct {
		name      string
		config    ServerConfig
		expectErr bool
	}{
		{
			name: "Valid config",
			config: ServerConfig{
				Protocol:          ProtocolREST,
				Host:              "localhost",
				Port:              8000,
				KeepaliveInterval: 30 * time.Second,
				MaxConnIdleTime:   5 * time.Minute,
			},
			expectErr: false,
		},
		{
			name: "Missing host",
			config: ServerConfig{
				Protocol:          ProtocolREST,
				Port:              8000,
				KeepaliveInterval: 30 * time.Second,
			},
			expectErr: true,
		},
		{
			name: "Invalid port",
			config: ServerConfig{
				Protocol: ProtocolREST,
				Host:     "localhost",
				Port:     70000,
			},
			expectErr: true,
		},
		{
			name: "Invalid protocol",
			config: ServerConfig{
				Protocol: "invalid",
				Host:     "localhost",
				Port:     8000,
			},
			expectErr: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.config.Validate()
			if (err != nil) != tc.expectErr {
				t.Errorf("expected error=%v, got error=%v (%v)", tc.expectErr, err != nil, err)
			}
		})
	}
}

func TestLoadFromEnvironment(t *testing.T) {
	// Save original env vars
	originalHost := os.Getenv("RYZANSTEIN_SERVER_HOST")
	originalPort := os.Getenv("RYZANSTEIN_SERVER_PORT")

	defer func() {
		// Restore original env vars
		os.Setenv("RYZANSTEIN_SERVER_HOST", originalHost)
		os.Setenv("RYZANSTEIN_SERVER_PORT", originalPort)
	}()

	// Test environment variable loading
	os.Setenv("RYZANSTEIN_SERVER_HOST", "example.com")
	os.Setenv("RYZANSTEIN_SERVER_PORT", "9000")

	config := DefaultAppConfig()
	err := LoadFromEnvironment(&config)

	if err != nil {
		t.Fatalf("failed to load from environment: %v", err)
	}

	if config.Server.Host != "example.com" {
		t.Errorf("expected host 'example.com', got %s", config.Server.Host)
	}

	if config.Server.Port != 9000 {
		t.Errorf("expected port 9000, got %d", config.Server.Port)
	}
}

func TestGetModel(t *testing.T) {
	config := DefaultAppConfig()

	// Get existing model
	model := config.GetModel("bitnet-7b")
	if model == nil {
		t.Error("expected model to be found")
	}

	if model.ID != "bitnet-7b" {
		t.Errorf("expected model ID 'bitnet-7b', got %s", model.ID)
	}

	// Get non-existing model
	model = config.GetModel("nonexistent")
	if model != nil {
		t.Error("expected nil for non-existing model")
	}
}

func TestGetEnabledModels(t *testing.T) {
	config := DefaultAppConfig()

	// Add a disabled model
	config.Models["disabled-model"] = ModelConfig{
		ID:              "disabled-model",
		ContextWindow:   4096,
		MaxOutputTokens: 2048,
		Enabled:         false,
	}

	enabled := config.GetEnabledModels()

	if len(enabled) != 1 {
		t.Errorf("expected 1 enabled model, got %d", len(enabled))
	}

	if enabled[0].ID != "bitnet-7b" {
		t.Errorf("expected enabled model to be 'bitnet-7b', got %s", enabled[0].ID)
	}
}

func TestMergeConfig(t *testing.T) {
	base := DefaultAppConfig()
	new := DefaultAppConfig()

	new.Server.Host = "merged.example.com"
	new.Inference.Temperature = 0.5

	merged := MergeConfig(&base, &new)

	if merged.Server.Host != "merged.example.com" {
		t.Errorf("expected merged host 'merged.example.com', got %s", merged.Server.Host)
	}

	if merged.Inference.Temperature != 0.5 {
		t.Errorf("expected merged temperature 0.5, got %f", merged.Inference.Temperature)
	}
}

func TestCloneConfig(t *testing.T) {
	original := DefaultAppConfig()
	original.Server.Host = "original.com"

	clone := original.Clone()
	clone.Server.Host = "cloned.com"

	if original.Server.Host != "original.com" {
		t.Error("clone operation modified original config")
	}

	if clone.Server.Host != "cloned.com" {
		t.Errorf("expected cloned host 'cloned.com', got %s", clone.Server.Host)
	}
}

func TestSaveAndLoadConfig(t *testing.T) {
	// Create temporary file
	tmpFile, err := os.CreateTemp("", "config-*.yaml")
	if err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}
	defer os.Remove(tmpFile.Name())
	tmpFile.Close()

	// Save config
	config := DefaultAppConfig()
	config.Server.Host = "test.example.com"

	if err := config.SaveToFile(tmpFile.Name()); err != nil {
		t.Fatalf("failed to save config: %v", err)
	}

	// Load config
	loader := NewLoader(tmpFile.Name())
	loaded, err := loader.Load()

	if err != nil {
		t.Fatalf("failed to load config: %v", err)
	}

	if loaded.Server.Host != "test.example.com" {
		t.Errorf("expected host 'test.example.com', got %s", loaded.Server.Host)
	}
}

func TestParseBool(t *testing.T) {
	testCases := []struct {
		input    string
		expected bool
	}{
		{"true", true},
		{"1", true},
		{"yes", true},
		{"on", true},
		{"enabled", true},
		{"false", false},
		{"0", false},
		{"no", false},
		{"off", false},
		{"disabled", false},
		{"invalid", false},
		{"", false},
	}

	for _, tc := range testCases {
		t.Run(tc.input, func(t *testing.T) {
			result := parseBool(tc.input)
			if result != tc.expected {
				t.Errorf("parseBool(%q) = %v, expected %v", tc.input, result, tc.expected)
			}
		})
	}
}

func TestProtocolValidation(t *testing.T) {
	testCases := []struct {
		name      string
		protocol  Protocol
		expectErr bool
	}{
		{"REST protocol", ProtocolREST, false},
		{"gRPC protocol", ProtocolGRPC, false},
		{"Invalid protocol", "invalid", true},
		{"Empty protocol", "", true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			config := DefaultAppConfig()
			config.Server.Protocol = tc.protocol

			err := config.Server.Validate()
			if (err != nil) != tc.expectErr {
				t.Errorf("expected error=%v, got error=%v", tc.expectErr, err != nil)
			}
		})
	}
}

func TestLoaderWithDefaults(t *testing.T) {
	// Test with non-existent file (should use defaults)
	config, err := LoadWithDefaults("/nonexistent/path.yaml")

	if err != nil {
		t.Fatalf("expected default config to be returned, got error: %v", err)
	}

	if config.Server.Host != "localhost" {
		t.Errorf("expected default host 'localhost', got %s", config.Server.Host)
	}
}

func TestAppConfigValidation(t *testing.T) {
	// Valid config
	config := DefaultAppConfig()
	if err := config.Validate(); err != nil {
		t.Fatalf("default config should be valid: %v", err)
	}

	// Invalid: no models
	config.Models = make(map[string]ModelConfig)
	if err := config.Validate(); err == nil {
		t.Error("expected error for config with no models")
	}

	// Invalid: bad log level
	config = DefaultAppConfig()
	config.LogLevel = "invalid"
	if err := config.Validate(); err == nil {
		t.Error("expected error for invalid log level")
	}
}
