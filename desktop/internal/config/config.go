package config

import (
	"fmt"
	"time"
)

// Protocol defines the communication protocol to use
type Protocol string

const (
	// ProtocolREST uses HTTP/REST for communication
	ProtocolREST Protocol = "rest"

	// ProtocolGRPC uses gRPC for communication
	ProtocolGRPC Protocol = "grpc"
)

// ModelConfig contains configuration for a specific model
type ModelConfig struct {
	// ID is the unique identifier for the model
	ID string `yaml:"id" toml:"id"`

	// Name is the human-readable name for the model
	Name string `yaml:"name" toml:"name"`

	// Type is the model type (e.g., "bitnet", "transformer")
	Type string `yaml:"type" toml:"type"`

	// Quantization specifies the quantization method (e.g., "ternary", "int4")
	Quantization string `yaml:"quantization" toml:"quantization"`

	// ContextWindow is the maximum context window size
	ContextWindow int `yaml:"context_window" toml:"context_window"`

	// MaxOutputTokens is the maximum number of output tokens
	MaxOutputTokens int `yaml:"max_output_tokens" toml:"max_output_tokens"`

	// Enabled indicates if this model is available for use
	Enabled bool `yaml:"enabled" toml:"enabled"`

	// Metadata contains additional model-specific configuration
	Metadata map[string]string `yaml:"metadata" toml:"metadata"`
}

// Validate checks if the model configuration is valid
func (m *ModelConfig) Validate() error {
	if m.ID == "" {
		return fmt.Errorf("model config: ID is required")
	}

	if m.ContextWindow <= 0 {
		return fmt.Errorf("model config: context_window must be positive")
	}

	if m.MaxOutputTokens <= 0 {
		return fmt.Errorf("model config: max_output_tokens must be positive")
	}

	return nil
}

// InferenceConfig contains configuration for inference behavior
type InferenceConfig struct {
	// DefaultModel is the ID of the model to use if not specified
	DefaultModel string `yaml:"default_model" toml:"default_model"`

	// Timeout is the maximum time to wait for inference results
	Timeout time.Duration `yaml:"timeout" toml:"timeout"`

	// MaxRetries is the maximum number of retry attempts for failed requests
	MaxRetries int `yaml:"max_retries" toml:"max_retries"`

	// RetryBackoff is the initial backoff duration for retries
	RetryBackoff time.Duration `yaml:"retry_backoff" toml:"retry_backoff"`

	// MaxBackoff is the maximum backoff duration for exponential backoff
	MaxBackoff time.Duration `yaml:"max_backoff" toml:"max_backoff"`

	// Temperature controls the randomness of responses (0.0 to 2.0)
	Temperature float32 `yaml:"temperature" toml:"temperature"`

	// TopP controls diversity via nucleus sampling (0.0 to 1.0)
	TopP float32 `yaml:"top_p" toml:"top_p"`

	// MaxTokens is the default maximum tokens for inference
	MaxTokens int `yaml:"max_tokens" toml:"max_tokens"`

	// StreamResults indicates whether to stream results
	StreamResults bool `yaml:"stream_results" toml:"stream_results"`
}

// Validate checks if the inference configuration is valid
func (i *InferenceConfig) Validate() error {
	if i.Timeout <= 0 {
		return fmt.Errorf("inference config: timeout must be positive")
	}

	if i.MaxRetries < 0 {
		return fmt.Errorf("inference config: max_retries must be non-negative")
	}

	if i.RetryBackoff <= 0 {
		return fmt.Errorf("inference config: retry_backoff must be positive")
	}

	if i.MaxBackoff < i.RetryBackoff {
		return fmt.Errorf("inference config: max_backoff must be >= retry_backoff")
	}

	if i.Temperature < 0 || i.Temperature > 2.0 {
		return fmt.Errorf("inference config: temperature must be between 0.0 and 2.0")
	}

	if i.TopP < 0 || i.TopP > 1.0 {
		return fmt.Errorf("inference config: top_p must be between 0.0 and 1.0")
	}

	if i.MaxTokens <= 0 {
		return fmt.Errorf("inference config: max_tokens must be positive")
	}

	return nil
}

// ServerConfig contains configuration for the server connection
type ServerConfig struct {
	// Protocol specifies which protocol to use (rest or grpc)
	Protocol Protocol `yaml:"protocol" toml:"protocol"`

	// Host is the server hostname or IP address
	Host string `yaml:"host" toml:"host"`

	// Port is the server port
	Port int `yaml:"port" toml:"port"`

	// TLS indicates whether to use TLS for the connection
	TLS bool `yaml:"tls" toml:"tls"`

	// TLSVerify indicates whether to verify the server certificate
	TLSVerify bool `yaml:"tls_verify" toml:"tls_verify"`

	// CertFile is the path to the client certificate (if needed)
	CertFile string `yaml:"cert_file" toml:"cert_file"`

	// KeyFile is the path to the client key (if needed)
	KeyFile string `yaml:"key_file" toml:"key_file"`

	// CAFile is the path to the CA certificate for verification
	CAFile string `yaml:"ca_file" toml:"ca_file"`

	// KeepaliveInterval is the interval for keepalive pings
	KeepaliveInterval time.Duration `yaml:"keepalive_interval" toml:"keepalive_interval"`

	// MaxConnIdleTime is the maximum time a connection can be idle
	MaxConnIdleTime time.Duration `yaml:"max_conn_idle_time" toml:"max_conn_idle_time"`

	// Metadata contains additional server configuration
	Metadata map[string]string `yaml:"metadata" toml:"metadata"`
}

// Validate checks if the server configuration is valid
func (s *ServerConfig) Validate() error {
	if s.Host == "" {
		return fmt.Errorf("server config: host is required")
	}

	if s.Port <= 0 || s.Port > 65535 {
		return fmt.Errorf("server config: port must be between 1 and 65535")
	}

	if s.Protocol != ProtocolREST && s.Protocol != ProtocolGRPC {
		return fmt.Errorf("server config: invalid protocol '%s' (must be 'rest' or 'grpc')", s.Protocol)
	}

	if s.KeepaliveInterval <= 0 {
		return fmt.Errorf("server config: keepalive_interval must be positive")
	}

	if s.MaxConnIdleTime <= 0 {
		return fmt.Errorf("server config: max_conn_idle_time must be positive")
	}

	return nil
}

// AppConfig is the complete application configuration
type AppConfig struct {
	// Server contains server connection settings
	Server ServerConfig `yaml:"server" toml:"server"`

	// Inference contains inference behavior settings
	Inference InferenceConfig `yaml:"inference" toml:"inference"`

	// Models contains available model configurations
	Models map[string]ModelConfig `yaml:"models" toml:"models"`

	// LogLevel sets the logging level (debug, info, warn, error)
	LogLevel string `yaml:"log_level" toml:"log_level"`

	// LogFormat sets the logging format (json, text)
	LogFormat string `yaml:"log_format" toml:"log_format"`

	// MetricsEnabled indicates whether to collect metrics
	MetricsEnabled bool `yaml:"metrics_enabled" toml:"metrics_enabled"`

	// MetricsPort is the port for the metrics endpoint
	MetricsPort int `yaml:"metrics_port" toml:"metrics_port"`

	// Version is the configuration version (for schema compatibility)
	Version string `yaml:"version" toml:"version"`
}

// Validate checks if the complete configuration is valid
func (a *AppConfig) Validate() error {
	if err := a.Server.Validate(); err != nil {
		return err
	}

	if err := a.Inference.Validate(); err != nil {
		return err
	}

	if len(a.Models) == 0 {
		return fmt.Errorf("app config: at least one model must be configured")
	}

	for id, model := range a.Models {
		if err := model.Validate(); err != nil {
			return fmt.Errorf("app config: invalid model config for '%s': %w", id, err)
		}
	}

	if a.LogLevel != "" {
		validLevels := map[string]bool{
			"debug": true,
			"info":  true,
			"warn":  true,
			"error": true,
		}
		if !validLevels[a.LogLevel] {
			return fmt.Errorf("app config: invalid log_level '%s'", a.LogLevel)
		}
	}

	if a.LogFormat != "" {
		validFormats := map[string]bool{
			"json": true,
			"text": true,
		}
		if !validFormats[a.LogFormat] {
			return fmt.Errorf("app config: invalid log_format '%s'", a.LogFormat)
		}
	}

	if a.MetricsEnabled && (a.MetricsPort <= 0 || a.MetricsPort > 65535) {
		return fmt.Errorf("app config: invalid metrics_port '%d'", a.MetricsPort)
	}

	return nil
}

// GetModel returns the model configuration for the given ID
func (a *AppConfig) GetModel(id string) *ModelConfig {
	if m, exists := a.Models[id]; exists {
		return &m
	}
	return nil
}

// GetEnabledModels returns all enabled models
func (a *AppConfig) GetEnabledModels() []ModelConfig {
	var enabled []ModelConfig
	for _, m := range a.Models {
		if m.Enabled {
			enabled = append(enabled, m)
		}
	}
	return enabled
}

// DefaultAppConfig returns a sensible default configuration
func DefaultAppConfig() AppConfig {
	return AppConfig{
		Server: ServerConfig{
			Protocol:          ProtocolREST,
			Host:              "localhost",
			Port:              8000,
			TLS:               false,
			TLSVerify:         true,
			KeepaliveInterval: 30 * time.Second,
			MaxConnIdleTime:   5 * time.Minute,
		},
		Inference: InferenceConfig{
			DefaultModel:  "default",
			Timeout:       30 * time.Second,
			MaxRetries:    3,
			RetryBackoff:  100 * time.Millisecond,
			MaxBackoff:    10 * time.Second,
			Temperature:   0.7,
			TopP:          0.9,
			MaxTokens:     2048,
			StreamResults: false,
		},
		Models: map[string]ModelConfig{
			"default": {
				ID:              "bitnet-7b",
				Name:            "BitNet b1.58 7B",
				Type:            "bitnet",
				Quantization:    "ternary",
				ContextWindow:   4096,
				MaxOutputTokens: 2048,
				Enabled:         true,
			},
		},
		LogLevel:       "info",
		LogFormat:      "json",
		MetricsEnabled: true,
		MetricsPort:    9090,
		Version:        "1.0",
	}
}
