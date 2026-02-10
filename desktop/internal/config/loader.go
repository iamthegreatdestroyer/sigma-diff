package config

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"gopkg.in/yaml.v3"
)

// Loader handles configuration loading from various sources
type Loader struct {
	// configPath is the path to the configuration file
	configPath string

	// config is the loaded configuration
	config *AppConfig
}

// NewLoader creates a new configuration loader
func NewLoader(configPath string) *Loader {
	return &Loader{
		configPath: configPath,
	}
}

// LoadFromFile loads configuration from a YAML or TOML file
func (l *Loader) LoadFromFile(filePath string) (*AppConfig, error) {
	// Check if file exists
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		return nil, fmt.Errorf("config: file not found: %s", filePath)
	}

	// Read file
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("config: failed to read file: %w", err)
	}

	// Parse based on file extension
	ext := filepath.Ext(filePath)
	var config AppConfig

	switch ext {
	case ".yaml", ".yml":
		if err := yaml.Unmarshal(data, &config); err != nil {
			return nil, fmt.Errorf("config: failed to parse YAML: %w", err)
		}

	case ".toml":
		// For TOML support, we would use a TOML library
		// For now, we'll support YAML as the primary format
		return nil, fmt.Errorf("config: TOML support not yet implemented, use YAML instead")

	default:
		return nil, fmt.Errorf("config: unsupported file format: %s", ext)
	}

	// Validate configuration
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("config: validation failed: %w", err)
	}

	l.config = &config
	return &config, nil
}

// LoadFromEnvironment loads configuration overrides from environment variables
func LoadFromEnvironment(config *AppConfig) error {
	// Server settings
	if host := os.Getenv("RYZANSTEIN_SERVER_HOST"); host != "" {
		config.Server.Host = host
	}

	if port := os.Getenv("RYZANSTEIN_SERVER_PORT"); port != "" {
		var portNum int
		if _, err := fmt.Sscanf(port, "%d", &portNum); err != nil {
			return fmt.Errorf("config: invalid RYZANSTEIN_SERVER_PORT: %w", err)
		}
		config.Server.Port = portNum
	}

	if protocol := os.Getenv("RYZANSTEIN_SERVER_PROTOCOL"); protocol != "" {
		config.Server.Protocol = Protocol(protocol)
	}

	if tls := os.Getenv("RYZANSTEIN_SERVER_TLS"); tls != "" {
		config.Server.TLS = parseBool(tls)
	}

	if tlsVerify := os.Getenv("RYZANSTEIN_SERVER_TLS_VERIFY"); tlsVerify != "" {
		config.Server.TLSVerify = parseBool(tlsVerify)
	}

	// Inference settings
	if model := os.Getenv("RYZANSTEIN_DEFAULT_MODEL"); model != "" {
		config.Inference.DefaultModel = model
	}

	if timeout := os.Getenv("RYZANSTEIN_INFERENCE_TIMEOUT"); timeout != "" {
		duration, err := time.ParseDuration(timeout)
		if err != nil {
			return fmt.Errorf("config: invalid RYZANSTEIN_INFERENCE_TIMEOUT: %w", err)
		}
		config.Inference.Timeout = duration
	}

	if maxRetries := os.Getenv("RYZANSTEIN_MAX_RETRIES"); maxRetries != "" {
		var retries int
		if _, err := fmt.Sscanf(maxRetries, "%d", &retries); err != nil {
			return fmt.Errorf("config: invalid RYZANSTEIN_MAX_RETRIES: %w", err)
		}
		config.Inference.MaxRetries = retries
	}

	if temp := os.Getenv("RYZANSTEIN_TEMPERATURE"); temp != "" {
		var temperature float32
		if _, err := fmt.Sscanf(temp, "%f", &temperature); err != nil {
			return fmt.Errorf("config: invalid RYZANSTEIN_TEMPERATURE: %w", err)
		}
		config.Inference.Temperature = temperature
	}

	if topP := os.Getenv("RYZANSTEIN_TOP_P"); topP != "" {
		var topPValue float32
		if _, err := fmt.Sscanf(topP, "%f", &topPValue); err != nil {
			return fmt.Errorf("config: invalid RYZANSTEIN_TOP_P: %w", err)
		}
		config.Inference.TopP = topPValue
	}

	if maxTokens := os.Getenv("RYZANSTEIN_MAX_TOKENS"); maxTokens != "" {
		var tokens int
		if _, err := fmt.Sscanf(maxTokens, "%d", &tokens); err != nil {
			return fmt.Errorf("config: invalid RYZANSTEIN_MAX_TOKENS: %w", err)
		}
		config.Inference.MaxTokens = tokens
	}

	// Logging settings
	if logLevel := os.Getenv("RYZANSTEIN_LOG_LEVEL"); logLevel != "" {
		config.LogLevel = logLevel
	}

	if logFormat := os.Getenv("RYZANSTEIN_LOG_FORMAT"); logFormat != "" {
		config.LogFormat = logFormat
	}

	// Metrics settings
	if metricsEnabled := os.Getenv("RYZANSTEIN_METRICS_ENABLED"); metricsEnabled != "" {
		config.MetricsEnabled = parseBool(metricsEnabled)
	}

	if metricsPort := os.Getenv("RYZANSTEIN_METRICS_PORT"); metricsPort != "" {
		var port int
		if _, err := fmt.Sscanf(metricsPort, "%d", &port); err != nil {
			return fmt.Errorf("config: invalid RYZANSTEIN_METRICS_PORT: %w", err)
		}
		config.MetricsPort = port
	}

	// Final validation
	if err := config.Validate(); err != nil {
		return fmt.Errorf("config: validation after environment override failed: %w", err)
	}

	return nil
}

// Load loads configuration from file and applies environment overrides
func (l *Loader) Load() (*AppConfig, error) {
	// Start with defaults
	config := DefaultAppConfig()

	// Load from file if path provided
	if l.configPath != "" {
		fileConfig, err := l.LoadFromFile(l.configPath)
		if err != nil {
			return nil, err
		}
		config = *fileConfig
	}

	// Apply environment overrides
	if err := LoadFromEnvironment(&config); err != nil {
		return nil, err
	}

	l.config = &config
	return &config, nil
}

// LoadWithDefaults loads configuration with fallback to defaults
func LoadWithDefaults(filePath string) (*AppConfig, error) {
	loader := NewLoader(filePath)
	return loader.Load()
}

// MergeConfig merges a new config into an existing one
// New values take precedence over existing values
func MergeConfig(base, new *AppConfig) *AppConfig {
	merged := *base

	// Merge server config
	if new.Server.Host != "" {
		merged.Server.Host = new.Server.Host
	}
	if new.Server.Port != 0 {
		merged.Server.Port = new.Server.Port
	}
	if new.Server.Protocol != "" {
		merged.Server.Protocol = new.Server.Protocol
	}

	// Merge inference config
	if new.Inference.DefaultModel != "" {
		merged.Inference.DefaultModel = new.Inference.DefaultModel
	}
	if new.Inference.Timeout != 0 {
		merged.Inference.Timeout = new.Inference.Timeout
	}
	if new.Inference.MaxRetries != 0 {
		merged.Inference.MaxRetries = new.Inference.MaxRetries
	}
	if new.Inference.Temperature != 0 {
		merged.Inference.Temperature = new.Inference.Temperature
	}

	// Merge models
	if len(new.Models) > 0 {
		merged.Models = new.Models
	}

	// Merge logging config
	if new.LogLevel != "" {
		merged.LogLevel = new.LogLevel
	}
	if new.LogFormat != "" {
		merged.LogFormat = new.LogFormat
	}

	return &merged
}

// parseBool parses a boolean value from string
func parseBool(s string) bool {
	switch s {
	case "true", "1", "yes", "on", "enabled":
		return true
	case "false", "0", "no", "off", "disabled":
		return false
	default:
		return false
	}
}

// SaveToFile saves configuration to a file
func (a *AppConfig) SaveToFile(filePath string) error {
	// Marshal to YAML
	data, err := yaml.Marshal(a)
	if err != nil {
		return fmt.Errorf("config: failed to marshal YAML: %w", err)
	}

	// Create directory if needed
	dir := filepath.Dir(filePath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("config: failed to create directory: %w", err)
	}

	// Write file
	if err := os.WriteFile(filePath, data, 0644); err != nil {
		return fmt.Errorf("config: failed to write file: %w", err)
	}

	return nil
}

// Clone creates a deep copy of the configuration
func (a *AppConfig) Clone() *AppConfig {
	clone := *a

	// Clone server metadata
	if a.Server.Metadata != nil {
		clone.Server.Metadata = make(map[string]string)
		for k, v := range a.Server.Metadata {
			clone.Server.Metadata[k] = v
		}
	}

	// Clone models
	clone.Models = make(map[string]ModelConfig)
	for k, v := range a.Models {
		modelClone := v
		if v.Metadata != nil {
			modelClone.Metadata = make(map[string]string)
			for mk, mv := range v.Metadata {
				modelClone.Metadata[mk] = mv
			}
		}
		clone.Models[k] = modelClone
	}

	return &clone
}
