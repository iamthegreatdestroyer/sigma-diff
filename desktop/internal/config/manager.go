package config

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
)

// Manager handles application configuration
type Manager struct {
	config   ConfigData
	filePath string
}

// ConfigData represents application configuration
type ConfigData struct {
	Theme             string `json:"theme"`
	DefaultModel      string `json:"defaultModel"`
	DefaultAgent      string `json:"defaultAgent"`
	RyzansteinAPIURL  string `json:"ryzansteinApiUrl"`
	MCPServerURL      string `json:"mcpServerUrl"`
	AutoLoadLastModel bool   `json:"autoLoadLastModel"`
	EnableSystemTray  bool   `json:"enableSystemTray"`
	MinimizeToTray    bool   `json:"minimizeToTray"`
}

// NewManager creates a new config manager
func NewManager() (*Manager, error) {
	log.Println("[ConfigManager] Initializing")

	// Get config directory
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}

	configDir := filepath.Join(home, ".ryzanstein")
	if err := os.MkdirAll(configDir, 0755); err != nil {
		return nil, err
	}

	configFile := filepath.Join(configDir, "config.json")

	m := &Manager{
		filePath: configFile,
		config: ConfigData{
			Theme:             "dark",
			DefaultModel:      "ryzanstein-7b",
			DefaultAgent:      "@APEX",
			RyzansteinAPIURL:  "http://localhost:8000",
			MCPServerURL:      "localhost:50051",
			AutoLoadLastModel: true,
			EnableSystemTray:  true,
			MinimizeToTray:    true,
		},
	}

	// Load existing config if available
	if _, err := os.Stat(configFile); err == nil {
		data, err := os.ReadFile(configFile)
		if err == nil {
			json.Unmarshal(data, &m.config)
			log.Printf("[ConfigManager] Loaded config from %s\n", configFile)
		}
	} else {
		// Save default config
		m.SaveConfig(m.config)
	}

	return m, nil
}

// GetConfig returns current configuration
func (m *Manager) GetConfig() ConfigData {
	return m.config
}

// SaveConfig saves configuration to disk
func (m *Manager) SaveConfig(cfg ConfigData) error {
	log.Printf("[ConfigManager] Saving config to %s\n", m.filePath)

	m.config = cfg

	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}

	if err := os.WriteFile(m.filePath, data, 0644); err != nil {
		return fmt.Errorf("failed to save config: %w", err)
	}

	return nil
}

// GetString returns a string config value
func (m *Manager) GetString(key string) string {
	switch key {
	case "theme":
		return m.config.Theme
	case "defaultModel":
		return m.config.DefaultModel
	case "defaultAgent":
		return m.config.DefaultAgent
	case "ryzansteinApiUrl":
		return m.config.RyzansteinAPIURL
	case "mcpServerUrl":
		return m.config.MCPServerURL
	default:
		return ""
	}
}

// SetString sets a string config value
func (m *Manager) SetString(key, value string) {
	switch key {
	case "theme":
		m.config.Theme = value
	case "defaultModel":
		m.config.DefaultModel = value
	case "defaultAgent":
		m.config.DefaultAgent = value
	case "ryzansteinApiUrl":
		m.config.RyzansteinAPIURL = value
	case "mcpServerUrl":
		m.config.MCPServerURL = value
	}
}

// GetBool returns a boolean config value
func (m *Manager) GetBool(key string) bool {
	switch key {
	case "autoLoadLastModel":
		return m.config.AutoLoadLastModel
	case "enableSystemTray":
		return m.config.EnableSystemTray
	case "minimizeToTray":
		return m.config.MinimizeToTray
	default:
		return false
	}
}

// SetBool sets a boolean config value
func (m *Manager) SetBool(key string, value bool) {
	switch key {
	case "autoLoadLastModel":
		m.config.AutoLoadLastModel = value
	case "enableSystemTray":
		m.config.EnableSystemTray = value
	case "minimizeToTray":
		m.config.MinimizeToTray = value
	}
}
