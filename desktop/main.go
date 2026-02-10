package main

// This file is the entry point for Wails
// The actual implementation is in cmd/ryzanstein/main.go

import (
	"context"
	"embed"
	"fmt"
	"log"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/agents"
	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/chat"
	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/config"
	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/ipc"
	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/models"
	"github.com/wailsapp/wails/v2"
	"github.com/wailsapp/wails/v2/pkg/options"
	"github.com/wailsapp/wails/v2/pkg/runtime"
)

//go:embed all:frontend/dist
var assets embed.FS

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

// App struct is where we bind all application methods
type App struct {
	ctx       context.Context
	chat      *chat.Service
	models    *models.Service
	agents    *agents.Service
	config    *config.Manager
	ipc       *ipc.Server
	mu        sync.RWMutex
	isRunning bool
}

// NewApp creates a new App application struct
func NewApp() *App {
	return &App{
		isRunning: false,
	}
}

// Startup is called at application startup
func (a *App) Startup(ctx context.Context) {
	a.ctx = ctx
	a.isRunning = true
	log.Println("[Desktop] Application starting up")

	var err error
	a.config, err = config.NewManager()
	if err != nil {
		log.Printf("[Desktop] Failed to load config: %v\n", err)
		runtime.MessageDialog(ctx, runtime.MessageDialogOptions{
			Type:    runtime.ErrorDialog,
			Title:   "Startup Error",
			Message: fmt.Sprintf("Failed to load config: %v", err),
		})
		return
	}

	a.chat = chat.NewService()
	a.models = models.NewService(a.config)
	a.agents = agents.NewService()
	a.ipc = ipc.NewServer()

	go a.startIPCServer()
	go a.models.LoadInstalledModels()

	runtime.EventsEmit(ctx, "app:ready", map[string]interface{}{
		"version":   "1.0.0",
		"timestamp": time.Now().Unix(),
	})

	log.Println("[Desktop] Application ready")
}

// Shutdown is called at application termination
func (a *App) Shutdown(ctx context.Context) {
	log.Println("[Desktop] Application shutting down")
	a.isRunning = false

	a.chat.Close()
	a.ipc.Close()

	log.Println("[Desktop] Application shutdown complete")
}

// ============================================================================
// Chat Service Methods
// ============================================================================

type Message struct {
	ID        string                 `json:"id"`
	Role      string                 `json:"role"`
	Content   string                 `json:"content"`
	Timestamp int64                  `json:"timestamp"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

func (a *App) SendMessage(userMessage string, modelID string, agentCodename string) (string, error) {
	log.Printf("[Chat] Sending message: %s (model: %s, agent: %s)\n",
		userMessage, modelID, agentCodename)

	ctx, cancel := context.WithTimeout(a.ctx, 30*time.Second)
	defer cancel()

	// Create inference request (for future use)
	_ = &InferenceRequest{
		ModelID:     modelID,
		Prompt:      userMessage,
		MaxTokens:   2048,
		Temperature: 0.7,
		TopP:        0.9,
		Metadata: map[string]interface{}{
			"agent": agentCodename,
		},
	}

	// Execute inference (currently mock, but structured for real inference)
	var response *InferenceResponse

	// For now, provide a mock response that acknowledges the agent
	// TODO: Connect to real inference service
	responseText := fmt.Sprintf("🤖 %s here! I've analyzed your request: '%s'. This is currently a simulated response - real LLM inference will be connected soon.",
		agentCodename, userMessage)

	response = &InferenceResponse{
		Text:     responseText,
		Tokens:   len(strings.Split(responseText, " ")), // Rough token count
		Duration: 150 * time.Millisecond,                // Mock duration
		Model:    modelID,
		Metadata: map[string]interface{}{
			"timestamp": time.Now(),
			"agent":     agentCodename,
		},
	}

	// Add to chat history
	a.chat.AddMessage(ctx, "user", userMessage, modelID, agentCodename)
	a.chat.AddMessage(ctx, "assistant", response.Text, modelID, agentCodename)

	runtime.EventsEmit(a.ctx, "chat:message", Message{
		ID:        fmt.Sprintf("msg_%d", time.Now().UnixNano()),
		Role:      "user",
		Content:   userMessage,
		Timestamp: time.Now().Unix(),
	})

	runtime.EventsEmit(a.ctx, "chat:response", Message{
		ID:        fmt.Sprintf("msg_%d", time.Now().UnixNano()),
		Role:      "assistant",
		Content:   response.Text,
		Timestamp: time.Now().Unix(),
	})

	return response.Text, nil
}

func (a *App) GetHistory(limit int) ([]Message, error) {
	log.Printf("[Chat] Fetching history (limit: %d)\n", limit)
	chatHistory := a.chat.GetHistory(limit)
	messages := make([]Message, len(chatHistory))
	for i, h := range chatHistory {
		messages[i] = Message{
			ID:        h.ID,
			Role:      h.Role,
			Content:   h.Content,
			Timestamp: h.Timestamp,
		}
	}
	return messages, nil
}

func (a *App) ClearHistory() error {
	log.Println("[Chat] Clearing history")
	a.chat.ClearHistory()
	runtime.EventsEmit(a.ctx, "chat:cleared", nil)
	return nil
}

// ============================================================================
// Model Service Methods
// ============================================================================

type ModelInfo struct {
	ID            string `json:"id"`
	Name          string `json:"name"`
	Size          string `json:"size"`
	ContextLength int    `json:"contextLength"`
	Loaded        bool   `json:"loaded"`
	Status        string `json:"status"`
}

func (a *App) ListModels() ([]ModelInfo, error) {
	log.Println("[Models] Listing models")
	svcModels := a.models.ListModels()
	models := make([]ModelInfo, len(svcModels))
	for i, m := range svcModels {
		models[i] = ModelInfo{
			ID:            m.ID,
			Name:          m.Name,
			Size:          m.Size,
			ContextLength: m.ContextLength,
			Loaded:        m.Loaded,
			Status:        m.Status,
		}
	}
	return models, nil
}

func (a *App) LoadModel(modelID string) error {
	log.Printf("[Models] Loading model: %s\n", modelID)
	go func() {
		err := a.models.LoadModel(modelID)
		if err != nil {
			log.Printf("[Models] Error loading model: %v\n", err)
			runtime.EventsEmit(a.ctx, "model:loadError", map[string]interface{}{
				"modelID": modelID,
				"error":   err.Error(),
			})
			return
		}
		log.Printf("[Models] Successfully loaded: %s\n", modelID)
		runtime.EventsEmit(a.ctx, "model:loaded", map[string]interface{}{
			"modelID": modelID,
		})
	}()
	return nil
}

func (a *App) UnloadModel(modelID string) error {
	log.Printf("[Models] Unloading model: %s\n", modelID)
	return a.models.UnloadModel(modelID)
}

// ============================================================================
// Agent Service Methods
// ============================================================================

type AgentInfo struct {
	Codename       string   `json:"codename"`
	Name           string   `json:"name"`
	Tier           int      `json:"tier"`
	Philosophy     string   `json:"philosophy"`
	Capabilities   []string `json:"capabilities"`
	MasteryDomains []string `json:"masteryDomains"`
}

func (a *App) ListAgents() ([]string, error) {
	log.Println("[Agents] Listing agents")
	return a.agents.ListAgents(), nil
}

func (a *App) InvokeAgent(agentCodename string, toolName string, parameters map[string]interface{}) (interface{}, error) {
	log.Printf("[Agents] Invoking %s.%s\n", agentCodename, toolName)
	ctx, cancel := context.WithTimeout(a.ctx, 30*time.Second)
	defer cancel()
	result, err := a.agents.InvokeTool(ctx, agentCodename, toolName, parameters)
	if err != nil {
		log.Printf("[Agents] Error invoking agent: %v\n", err)
		return nil, err
	}
	runtime.EventsEmit(a.ctx, "agent:invoked", map[string]interface{}{
		"agent":  agentCodename,
		"tool":   toolName,
		"result": result,
	})
	return result, nil
}

// ============================================================================
// Config Service Methods
// ============================================================================

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

func (a *App) GetConfig() (ConfigData, error) {
	log.Println("[Config] Fetching configuration")
	svcCfg := a.config.GetConfig()
	return ConfigData{
		Theme:             svcCfg.Theme,
		DefaultModel:      svcCfg.DefaultModel,
		DefaultAgent:      svcCfg.DefaultAgent,
		RyzansteinAPIURL:  svcCfg.RyzansteinAPIURL,
		MCPServerURL:      svcCfg.MCPServerURL,
		AutoLoadLastModel: svcCfg.AutoLoadLastModel,
		EnableSystemTray:  svcCfg.EnableSystemTray,
		MinimizeToTray:    svcCfg.MinimizeToTray,
	}, nil
}

func (a *App) SaveConfig(cfg ConfigData) error {
	log.Println("[Config] Saving configuration")
	svcCfg := config.ConfigData{
		Theme:             cfg.Theme,
		DefaultModel:      cfg.DefaultModel,
		DefaultAgent:      cfg.DefaultAgent,
		RyzansteinAPIURL:  cfg.RyzansteinAPIURL,
		MCPServerURL:      cfg.MCPServerURL,
		AutoLoadLastModel: cfg.AutoLoadLastModel,
		EnableSystemTray:  cfg.EnableSystemTray,
		MinimizeToTray:    cfg.MinimizeToTray,
	}
	err := a.config.SaveConfig(svcCfg)
	if err == nil {
		runtime.EventsEmit(a.ctx, "config:saved", cfg)
	}
	return err
}

// ============================================================================
// System Methods
// ============================================================================

func (a *App) Greet(name string) string {
	log.Printf("[System] Greet: %s\n", name)
	return fmt.Sprintf("Hello %s, let's build amazing AI applications!", name)
}

func (a *App) GetVersion() string {
	return "1.0.0"
}

func (a *App) GetSystemInfo() map[string]interface{} {
	return map[string]interface{}{
		"arch": os.Getenv("PROCESSOR_ARCHITECTURE"),
	}
}

// ============================================================================
// IPC Server
// ============================================================================

func (a *App) startIPCServer() {
	err := a.ipc.Start()
	if err != nil {
		log.Printf("[IPC] Error starting server: %v\n", err)
		runtime.EventsEmit(a.ctx, "ipc:error", err.Error())
		return
	}
	log.Println("[IPC] Server started")
}

// ============================================================================
// Main Entry Point
// ============================================================================

func main() {
	app := NewApp()

	// Create application with options
	err := wails.Run(&options.App{
		Title:      "Ryzanstein",
		Width:      1400,
		Height:     900,
		MinWidth:   800,
		MinHeight:  600,
		Assets:     assets,
		OnStartup:  app.Startup,
		OnShutdown: app.Shutdown,
		OnDomReady: app.onDomReady,
		Bind: []interface{}{
			app,
		},
	})

	if err != nil {
		log.Fatalf("Fatal error: %v", err)
	}
}

func (a *App) onDomReady(ctx context.Context) {
	log.Println("[Desktop] DOM ready")
	runtime.EventsEmit(ctx, "dom:ready", nil)
}
