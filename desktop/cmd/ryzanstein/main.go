package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"sync"
	"time"

	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/agents"
	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/chat"
	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/config"
	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/ipc"
	"github.com/iamthegreatdestroyer/Ryzanstein/desktop/internal/models"
	"github.com/wailsapp/wails/v2/pkg/assetserver"
	"github.com/wailsapp/wails/v2/pkg/options"
	"github.com/wailsapp/wails/v2/pkg/runtime"
)

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

	// Initialize config manager
	var err error
	a.config, err = config.NewManager()
	if err != nil {
		log.Printf("[Desktop] Failed to load config: %v\n", err)
		runtime.MessageDialog(ctx, runtime.MessageDialogOptions{
			Type:    runtime.ErrorDialog,
			Title:   "Configuration Error",
			Message: fmt.Sprintf("Failed to load configuration: %v", err),
		})
		return
	}

	// Initialize services
	a.chat = chat.NewService()
	a.models = models.NewService(a.config)
	a.agents = agents.NewService()
	a.ipc = ipc.NewServer()

	// Start IPC server
	go a.startIPCServer()

	// Load models in background
	go a.models.LoadInstalledModels()

	// Update UI with initial state
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

	// Graceful shutdown
	a.chat.Close()
	a.ipc.Close()

	log.Println("[Desktop] Application shutdown complete")
}

// ============================================================================
// Chat Service Methods
// ============================================================================

// Message represents a chat message
type Message struct {
	ID        string                 `json:"id"`
	Role      string                 `json:"role"` // "user" or "assistant"
	Content   string                 `json:"content"`
	Timestamp int64                  `json:"timestamp"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// SendMessage sends a message and returns the response
func (a *App) SendMessage(userMessage string, modelID string, agentCodename string) (string, error) {
	log.Printf("[Chat] Sending message: %s (model: %s, agent: %s)\n",
		userMessage, modelID, agentCodename)

	// Create message context
	ctx, cancel := context.WithTimeout(a.ctx, 30*time.Second)
	defer cancel()

	// Send through chat service
	response, err := a.chat.SendMessage(ctx, userMessage, modelID, agentCodename)
	if err != nil {
		log.Printf("[Chat] Error sending message: %v\n", err)
		return "", err
	}

	// Emit message sent event
	runtime.EventsEmit(a.ctx, "chat:message", Message{
		ID:        fmt.Sprintf("msg_%d", time.Now().UnixNano()),
		Role:      "user",
		Content:   userMessage,
		Timestamp: time.Now().Unix(),
	})

	// Emit response event
	runtime.EventsEmit(a.ctx, "chat:response", Message{
		ID:        fmt.Sprintf("msg_%d", time.Now().UnixNano()),
		Role:      "assistant",
		Content:   response,
		Timestamp: time.Now().Unix(),
	})

	return response, nil
}

// GetHistory returns chat history
func (a *App) GetHistory(limit int) ([]Message, error) {
	log.Printf("[Chat] Fetching history (limit: %d)\n", limit)
	return a.chat.GetHistory(limit), nil
}

// ClearHistory clears chat history
func (a *App) ClearHistory() error {
	log.Println("[Chat] Clearing history")
	a.chat.ClearHistory()
	runtime.EventsEmit(a.ctx, "chat:cleared", nil)
	return nil
}

// ============================================================================
// Model Service Methods
// ============================================================================

// ModelInfo represents model information
type ModelInfo struct {
	ID            string `json:"id"`
	Name          string `json:"name"`
	Size          string `json:"size"`
	ContextLength int    `json:"contextLength"`
	Loaded        bool   `json:"loaded"`
	Status        string `json:"status"`
}

// ListModels returns available models
func (a *App) ListModels() ([]ModelInfo, error) {
	log.Println("[Models] Listing models")
	return a.models.ListModels(), nil
}

// LoadModel loads a model
func (a *App) LoadModel(modelID string) error {
	log.Printf("[Models] Loading model: %s\n", modelID)

	// Load model in background
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

// UnloadModel unloads a model
func (a *App) UnloadModel(modelID string) error {
	log.Printf("[Models] Unloading model: %s\n", modelID)
	return a.models.UnloadModel(modelID)
}

// ============================================================================
// Agent Service Methods
// ============================================================================

// AgentInfo represents agent information
type AgentInfo struct {
	Codename       string   `json:"codename"`
	Name           string   `json:"name"`
	Tier           int      `json:"tier"`
	Philosophy     string   `json:"philosophy"`
	Capabilities   []string `json:"capabilities"`
	MasteryDomains []string `json:"masteryDomains"`
}

// ListAgents returns available agents
func (a *App) ListAgents() ([]AgentInfo, error) {
	log.Println("[Agents] Listing agents")
	return a.agents.ListAgents(), nil
}

// InvokeAgent invokes an agent tool
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

// ConfigData represents configuration
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

// GetConfig returns current configuration
func (a *App) GetConfig() (ConfigData, error) {
	log.Println("[Config] Fetching configuration")
	return a.config.GetConfig(), nil
}

// SaveConfig saves configuration
func (a *App) SaveConfig(cfg ConfigData) error {
	log.Println("[Config] Saving configuration")
	err := a.config.SaveConfig(cfg)
	if err == nil {
		runtime.EventsEmit(a.ctx, "config:saved", cfg)
	}
	return err
}

// ============================================================================
// System Methods
// ============================================================================

// Greet returns a greeting
func (a *App) Greet(name string) string {
	log.Printf("[System] Greet: %s\n", name)
	return fmt.Sprintf("Hello %s, let's build amazing AI applications!", name)
}

// GetVersion returns application version
func (a *App) GetVersion() string {
	return "1.0.0"
}

// GetSystemInfo returns system information
func (a *App) GetSystemInfo() map[string]interface{} {
	return map[string]interface{}{
		"platform": runtime.Environment(a.ctx).Platform,
		"os":       runtime.Environment(a.ctx).BuildType,
		"arch":     os.Getenv("PROCESSOR_ARCHITECTURE"),
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
	err := runtime.Run(&runtime.Options{
		Title:             "Ryzanstein",
		Width:             1400,
		Height:            900,
		MinWidth:          800,
		MinHeight:         600,
		MaxWidth:          2560,
		MaxHeight:         1440,
		DisableFrameless:  false,
		Fullscreen:        false,
		AlwaysOnTop:       false,
		Frameless:         false,
		EnableDragAndDrop: true,
		AppType:           runtime.Desktop,
		AssetServer: &assetserver.AssetServer{
			Assets: assets,
		},
		BackgroundColour: &options.RGBA{R: 27, G: 38, B: 54, A: 1},
		OnStartup:        app.Startup,
		OnShutdown:       app.Shutdown,
		OnDomReady:       app.onDomReady,
		Bind: []interface{}{
			app,
		},
	})

	if err != nil {
		log.Fatalf("Fatal error: %v", err)
	}
}

// onDomReady is called when the DOM is ready
func (a *App) onDomReady(ctx context.Context) {
	log.Println("[Desktop] DOM ready")
	// Emit ready event to frontend
	runtime.EventsEmit(ctx, "dom:ready", nil)
}
