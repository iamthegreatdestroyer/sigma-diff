# ============================================================================
# 🚀 RYZANSTEIN DESKTOP APP - MASTER SETUP SCRIPT
# ============================================================================
# 
# Purpose: Fully automated setup of Desktop Application (Wails + Go + React)
# Author: ARCHITECT Mode
# Date: January 8, 2026
# 
# This script handles:
# - Dependency verification (Go, Node.js, Wails)
# - Project structure initialization
# - File generation for all components
# - NPM dependency installation
# - Build configuration
# - Compilation and packaging
#
# Usage: .\SETUP_DESKTOP_APP_MASTER.ps1
# ============================================================================

param(
    [switch]$SkipDependencies = $false,
    [switch]$DevelopmentOnly = $false,
    [string]$WailsVersion = "latest"
)

# ============================================================================
# CONFIGURATION
# ============================================================================

$ErrorActionPreference = "Stop"
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = $scriptPath
$frontendPath = Join-Path $projectRoot "packages\desktop"
$backendPath = Join-Path $projectRoot "cmd\ryzanstein"
$internalPath = Join-Path $projectRoot "internal"

$colors = @{
    Success  = "Green"
    Warning  = "Yellow"
    Error    = "Red"
    Info     = "Cyan"
    Progress = "Magenta"
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

function Write-Header {
    param([string]$message)
    Write-Host ""
    Write-Host "╔════════════════════════════════════════════════════════════════════════════════╗" -ForegroundColor $colors.Info
    Write-Host "║ $($message.PadRight(78)) ║" -ForegroundColor $colors.Info
    Write-Host "╚════════════════════════════════════════════════════════════════════════════════╝" -ForegroundColor $colors.Info
    Write-Host ""
}

function Write-Step {
    param([string]$message)
    Write-Host "▶ $message" -ForegroundColor $colors.Progress
}

function Write-Success {
    param([string]$message)
    Write-Host "✓ $message" -ForegroundColor $colors.Success
}

function Write-Warning {
    param([string]$message)
    Write-Host "⚠ $message" -ForegroundColor $colors.Warning
}

function Write-Error {
    param([string]$message)
    Write-Host "✗ $message" -ForegroundColor $colors.Error
}

function Test-CommandExists {
    param([string]$command)
    $null = Get-Command $command -ErrorAction SilentlyContinue
    return $?
}

function Ensure-Directory {
    param([string]$path)
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
        Write-Success "Created directory: $path"
    }
}

function Create-FileIfNotExists {
    param([string]$path, [string]$content)
    if (Test-Path $path) {
        Write-Warning "File already exists: $path (skipping)"
        return
    }
    
    $dir = Split-Path -Parent $path
    Ensure-Directory $dir
    
    Set-Content -Path $path -Value $content -Encoding UTF8
    Write-Success "Created file: $path"
}

# ============================================================================
# DEPENDENCY VERIFICATION
# ============================================================================

function Verify-Dependencies {
    Write-Header "STEP 1: VERIFYING DEPENDENCIES"
    
    $allGood = $true
    
    # Check Go
    if (Test-CommandExists "go") {
        $goVersion = (go version).Split()[2]
        Write-Success "Go installed: $goVersion"
    }
    else {
        Write-Error "Go not installed. Download from https://golang.org"
        $allGood = $false
    }
    
    # Check Node.js
    if (Test-CommandExists "node") {
        $nodeVersion = (node --version)
        Write-Success "Node.js installed: $nodeVersion"
    }
    else {
        Write-Error "Node.js not installed. Download from https://nodejs.org"
        $allGood = $false
    }
    
    # Check Wails
    if (Test-CommandExists "wails") {
        $wailsVersion = (wails version)
        Write-Success "Wails installed: $wailsVersion"
    }
    else {
        if ($SkipDependencies) {
            Write-Warning "Wails not installed. Run: go install github.com/wailsapp/wails/v3/cmd/wails@latest"
            $allGood = $false
        }
        else {
            Write-Step "Installing Wails..."
            go install github.com/wailsapp/wails/v3/cmd/wails@latest
            Write-Success "Wails installed"
        }
    }
    
    if (-not $allGood) {
        throw "Dependencies verification failed. Please install missing dependencies."
    }
}

# ============================================================================
# DIRECTORY STRUCTURE
# ============================================================================

function Initialize-DirectoryStructure {
    Write-Header "STEP 2: INITIALIZING DIRECTORY STRUCTURE"
    
    # Backend directories
    Ensure-Directory (Join-Path $internalPath "app")
    Ensure-Directory (Join-Path $internalPath "handlers")
    Ensure-Directory (Join-Path $internalPath "services")
    Ensure-Directory (Join-Path $internalPath "config")
    
    # Frontend directories
    Ensure-Directory (Join-Path $frontendPath "src\components")
    Ensure-Directory (Join-Path $frontendPath "src\hooks")
    Ensure-Directory (Join-Path $frontendPath "src\store")
    Ensure-Directory (Join-Path $frontendPath "src\services")
    Ensure-Directory (Join-Path $frontendPath "src\types")
    Ensure-Directory (Join-Path $frontendPath "src\utils")
    
    Write-Success "Directory structure initialized"
}

# ============================================================================
# BACKEND FILE GENERATION
# ============================================================================

function Generate-BackendFiles {
    Write-Header "STEP 3: GENERATING BACKEND FILES"
    
    # Main.go
    $mainGo = @'
package main

import (
	"context"
	"fmt"

	"s:\Ryot\internal\app"
	"github.com/wailsapp/wails/v3/pkg/application"
)

func main() {
	app := application.New(application.Options{
		Title: "Ryzanstein",
		Width: 1400,
		Height: 900,
		MinWidth: 800,
		MinHeight: 600,
	})

	app.OnStartup(func(ctx context.Context) {
		fmt.Println("Ryzanstein Desktop App starting...")
	})

	if err := app.Run(); err != nil {
		panic(err)
	}
}
'@
    Create-FileIfNotExists (Join-Path $backendPath "main.go") $mainGo
    
    # App.go
    $appGo = @'
package app

import (
	"context"
)

type App struct {
	ctx context.Context
}

func NewApp() *App {
	return &App{}
}

func (a *App) Startup(ctx context.Context) {
	a.ctx = ctx
}

func (a *App) Shutdown(ctx context.Context) {
}

func (a *App) Greet(name string) string {
	return "Hello " + name
}
'@
    Create-FileIfNotExists (Join-Path $internalPath "app\app.go") $appGo
    
    # Chat Handler
    $chatHandler = @'
package handlers

import (
	"context"
	"encoding/json"
)

type ChatRequest struct {
	Message string `json:"message"`
	AgentID string `json:"agent_id"`
	ModelID string `json:"model_id"`
}

type ChatResponse struct {
	Response string `json:"response"`
	TraceID  string `json:"trace_id"`
}

type ChatHandler struct{}

func NewChatHandler() *ChatHandler {
	return &ChatHandler{}
}

func (ch *ChatHandler) SendMessage(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	// Implementation will connect to REST API
	return &ChatResponse{
		Response: "Echo: " + req.Message,
		TraceID:  "trace-123",
	}, nil
}

func (ch *ChatHandler) GetHistory(ctx context.Context, agentID string) ([]map[string]interface{}, error) {
	return []map[string]interface{}{}, nil
}
'@
    Create-FileIfNotExists (Join-Path $internalPath "handlers\chat.go") $chatHandler
    
    # Models Handler
    $modelsHandler = @'
package handlers

import (
	"context"
)

type Model struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

type ModelsHandler struct{}

func NewModelsHandler() *ModelsHandler {
	return &ModelsHandler{}
}

func (mh *ModelsHandler) ListModels(ctx context.Context) ([]Model, error) {
	return []Model{
		{ID: "model-1", Name: "Llama 2"},
		{ID: "model-2", Name: "Mistral"},
	}, nil
}

func (mh *ModelsHandler) LoadModel(ctx context.Context, modelID string) error {
	return nil
}
'@
    Create-FileIfNotExists (Join-Path $internalPath "handlers\models.go") $modelsHandler
    
    # Agents Handler
    $agentsHandler = @'
package handlers

import (
	"context"
)

type Agent struct {
	ID   string `json:"id"`
	Name string `json:"name"`
	Type string `json:"type"`
}

type AgentsHandler struct{}

func NewAgentsHandler() *AgentsHandler {
	return &AgentsHandler{}
}

func (ah *AgentsHandler) ListAgents(ctx context.Context) ([]Agent, error) {
	return []Agent{
		{ID: "agent-1", Name: "APEX", Type: "Engineering"},
		{ID: "agent-2", Name: "ARCHITECT", Type: "Design"},
	}, nil
}

func (ah *AgentsHandler) InvokeAgent(ctx context.Context, agentID string, prompt string) (string, error) {
	return "Response from " + agentID, nil
}
'@
    Create-FileIfNotExists (Join-Path $internalPath "handlers\agents.go") $agentsHandler
    
    Write-Success "Backend files generated"
}

# ============================================================================
# FRONTEND FILE GENERATION
# ============================================================================

function Generate-FrontendFiles {
    Write-Header "STEP 4: GENERATING FRONTEND FILES"
    
    # ChatPanel Component
    $chatPanel = @'
import React, { useState, useRef, useEffect } from 'react';
import { useChat } from '../hooks/useChat';
import './ChatPanel.css';

export const ChatPanel: React.FC = () => {
    const { messages, sendMessage, loading } = useChat();
    const [input, setInput] = useState('');
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const handleSend = async () => {
        if (input.trim()) {
            await sendMessage(input);
            setInput('');
        }
    };

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    return (
        <div className="chat-panel">
            <div className="messages">
                {messages.map((msg, idx) => (
                    <div key={idx} className={`message ${msg.role}`}>
                        {msg.content}
                    </div>
                ))}
                <div ref={messagesEndRef} />
            </div>
            <div className="input-area">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                    placeholder="Type a message..."
                    disabled={loading}
                />
                <button onClick={handleSend} disabled={loading}>
                    {loading ? 'Sending...' : 'Send'}
                </button>
            </div>
        </div>
    );
};
'@
    Create-FileIfNotExists (Join-Path $frontendPath "src\components\ChatPanel.tsx") $chatPanel
    
    # useChat Hook
    $useChat = @'
import { useState, useCallback } from 'react';
import { RyzansteinAPI } from '../services/api';

interface Message {
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp?: Date;
}

export const useChat = () => {
    const api = new RyzansteinAPI();
    const [messages, setMessages] = useState<Message[]>([]);
    const [loading, setLoading] = useState(false);

    const sendMessage = useCallback(async (content: string) => {
        setLoading(true);
        try {
            setMessages(prev => [...prev, { role: 'user', content, timestamp: new Date() }]);

            const response = await api.chat(content, 'default', 'default');
            setMessages(prev => [...prev, { 
                role: 'assistant', 
                content: response.response,
                timestamp: new Date() 
            }]);
        } catch (error) {
            setMessages(prev => [...prev, { 
                role: 'system', 
                content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
                timestamp: new Date()
            }]);
        } finally {
            setLoading(false);
        }
    }, []);

    return { messages, sendMessage, loading };
};
'@
    Create-FileIfNotExists (Join-Path $frontendPath "src\hooks\useChat.ts") $useChat
    
    # API Service
    $apiService = @'
import axios from 'axios';

export class RyzansteinAPI {
    private baseURL: string = 'http://localhost:8000';
    private client = axios.create({
        baseURL: this.baseURL,
        timeout: 30000,
    });

    async chat(message: string, agentId: string, modelId: string) {
        const response = await this.client.post('/chat', {
            message,
            agent_id: agentId,
            model_id: modelId,
        });
        return response.data;
    }

    async listModels() {
        const response = await this.client.get('/models');
        return response.data;
    }

    async listAgents() {
        const response = await this.client.get('/agents');
        return response.data;
    }

    async loadModel(modelId: string) {
        return this.client.post('/models/load', { model_id: modelId });
    }
}
'@
    Create-FileIfNotExists (Join-Path $frontendPath "src\services\api.ts") $apiService
    
    # Chat Store (Zustand)
    $chatStore = @'
import create from 'zustand';

interface Message {
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp?: Date;
}

interface ChatStore {
    messages: Message[];
    addMessage: (message: Message) => void;
    clearMessages: () => void;
}

export const useChatStore = create<ChatStore>((set) => ({
    messages: [],
    addMessage: (message) =>
        set((state) => ({
            messages: [...state.messages, { ...message, timestamp: new Date() }],
        })),
    clearMessages: () => set({ messages: [] }),
}));
'@
    Create-FileIfNotExists (Join-Path $frontendPath "src\store\chatStore.ts") $chatStore
    
    # CSS
    $chatPanelCss = @'
.chat-panel {
    display: flex;
    flex-direction: column;
    height: 100vh;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.messages {
    flex: 1;
    overflow-y: auto;
    padding: 1rem;
    background: var(--vscode-editor-background, #1e1e1e);
    color: var(--vscode-editor-foreground, #d4d4d4);
}

.message {
    margin: 0.5rem 0;
    padding: 0.75rem;
    border-radius: 4px;
    word-wrap: break-word;
}

.message.user {
    background: var(--vscode-button-background, #0e639c);
    color: var(--vscode-button-foreground, #fff);
    margin-left: 2rem;
}

.message.assistant {
    background: var(--vscode-editor-lineHighlightBackground, #2d2d30);
    margin-right: 2rem;
}

.input-area {
    display: flex;
    gap: 0.5rem;
    padding: 1rem;
    border-top: 1px solid var(--vscode-panel-border, #3e3e42);
    background: var(--vscode-panel-background, #252526);
}

.input-area input {
    flex: 1;
    padding: 0.75rem;
    border: 1px solid var(--vscode-input-border, #3e3e42);
    background: var(--vscode-input-background, #3c3c3c);
    color: var(--vscode-input-foreground, #d4d4d4);
    border-radius: 4px;
    font-size: 14px;
}

.input-area button {
    padding: 0.75rem 1.5rem;
    background: var(--vscode-button-background, #0e639c);
    color: var(--vscode-button-foreground, #fff);
    border: none;
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.2s;
}

.input-area button:hover {
    background: var(--vscode-button-hoverBackground, #1177bb);
}

.input-area button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}
'@
    Create-FileIfNotExists (Join-Path $frontendPath "src\components\ChatPanel.css") $chatPanelCss
    
    Write-Success "Frontend files generated"
}

# ============================================================================
# NPM SETUP
# ============================================================================

function Setup-NPM {
    Write-Header "STEP 5: SETTING UP NPM DEPENDENCIES"
    
    Write-Step "Installing npm packages..."
    Push-Location $frontendPath
    
    try {
        npm install axios zustand react-router-dom --save
        npm install -D @types/react @types/react-dom typescript --save-dev
        Write-Success "NPM dependencies installed"
    }
    finally {
        Pop-Location
    }
}

# ============================================================================
# PACKAGE.JSON SETUP
# ============================================================================

function Setup-PackageJson {
    Write-Header "STEP 6: CONFIGURING PACKAGE.JSON"
    
    $packageJsonPath = Join-Path $frontendPath "package.json"
    
    if (-not (Test-Path $packageJsonPath)) {
        $packageJson = @{
            name            = "ryzanstein-desktop"
            version         = "1.0.0"
            type            = "module"
            scripts         = @{
                dev     = "vite"
                build   = "tsc && vite build"
                preview = "vite preview"
            }
            dependencies    = @{
                react       = "^18.0.0"
                "react-dom" = "^18.0.0"
                axios       = "^1.6.0"
                zustand     = "^4.4.0"
            }
            devDependencies = @{
                typescript         = "^5.0.0"
                vite               = "^5.0.0"
                "@types/react"     = "^18.0.0"
                "@types/react-dom" = "^18.0.0"
            }
        } | ConvertTo-Json
        
        Set-Content -Path $packageJsonPath -Value $packageJson -Encoding UTF8
        Write-Success "Created package.json"
    }
}

# ============================================================================
# GO MOD SETUP
# ============================================================================

function Setup-GoMod {
    Write-Header "STEP 7: CONFIGURING GO MODULE"
    
    $goModPath = Join-Path $projectRoot "go.mod"
    
    if (-not (Test-Path $goModPath)) {
        Push-Location $projectRoot
        try {
            go mod init github.com/iamthegreatdestroyer/ryzanstein
            go get github.com/wailsapp/wails/v3@latest
            Write-Success "Go module initialized"
        }
        finally {
            Pop-Location
        }
    }
    else {
        Write-Success "go.mod already exists"
    }
}

# ============================================================================
# WAILS CONFIGURATION
# ============================================================================

function Setup-WailsConfig {
    Write-Header "STEP 8: CONFIGURING WAILS"
    
    $wailsJsonPath = Join-Path $projectRoot "wails.json"
    
    if (-not (Test-Path $wailsJsonPath)) {
        $wailsJson = @{
            name       = "Ryzanstein"
            type       = "desktop"
            outputType = "desktop"
            frontend   = "packages/desktop"
            build      = @{
                appType  = "desktop"
                frontend = @{
                    dir     = "packages/desktop"
                    install = "npm install"
                    build   = "npm run build"
                    dev     = "npm run dev"
                }
                backend  = @{
                    main = "cmd/ryzanstein/main.go"
                }
            }
            app        = @{
                title     = "Ryzanstein"
                width     = 1400
                height    = 900
                minWidth  = 800
                minHeight = 600
                menu      = "AppMenu"
            }
        } | ConvertTo-Json -Depth 10
        
        Set-Content -Path $wailsJsonPath -Value $wailsJson -Encoding UTF8
        Write-Success "Created wails.json"
    }
    else {
        Write-Success "wails.json already exists"
    }
}

# ============================================================================
# BUILD
# ============================================================================

function Build-Application {
    Write-Header "STEP 9: BUILDING APPLICATION"
    
    if ($DevelopmentOnly) {
        Write-Step "Building for development..."
        Push-Location $projectRoot
        try {
            wails dev
        }
        finally {
            Pop-Location
        }
    }
    else {
        Write-Step "Building for production..."
        Push-Location $projectRoot
        try {
            wails build -nsis
            Write-Success "Build completed successfully"
        }
        finally {
            Pop-Location
        }
    }
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

function Main {
    Clear-Host
    Write-Header "RYZANSTEIN DESKTOP APP - AUTOMATED SETUP"
    
    try {
        Verify-Dependencies
        Initialize-DirectoryStructure
        Generate-BackendFiles
        Generate-FrontendFiles
        Setup-PackageJson
        Setup-GoMod
        Setup-WailsConfig
        Setup-NPM
        
        Write-Header "✅ SETUP COMPLETE"
        
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor $colors.Info
        Write-Host "  1. cd s:\Ryot\desktop" -ForegroundColor $colors.Progress
        Write-Host "  2. .\SETUP_DESKTOP_APP_MASTER.ps1 -DevelopmentOnly" -ForegroundColor $colors.Progress
        Write-Host "     (or .\SETUP_DESKTOP_APP_MASTER.ps1 for production build)" -ForegroundColor $colors.Progress
        Write-Host ""
        
    }
    catch {
        Write-Error "Setup failed: $_"
        exit 1
    }
}

# Run main
Main
