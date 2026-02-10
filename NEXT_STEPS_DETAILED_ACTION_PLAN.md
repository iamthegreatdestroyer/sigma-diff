# 📋 DETAILED NEXT STEPS ACTION PLAN

## Desktop App + VS Code Extension Integration

**Date**: January 8, 2026  
**Current Branch**: sprint6/api-integration  
**Status**: Post-Deployment - Ready for Next Phase  
**Created By**: ARCHITECT Mode

---

## 🎯 OVERALL VISION

Integrate the Ryzanstein backend (REST/gRPC APIs) with both the **Desktop Application** (Wails + Go) and **VS Code Extension** (TypeScript) to provide unified AI agent access across multiple platforms.

---

## 📱 PHASE 1: DESKTOP APPLICATION DEVELOPMENT

### Current State

```
desktop/
├── build.sh
├── cmd/                    (Go backend)
├── internal/               (Backend logic)
├── packages/               (Frontend - React/Vue)
├── go.mod                  (Go dependencies)
└── wails.json             (Wails configuration)
```

### Architecture

- **Backend**: Go (Wails IPC bridge)
- **Frontend**: React/TypeScript (desktop UI)
- **API Layer**: HTTP client to REST/gRPC endpoints
- **State Management**: Redux/Zustand

---

## 🖥️ DESKTOP APP - STEP-BY-STEP INSTRUCTIONS

### **STEP 1: Setup Desktop Development Environment** (30 min)

#### Prerequisites

```powershell
# Install Wails
go install github.com/wailsapp/wails/v3/cmd/wails@latest

# Verify installation
wails version

# Install Node.js dependencies
cd s:\Ryot\desktop\packages\desktop
npm install
```

#### Tasks

1. **Initialize Wails Project Structure**

   ```bash
   cd s:\Ryot\desktop
   wails doctor  # Verify all dependencies
   ```

2. **Setup Frontend Dependencies**

   ```bash
   cd packages/desktop
   npm install axios zustand react-router-dom
   npm run dev  # Start dev server
   ```

3. **Create Frontend Folder Structure**

   ```
   packages/desktop/src/
   ├── components/
   │   ├── ChatPanel.tsx
   │   ├── AgentSelector.tsx
   │   ├── ModelManager.tsx
   │   └── Settings.tsx
   ├── hooks/
   │   ├── useChat.ts
   │   ├── useAgents.ts
   │   └── useModels.ts
   ├── store/
   │   ├── chatStore.ts
   │   ├── agentStore.ts
   │   └── configStore.ts
   ├── services/
   │   ├── api.ts
   │   └── ryzansteinAPI.ts
   ├── types/
   │   └── index.ts
   ├── App.tsx
   └── main.tsx
   ```

4. **Create Go Backend Structure**

   ```
   cmd/ryzanstein/
   └── main.go               (Entry point)

   internal/
   ├── app/
   │   └── app.go           (Wails app struct)
   ├── handlers/
   │   ├── chat.go
   │   ├── models.go
   │   ├── agents.go
   │   └── config.go
   ├── services/
   │   ├── api_client.go    (HTTP/gRPC client)
   │   └── model_manager.go
   └── config/
       └── config.go
   ```

---

### **STEP 2: Build API Client Layer** (45 min)

#### Create Go API Client (`internal/services/api_client.go`)

```go
package services

import (
    "context"
    "encoding/json"
    "fmt"
    "net/http"
    "time"
)

type RyzansteinClient struct {
    baseURL string
    client  *http.Client
}

type ChatRequest struct {
    Message  string `json:"message"`
    AgentID  string `json:"agent_id"`
    ModelID  string `json:"model_id"`
    History  []Message `json:"history"`
}

type ChatResponse struct {
    Response string    `json:"response"`
    TraceID  string    `json:"trace_id"`
    Duration time.Duration `json:"duration"`
}

type Message struct {
    Role    string `json:"role"`
    Content string `json:"content"`
}

func NewRyzansteinClient(baseURL string) *RyzansteinClient {
    return &RyzansteinClient{
        baseURL: baseURL,
        client: &http.Client{
            Timeout: 30 * time.Second,
        },
    }
}

// Chat sends a message to the chat API
func (rc *RyzansteinClient) Chat(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
    // Implementation details...
}

// ListModels fetches available models
func (rc *RyzansteinClient) ListModels(ctx context.Context) ([]Model, error) {
    // Implementation details...
}

// ListAgents fetches available agents
func (rc *RyzansteinClient) ListAgents(ctx context.Context) ([]Agent, error) {
    // Implementation details...
}

// LoadModel loads a specific model
func (rc *RyzansteinClient) LoadModel(ctx context.Context, modelID string) error {
    // Implementation details...
}
```

#### Create TypeScript API Service (`packages/desktop/src/services/api.ts`)

```typescript
import axios from "axios";

export class RyzansteinAPI {
  private baseURL: string;
  private client: axios.AxiosInstance;

  constructor(baseURL: string = "http://localhost:8000") {
    this.baseURL = baseURL;
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        "Content-Type": "application/json",
      },
    });
  }

  async chat(message: string, agentId: string, modelId: string): Promise<any> {
    // Implementation
  }

  async listModels(): Promise<any[]> {
    // Implementation
  }

  async listAgents(): Promise<any[]> {
    // Implementation
  }

  async loadModel(modelId: string): Promise<void> {
    // Implementation
  }

  async getConfig(): Promise<any> {
    // Implementation
  }

  async saveConfig(config: any): Promise<void> {
    // Implementation
  }
}
```

---

### **STEP 3: Create React Components** (60 min)

#### ChatPanel Component (`packages/desktop/src/components/ChatPanel.tsx`)

```typescript
import React, { useState, useRef, useEffect } from "react";
import { useChat } from "../hooks/useChat";
import "./ChatPanel.css";

export const ChatPanel: React.FC = () => {
  const { messages, sendMessage, loading } = useChat();
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const handleSend = async () => {
    if (input.trim()) {
      await sendMessage(input);
      setInput("");
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
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
          onKeyPress={(e) => e.key === "Enter" && handleSend()}
          placeholder="Type a message..."
          disabled={loading}
        />
        <button onClick={handleSend} disabled={loading}>
          {loading ? "Sending..." : "Send"}
        </button>
      </div>
    </div>
  );
};
```

#### Agent Selector Component (`packages/desktop/src/components/AgentSelector.tsx`)

```typescript
import React, { useEffect, useState } from "react";
import { useAgents } from "../hooks/useAgents";
import "./AgentSelector.css";

export const AgentSelector: React.FC = () => {
  const { agents, loading, selectAgent } = useAgents();
  const [selected, setSelected] = useState<string | null>(null);

  const handleSelect = (agentId: string) => {
    setSelected(agentId);
    selectAgent(agentId);
  };

  if (loading) return <div>Loading agents...</div>;

  return (
    <div className="agent-selector">
      <h3>Select Agent</h3>
      <div className="agents-list">
        {agents.map((agent) => (
          <div
            key={agent.id}
            className={`agent-card ${selected === agent.id ? "active" : ""}`}
            onClick={() => handleSelect(agent.id)}
          >
            <h4>{agent.name}</h4>
            <p>{agent.description}</p>
            <small>{agent.type}</small>
          </div>
        ))}
      </div>
    </div>
  );
};
```

---

### **STEP 4: Create Custom Hooks** (30 min)

#### useChat Hook (`packages/desktop/src/hooks/useChat.ts`)

```typescript
import { useState, useCallback } from "react";
import { RyzansteinAPI } from "../services/api";
import { useChatStore } from "../store/chatStore";

export const useChat = () => {
  const api = new RyzansteinAPI();
  const { messages, addMessage } = useChatStore();
  const { selectedAgent } = useAgentStore();
  const { selectedModel } = useModelStore();
  const [loading, setLoading] = useState(false);

  const sendMessage = useCallback(
    async (content: string) => {
      setLoading(true);
      try {
        addMessage({ role: "user", content });

        const response = await api.chat(content, selectedAgent, selectedModel);

        addMessage({ role: "assistant", content: response.response });
      } catch (error) {
        console.error("Chat error:", error);
        addMessage({
          role: "system",
          content: `Error: ${
            error instanceof Error ? error.message : "Unknown error"
          }`,
        });
      } finally {
        setLoading(false);
      }
    },
    [selectedAgent, selectedModel]
  );

  return { messages, sendMessage, loading };
};
```

#### useAgents Hook (`packages/desktop/src/hooks/useAgents.ts`)

```typescript
import { useState, useEffect } from "react";
import { RyzansteinAPI } from "../services/api";

export const useAgents = () => {
  const [agents, setAgents] = useState([]);
  const [loading, setLoading] = useState(true);
  const api = new RyzansteinAPI();

  useEffect(() => {
    const loadAgents = async () => {
      try {
        const data = await api.listAgents();
        setAgents(data);
      } catch (error) {
        console.error("Failed to load agents:", error);
      } finally {
        setLoading(false);
      }
    };

    loadAgents();
  }, []);

  const selectAgent = (agentId: string) => {
    // Dispatch to store
  };

  return { agents, loading, selectAgent };
};
```

---

### **STEP 5: Setup State Management with Zustand** (30 min)

#### Chat Store (`packages/desktop/src/store/chatStore.ts`)

```typescript
import create from "zustand";

interface Message {
  role: "user" | "assistant" | "system";
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
```

#### Agent Store (`packages/desktop/src/store/agentStore.ts`)

```typescript
import create from "zustand";

interface AgentStore {
  selectedAgent: string | null;
  selectAgent: (agentId: string) => void;
}

export const useAgentStore = create<AgentStore>((set) => ({
  selectedAgent: null,
  selectAgent: (agentId) => set({ selectedAgent: agentId }),
}));
```

---

### **STEP 6: Build & Package Desktop App** (30 min)

#### Development Build

```bash
cd s:\Ryot\desktop

# Start dev server
wails dev
```

#### Production Build

```bash
# Build for Windows
wails build -nsis

# Build for macOS
wails build -mac

# Build for Linux
wails build -linux all
```

#### Wails Configuration (`wails.json`)

```json
{
  "name": "Ryzanstein",
  "type": "desktop",
  "outputType": "desktop",
  "frontend": "packages/desktop",
  "build": {
    "appType": "desktop",
    "frontend": {
      "dir": "packages/desktop",
      "install": "npm install",
      "build": "npm run build",
      "dev": "npm run dev"
    },
    "backend": {
      "main": "cmd/ryzanstein/main.go"
    }
  },
  "bindings": [
    "main.App",
    "main.Chat",
    "main.Models",
    "main.Agents",
    "main.Config"
  ]
}
```

---

## 💻 PHASE 2: VS CODE EXTENSION DEVELOPMENT

### Current State

```
vscode-extension/
├── build.sh
├── package.json            (Extension metadata)
├── src/                    (TypeScript source)
└── dist/                   (Compiled output)
```

### Architecture

- **Language**: TypeScript
- **API**: VS Code Extension API
- **Backend Communication**: HTTP/WebSocket
- **Features**: Inline chat, code generation, agent tooling

---

## 📝 VS CODE EXTENSION - STEP-BY-STEP INSTRUCTIONS

### **STEP 1: Setup Extension Development Environment** (30 min)

#### Prerequisites

```bash
# Install dependencies
cd s:\Ryot\vscode-extension
npm install

# Install VS Code Extension CLI
npm install -g @vscode/vsce
```

#### Create Project Structure

```
src/
├── extension.ts            (Entry point)
├── commands/
│   ├── chatCommand.ts
│   ├── agentCommand.ts
│   └── codeGenCommand.ts
├── webview/
│   ├── chatPanel.ts
│   └── assets/
│       ├── index.html
│       ├── styles.css
│       └── script.js
├── services/
│   ├── ryzansteinAPI.ts
│   └── webviewManager.ts
├── types/
│   └── index.ts
└── utils/
    └── logger.ts

resources/
├── icon.png
└── dark/
    └── icon.png
```

---

### **STEP 2: Create Extension Entry Point** (20 min)

#### `src/extension.ts`

```typescript
import * as vscode from "vscode";
import { RyzansteinAPI } from "./services/ryzansteinAPI";
import { ChatPanel } from "./webview/chatPanel";

let ryzansteinAPI: RyzansteinAPI;
let chatPanel: ChatPanel | undefined;

export async function activate(context: vscode.ExtensionContext) {
  console.log("Ryzanstein extension activated");

  // Initialize API client
  const config = vscode.workspace.getConfiguration("ryzanstein");
  const apiUrl = config.get<string>("apiUrl") || "http://localhost:8000";
  ryzansteinAPI = new RyzansteinAPI(apiUrl);

  // Register commands
  registerCommands(context);

  // Update status bar
  updateStatusBar();
}

function registerCommands(context: vscode.ExtensionContext) {
  // Open Chat Command
  context.subscriptions.push(
    vscode.commands.registerCommand("ryzanstein.openChat", async () => {
      if (chatPanel) {
        chatPanel.reveal();
      } else {
        chatPanel = new ChatPanel(
          vscode.window.createWebviewPanel(
            "ryzansteinChat",
            "Ryzanstein Chat",
            vscode.ViewColumn.Beside,
            { enableScripts: true, retainContextWhenHidden: true }
          ),
          ryzansteinAPI
        );
      }
    })
  );

  // Select Agent Command
  context.subscriptions.push(
    vscode.commands.registerCommand("ryzanstein.selectAgent", async () => {
      const agents = await ryzansteinAPI.listAgents();
      const agentNames = agents.map((a) => a.name);

      const selected = await vscode.window.showQuickPick(agentNames);
      if (selected) {
        const agent = agents.find((a) => a.name === selected);
        if (agent) {
          vscode.workspace
            .getConfiguration("ryzanstein")
            .update("selectedAgent", agent.id);
        }
      }
    })
  );

  // Code Generation Command
  context.subscriptions.push(
    vscode.commands.registerCommand("ryzanstein.generateCode", async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) return;

      const prompt = await vscode.window.showInputBox({
        prompt: "Code generation prompt",
      });
      if (!prompt) return;

      // Call API to generate code
      const generated = await ryzansteinAPI.generateCode(prompt);
      editor.edit((editBuilder) => {
        editBuilder.insert(editor.selection.active, generated);
      });
    })
  );
}

function updateStatusBar() {
  const statusBar = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Right,
    100
  );
  statusBar.command = "ryzanstein.openChat";
  statusBar.text = "$(robot) Ryzanstein";
  statusBar.tooltip = "Open Ryzanstein Chat";
  statusBar.show();
}

export function deactivate() {
  console.log("Ryzanstein extension deactivated");
}
```

---

### **STEP 3: Create Webview Chat Panel** (45 min)

#### `src/webview/chatPanel.ts`

```typescript
import * as vscode from "vscode";
import { RyzansteinAPI } from "../services/ryzansteinAPI";

export class ChatPanel {
  constructor(private panel: vscode.WebviewPanel, private api: RyzansteinAPI) {
    this.panel.webview.html = this.getHtmlContent();
    this.setupMessageHandlers();
  }

  private setupMessageHandlers() {
    this.panel.webview.onDidReceiveMessage(async (message) => {
      switch (message.command) {
        case "sendMessage":
          await this.handleMessage(message.text);
          break;
        case "selectAgent":
          await this.selectAgent(message.agentId);
          break;
        case "getAgents":
          await this.getAgents();
          break;
      }
    });
  }

  private async handleMessage(text: string) {
    const config = vscode.workspace.getConfiguration("ryzanstein");
    const agentId = config.get<string>("selectedAgent") || "default";

    try {
      const response = await this.api.chat(text, agentId);
      this.panel.webview.postMessage({
        command: "addMessage",
        role: "assistant",
        content: response.response,
      });
    } catch (error) {
      this.panel.webview.postMessage({
        command: "error",
        message: error instanceof Error ? error.message : "Unknown error",
      });
    }
  }

  private async selectAgent(agentId: string) {
    vscode.workspace
      .getConfiguration("ryzanstein")
      .update("selectedAgent", agentId);
  }

  private async getAgents() {
    const agents = await this.api.listAgents();
    this.panel.webview.postMessage({
      command: "setAgents",
      agents: agents,
    });
  }

  private getHtmlContent(): string {
    return `
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ryzanstein Chat</title>
    <link rel="stylesheet" href="${this.getStylesheet()}">
</head>
<body>
    <div id="chat-container">
        <div id="messages"></div>
        <div id="input-area">
            <input type="text" id="message-input" placeholder="Type a message...">
            <button id="send-button">Send</button>
        </div>
    </div>
    <script src="${this.getScript()}"></script>
</body>
</html>
        `;
  }

  reveal() {
    this.panel.reveal(vscode.ViewColumn.Beside);
  }

  private getStylesheet(): string {
    // Return stylesheet URI
    return "";
  }

  private getScript(): string {
    // Return script URI
    return "";
  }
}
```

#### `src/webview/assets/index.html`

```html
<!DOCTYPE html>
<html>
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Ryzanstein Chat</title>
    <style>
      * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
      }

      body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
          sans-serif;
        background: var(--vscode-editor-background);
        color: var(--vscode-editor-foreground);
      }

      #chat-container {
        display: flex;
        flex-direction: column;
        height: 100vh;
      }

      #messages {
        flex: 1;
        overflow-y: auto;
        padding: 1rem;
      }

      .message {
        margin: 0.5rem 0;
        padding: 0.75rem;
        border-radius: 4px;
      }

      .message.user {
        background: var(--vscode-button-background);
        color: var(--vscode-button-foreground);
        margin-left: 2rem;
      }

      .message.assistant {
        background: var(--vscode-editor-lineHighlightBackground);
        margin-right: 2rem;
      }

      #input-area {
        display: flex;
        gap: 0.5rem;
        padding: 1rem;
        border-top: 1px solid var(--vscode-panel-border);
      }

      #message-input {
        flex: 1;
        padding: 0.75rem;
        border: 1px solid var(--vscode-input-border);
        background: var(--vscode-input-background);
        color: var(--vscode-input-foreground);
        border-radius: 4px;
        font-size: 14px;
      }

      #send-button {
        padding: 0.75rem 1.5rem;
        background: var(--vscode-button-background);
        color: var(--vscode-button-foreground);
        border: none;
        border-radius: 4px;
        cursor: pointer;
      }

      #send-button:hover {
        background: var(--vscode-button-hoverBackground);
      }
    </style>
  </head>
  <body>
    <div id="chat-container">
      <div id="messages"></div>
      <div id="input-area">
        <input type="text" id="message-input" placeholder="Type a message..." />
        <button id="send-button">Send</button>
      </div>
    </div>

    <script>
      const vscode = acquireVsCodeApi();
      const messagesDiv = document.getElementById("messages");
      const input = document.getElementById("message-input");
      const sendButton = document.getElementById("send-button");

      sendButton.addEventListener("click", () => {
        if (input.value.trim()) {
          vscode.postMessage({
            command: "sendMessage",
            text: input.value,
          });
          input.value = "";
        }
      });

      input.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
          sendButton.click();
        }
      });

      window.addEventListener("message", (event) => {
        const message = event.data;
        switch (message.command) {
          case "addMessage":
            const div = document.createElement("div");
            div.className = `message ${message.role}`;
            div.textContent = message.content;
            messagesDiv.appendChild(div);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            break;
        }
      });

      // Load agents on startup
      vscode.postMessage({ command: "getAgents" });
    </script>
  </body>
</html>
```

---

### **STEP 4: Create API Service** (30 min)

#### `src/services/ryzansteinAPI.ts`

```typescript
import axios, { AxiosInstance } from "axios";

export class RyzansteinAPI {
  private client: AxiosInstance;

  constructor(baseURL: string) {
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        "Content-Type": "application/json",
      },
    });
  }

  async chat(message: string, agentId: string): Promise<{ response: string }> {
    const response = await this.client.post("/chat", {
      message,
      agent_id: agentId,
    });
    return response.data;
  }

  async listAgents(): Promise<any[]> {
    const response = await this.client.get("/agents");
    return response.data;
  }

  async listModels(): Promise<any[]> {
    const response = await this.client.get("/models");
    return response.data;
  }

  async generateCode(prompt: string): Promise<string> {
    const response = await this.client.post("/generate", {
      prompt,
      type: "code",
    });
    return response.data.generated;
  }

  async refactorCode(code: string): Promise<string> {
    const response = await this.client.post("/refactor", {
      code,
    });
    return response.data.refactored;
  }
}
```

---

### **STEP 5: Configure Extension Manifest** (15 min)

#### `package.json` updates

```json
{
  "name": "ryzanstein",
  "displayName": "Ryzanstein - Elite AI Agent Assistant",
  "description": "Bring the Elite AI Agent Collective into VS Code",
  "version": "1.0.0",
  "publisher": "iamthegreatdestroyer",
  "engines": {
    "vscode": "^1.85.0"
  },
  "categories": ["AI", "Chat", "Code Generators"],
  "activationEvents": ["onStartupFinished"],
  "main": "./dist/extension.js",
  "contributes": {
    "commands": [
      {
        "command": "ryzanstein.openChat",
        "title": "Open Chat"
      },
      {
        "command": "ryzanstein.selectAgent",
        "title": "Select Agent"
      },
      {
        "command": "ryzanstein.generateCode",
        "title": "Generate Code"
      }
    ],
    "configuration": {
      "title": "Ryzanstein",
      "properties": {
        "ryzanstein.apiUrl": {
          "type": "string",
          "default": "http://localhost:8000",
          "description": "Ryzanstein API server URL"
        },
        "ryzanstein.selectedAgent": {
          "type": "string",
          "default": "default",
          "description": "Currently selected agent"
        }
      }
    }
  },
  "scripts": {
    "vscode:prepublish": "npm run compile",
    "compile": "tsc -p ./",
    "watch": "tsc -watch -p ./",
    "lint": "eslint src",
    "package": "vsce package",
    "publish": "vsce publish"
  }
}
```

---

### **STEP 6: Build & Package Extension** (15 min)

#### Development

```bash
cd s:\Ryot\vscode-extension

# Install dependencies
npm install

# Compile TypeScript
npm run compile

# Watch for changes
npm run watch

# Test extension locally
# Press F5 in VS Code to launch extension development host
```

#### Production

```bash
# Create .vsix package
npm run package

# Publish to Marketplace (requires publisher account)
npm run publish
```

---

## 🔗 INTEGRATION CHECKLIST

### Desktop App Integration

- [ ] API client layer connects to backend
- [ ] Chat messages send/receive correctly
- [ ] Agent selection works
- [ ] Model loading displays status
- [ ] Settings persist to config file
- [ ] Error handling & user feedback
- [ ] Performance metrics collected
- [ ] Built and tested on Windows
- [ ] Built and tested on macOS
- [ ] Built and tested on Linux

### VS Code Extension Integration

- [ ] Extension activates on startup
- [ ] Chat panel opens via command
- [ ] Messages send to backend
- [ ] Agent selection from quick pick
- [ ] Code generation works
- [ ] Status bar shows connection status
- [ ] Configuration settings work
- [ ] Error handling & logging
- [ ] Packaged as .vsix
- [ ] Ready for Marketplace submission

---

## 🚀 DEPLOYMENT CHECKLIST

### Pre-Release

- [ ] All tests passing (100% coverage)
- [ ] Code reviewed & approved
- [ ] Documentation complete
- [ ] Security audit completed
- [ ] Performance benchmarks passed
- [ ] Cross-platform testing done
- [ ] User guide written
- [ ] Known issues documented

### Release

- [ ] Version bumped in package.json
- [ ] Changelog updated
- [ ] Desktop app signed (Windows/macOS)
- [ ] VS Code extension published
- [ ] Release notes published
- [ ] Announcement posted

---

## 📞 SUPPORT & RESOURCES

### Desktop App

- **Wails Documentation**: https://wails.io
- **React Guide**: https://react.dev
- **Zustand Store**: https://github.com/pmndrs/zustand

### VS Code Extension

- **VS Code API**: https://code.visualstudio.com/api
- **Extension Guidelines**: https://code.visualstudio.com/api/extension-guides/overview
- **Webview Guide**: https://code.visualstudio.com/api/extension-guides/webview

---

## ⏱️ TIMELINE ESTIMATE

| Phase             | Task             | Duration | Status      |
| ----------------- | ---------------- | -------- | ----------- |
| 1                 | Desktop Setup    | 30 min   | Not Started |
| 2                 | API Client       | 45 min   | Not Started |
| 3                 | React Components | 60 min   | Not Started |
| 4                 | Custom Hooks     | 30 min   | Not Started |
| 5                 | State Management | 30 min   | Not Started |
| 6                 | Build & Package  | 30 min   | Not Started |
| **Desktop Total** | **~3.5 hours**   |          |             |
| 1                 | VS Code Setup    | 30 min   | Not Started |
| 2                 | Entry Point      | 20 min   | Not Started |
| 3                 | Chat Panel       | 45 min   | Not Started |
| 4                 | API Service      | 30 min   | Not Started |
| 5                 | Manifest Config  | 15 min   | Not Started |
| 6                 | Build & Package  | 15 min   | Not Started |
| **VS Code Total** | **~2.75 hours**  |          |             |
| **GRAND TOTAL**   | **~6.25 hours**  |          |             |

---

**Last Updated**: January 8, 2026  
**Next Review**: Upon completion of Phase 1
