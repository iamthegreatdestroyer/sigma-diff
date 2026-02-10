# ============================================================================
# 🚀 RYZANSTEIN VS CODE EXTENSION - MASTER SETUP SCRIPT
# ============================================================================
#
# Purpose: Fully automated setup of VS Code Extension
# Author: ARCHITECT Mode
# Date: January 8, 2026
#
# This script handles:
# - Dependency verification (Node.js, npm)
# - Project structure initialization
# - TypeScript configuration
# - Extension file generation
# - NPM dependency installation
# - TypeScript compilation
# - Extension packaging
#
# Usage: .\SETUP_VSCODE_EXTENSION_MASTER.ps1
# ============================================================================

param(
    [switch]$SkipDependencies = $false,
    [switch]$PackageOnly = $false,
    [switch]$PublishToMarketplace = $false
)

# ============================================================================
# CONFIGURATION
# ============================================================================

$ErrorActionPreference = "Stop"
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = $scriptPath
$srcPath = Join-Path $projectRoot "src"
$distPath = Join-Path $projectRoot "dist"

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
    
    # Check Node.js
    if (Test-CommandExists "node") {
        $nodeVersion = (node --version)
        Write-Success "Node.js installed: $nodeVersion"
    }
    else {
        Write-Error "Node.js not installed. Download from https://nodejs.org"
        $allGood = $false
    }
    
    # Check npm
    if (Test-CommandExists "npm") {
        $npmVersion = (npm --version)
        Write-Success "npm installed: $npmVersion"
    }
    else {
        Write-Error "npm not installed. Install Node.js which includes npm."
        $allGood = $false
    }
    
    # Check vsce (VS Code Extension CLI)
    if (Test-CommandExists "vsce") {
        Write-Success "vsce (VS Code Extension CLI) installed"
    }
    else {
        if ($SkipDependencies) {
            Write-Warning "vsce not installed. Run: npm install -g @vscode/vsce"
        }
        else {
            Write-Step "Installing vsce..."
            npm install -g @vscode/vsce
            Write-Success "vsce installed"
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
    
    Ensure-Directory (Join-Path $srcPath "commands")
    Ensure-Directory (Join-Path $srcPath "webview\assets")
    Ensure-Directory (Join-Path $srcPath "services")
    Ensure-Directory (Join-Path $srcPath "types")
    Ensure-Directory (Join-Path $srcPath "utils")
    
    Write-Success "Directory structure initialized"
}

# ============================================================================
# EXTENSION FILES GENERATION
# ============================================================================

function Generate-ExtensionFiles {
    Write-Header "STEP 3: GENERATING EXTENSION FILES"
    
    # Extension.ts - Entry Point
    $extensionTs = @'
import * as vscode from 'vscode';
import { RyzansteinAPI } from './services/ryzansteinAPI';
import { ChatPanel } from './webview/chatPanel';

let ryzansteinAPI: RyzansteinAPI;
let chatPanel: ChatPanel | undefined;

export async function activate(context: vscode.ExtensionContext) {
    console.log('Ryzanstein extension activated');

    const config = vscode.workspace.getConfiguration('ryzanstein');
    const apiUrl = config.get<string>('apiUrl') || 'http://localhost:8000';
    ryzansteinAPI = new RyzansteinAPI(apiUrl);

    registerCommands(context);
    updateStatusBar();
}

function registerCommands(context: vscode.ExtensionContext) {
    context.subscriptions.push(
        vscode.commands.registerCommand('ryzanstein.openChat', async () => {
            if (chatPanel) {
                chatPanel.reveal();
            } else {
                const panel = vscode.window.createWebviewPanel(
                    'ryzansteinChat',
                    'Ryzanstein Chat',
                    vscode.ViewColumn.Beside,
                    { enableScripts: true, retainContextWhenHidden: true }
                );
                chatPanel = new ChatPanel(panel, ryzansteinAPI);
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('ryzanstein.selectAgent', async () => {
            try {
                const agents = await ryzansteinAPI.listAgents();
                const agentNames = agents.map(a => a.name);
                const selected = await vscode.window.showQuickPick(agentNames);
                if (selected) {
                    const agent = agents.find(a => a.name === selected);
                    if (agent) {
                        vscode.workspace.getConfiguration('ryzanstein').update('selectedAgent', agent.id);
                        vscode.window.showInformationMessage(`Selected agent: ${selected}`);
                    }
                }
            } catch (error) {
                vscode.window.showErrorMessage(`Error loading agents: ${error}`);
            }
        })
    );
}

function updateStatusBar() {
    const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBar.command = 'ryzanstein.openChat';
    statusBar.text = '$(robot) Ryzanstein';
    statusBar.tooltip = 'Open Ryzanstein Chat';
    statusBar.show();
}

export function deactivate() {
    console.log('Ryzanstein extension deactivated');
}
'@
    Create-FileIfNotExists (Join-Path $srcPath "extension.ts") $extensionTs
    
    # ChatPanel.ts
    $chatPanelTs = @'
import * as vscode from 'vscode';
import { RyzansteinAPI } from '../services/ryzansteinAPI';

export class ChatPanel {
    constructor(
        private panel: vscode.WebviewPanel,
        private api: RyzansteinAPI
    ) {
        this.panel.webview.html = this.getHtmlContent();
        this.setupMessageHandlers();
    }

    private setupMessageHandlers() {
        this.panel.webview.onDidReceiveMessage(async (message) => {
            switch (message.command) {
                case 'sendMessage':
                    await this.handleMessage(message.text);
                    break;
                case 'selectAgent':
                    await this.selectAgent(message.agentId);
                    break;
                case 'getAgents':
                    await this.getAgents();
                    break;
            }
        });
    }

    private async handleMessage(text: string) {
        const config = vscode.workspace.getConfiguration('ryzanstein');
        const agentId = config.get<string>('selectedAgent') || 'default';

        try {
            const response = await this.api.chat(text, agentId);
            this.panel.webview.postMessage({
                command: 'addMessage',
                role: 'assistant',
                content: response.response,
            });
        } catch (error) {
            this.panel.webview.postMessage({
                command: 'error',
                message: error instanceof Error ? error.message : 'Unknown error',
            });
        }
    }

    private async selectAgent(agentId: string) {
        vscode.workspace.getConfiguration('ryzanstein').update('selectedAgent', agentId);
    }

    private async getAgents() {
        try {
            const agents = await this.api.listAgents();
            this.panel.webview.postMessage({
                command: 'setAgents',
                agents: agents,
            });
        } catch (error) {
            console.error('Error loading agents:', error);
        }
    }

    private getHtmlContent(): string {
        const nonce = this.getNonce();
        return `<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ryzanstein Chat</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--vscode-editor-background); color: var(--vscode-editor-foreground); }
        #chat-container { display: flex; flex-direction: column; height: 100vh; }
        #messages { flex: 1; overflow-y: auto; padding: 1rem; }
        .message { margin: 0.5rem 0; padding: 0.75rem; border-radius: 4px; }
        .message.user { background: var(--vscode-button-background); color: var(--vscode-button-foreground); margin-left: 2rem; }
        .message.assistant { background: var(--vscode-editor-lineHighlightBackground); margin-right: 2rem; }
        #input-area { display: flex; gap: 0.5rem; padding: 1rem; border-top: 1px solid var(--vscode-panel-border); }
        #message-input { flex: 1; padding: 0.75rem; border: 1px solid var(--vscode-input-border); background: var(--vscode-input-background); color: var(--vscode-input-foreground); border-radius: 4px; font-size: 14px; }
        #send-button { padding: 0.75rem 1.5rem; background: var(--vscode-button-background); color: var(--vscode-button-foreground); border: none; border-radius: 4px; cursor: pointer; }
        #send-button:hover { background: var(--vscode-button-hoverBackground); }
    </style>
</head>
<body>
    <div id="chat-container">
        <div id="messages"></div>
        <div id="input-area">
            <input type="text" id="message-input" placeholder="Type a message...">
            <button id="send-button">Send</button>
        </div>
    </div>
    <script nonce="${nonce}">
        const vscode = acquireVsCodeApi();
        const messagesDiv = document.getElementById('messages');
        const input = document.getElementById('message-input');
        const sendButton = document.getElementById('send-button');

        sendButton.addEventListener('click', () => {
            if (input.value.trim()) {
                vscode.postMessage({
                    command: 'sendMessage',
                    text: input.value,
                });
                input.value = '';
            }
        });

        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendButton.click();
            }
        });

        window.addEventListener('message', (event) => {
            const message = event.data;
            switch (message.command) {
                case 'addMessage':
                    const div = document.createElement('div');
                    div.className = \`message \${message.role}\`;
                    div.textContent = message.content;
                    messagesDiv.appendChild(div);
                    messagesDiv.scrollTop = messagesDiv.scrollHeight;
                    break;
            }
        });

        vscode.postMessage({ command: 'getAgents' });
    </script>
</body>
</html>`;
    }

    private getNonce(): string {
        let text = '';
        const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
        for (let i = 0; i < 32; i++) {
            text += possible.charAt(Math.floor(Math.random() * possible.length));
        }
        return text;
    }

    reveal() {
        this.panel.reveal(vscode.ViewColumn.Beside);
    }
}
'@
    Create-FileIfNotExists (Join-Path $srcPath "webview\chatPanel.ts") $chatPanelTs
    
    # RyzansteinAPI Service
    $apiService = @'
import axios from 'axios';

export class RyzansteinAPI {
    private client;

    constructor(baseURL: string) {
        this.client = axios.create({
            baseURL,
            timeout: 30000,
            headers: { 'Content-Type': 'application/json' },
        });
    }

    async chat(message: string, agentId: string): Promise<{ response: string }> {
        const response = await this.client.post('/chat', {
            message,
            agent_id: agentId,
        });
        return response.data;
    }

    async listAgents(): Promise<any[]> {
        const response = await this.client.get('/agents');
        return response.data;
    }

    async listModels(): Promise<any[]> {
        const response = await this.client.get('/models');
        return response.data;
    }

    async generateCode(prompt: string): Promise<string> {
        const response = await this.client.post('/generate', {
            prompt,
            type: 'code',
        });
        return response.data.generated;
    }
}
'@
    Create-FileIfNotExists (Join-Path $srcPath "services\ryzansteinAPI.ts") $apiService
    
    Write-Success "Extension files generated"
}

# ============================================================================
# TYPESCRIPT CONFIGURATION
# ============================================================================

function Setup-TypeScriptConfig {
    Write-Header "STEP 4: CONFIGURING TYPESCRIPT"
    
    $tsconfigPath = Join-Path $projectRoot "tsconfig.json"
    
    if (-not (Test-Path $tsconfigPath)) {
        $tsconfig = @{
            compilerOptions = @{
                target                           = "ES2020"
                module                           = "commonjs"
                lib                              = @("ES2020")
                outDir                           = "./dist"
                rootDir                          = "./src"
                strict                           = $true
                esModuleInterop                  = $true
                skipLibCheck                     = $true
                forceConsistentCasingInFileNames = $true
                resolveJsonModule                = $true
                declaration                      = $true
                declarationMap                   = $true
                sourceMap                        = $true
            }
            include         = @("src/**/*")
            exclude         = @("node_modules", "**/*.spec.ts")
        } | ConvertTo-Json -Depth 10
        
        Set-Content -Path $tsconfigPath -Value $tsconfig -Encoding UTF8
        Write-Success "Created tsconfig.json"
    }
    else {
        Write-Success "tsconfig.json already exists"
    }
}

# ============================================================================
# ESLINT CONFIGURATION
# ============================================================================

function Setup-ESLintConfig {
    Write-Header "STEP 5: CONFIGURING ESLINT"
    
    $eslintPath = Join-Path $projectRoot ".eslintrc.json"
    
    if (-not (Test-Path $eslintPath)) {
        $eslint = @{
            env           = @{
                browser = $true
                es2021  = $true
                node    = $true
            }
            extends       = @("eslint:recommended")
            parser        = "@typescript-eslint/parser"
            parserOptions = @{
                ecmaVersion = 12
                sourceType  = "module"
            }
            plugins       = @("@typescript-eslint")
            rules         = @{}
        } | ConvertTo-Json -Depth 10
        
        Set-Content -Path $eslintPath -Value $eslint -Encoding UTF8
        Write-Success "Created .eslintrc.json"
    }
    else {
        Write-Success ".eslintrc.json already exists"
    }
}

# ============================================================================
# NPM SETUP
# ============================================================================

function Setup-NPM {
    Write-Header "STEP 6: INSTALLING NPM DEPENDENCIES"
    
    Write-Step "Running npm install..."
    Push-Location $projectRoot
    
    try {
        npm install
        Write-Success "NPM dependencies installed"
    }
    finally {
        Pop-Location
    }
}

# ============================================================================
# TYPESCRIPT COMPILATION
# ============================================================================

function Compile-TypeScript {
    Write-Header "STEP 7: COMPILING TYPESCRIPT"
    
    Write-Step "Compiling TypeScript..."
    Push-Location $projectRoot
    
    try {
        npm run compile
        Write-Success "TypeScript compiled successfully"
    }
    finally {
        Pop-Location
    }
}

# ============================================================================
# EXTENSION PACKAGING
# ============================================================================

function Package-Extension {
    Write-Header "STEP 8: PACKAGING EXTENSION"
    
    Write-Step "Creating .vsix package..."
    Push-Location $projectRoot
    
    try {
        vsce package --no-update-package-json
        Write-Success "Extension packaged successfully (.vsix created)"
        
        # List the generated .vsix file
        $vsixFile = Get-ChildItem -Path $projectRoot -Filter "*.vsix" | Select-Object -First 1
        if ($vsixFile) {
            Write-Success "Package: $($vsixFile.Name)"
        }
    }
    finally {
        Pop-Location
    }
}

# ============================================================================
# MARKETPLACE PUBLISHING
# ============================================================================

function Publish-ToMarketplace {
    Write-Header "STEP 9: PUBLISHING TO MARKETPLACE"
    
    if (-not $PublishToMarketplace) {
        Write-Warning "Skipping marketplace publishing (use -PublishToMarketplace to enable)"
        return
    }
    
    Write-Step "Publishing to VS Code Marketplace..."
    Write-Warning "Make sure you have:"
    Write-Warning "  1. Created a publisher account on https://marketplace.visualstudio.com"
    Write-Warning "  2. Created a Personal Access Token (PAT)"
    Write-Warning "  3. Logged in with: vsce login <publisher>"
    
    Push-Location $projectRoot
    
    try {
        vsce publish
        Write-Success "Extension published to marketplace"
    }
    catch {
        Write-Error "Publishing failed: $_"
    }
    finally {
        Pop-Location
    }
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

function Main {
    Clear-Host
    Write-Header "RYZANSTEIN VS CODE EXTENSION - AUTOMATED SETUP"
    
    try {
        Verify-Dependencies
        Initialize-DirectoryStructure
        Generate-ExtensionFiles
        Setup-TypeScriptConfig
        Setup-ESLintConfig
        Setup-NPM
        
        if (-not $PackageOnly) {
            Compile-TypeScript
            Package-Extension
            Publish-ToMarketplace
        }
        
        Write-Header "✅ SETUP COMPLETE"
        
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor $colors.Info
        Write-Host "  1. Press F5 in VS Code to launch Extension Development Host" -ForegroundColor $colors.Progress
        Write-Host "  2. Or run: npm run compile" -ForegroundColor $colors.Progress
        Write-Host "  3. Or run: npm run package" -ForegroundColor $colors.Progress
        Write-Host ""
        
    }
    catch {
        Write-Error "Setup failed: $_"
        exit 1
    }
}

# Run main
Main
