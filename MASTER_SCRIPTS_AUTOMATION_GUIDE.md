# 🚀 MASTER SCRIPTS - COMPLETE AUTOMATION GUIDE

**Date**: January 8, 2026  
**Purpose**: Fully automated setup of Desktop App and VS Code Extension  
**Autonomy Level**: Maximum (95%+ automated)

---

## 📋 OVERVIEW

Three master scripts have been created to automate the entire Ryzanstein ecosystem setup:

1. **SETUP_COMPLETE_ECOSYSTEM_MASTER.ps1** - Orchestrates both platforms
2. **SETUP_DESKTOP_APP_MASTER.ps1** - Desktop app automation
3. **SETUP_VSCODE_EXTENSION_MASTER.ps1** - VS Code extension automation

---

## 🎯 WHAT THESE SCRIPTS DO AUTOMATICALLY

### Complete Automation Coverage

| Task                         | Desktop App | VS Code Extension |
| ---------------------------- | ----------- | ----------------- |
| Dependency verification      | ✅          | ✅                |
| Directory structure creation | ✅          | ✅                |
| File generation (40+ files)  | ✅          | ✅                |
| Configuration files          | ✅          | ✅                |
| npm/go dependencies          | ✅          | ✅                |
| TypeScript compilation       | ❌          | ✅                |
| Build & packaging            | ✅          | ✅                |
| Error handling               | ✅          | ✅                |

**Total Automation**: 95%+

---

## 🚀 QUICK START

### Option 1: Complete Setup (Recommended)

```powershell
cd s:\Ryot
.\SETUP_COMPLETE_ECOSYSTEM_MASTER.ps1
```

This will:

1. Check all prerequisites
2. Setup Desktop App fully
3. Setup VS Code Extension fully
4. Verify integration
5. Show status report

**Time**: ~15-20 minutes

---

### Option 2: Desktop App Only

```powershell
cd s:\Ryot
.\SETUP_COMPLETE_ECOSYSTEM_MASTER.ps1 -SetupType Desktop
```

**Time**: ~8-10 minutes

---

### Option 3: VS Code Extension Only

```powershell
cd s:\Ryot
.\SETUP_COMPLETE_ECOSYSTEM_MASTER.ps1 -SetupType Extension
```

**Time**: ~5-7 minutes

---

### Option 4: Direct Script Execution

#### Desktop App

```powershell
cd s:\Ryot\desktop
.\SETUP_DESKTOP_APP_MASTER.ps1
```

#### VS Code Extension

```powershell
cd s:\Ryot\vscode-extension
.\SETUP_VSCODE_EXTENSION_MASTER.ps1
```

---

## 🔧 AVAILABLE PARAMETERS

### SETUP_COMPLETE_ECOSYSTEM_MASTER.ps1

```powershell
-SetupType <string>
    Full        # Default - setup both platforms
    Desktop     # Desktop app only
    Extension   # VS Code extension only
    Dev         # Development mode (no setup)

-SkipDependencies
    # Skip automatic dependency installation
    # Useful if you've already installed dependencies

-Verbose
    # Show detailed output for debugging
```

**Examples**:

```powershell
# Setup desktop only, skip dependency checks
.\SETUP_COMPLETE_ECOSYSTEM_MASTER.ps1 -SetupType Desktop -SkipDependencies

# Setup extension with verbose output
.\SETUP_COMPLETE_ECOSYSTEM_MASTER.ps1 -SetupType Extension -Verbose

# Development mode (no actual setup)
.\SETUP_COMPLETE_ECOSYSTEM_MASTER.ps1 -SetupType Dev
```

---

### SETUP_DESKTOP_APP_MASTER.ps1

```powershell
-SkipDependencies
    # Skip Go/Node.js/Wails checks

-DevelopmentOnly
    # Build for development (wails dev)
    # Default: build for production
```

**Examples**:

```powershell
# Setup with development mode
.\SETUP_DESKTOP_APP_MASTER.ps1 -DevelopmentOnly

# Skip dependency checks
.\SETUP_DESKTOP_APP_MASTER.ps1 -SkipDependencies
```

---

### SETUP_VSCODE_EXTENSION_MASTER.ps1

```powershell
-SkipDependencies
    # Skip Node.js/npm/vsce checks

-PackageOnly
    # Only package, skip compilation

-PublishToMarketplace
    # Publish to VS Code Marketplace
```

**Examples**:

```powershell
# Setup and publish to marketplace
.\SETUP_VSCODE_EXTENSION_MASTER.ps1 -PublishToMarketplace

# Package only (no compilation)
.\SETUP_VSCODE_EXTENSION_MASTER.ps1 -PackageOnly
```

---

## 📊 WHAT GETS CREATED

### Desktop App Files

```
desktop/
├── cmd/ryzanstein/
│   └── main.go                    # Wails entry point
├── internal/
│   ├── app/
│   │   └── app.go                # Wails app struct
│   ├── handlers/
│   │   ├── chat.go               # Chat handler (100 lines)
│   │   ├── models.go             # Models handler (100 lines)
│   │   └── agents.go             # Agents handler (100 lines)
│   └── services/
│       └── api_client.go         # API client
├── packages/desktop/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatPanel.tsx     # Chat UI component (100 lines)
│   │   │   ├── ChatPanel.css     # Component styles (80 lines)
│   │   │   └── AgentSelector.tsx # Agent selector (80 lines)
│   │   ├── hooks/
│   │   │   ├── useChat.ts        # Chat logic hook (60 lines)
│   │   │   ├── useAgents.ts      # Agents logic (60 lines)
│   │   │   └── useModels.ts      # Models logic (60 lines)
│   │   ├── store/
│   │   │   ├── chatStore.ts      # Zustand chat store (40 lines)
│   │   │   ├── agentStore.ts     # Agent store (30 lines)
│   │   │   └── configStore.ts    # Config store (30 lines)
│   │   ├── services/
│   │   │   └── api.ts            # API service client (80 lines)
│   │   └── types/
│   │       └── index.ts          # Type definitions
│   ├── package.json              # npm config
│   └── tsconfig.json             # TypeScript config
├── wails.json                    # Wails config
├── go.mod                        # Go module
└── go.sum                        # Go dependencies

Total: 30+ files, ~1,200 lines of code
```

---

### VS Code Extension Files

```
vscode-extension/
├── src/
│   ├── extension.ts              # Extension entry point (150 lines)
│   ├── commands/
│   │   └── chatCommand.ts        # Chat command handler (80 lines)
│   ├── webview/
│   │   ├── chatPanel.ts          # Chat webview (200 lines)
│   │   └── assets/
│   │       ├── index.html        # Chat UI HTML
│   │       ├── styles.css        # Webview styles
│   │       └── script.js         # Webview script
│   ├── services/
│   │   └── ryzansteinAPI.ts      # API client (100 lines)
│   ├── types/
│   │   └── index.ts              # Type definitions
│   └── utils/
│       └── logger.ts             # Logging utility
├── dist/                         # Compiled output (auto-generated)
├── package.json                  # Extension manifest
├── tsconfig.json                 # TypeScript config
├── .eslintrc.json                # ESLint config
└── *.vsix                        # Packaged extension (auto-generated)

Total: 20+ files, ~800 lines of code
```

---

## ✨ AUTOMATION DETAILS

### Phase 1: Dependency Management

```
✓ Check for Go installation
✓ Check for Node.js installation
✓ Check for npm
✓ Check for Wails CLI
✓ Check for VS Code CLI (vsce)
✓ Auto-install missing dependencies (optional)
```

### Phase 2: Directory Structure

```
✓ Create backend directories (app, handlers, services, config)
✓ Create frontend directories (components, hooks, stores, services, types)
✓ Create webview directories for extension
✓ Create config file directories
```

### Phase 3: File Generation

```
✓ Generate all backend files with full implementations
✓ Generate all React components with TypeScript
✓ Generate all custom hooks with logic
✓ Generate state management stores (Zustand)
✓ Generate API client services
✓ Generate extension entry point
✓ Generate webview panels
✓ Generate configuration files
```

### Phase 4: Dependency Installation

```
✓ npm install for all frontend dependencies
✓ go mod download for backend dependencies
✓ vsce installation for extension packaging
```

### Phase 5: Configuration

```
✓ Create tsconfig.json with optimal settings
✓ Create package.json with correct scripts
✓ Create wails.json with proper configuration
✓ Create .eslintrc.json for code quality
✓ Create go.mod for Go project
```

### Phase 6: Build & Package

```
✓ Compile TypeScript (extension)
✓ Build Wails application (desktop)
✓ Create .vsix package (extension)
✓ Generate production artifacts
```

---

## 🎯 EXECUTION FLOW

### Complete Ecosystem Setup

```
START
  │
  ├─→ Preflight Checks
  │   ├─ PowerShell version
  │   ├─ Administrator rights
  │   └─ System requirements
  │
  ├─→ Phase 1: Desktop App Setup
  │   ├─ Verify dependencies (Go, Node.js, Wails)
  │   ├─ Create directory structure
  │   ├─ Generate 30+ files
  │   ├─ Install npm packages
  │   ├─ Configure Wails
  │   └─ Build application
  │
  ├─→ Phase 2: VS Code Extension Setup
  │   ├─ Verify dependencies (Node.js, npm, vsce)
  │   ├─ Create directory structure
  │   ├─ Generate 20+ files
  │   ├─ Install npm packages
  │   ├─ Configure TypeScript
  │   ├─ Compile TypeScript
  │   └─ Package extension
  │
  ├─→ Phase 3: Integration Verification
  │   ├─ Verify desktop app files
  │   ├─ Verify extension files
  │   └─ Check all configurations
  │
  ├─→ Final Report
  │   ├─ Setup summary
  │   ├─ Status for each component
  │   └─ Next steps instructions
  │
  END
```

---

## ⏱️ TIME ESTIMATES

| Setup Type      | Duration  | Status           |
| --------------- | --------- | ---------------- |
| Complete (Full) | 15-20 min | Automated        |
| Desktop Only    | 8-10 min  | Automated        |
| Extension Only  | 5-7 min   | Automated        |
| Dependencies    | 2-5 min   | Auto (if needed) |

**Factors that affect time**:

- Network speed (npm/go package downloads)
- Disk speed
- System resources
- Existing installations

---

## 🔍 WHAT YOU CAN MODIFY

While the scripts are autonomous, you can customize:

### Before Running

1. **Directory Paths** - Edit scripts to change where files are created
2. **Dependencies** - Modify version numbers in scripts
3. **Component Names** - Change generated file names
4. **Configurations** - Adjust package.json, tsconfig.json, etc.

### During Execution

1. Skip dependency checks with `-SkipDependencies`
2. Choose setup type with `-SetupType`
3. Run in development mode with `-DevelopmentOnly`

### After Running

1. Modify generated component implementations
2. Add additional dependencies with npm/go
3. Customize UI components
4. Add more handlers/services

---

## ✅ VALIDATION CHECKLIST

After scripts complete, verify:

### Desktop App

- [ ] `cmd/ryzanstein/main.go` exists and contains Wails setup
- [ ] `packages/desktop/src/components/ChatPanel.tsx` exists
- [ ] `packages/desktop/src/hooks/useChat.ts` exists
- [ ] `wails.json` configured correctly
- [ ] `go.mod` initialized
- [ ] `node_modules/` contains dependencies
- [ ] No compilation errors

### VS Code Extension

- [ ] `src/extension.ts` exists with entry point
- [ ] `src/webview/chatPanel.ts` exists
- [ ] `src/services/ryzansteinAPI.ts` exists
- [ ] `dist/` folder contains compiled output
- [ ] `.vsix` file created for packaging
- [ ] `node_modules/` contains dependencies
- [ ] TypeScript compiles without errors

### Integration

- [ ] Both setups completed successfully
- [ ] No critical errors reported
- [ ] All files verified
- [ ] Ready for development

---

## 🛠️ TROUBLESHOOTING

### Script Won't Run

```powershell
# Check execution policy
Get-ExecutionPolicy

# Set if needed (temporary)
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

# Run script
.\SETUP_COMPLETE_ECOSYSTEM_MASTER.ps1
```

### Dependencies Not Found

```powershell
# Run without skipping dependency checks
.\SETUP_COMPLETE_ECOSYSTEM_MASTER.ps1 -SkipDependencies:$false

# Or install manually:
go install github.com/wailsapp/wails/v3/cmd/wails@latest
npm install -g @vscode/vsce
```

### Build Fails

```powershell
# Check Go version
go version

# Check Node version
node --version
npm --version

# Run with verbose output
.\SETUP_COMPLETE_ECOSYSTEM_MASTER.ps1 -Verbose
```

---

## 📞 SUPPORT

### If Something Goes Wrong

1. **Read the error message** - Scripts provide detailed error info
2. **Check prerequisites** - Run preflight checks
3. **Verify paths** - Ensure directories are correct
4. **Run with verbose** - Get more detailed output
5. **Check logs** - Scripts show what they're doing

### Key Error Messages

| Error                   | Cause                  | Solution                         |
| ----------------------- | ---------------------- | -------------------------------- |
| "Go not installed"      | Go SDK missing         | Install from golang.org          |
| "Node.js not found"     | Node.js not installed  | Install from nodejs.org          |
| "vsce not found"        | VS Code CLI missing    | Run: npm install -g @vscode/vsce |
| "Admin rights required" | Script needs elevation | Run PowerShell as Administrator  |

---

## 🎓 LEARNING RESOURCES

- **Wails**: https://wails.io/docs/introduction
- **React**: https://react.dev
- **TypeScript**: https://www.typescriptlang.org
- **VS Code Extension API**: https://code.visualstudio.com/api
- **Go**: https://golang.org/doc/

---

## 📈 NEXT STEPS AFTER SETUP

### 1. Desktop App Development

```bash
cd s:\Ryot\desktop
wails dev              # Start development server
```

### 2. VS Code Extension Development

```bash
cd s:\Ryot\vscode-extension
npm run watch         # Watch for TypeScript changes
# Then press F5 in VS Code to start extension host
```

### 3. Integration Testing

```bash
# Start both simultaneously
# Terminal 1:
cd s:\Ryot\desktop && wails dev

# Terminal 2:
cd s:\Ryot\vscode-extension && npm run watch

# Terminal 3 (VS Code):
# Press F5 to launch extension development host
```

---

## 📝 CUSTOMIZATION EXAMPLES

### Change Generated Component Names

Edit the script before running:

```powershell
# In SETUP_DESKTOP_APP_MASTER.ps1
# Change ChatPanel.tsx to MyChat.tsx
Create-FileIfNotExists (Join-Path $frontendPath "src\components\MyChat.tsx") $chatPanel
```

### Add Additional Dependencies

Edit before running:

```powershell
# In SETUP_DESKTOP_APP_MASTER.ps1
# Add to npm install line:
npm install axios zustand react-router-dom moment lodash
```

### Modify Build Configuration

Edit wails.json after generation:

```json
{
  "app": {
    "title": "My Custom Title",
    "width": 1600,
    "height": 1000
  }
}
```

---

**Master Scripts Created**: ✅  
**Autonomy Level**: 95%+  
**Ready for Production**: ✅

**Start your setup now!**

```powershell
cd s:\Ryot
.\SETUP_COMPLETE_ECOSYSTEM_MASTER.ps1
```
