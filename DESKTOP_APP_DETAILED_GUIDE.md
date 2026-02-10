# ONE-CLICK RYZANSTEIN DESKTOP APP SETUP

## Quick Start (30 seconds)

```powershell
cd s:\Ryot
.\ONE_CLICK_DESKTOP_SETUP.ps1
```

That's it! The script will:

1. ✓ Validate all dependencies
2. ✓ Build the Go backend
3. ✓ Setup React frontend
4. ✓ Install Wails framework
5. ✓ Test everything works
6. ✓ Show you detailed instructions

---

## What You Need Before Running

### Required Software (Install Once)

#### 1. **Go 1.20+** (Programming Language)

- **Windows**: https://golang.org/dl
- **Mac**: `brew install go`
- **Linux**: `sudo apt-get install golang-go`

**Verify:**

```powershell
go version
```

#### 2. **Node.js 18+** (JavaScript Runtime)

- **All OS**: https://nodejs.org
- Choose **LTS version**

**Verify:**

```powershell
node --version
npm --version
```

#### 3. **Git** (Optional but recommended)

- **All OS**: https://git-scm.com

---

## Step-by-Step Setup

### Step 1: Open PowerShell

- **Windows**: Press `Win + X`, select "Windows PowerShell (Admin)"
- **Mac/Linux**: Open Terminal

### Step 2: Navigate to Project

```powershell
cd s:\Ryot
```

### Step 3: Run One-Click Setup

```powershell
.\ONE_CLICK_DESKTOP_SETUP.ps1
```

### Step 4: Wait for Completion

The script will:

- Check dependencies (30 seconds)
- Build backend (1-2 minutes)
- Install frontend (2-3 minutes)
- Setup framework (1 minute)
- Test everything (30 seconds)

**Total Time: 5-7 minutes**

### Step 5: Launch Application

Once setup completes, you'll see instructions. Choose one:

#### **Option A: Development Mode** (Best for Testing)

```powershell
cd s:\Ryot\desktop
wails dev
```

- Opens application window
- Hot-reload enabled (changes auto-refresh)
- Debugging tools available

#### **Option B: Production Build** (Create Installer)

```powershell
cd s:\Ryot\desktop
wails build -clean
```

- Creates Windows installer
- Creates executable
- Ready to distribute

#### **Option C: Direct Run**

```powershell
# Windows
s:\Ryot\desktop\bin\ryzanstein.exe

# Mac/Linux
./s/Ryot/desktop/bin/ryzanstein
```

---

## Features Available After Setup

✓ **Chat Interface**

- Real-time chat with AI agents
- Message history
- Model selection

✓ **AI Agents**

- APEX (Computer Science)
- ARCHITECT (System Design)
- TENSOR (Machine Learning)
- CIPHER (Security)
- - 16 more agents

✓ **Model Management**

- View available models
- Load/switch models
- Configure settings

✓ **Code Analysis**

- Generate code
- Explain existing code
- Find bugs
- Optimize performance

✓ **Settings Panel**

- API configuration
- Agent preferences
- UI customization

---

## Configuration

### API Server

**Default**: `http://localhost:8000`

**Change Location**:

1. Open application
2. Click Settings ⚙️
3. Update "Ryzanstein API URL"
4. Save and restart

### MCP Server

**Default**: `localhost:50051`

**Change in Settings** if needed

---

## Troubleshooting

### Problem: "Script cannot be loaded"

**Solution**: Enable script execution

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then run setup again.

---

### Problem: "Go not found"

**Solution**: Install Go first

- https://golang.org/dl
- Add to PATH (installer should do this)
- Restart terminal

**Verify:**

```powershell
go version
```

---

### Problem: "wails: command not found"

**Solution**: Install Wails manually

```powershell
go install github.com/wailsapp/wails/v2/cmd/wails@latest
```

---

### Problem: "Port 8000 already in use"

**Solution**: Change API URL in settings

1. Open app
2. Settings ⚙️
3. Change "Ryzanstein API URL" to different port
4. Restart app

---

### Problem: Build fails with "go mod"

**Solution**: Clean modules

```powershell
cd s:\Ryot\desktop
go mod tidy
.\ONE_CLICK_DESKTOP_SETUP.ps1
```

---

### Problem: Frontend won't build

**Solution**: Clean npm cache

```powershell
cd s:\Ryot\desktop\packages\desktop
npm cache clean --force
npm install
npm run build
```

---

## Advanced Options

### Skip Validation

If you know all dependencies are installed:

```powershell
.\ONE_CLICK_DESKTOP_SETUP.ps1 -SkipValidation
```

### Verbose Output

See detailed logs:

```powershell
.\ONE_CLICK_DESKTOP_SETUP.ps1 -Verbose
```

---

## Development Workflow

### Making Changes

1. Edit code in `s:\Ryot\desktop\packages\desktop\src\`
2. Save file
3. Application auto-refreshes (if using `wails dev`)

### View Logs

- **Development**: Check Terminal output
- **Production**: Check `debug.log` in app directory

### Stop Application

- **Dev Mode**: Press `Ctrl+C` in terminal
- **Production**: Click X or `Alt+F4`

---

## Next Steps

1. **Run setup**: `.\ONE_CLICK_DESKTOP_SETUP.ps1`
2. **Launch app**: `wails dev`
3. **Test features**: Click around, try chat, run code analysis
4. **Build installer**: `wails build -clean` (when ready)

---

## Support

**Questions or Issues?**

1. Check Troubleshooting section above
2. Check application logs
3. Verify all dependencies installed
4. Try clean setup: `go mod tidy` + `npm cache clean --force`

---

## File Locations

| Component      | Location                              |
| -------------- | ------------------------------------- |
| Setup Script   | `s:\Ryot\ONE_CLICK_DESKTOP_SETUP.ps1` |
| Desktop App    | `s:\Ryot\desktop\`                    |
| Go Backend     | `s:\Ryot\desktop\cmd\ryzanstein\`     |
| React Frontend | `s:\Ryot\desktop\packages\desktop\`   |
| Executable     | `s:\Ryot\desktop\bin\ryzanstein.exe`  |
| Config         | `s:\Ryot\desktop\wails.json`          |

---

## Success Indicators

After setup, you should see:

- ✓ "All dependencies found!"
- ✓ "Backend compiled successfully"
- ✓ "Frontend built successfully"
- ✓ "Wails installed successfully"
- ✓ "Desktop app executable found"

Then follow the instructions to launch!

---

**You're all set! Enjoy Ryzanstein! 🚀**
