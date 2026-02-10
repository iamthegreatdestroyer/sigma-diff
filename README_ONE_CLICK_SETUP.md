# 🎉 ONE-CLICK RYZANSTEIN DESKTOP APP - COMPLETE SETUP

## ⚡ FASTEST WAY TO GET STARTED (30 seconds to launch)

```powershell
cd s:\Ryot
.\ONE_CLICK_DESKTOP_SETUP.ps1
```

**That's literally all you need to do.**

---

## 📋 WHAT YOU'LL GET

### Automated Installation

- ✓ Dependency validation
- ✓ Go backend compilation
- ✓ React frontend setup
- ✓ Wails framework installation
- ✓ Automatic testing
- ✓ Clear next steps

### Time Breakdown

| Phase              | Duration     |
| ------------------ | ------------ |
| Dependencies check | ~30 sec      |
| Go compilation     | ~1-2 min     |
| npm install        | ~2-3 min     |
| Final setup        | ~1 min       |
| **Total**          | **~5-7 min** |

### Features Available After Setup

- Elite AI Agent Collective (20+ agents)
- Real-time chat interface
- Code generation
- Model management
- Settings panel
- Hot-reload (development mode)
- Installer creation (production mode)

---

## 🚀 THREE WAYS TO LAUNCH

### Option 1: Development Mode (RECOMMENDED)

```powershell
cd s:\Ryot\desktop
wails dev
```

**Best for:** Testing, development, debugging
**Features:** Hot-reload, debugging tools, live code refresh

### Option 2: Production Build

```powershell
cd s:\Ryot\desktop
wails build -clean
```

**Best for:** Distribution, final release
**Creates:** Windows installer + standalone .exe

### Option 3: Direct Execution

```powershell
s:\Ryot\desktop\bin\ryzanstein.exe
```

**Best for:** Quick testing
**Note:** Runs built executable directly

---

## ✅ REQUIREMENTS (Install Once)

### Go 1.20+

- **Download:** https://golang.org/dl
- **Verify:** `go version`
- **Mac:** `brew install go`
- **Linux:** `sudo apt-get install golang-go`

### Node.js 18+

- **Download:** https://nodejs.org (LTS version)
- **Verify:** `node --version` && `npm --version`

### Git (Optional)

- **Download:** https://git-scm.com

---

## 🎯 STEP-BY-STEP SETUP

### Step 1: Open PowerShell (Admin)

- **Windows:** Press `Win + X`
- **Select:** "Windows PowerShell (Admin)"
- **Mac/Linux:** Open Terminal

### Step 2: Navigate to Project

```powershell
cd s:\Ryot
```

### Step 3: Run Setup

```powershell
.\ONE_CLICK_DESKTOP_SETUP.ps1
```

### Step 4: Watch Magic Happen

The script will:

- Check all dependencies ✓
- Compile Go backend ✓
- Install npm packages ✓
- Setup Wails framework ✓
- Test everything ✓
- Show instructions ✓

### Step 5: Choose Launch Option

Follow the instructions displayed in terminal

---

## 🔧 TROUBLESHOOTING (< 2 minutes each)

### Script won't run

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\ONE_CLICK_DESKTOP_SETUP.ps1
```

### "Go not found"

- Download from https://golang.org/dl
- Restart terminal after install
- Verify: `go version`

### "wails: command not found"

```powershell
go install github.com/wailsapp/wails/v2/cmd/wails@latest
```

### "Port 8000 in use"

1. Open app settings ⚙️
2. Change "Ryzanstein API URL"
3. Restart app

### Build fails

```powershell
cd s:\Ryot\desktop
go mod tidy
npm cache clean --force
.\ONE_CLICK_DESKTOP_SETUP.ps1
```

---

## 📚 DOCUMENTATION FILES

| File                            | Size | Purpose                     |
| ------------------------------- | ---- | --------------------------- |
| `ONE_CLICK_DESKTOP_SETUP.ps1`   | 13KB | Automated Windows setup     |
| `ONE_CLICK_DESKTOP_SETUP.sh`    | 6KB  | Automated Mac/Linux setup   |
| `QUICK_START.md`                | 4KB  | One-page reference card     |
| `DESKTOP_APP_DETAILED_GUIDE.md` | 6KB  | 450+ lines of detailed docs |

**Pick ONE file to read:**

- **Busy?** → Read `QUICK_START.md` (5 min)
- **Want details?** → Read `DESKTOP_APP_DETAILED_GUIDE.md` (15 min)
- **Just want to go?** → Just run the script!

---

## 🎮 WHAT TO DO AFTER LAUNCH

1. **Open Chat** - Click chat icon
2. **Select Agent** - Pick an AI agent (APEX, ARCHITECT, etc.)
3. **Select Model** - Choose a model
4. **Start Chatting** - Send your first message!
5. **Try Features** - Test code generation, explanations, etc.

---

## 📊 FEATURE SHOWCASE

| Feature    | Status  | How to Access          |
| ---------- | ------- | ---------------------- |
| Chat       | ✓ Ready | Main window            |
| AI Agents  | ✓ Ready | Agent panel            |
| Code Gen   | ✓ Ready | Chat + Generate button |
| Model Mgmt | ✓ Ready | Model panel            |
| Settings   | ✓ Ready | ⚙️ Icon                |
| Hot Reload | ✓ Ready | Dev mode only          |
| Installer  | ✓ Ready | `wails build -clean`   |

---

## ❓ FAQ

**Q: Do I need to install anything before running the script?**
A: Yes, Go 1.20+, Node.js 18+, and npm. The script checks for these.

**Q: How long does setup take?**
A: Usually 5-7 minutes total.

**Q: Can I skip the setup and just run the executable?**
A: Only if you've already run setup before.

**Q: What if something fails?**
A: Check Troubleshooting section above. Most issues have 2-minute fixes.

**Q: Can I use this on Mac/Linux?**
A: Yes! Use `ONE_CLICK_DESKTOP_SETUP.sh` instead.

**Q: What ports does it use?**
A: By default: API on 8000, MCP on 50051 (configurable in settings).

**Q: Can I customize the AI agents?**
A: Yes! Full settings panel available in app.

**Q: Is the installer safe?**
A: Yes, created by Wails (trusted framework).

**Q: Can I uninstall easily?**
A: Yes, standard Windows uninstaller.

---

## 🎯 SUCCESS INDICATORS

After running the setup script, you should see:

```
✓ All dependencies found!
✓ Backend compiled successfully
✓ Frontend built successfully
✓ Wails installed successfully
✓ Desktop app executable found
```

Then you're ready to launch!

---

## 📁 FILE LOCATIONS

```
s:\Ryot\
├── ONE_CLICK_DESKTOP_SETUP.ps1       ← Run this!
├── ONE_CLICK_DESKTOP_SETUP.sh        ← Or this (Mac/Linux)
├── QUICK_START.md                    ← Quick reference
├── DESKTOP_APP_DETAILED_GUIDE.md     ← Full instructions
├── desktop/
│   ├── cmd/ryzanstein/               ← Go backend
│   ├── packages/desktop/             ← React frontend
│   ├── bin/ryzanstein.exe            ← Built executable
│   └── wails.json                    ← Configuration
└── vscode-extension/                 ← VS Code extension
```

---

## 🚀 READY TO GO!

**Everything is simple, flawless, and automatic.**

Just run the one-click setup and you're done.

```powershell
cd s:\Ryot
.\ONE_CLICK_DESKTOP_SETUP.ps1
```

**Enjoy Ryzanstein!** ✨

---

## 📞 NEED HELP?

| Issue             | Solution                            |
| ----------------- | ----------------------------------- |
| Script errors     | See Troubleshooting section         |
| Setup takes long  | Normal - Go compilation is slow     |
| App won't launch  | Check error messages                |
| Want to customize | Edit settings in app                |
| Need more info    | See `DESKTOP_APP_DETAILED_GUIDE.md` |

---

**Everything you need is in place. The setup is simple. Just run it and enjoy!** 🎉
