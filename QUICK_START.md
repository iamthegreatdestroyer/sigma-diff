# RYZANSTEIN DESKTOP APP - QUICK START

## 🚀 ONE-CLICK LAUNCH (Windows)

```powershell
cd s:\Ryot
.\ONE_CLICK_DESKTOP_SETUP.ps1
```

**That's it!** The script automatically:

- ✓ Checks dependencies
- ✓ Builds backend
- ✓ Installs frontend
- ✓ Configures Wails
- ✓ Tests everything
- ✓ Shows next steps

---

## ⏱️ TIMING

| Phase            | Time        |
| ---------------- | ----------- |
| Dependency check | 30 sec      |
| Go build         | 1-2 min     |
| npm install      | 2-3 min     |
| Setup            | 1 min       |
| Test             | 30 sec      |
| **TOTAL**        | **5-7 min** |

---

## 📋 CHECKLIST BEFORE SETUP

- [ ] Windows/Mac/Linux with admin access
- [ ] Go 1.20+ installed
- [ ] Node.js 18+ installed
- [ ] npm 8+ installed
- [ ] Internet connection (for dependencies)
- [ ] 500MB free disk space

---

## 🎯 LAUNCH OPTIONS (After Setup)

### Development Mode (HOT RELOAD)

```powershell
cd s:\Ryot\desktop
wails dev
```

**Best for:** Testing, development, debugging

### Production Build (Create Installer)

```powershell
cd s:\Ryot\desktop
wails build -clean
```

**Best for:** Distribution, final build

### Direct Run

```powershell
s:\Ryot\desktop\bin\ryzanstein.exe
```

**Best for:** Quick testing

---

## 🎨 FEATURES

| Feature          | Status     |
| ---------------- | ---------- |
| Chat Interface   | ✓ Included |
| AI Agents (20+)  | ✓ Included |
| Code Generation  | ✓ Included |
| Model Management | ✓ Included |
| Settings Panel   | ✓ Included |
| Hot Reload (Dev) | ✓ Included |
| Installer (Prod) | ✓ Included |

---

## ⚡ TROUBLESHOOTING (2-MINUTE FIXES)

### Script won't run

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### "Go not found"

- Download: https://golang.org/dl
- Restart terminal after install

### "wails: command not found"

```powershell
go install github.com/wailsapp/wails/v2/cmd/wails@latest
```

### "Port 8000 in use"

- Open Settings in app
- Change "Ryzanstein API URL"

### Build fails

```powershell
cd s:\Ryot\desktop
go mod tidy
npm cache clean --force
.\ONE_CLICK_DESKTOP_SETUP.ps1
```

---

## 📁 FILE STRUCTURE

```
s:\Ryot\
├── ONE_CLICK_DESKTOP_SETUP.ps1          ← Run this!
├── DESKTOP_APP_DETAILED_GUIDE.md        ← Full instructions
├── desktop/
│   ├── cmd/ryzanstein/                  ← Go backend
│   ├── packages/desktop/                ← React frontend
│   ├── bin/ryzanstein.exe               ← Built executable
│   └── wails.json                       ← Configuration
└── vscode-extension/                    ← VS Code extension
```

---

## 🔍 SUCCESS SIGNS

After running setup, you should see:

```
✓ All dependencies found!
✓ Backend compiled successfully
✓ Frontend built successfully
✓ Wails installed successfully
✓ Desktop app executable found
```

Then follow the instructions displayed!

---

## 💡 WHAT IF SOMETHING GOES WRONG?

1. **Check error message** - Most are self-explanatory
2. **See full guide** - Open `DESKTOP_APP_DETAILED_GUIDE.md`
3. **Run cleanup** - `go mod tidy && npm cache clean --force`
4. **Try again** - Run `ONE_CLICK_DESKTOP_SETUP.ps1` again

---

## 🎓 NEXT STEPS

| Step | Action               |
| ---- | -------------------- |
| 1    | Run setup script     |
| 2    | Choose launch option |
| 3    | Test chat feature    |
| 4    | Select an AI agent   |
| 5    | Send a message       |

---

## 📞 QUICK REFERENCE

| Need            | Command                                  |
| --------------- | ---------------------------------------- |
| Setup app       | `.\ONE_CLICK_DESKTOP_SETUP.ps1`          |
| Dev mode        | `wails dev` (in desktop folder)          |
| Build installer | `wails build -clean`                     |
| Clean rebuild   | `go mod tidy && npm cache clean --force` |
| Check Go        | `go version`                             |
| Check Node      | `node --version`                         |

---

## ✨ YOU'RE READY!

The one-click setup is **simple, fast, and flawless**.

No manual configuration. No hidden steps. Just **one command** and you're done.

**Enjoy Ryzanstein!** 🚀
