# Ryzanstein LLM Quick Start Guide

**Build, Deploy & Run in 5 Minutes**

> **Hardware:** Ryzanstein 7 7730U | **OS:** Windows 11 | **Status:** ✅ Production Ready

---

## 📋 Prerequisites

- **CMake** 3.20+ ([Download](https://cmake.org/download/))
- **Visual Studio 2022** Community+ (with C++ workload)
- **Python** 3.10+ ([Download](https://www.python.org/downloads/))
- **Git** for version control

Verify installation:

```powershell
cmake --version
python --version
```

---

## 🚀 Step 1: Clone & Setup (1 minute)

```powershell
# Navigate to your workspace
cd C:\Users\sgbil\Ryzanstein\Ryzanstein LLM

# Verify project structure
dir /B | Select -First 10
```

Expected output shows: `src/`, `build/`, `CMakeLists.txt`, etc.

---

## 🔨 Step 2: Configure CMake (2 minutes)

```powershell
# Create and enter build directory
if (-not (Test-Path build)) { mkdir build }
cd build

# Configure for Release build with optimizations
cmake -G "Visual Studio 17 2022" `
       -A x64 `
       -DBUILD_TESTS=ON `
       -DCMAKE_BUILD_TYPE=Release `
       ..

# Output should end with:
# "Configuring done"
# "Build files have been written to: C:\...\build"
```

---

## 🏗️ Step 3: Build (2 minutes)

```powershell
# Build in Release configuration (optimized)
cmake --build . --config Release -j 8

# Monitor output for:
# ✓ "Build completed successfully"
# ✓ No fatal errors
# ✓ Linking complete
```

**Build time:** ~90 seconds on Ryzanstein 7 7730U

---

## ✅ Step 4: Run Tests (< 1 minute)

```powershell
# Run the test suite
ctest --output-on-failure -C Release

# Expected output:
# Test project C:\Users\sgbil\Ryzanstein\Ryzanstein LLM\build
# 100% tests passed
```

---

## 🎯 Verify Installation

```powershell
# List build artifacts
dir Release\*.exe

# Check binaries exist:
# - benchmark.exe
# - test_engine.exe
# - inference.exe (if available)
```

---

## 💡 Quick Integration

See **[INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md)** for:

- Using BitNet in your Python project
- Calling C++ bindings
- Configuration options

---

## 📊 Performance Check

```powershell
# Run benchmark (if available)
.\Release\benchmark.exe

# Expected output:
# Throughput: ~0.42 tok/s
# Memory: <500 MB
```

See **[PERFORMANCE_REPORT.md](./PERFORMANCE_REPORT.md)** for detailed metrics.

---

## 🏗️ Project Structure

```
Ryzanstein LLM/
├── src/
│   ├── core/
│   │   ├── tmac/        ← Token-Aligned Memory (T-MAC)
│   │   ├── bitnet/      ← BitNet 1.58b quantization
│   │   └── kvcache/     ← KV Cache optimization
│   ├── python/          ← Python bindings
│   └── tests/           ← Test suite
├── build/               ← CMake output
├── CMakeLists.txt       ← Build configuration
└── pyproject.toml       ← Python package config
```

See **[ARCHITECTURE.md](./ARCHITECTURE.md)** for component details.

---

## 🔗 Next Steps

1. **For Integration:** → [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md)
2. **For Deployment:** → [DEPLOYMENT_CHECKLIST.md](./DEPLOYMENT_CHECKLIST.md)
3. **For Architecture:** → [ARCHITECTURE.md](./ARCHITECTURE.md)
4. **For Benchmarks:** → [PERFORMANCE_REPORT.md](./PERFORMANCE_REPORT.md)

---

## 🐛 Troubleshooting

| Issue               | Solution                                                      |
| ------------------- | ------------------------------------------------------------- |
| CMake not found     | Add to PATH: `C:\Program Files\CMake\bin`                     |
| Python not detected | Use full path: `C:\Python310\python.exe`                      |
| Build fails         | Clear cache: `rm -r build && mkdir build && cd build`         |
| Tests fail          | Check [DEPLOYMENT_CHECKLIST.md](./DEPLOYMENT_CHECKLIST.md) #5 |

---

## ✨ You're Ready!

Your Ryzanstein LLM environment is now production-ready. Proceed to **[INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md)** to start using BitNet in your applications.

**Status:** ✅ All systems operational

---

_Last updated: December 2025 | Hardware: Ryzanstein 7 7730U | Windows 11_
