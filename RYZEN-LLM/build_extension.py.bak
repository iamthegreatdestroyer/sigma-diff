#!/usr/bin/env python3
"""
Build script for Ryzen LLM C++ extension
Handles setup of build environment and compilation
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status"""
    print(f"\n🔧 {description}")
    print(f"Command: {cmd}")

    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✓ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def check_build_tools():
    """Check if required build tools are available"""
    print("🔍 Checking build tools...")

    tools = {
        "cmake": "CMake",
        "g++": "C++ Compiler (MinGW/Clang)",
        "gcc": "C Compiler (MinGW/Clang)"
    }

    missing = []
    for tool, name in tools.items():
        try:
            result = subprocess.run(f'where {tool}', shell=True, capture_output=True)
            if result.returncode != 0:
                missing.append(name)
        except:
            missing.append(name)

    if missing:
        print(f"❌ Missing build tools: {', '.join(missing)}")
        print("\n📦 Please install:")
        print("- LLVM MinGW (winget install LLVM.LLVM)")
        print("- CMake (winget install Kitware.CMake)")
        print("- Ensure they're in your PATH")
        return False

    print("✓ All build tools found")
    return True

def setup_build_directory():
    """Set up build directory"""
    build_dir = Path("build")
    if not build_dir.exists():
        build_dir.mkdir()
        print("✓ Created build directory")

    cpp_build_dir = build_dir / "cpp"
    if not cpp_build_dir.exists():
        cpp_build_dir.mkdir()
        print("✓ Created C++ build directory")

    return build_dir, cpp_build_dir

def configure_cmake(build_dir: Path, cpp_build_dir: Path):
    """Configure CMake build"""
    print(f"\n🔧 Configuring CMake in {cpp_build_dir}")

    # Change to build directory
    os.chdir(cpp_build_dir)

    # Run CMake configuration
    cmake_cmd = f'cmake -S "../.." -B "." -DCMAKE_BUILD_TYPE=Release'

    # Add Python-specific configuration with forward slashes for MinGW compatibility
    python_executable = sys.executable.replace('\\', '/')
    python_include = sys.base_prefix.replace('\\', '/') + "/include"
    python_library = sys.base_prefix.replace('\\', '/') + "/libs/python313.lib"

    cmake_cmd += f' -DPython_EXECUTABLE="{python_executable}"'
    cmake_cmd += f' -DPython_INCLUDE_DIR="{python_include}"'
    cmake_cmd += f' -DPython_LIBRARY="{python_library}"'

    # Specify MinGW generator for Windows
    if platform.system() == "Windows":
        cmake_cmd += ' -G "MinGW Makefiles"'

    success = run_command(cmake_cmd, "Configuring CMake")

    # Change back
    os.chdir("../../")

    return success

def build_extension(build_dir: Path, cpp_build_dir: Path):
    """Build the C++ extension"""
    print(f"\n🔧 Building extension in {cpp_build_dir}")

    # Change to build directory
    os.chdir(cpp_build_dir)

    # Run build
    build_cmd = 'cmake --build . --config Release --parallel'

    success = run_command(build_cmd, "Building C++ extension")

    # Change back
    os.chdir("../../")

    return success

def install_extension(build_dir: Path, cpp_build_dir: Path):
    """Install the built extension"""
    print("\n🔧 Installing extension")

    # The CMakeLists.txt already puts the extension in the correct location
    # Check if it exists
    python_build_dir = build_dir / "python" / "ryzen_llm"
    extension_files = list(python_build_dir.glob("*.pyd")) + list(python_build_dir.glob("*.dll"))

    if not extension_files:
        print("❌ No extension file (.pyd or .dll) found in build directory")
        return False

    extension_file = extension_files[0]
    print(f"✓ Extension installed to {extension_file}")
    return True

def test_extension():
    """Test the built extension"""
    print("\n🧪 Testing extension")

    # Add build directory to Python path
    build_python_dir = Path("build/python")
    if str(build_python_dir) not in sys.path:
        sys.path.insert(0, str(build_python_dir))

    try:
        import ryzen_llm_bindings
        print("✓ Extension imports successfully")

        # Check for test function
        has_test_function = hasattr(ryzen_llm_bindings, 'test_function')

        if has_test_function:
            print("✓ Test function available")
            result = ryzen_llm_bindings.test_function()
            print(f"✓ Test function returned: {result}")
            return True
        else:
            print("⚠ Test function not exposed")
            available = [attr for attr in dir(ryzen_llm_bindings) if not attr.startswith('_')]
            print(f"Available functions: {available}")
            return False

    except ImportError as e:
        print(f"❌ Extension import failed: {e}")
        return False

def main():
    print("=== Ryzen LLM C++ Extension Build Script ===\n")

    # Check build tools
    if not check_build_tools():
        return 1

    # Setup directories
    build_dir, cpp_build_dir = setup_build_directory()

    # Configure CMake
    if not configure_cmake(build_dir, cpp_build_dir):
        return 1

    # Build extension
    if not build_extension(build_dir, cpp_build_dir):
        return 1

    # Install extension
    if not install_extension(build_dir, cpp_build_dir):
        return 1

    # Test extension
    if not test_extension():
        return 1

    print("\n🎉 Build completed successfully!")
    print("\n📊 Summary:")
    print("- ✅ Build tools verified")
    print("- ✅ CMake configuration completed")
    print("- ✅ C++ extension compiled")
    print("- ✅ Extension installed")
    print("- ✅ Basic import test passed")

    print("\n🚀 Next steps:")
    print("- Run 'python test_cpp_extension.py' to validate implementations")
    print("- Run 'python test_quantization_performance.py' for full testing")

    return 0

if __name__ == "__main__":
    sys.exit(main())