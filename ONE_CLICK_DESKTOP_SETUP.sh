#!/bin/bash
# One-Click Desktop Setup for Mac/Linux

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DESKTOP_PATH="$PROJECT_ROOT/desktop"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Functions
write_header() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════════════╗"
    echo "║ $1"
    echo "╚════════════════════════════════════════════════════════════════════════════════╝"
    echo ""
}

write_step() {
    echo ""
    echo -e "${MAGENTA}Step $1 : $2${NC}"
    echo "─────────────────────────────────────────────────────────────────────────────────"
}

write_success() {
    echo -e "${GREEN}  ✓ $1${NC}"
}

write_info() {
    echo -e "${CYAN}  ℹ $1${NC}"
}

# Validate prerequisites
validate_prerequisites() {
    write_step 1 "Validating Prerequisites"
    
    local all_valid=true
    
    write_info "Checking required dependencies..."
    echo ""
    
    # Check Go
    if command -v go &> /dev/null; then
        write_success "Go installed"
    else
        echo -e "${RED}  ✗ Go not found - Install from https://golang.org/dl${NC}"
        all_valid=false
    fi
    
    # Check Node
    if command -v node &> /dev/null; then
        write_success "Node.js installed"
    else
        echo -e "${RED}  ✗ Node.js not found - Install from https://nodejs.org${NC}"
        all_valid=false
    fi
    
    # Check npm
    if command -v npm &> /dev/null; then
        write_success "npm installed"
    else
        echo -e "${RED}  ✗ npm not found - Install Node.js first${NC}"
        all_valid=false
    fi
    
    echo ""
    
    if [ "$all_valid" = false ]; then
        echo -e "${RED}Missing required dependencies${NC}"
        exit 1
    fi
    
    write_success "All dependencies found!"
}

# Setup backend
setup_backend() {
    write_step 2 "Setting Up Backend (Go)"
    
    write_info "Building Go application..."
    
    cd "$DESKTOP_PATH"
    
    go mod download
    go build -o bin/ryzanstein ./cmd/ryzanstein
    
    write_success "Backend compiled successfully"
}

# Setup frontend
setup_frontend() {
    write_step 3 "Setting Up Frontend (React + Wails)"
    
    FRONTEND_PATH="$DESKTOP_PATH/packages/desktop"
    
    if [ ! -d "$FRONTEND_PATH" ]; then
        echo -e "${YELLOW}Frontend directory not found, skipping${NC}"
        return
    fi
    
    cd "$FRONTEND_PATH"
    
    write_info "Installing npm dependencies..."
    npm install
    
    write_info "Building React application..."
    npm run build
    
    write_success "Frontend built successfully"
}

# Setup Wails
setup_wails() {
    write_step 4 "Setting Up Wails Framework"
    
    write_info "Installing Wails CLI..."
    go install github.com/wailsapp/wails/v2/cmd/wails@latest
    
    write_success "Wails installed successfully"
}

# Show instructions
show_instructions() {
    write_header "SETUP COMPLETE - NEXT STEPS"
    
    echo -e "${GREEN}🎉 Your Ryzanstein Desktop App is ready!${NC}"
    echo ""
    
    echo "╔════════════════════════════════════════════════════════════════════════════════╗"
    echo "║ LAUNCH THE APPLICATION                                                        ║"
    echo "╚════════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    echo -e "${MAGENTA}Option 1: Development Mode (Recommended)${NC}"
    echo -e "${CYAN}  1. cd $DESKTOP_PATH${NC}"
    echo -e "${CYAN}  2. wails dev${NC}"
    echo -e "${CYAN}  → Application launches with hot-reload${NC}"
    echo ""
    
    echo -e "${MAGENTA}Option 2: Production Build${NC}"
    echo -e "${CYAN}  1. cd $DESKTOP_PATH${NC}"
    echo -e "${CYAN}  2. wails build -clean${NC}"
    echo -e "${CYAN}  → Creates package for your OS${NC}"
    echo ""
    
    echo "╔════════════════════════════════════════════════════════════════════════════════╗"
    echo "║ FEATURES INCLUDED                                                              ║"
    echo "╚════════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo -e "${GREEN}  ✓ Elite AI Agent Collective${NC}"
    echo -e "${GREEN}  ✓ Real-time Chat Interface${NC}"
    echo -e "${GREEN}  ✓ Model Management${NC}"
    echo -e "${GREEN}  ✓ Code Generation & Analysis${NC}"
    echo ""
}

# Main
main() {
    write_header "RYZANSTEIN DESKTOP APPLICATION - ONE-CLICK SETUP"
    
    validate_prerequisites
    setup_backend
    setup_frontend
    setup_wails
    show_instructions
    
    echo ""
    write_header "✨ SETUP SUCCESSFUL ✨"
    echo -e "${GREEN}Your application is ready to launch!${NC}"
    echo ""
}

main
