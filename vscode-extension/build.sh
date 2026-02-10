#!/bin/bash
# VS Code Extension Build Script
# Packages Ryzanstein extension for VS Code marketplace

set -e

echo "🔨 Building Ryzanstein VS Code Extension..."

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
VERSION=$(grep -m 1 '"version"' package.json | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
PUBLISHER="iamthegreatdestroyer"
NAME="ryzanstein"

echo -e "${BLUE}📦 Version: $VERSION${NC}"
echo -e "${BLUE}👤 Publisher: $PUBLISHER${NC}"

# Install dependencies
echo -e "${BLUE}📥 Installing dependencies...${NC}"
npm ci

# Type check
echo -e "${BLUE}✓ Type checking...${NC}"
npx tsc --noEmit

# Lint
echo -e "${BLUE}✓ Linting...${NC}"
npm run lint || true

# Build extension
echo -e "${BLUE}⚙️  Building extension...${NC}"
npm run compile

# Create VSIX package
echo -e "${BLUE}📦 Creating VSIX package...${NC}"
if command -v vsce &> /dev/null; then
    vsce package -o "${NAME}-${VERSION}.vsix"
else
    echo -e "${BLUE}Installing vsce...${NC}"
    npm install -g @vscode/vsce
    vsce package -o "${NAME}-${VERSION}.vsix"
fi

# Summary
echo -e "${GREEN}✅ Build complete!${NC}"
echo ""
echo "Extension Package: ${NAME}-${VERSION}.vsix"
echo ""
echo "To publish to marketplace:"
echo "  vsce publish -p <access-token>"
echo ""
echo "To install locally:"
echo "  code --install-extension ${NAME}-${VERSION}.vsix"
