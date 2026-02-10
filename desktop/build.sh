#!/bin/bash
# Desktop App Build Script
# Builds Ryzanstein desktop application for all platforms

set -e

echo "🔨 Building Ryzanstein Desktop Application..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="Ryzanstein"
VERSION=$(grep -m 1 'version' package.json | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
BUILD_DIR="build"
DIST_DIR="dist"

# Create build directory
mkdir -p $BUILD_DIR

# Detect platform
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="linux"
    EXTENSION="AppImage"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="darwin"
    EXTENSION="dmg"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    PLATFORM="windows"
    EXTENSION="exe"
else
    PLATFORM="unknown"
fi

echo -e "${BLUE}🖥️  Platform: $PLATFORM${NC}"
echo -e "${BLUE}📦 Version: $VERSION${NC}"

# Install dependencies
echo -e "${BLUE}📥 Installing dependencies...${NC}"
npm ci

# Build frontend
echo -e "${BLUE}🎨 Building frontend...${NC}"
cd packages/desktop
npm run build
cd ../..

# Build with Wails
echo -e "${BLUE}⚙️  Building with Wails...${NC}"
wails build -o $APP_NAME -platform $PLATFORM -ldflags "-X 'main.version=$VERSION'"

# Create distribution package
echo -e "${BLUE}📦 Creating distribution package...${NC}"
mkdir -p $DIST_DIR
cp -r $BUILD_DIR/bin/* $DIST_DIR/

# Generate checksums
echo -e "${BLUE}🔐 Generating checksums...${NC}"
cd $DIST_DIR
if command -v sha256sum &> /dev/null; then
    sha256sum * > SHA256SUMS
else
    shasum -a 256 * > SHA256SUMS
fi
cd ..

# Build summary
echo -e "${GREEN}✅ Build complete!${NC}"
echo -e "${GREEN}📍 Output: $DIST_DIR/${NC}"
echo ""
echo "Platform: $PLATFORM"
echo "Version: $VERSION"
echo "Binary: $APP_NAME.$EXTENSION"
echo ""
echo "To run the application:"
if [[ "$PLATFORM" == "windows" ]]; then
    echo "  $DIST_DIR/$APP_NAME.exe"
elif [[ "$PLATFORM" == "darwin" ]]; then
    echo "  open $DIST_DIR/$APP_NAME.dmg"
else
    echo "  $DIST_DIR/$APP_NAME.AppImage"
fi
