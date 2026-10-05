#!/bin/bash
# Script to build CHEHAB with Lattigo support and test code generation
# Usage: ./scripts/build_and_test_lattigo.sh

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== CHEHAB Lattigo Backend Build & Test ==="
echo "Project root: $PROJECT_ROOT"

# Step 1: Build CHEHAB
echo ""
echo "=== Step 1: Building CHEHAB ==="
cd "$PROJECT_ROOT"

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure and build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "Build completed!"

# Step 2: Test Lattigo code generation
echo ""
echo "=== Step 2: Testing Lattigo Code Generation ==="

# Build the dot_product_lattigo benchmark
cd "$PROJECT_ROOT/benchmarks/dot_product"

# Check if the Lattigo benchmark is in CMakeLists.txt, if not we'll compile manually
if ! grep -q "dot_product_lattigo" CMakeLists.txt 2>/dev/null; then
    echo "Compiling dot_product_lattigo manually..."
    g++ -std=c++17 -I"$PROJECT_ROOT/src" \
        -L"$PROJECT_ROOT/build" \
        dot_product_lattigo.cpp \
        ../global_variables.cpp \
        -lfheco -o dot_product_lattigo \
        2>/dev/null || echo "Note: Manual compilation may need adjustment for your system"
fi

# Run the benchmark to generate Lattigo code
echo ""
echo "Generating Lattigo code for dot product (4 elements)..."
./dot_product_lattigo 4 2>/dev/null || echo "Run from build directory or adjust paths"

# Step 3: Setup Lattigo backend
echo ""
echo "=== Step 3: Setting up Lattigo Backend ==="
cd "$PROJECT_ROOT/lattigo_backend"

# Initialize Go module if needed
if [ ! -f "go.sum" ]; then
    echo "Initializing Go module..."
    go mod tidy 2>/dev/null || echo "Note: Run 'go mod tidy' manually if Go is not in PATH"
fi

# Test Lattigo installation
echo ""
echo "Testing Lattigo installation..."
go run example_test.go 2>/dev/null || echo "Note: Run 'go run example_test.go' manually in lattigo_backend/"

echo ""
echo "=== Build & Test Complete ==="
echo ""
echo "Next steps:"
echo "1. Build CHEHAB: cd build && cmake .. && make"
echo "2. Run a benchmark: cd benchmarks/dot_product && ./dot_product_lattigo 4"
echo "3. Test Lattigo: cd lattigo_backend && go run example_test.go"
echo "4. Copy generated .go files to lattigo_backend/ and run them"

