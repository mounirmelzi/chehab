#!/bin/bash
# ============================================================================
#  HEonGPU local installation script for HPC
#  Installs HEonGPU into $HOME/local/heongpu (user-local, no root needed)
# ============================================================================
set -euo pipefail

INSTALL_PREFIX="${HOME}/local/heongpu"
BUILD_DIR="/tmp/${USER}_heongpu_build"
HEONGPU_REPO="https://github.com/Alisah-Ozcan/HEonGPU.git"
HEONGPU_BRANCH="main"

echo "═══════════════════════════════════════════════════════════"
echo "  HEonGPU Installation Script"
echo "  Install prefix: ${INSTALL_PREFIX}"
echo "═══════════════════════════════════════════════════════════"

module load cuda/12.2.0 2>/dev/null || module load cuda 2>/dev/null || {
    echo "WARNING: Could not load CUDA module. Checking if nvcc is available..."
    if ! command -v nvcc &>/dev/null; then
        echo "ERROR: CUDA toolkit not found. Please load the CUDA module manually."
        echo "  Try: module avail cuda"
        exit 1
    fi
}

module load cmake 2>/dev/null || true
module load gcc 2>/dev/null || true
# Do NOT load gmp/ntl modules -- we use spack GMP 6.3.0 + local NTL to avoid version conflicts

# Build NTL locally using spack GMP 6.3.0 (headers + lib from same source)
GMP_PREFIX="/share/apps/NYUAD6/spack/spack-0.23.0/opt/spack/linux-rocky8-zen/gcc-8.5.0/gmp-6.3.0-zims4vx7m6ggtn3ava2r2ksidghaly5v"
echo "Using GMP from: ${GMP_PREFIX}"
export LD_LIBRARY_PATH="${GMP_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="${GMP_PREFIX}/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"

if [ ! -d "${HOME}/local/ntl/include/NTL" ]; then
    echo "Installing NTL locally..."
    rm -rf "${HOME}/local/ntl"
    NTL_BUILD="/tmp/${USER}_ntl_build"
    rm -rf "$NTL_BUILD"
    mkdir -p "$NTL_BUILD" && cd "$NTL_BUILD"
    curl -L -o ntl.tar.gz https://libntl.org/ntl-11.5.1.tar.gz
    tar xzf ntl.tar.gz && cd ntl-11.5.1/src
    ./configure PREFIX="${HOME}/local/ntl" GMP_PREFIX="${GMP_PREFIX}" NTL_THREADS=on SHARED=on
    make -j$(nproc)
    make install
    cd / && rm -rf "$NTL_BUILD"
    echo "NTL installed at: ${HOME}/local/ntl"
else
    echo "NTL already installed at: ${HOME}/local/ntl"
fi

export LD_LIBRARY_PATH="${HOME}/local/ntl/lib:${LD_LIBRARY_PATH}"

echo ""
echo "CUDA version: $(nvcc --version | grep release)"
echo "CMake version: $(cmake --version | head -1)"
echo "GCC version: $(g++ --version | head -1)"
echo ""

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "Cloning HEonGPU..."
git clone --depth 1 --branch "${HEONGPU_BRANCH}" "${HEONGPU_REPO}" heongpu
cd heongpu

CUDA_ROOT=$(dirname $(dirname $(which nvcc)))
THRUST_DIR="${CUDA_ROOT}/include"
if [ ! -d "${THRUST_DIR}/thrust" ]; then
    # Try alternate locations
    for candidate in /share/apps/NYUAD5/cuda/*/include; do
        if [ -d "${candidate}/thrust" ]; then
            THRUST_DIR="${candidate}"
            break
        fi
    done
fi
echo "CUDA root: ${CUDA_ROOT}"
echo "Thrust dir: ${THRUST_DIR}"

echo ""
echo "Configuring HEonGPU..."
cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="70" \
    -DTHRUST_INCLUDE_DIR="${THRUST_DIR}" \
    -DGMP_ROOT="${GMP_PREFIX}" \
    -DGMP_DIR="${GMP_PREFIX}" \
    -DGMP_INCLUDE_DIR="${GMP_PREFIX}/include" \
    -DGMP_LIBRARIES="${GMP_PREFIX}/lib/libgmp.so" \
    -DNTL_ROOT="${HOME}/local/ntl" \
    -DNTL_DIR="${HOME}/local/ntl" \
    -DNTL_INCLUDE_DIR="${HOME}/local/ntl/include" \
    -DNTL_LIBRARIES="${HOME}/local/ntl/lib/libntl.so" \
    -DCMAKE_CXX_FLAGS="-I${HOME}/local/ntl/include -I${GMP_PREFIX}/include" \
    -DCMAKE_CUDA_FLAGS="-I${HOME}/local/ntl/include -I${GMP_PREFIX}/include" \
    -DCMAKE_EXE_LINKER_FLAGS="-L${HOME}/local/ntl/lib -L${GMP_PREFIX}/lib" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L${HOME}/local/ntl/lib -L${GMP_PREFIX}/lib" \
    -DHEONGPU_BUILD_EXAMPLES=OFF \
    -DHEONGPU_BUILD_TESTS=OFF

echo ""
echo "Building HEonGPU (this may take 10-20 minutes)..."
cmake --build build -j$(nproc)

echo ""
echo "Installing to ${INSTALL_PREFIX}..."
cmake --install build

echo ""
echo "Cleaning up build directory..."
rm -rf "${BUILD_DIR}"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  HEonGPU installed successfully!"
echo "  Location: ${INSTALL_PREFIX}"
echo ""
echo "  To use with CMake, add:"
echo "    -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX}"
echo "═══════════════════════════════════════════════════════════"
