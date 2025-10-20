#!/usr/bin/env fish
# build.fish

set -x CUDA_HOME /usr/local/cuda
set -x NVDIFF_ROOT (pwd)/nvdiffrast

# Clone nvdiffrast if needed
if not test -d $NVDIFF_ROOT
    git clone https://github.com/NVlabs/nvdiffrast.git $NVDIFF_ROOT
    rm -rf nvdiffrast/.git
end

mkdir -p build

# Compile the wrapper
nvcc -O3 -std=c++17 \
    -Xcompiler -fPIC \
    --shared \
    -arch=sm_89 \
    -I$NVDIFF_ROOT/nvdiffrast/common \
    -I$CUDA_HOME/include \
    -L$CUDA_HOME/lib64 \
    -lcudart -lcuda \
    src/nvdiffrast.cu \
    -o build/libnvdiffrast.so

echo "Build complete: build/libnvdiffrast.so"
