#!/usr/bin/env fish

set -l CUDA_PATH /usr/local/cuda
set -l TRT_INCLUDE /usr/include/x86_64-linux-gnu
set -l TRT_LIB /usr/lib/x86_64-linux-gnu

set -l CXX clang++
set -l CXXFLAGS \
    -std=c++17 \
    -fPIC \
    -Ofast \
    -march=native \
    -mtune=native \
    -flto=full \
    -fwhole-program-vtables \
    -funroll-loops \
    -fvectorize \
    -fslp-vectorize \
    -finline-functions \
    -fomit-frame-pointer \
    -fno-exceptions \
    -fno-rtti \
    -fno-stack-protector \
    -fno-unwind-tables \
    -fno-asynchronous-unwind-tables \
    -fmerge-all-constants \
    -pthread \
    -DNDEBUG

set -l INCLUDES -I$TRT_INCLUDE -I$CUDA_PATH/include
set -l LDFLAGS \
    -L$TRT_LIB \
    -L$CUDA_PATH/lib64 \
    -flto=full \
    -fuse-ld=lld \
    -Wl,-O3 \
    -Wl,--lto-O3 \
    -Wl,--as-needed \
    -Wl,--gc-sections \
    -Wl,--icf=all
set -l LIBS -lnvinfer -lnvonnxparser -lcudart

mkdir -p build

echo "Building TensorRT wrapper (this will take longer)..."
$CXX $CXXFLAGS $INCLUDES -c src/trt.cpp -o build/trt.o
or exit 1

echo "Creating shared library with full LTO..."
$CXX -shared $CXXFLAGS build/trt.o $LDFLAGS $LIBS -o build/libtrt.so
or exit 1

strip --strip-unneeded build/libtrt.so

echo "Build complete!"
ls -lh build/libtrt.so
