// nvdiffrast_wrapper.cu
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstring>
#include <cmath>

// Rasterization kernel
__global__ void rasterizeKernel(
    const float* __restrict__ pos,
    const int32_t* __restrict__ tri,
    int num_tri,
    int height,
    int width,
    float* __restrict__ out,
    int32_t* __restrict__ out_db
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

    int idx = py * width + px;
    float px_ndc = (2.0f * px / width) - 1.0f;
    float py_ndc = (2.0f * py / height) - 1.0f;

    float min_depth = 1e10f;
    int hit_tri = -1;
    float bary_u = 0, bary_v = 0, bary_w = 0;

    // Test all triangles
    for (int t = 0; t < num_tri; t++) {
        int i0 = tri[t * 3 + 0];
        int i1 = tri[t * 3 + 1];
        int i2 = tri[t * 3 + 2];

        // Get vertices in clip space
        float v0x = pos[i0 * 4 + 0];
        float v0y = pos[i0 * 4 + 1];
        float v0z = pos[i0 * 4 + 2];
        float v0w = pos[i0 * 4 + 3];

        float v1x = pos[i1 * 4 + 0];
        float v1y = pos[i1 * 4 + 1];
        float v1z = pos[i1 * 4 + 2];
        float v1w = pos[i1 * 4 + 3];

        float v2x = pos[i2 * 4 + 0];
        float v2y = pos[i2 * 4 + 1];
        float v2z = pos[i2 * 4 + 2];
        float v2w = pos[i2 * 4 + 3];

        // Perspective divide
        if (v0w <= 0 || v1w <= 0 || v2w <= 0) continue;

        v0x /= v0w; v0y /= v0w; v0z /= v0w;
        v1x /= v1w; v1y /= v1w; v1z /= v1w;
        v2x /= v2w; v2y /= v2w; v2z /= v2w;

        // Barycentric coordinates
        float denom = (v1y - v2y) * (v0x - v2x) + (v2x - v1x) * (v0y - v2y);
        if (fabsf(denom) < 1e-10f) continue;

        float u = ((v1y - v2y) * (px_ndc - v2x) + (v2x - v1x) * (py_ndc - v2y)) / denom;
        float v = ((v2y - v0y) * (px_ndc - v2x) + (v0x - v2x) * (py_ndc - v2y)) / denom;
        float w = 1.0f - u - v;

        // Check if inside triangle
        if (u >= 0 && v >= 0 && w >= 0) {
            float depth = u * v0z + v * v1z + w * v2z;
            if (depth < min_depth) {
                min_depth = depth;
                hit_tri = t;
                bary_u = u;
                bary_v = v;
                bary_w = w;
            }
        }
    }

    // Write output
    if (hit_tri >= 0) {
        out[idx * 4 + 0] = bary_u;
        out[idx * 4 + 1] = bary_v;
        out[idx * 4 + 2] = bary_w;
        out[idx * 4 + 3] = 1.0f;
        out_db[idx * 4 + 0] = hit_tri;
        out_db[idx * 4 + 1] = tri[hit_tri * 3 + 0];
        out_db[idx * 4 + 2] = tri[hit_tri * 3 + 1];
        out_db[idx * 4 + 3] = tri[hit_tri * 3 + 2];
    } else {
        out[idx * 4 + 0] = 0;
        out[idx * 4 + 1] = 0;
        out[idx * 4 + 2] = 0;
        out[idx * 4 + 3] = 0;
        out_db[idx * 4 + 0] = -1;
        out_db[idx * 4 + 1] = 0;
        out_db[idx * 4 + 2] = 0;
        out_db[idx * 4 + 3] = 0;
    }
}

// Interpolation kernel
__global__ void interpolateKernel(
    const float* __restrict__ attr,
    const float* __restrict__ rast,
    const int32_t* __restrict__ rast_db,
    const int32_t* __restrict__ tri,
    int num_attr,
    int height,
    int width,
    float* __restrict__ out
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

    int idx = py * width + px;
    float u = rast[idx * 4 + 0];
    float v = rast[idx * 4 + 1];
    float w = rast[idx * 4 + 2];
    float alpha = rast[idx * 4 + 3];

    if (alpha > 0) {
        // Get triangle index from rast_db
        int tri_idx = rast_db[idx * 4 + 0];

        if (tri_idx >= 0) {
            // Get vertex indices from rast_db (stored there by rasterizer)
            int i0 = rast_db[idx * 4 + 1];
            int i1 = rast_db[idx * 4 + 2];
            int i2 = rast_db[idx * 4 + 3];

            // Interpolate attributes
            for (int c = 0; c < num_attr; c++) {
                float a0 = attr[i0 * num_attr + c];
                float a1 = attr[i1 * num_attr + c];
                float a2 = attr[i2 * num_attr + c];
                out[idx * num_attr + c] = u * a0 + v * a1 + w * a2;
            }
        } else {
            for (int c = 0; c < num_attr; c++) {
                out[idx * num_attr + c] = 0;
            }
        }
    } else {
        for (int c = 0; c < num_attr; c++) {
            out[idx * num_attr + c] = 0;
        }
    }
}

// Texture sampling kernel
__global__ void textureKernel(
    const float* __restrict__ tex,
    const float* __restrict__ uv,
    int tex_height,
    int tex_width,
    int channels,
    int height,
    int width,
    float* __restrict__ out
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

    int idx = py * width + px;
    float u = uv[idx * 2 + 0];
    float v = uv[idx * 2 + 1];

    // Wrap UV coordinates
    u = u - floorf(u);
    v = v - floorf(v);

    // Bilinear sampling
    float fx = u * (tex_width - 1);
    float fy = v * (tex_height - 1);

    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);
    int x1 = min(x0 + 1, tex_width - 1);
    int y1 = min(y0 + 1, tex_height - 1);

    float wx = fx - x0;
    float wy = fy - y0;

    for (int c = 0; c < channels; c++) {
        float t00 = tex[(y0 * tex_width + x0) * channels + c];
        float t01 = tex[(y0 * tex_width + x1) * channels + c];
        float t10 = tex[(y1 * tex_width + x0) * channels + c];
        float t11 = tex[(y1 * tex_width + x1) * channels + c];

        float t0 = t00 * (1 - wx) + t01 * wx;
        float t1 = t10 * (1 - wx) + t11 * wx;

        out[idx * channels + c] = t0 * (1 - wy) + t1 * wy;
    }
}

extern "C" {

typedef struct {
    cudaStream_t stream;
} NVDRContext;

NVDRContext* nvdr_create_context() {
    NVDRContext* ctx = new NVDRContext();
    cudaStreamCreate(&ctx->stream);
    return ctx;
}

void nvdr_destroy_context(NVDRContext* ctx) {
    if (ctx->stream) cudaStreamDestroy(ctx->stream);
    delete ctx;
}

int nvdr_rasterize(
    NVDRContext* ctx,
    const float* pos,
    const int32_t* tri,
    int num_vtx,
    int num_tri,
    int height,
    int width,
    float* out,
    int32_t* out_db
) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);

    rasterizeKernel<<<gridSize, blockSize, 0, ctx->stream>>>(
        pos, tri, num_tri, height, width, out, out_db
    );

    cudaError_t err = cudaStreamSynchronize(ctx->stream);
    return (err == cudaSuccess) ? 0 : -1;
}

int nvdr_interpolate(
    NVDRContext* ctx,
    const float* attr,
    const float* rast,
    const int32_t* rast_db,
    const int32_t* tri,
    int num_vtx,
    int num_tri,
    int num_attr,
    int height,
    int width,
    float* out
) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);

    interpolateKernel<<<gridSize, blockSize, 0, ctx->stream>>>(
        attr, rast, rast_db, tri, num_attr, height, width, out
    );

    cudaError_t err = cudaStreamSynchronize(ctx->stream);
    return (err == cudaSuccess) ? 0 : -1;
}

int nvdr_texture(
    NVDRContext* ctx,
    const float* tex,
    const float* uv,
    int tex_height,
    int tex_width,
    int channels,
    int height,
    int width,
    int filter_mode,
    int boundary_mode,
    float* out
) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);

    textureKernel<<<gridSize, blockSize, 0, ctx->stream>>>(
        tex, uv, tex_height, tex_width, channels, height, width, out
    );

    cudaError_t err = cudaStreamSynchronize(ctx->stream);
    return (err == cudaSuccess) ? 0 : -1;
}

int nvdr_antialias(
    NVDRContext* ctx,
    const float* color,
    const float* rast,
    const float* pos,
    const int32_t* tri,
    int num_vtx,
    int num_tri,
    int channels,
    int height,
    int width,
    float* out
) {
    // Simple copy for now - full AA is complex
    size_t size = height * width * channels * sizeof(float);
    cudaMemcpyAsync(out, color, size, cudaMemcpyDeviceToDevice, ctx->stream);
    cudaError_t err = cudaStreamSynchronize(ctx->stream);
    return (err == cudaSuccess) ? 0 : -1;
}

} // extern "C"
