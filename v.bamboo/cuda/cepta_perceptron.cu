#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common_kernels.cuh"

#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

extern "C" {

struct CeptaPerceptronState {
    int P;          // number of neurons/paths
    int d;          // input dimension for dense path
    int alpha;      // number of output branches per neuron
    int vocab;      // vocabulary size for index path

    // Hyperparameters
    float eta_bp_w;
    float eta_bp_f;
    float eta_w;
    float eta_f;
    float eta_sp;
    float w_min, w_max;
    float f_min, f_max;
    float sp_min, sp_max;
    float z_ref, y_ref;
    float eps_z, eps_y;
    float beta_r, beta_m;
    float r_target, m_target;
    float lambda_r, lambda_m;
    int mask_w_mode;  // 0=all,1=active,2=inactive
    int mask_f_mode;  // 0=all,1=active,2=inactive
    int use_ste;
    float ste_slope;

    // Parameters on device
    float* w_dense;   // [P, d]
    float* w_index;   // [P, vocab]
    float* f_out;     // [P, alpha]
    float* sp;        // [P]
    float* r_bar;     // [P]
    float* m_bar;     // [P]

    // Buffers for latest forward/backward
    float* buf_X;     // [B, d] for dense
    int* buf_tok;     // [B] for index
    float* buf_Z;     // [B, P]
    int* buf_F;       // [B, P]
    float* buf_Y;     // [B, P, alpha]
    float* buf_t;     // [B, P]
    float* buf_deltaY;    // [B, P, alpha]
    float* buf_grad_w;    // [P, d] dense
    float* buf_grad_w_index; // [P, vocab]
    float* buf_grad_f;    // [P, alpha]
    float* buf_grad_sp;   // [P]
    float* buf_grad_X;    // [B, d]
    float* buf_row_mask_w; // [P]
    float* buf_row_mask_f; // [P]
    float* buf_r_mean;     // [P]
    float* buf_m_mean;     // [P]

    int last_B;
    int last_mode; // 0 none, 1 dense, 2 index
    int capacity_B;
};

// Utility allocation helpers
static void alloc_or_resize(float** ptr, size_t bytes) {
    if (*ptr) {
        float* tmp = nullptr;
        CUDA_CHECK(cudaMalloc(&tmp, bytes));
        CUDA_CHECK(cudaFree(*ptr));
        *ptr = tmp;
    } else {
        CUDA_CHECK(cudaMalloc(ptr, bytes));
    }
}

static void alloc_or_resize_int(int** ptr, size_t bytes) {
    if (*ptr) {
        int* tmp = nullptr;
        CUDA_CHECK(cudaMalloc(&tmp, bytes));
        CUDA_CHECK(cudaFree(*ptr));
        *ptr = tmp;
    } else {
        CUDA_CHECK(cudaMalloc(ptr, bytes));
    }
}

__device__ __forceinline__ uint32_t hash32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

__global__ void init_params_kernel(float* arr, int n, uint32_t seed, float scale, float shift) {
    GRID_STRIDE_LOOP(i, n) {
        uint32_t h = hash32(static_cast<uint32_t>(i) ^ seed);
        float r = (h & 0xFFFF) / 65535.0f; // [0,1]
        arr[i] = (r * 2.f - 1.f) * scale + shift;
    }
}

__global__ void fill_kernel(float* arr, int n, float value) {
    GRID_STRIDE_LOOP(i, n) {
        arr[i] = value;
    }
}

// forward dense kernel
__global__ void forward_dense_kernel(const float* X, const float* W, const float* Fw,
                                     const float* SP, int B, int P, int d, int alpha,
                                     float* Z, int* F, float* Y, float* t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * P;
    if (idx >= total) return;
    int b = idx / P;
    int p = idx - b * P;
    const float* xrow = X + b * d;
    const float* wrow = W + p * d;
    float acc = 0.f;
    for (int k = 0; k < d; ++k) {
        acc += xrow[k] * wrow[k];
    }
    Z[idx] = acc;
    int fire = acc >= SP[p];
    F[idx] = fire;
    float zfire = fire ? acc : 0.f;
    t[idx] = zfire;
    int yoff = (idx * alpha);
    for (int g = 0; g < alpha; ++g) {
        float fval = Fw[p * alpha + g];
        Y[yoff + g] = zfire * fval;
    }
}

// forward index kernel
__global__ void forward_index_kernel(const int* tok, const float* Widx, const float* Fw,
                                     const float* SP, int B, int P, int vocab, int alpha,
                                     float* Z, int* F, float* Y, float* t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * P;
    if (idx >= total) return;
    int b = idx / P;
    int p = idx - b * P;
    int tok_id = tok[b];
    if (tok_id < 0 || tok_id >= vocab) tok_id = 0; // safety
    const float* wcol = Widx + p * vocab;
    float z = wcol[tok_id];
    Z[idx] = z;
    int fire = z >= SP[p];
    F[idx] = fire;
    float zfire = fire ? z : 0.f;
    t[idx] = zfire;
    int yoff = idx * alpha;
    for (int g = 0; g < alpha; ++g) {
        float fval = Fw[p * alpha + g];
        Y[yoff + g] = zfire * fval;
    }
}

// backward dense kernel
__global__ void backward_dense_kernel(const float* deltaY, const float* delta_t,
                                      const float* X, const float* W,
                                      const float* Fw, const float* Z, const int* F,
                                      int B, int P, int d, int alpha,
                                      float* grad_w, float* grad_f, float* grad_sp,
                                      float* grad_X, int use_ste, float ste_slope) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * P;
    if (idx >= total) return;
    int b = idx / P;
    int p = idx - b * P;
    int fflag = F[idx];
    float z = Z[idx];
    const float* xrow = X + b * d;
    float S = 0.f;
    const float* dy = deltaY ? (deltaY + idx * alpha) : nullptr;
    if (dy) {
        for (int g = 0; g < alpha; ++g) {
            S += dy[g] * Fw[p * alpha + g];
        }
    }
    if (delta_t) S += delta_t[idx];
    float dL_dZ = (float)fflag * S;
    // Grad wrt f
    int foff = p * alpha;
    if (dy) {
        for (int g = 0; g < alpha; ++g) {
            float val = dy[g] * (float)fflag * z;
            atomicAdd(grad_f + foff + g, val);
        }
    }
    // Grad wrt w
    int woff = p * d;
    for (int k = 0; k < d; ++k) {
        atomicAdd(grad_w + woff + k, dL_dZ * xrow[k]);
    }
    // Grad wrt X
    int xoff = b * d;
    for (int k = 0; k < d; ++k) {
        atomicAdd(grad_X + xoff + k, dL_dZ * W[woff + k]);
    }
    // Grad wrt SP via STE
    if (use_ste) {
        float gate_grad = -ste_slope;
        atomicAdd(grad_sp + p, gate_grad * S * z);
    }
}

// backward index kernel
__global__ void backward_index_kernel(const float* deltaY, const float* delta_t,
                                      const int* tok, const float* Widx,
                                      const float* Fw, const float* Z, const int* F,
                                      int B, int P, int vocab, int alpha,
                                      float* grad_widx, float* grad_f, float* grad_sp,
                                      int use_ste, float ste_slope) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * P;
    if (idx >= total) return;
    int b = idx / P;
    int p = idx - b * P;
    int tok_id = tok[b];
    if (tok_id < 0 || tok_id >= vocab) tok_id = 0;
    int fflag = F[idx];
    float z = Z[idx];
    float S = 0.f;
    const float* dy = deltaY ? (deltaY + idx * alpha) : nullptr;
    if (dy) {
        for (int g = 0; g < alpha; ++g) {
            S += dy[g] * Fw[p * alpha + g];
        }
    }
    if (delta_t) S += delta_t[idx];
    float dL_dZ = (float)fflag * S;
    // Grad wrt f
    int foff = p * alpha;
    if (dy) {
        for (int g = 0; g < alpha; ++g) {
            float val = dy[g] * (float)fflag * z;
            atomicAdd(grad_f + foff + g, val);
        }
    }
    // Grad wrt Widx
    atomicAdd(grad_widx + p * vocab + tok_id, dL_dZ);
    if (use_ste) {
        float gate_grad = -ste_slope;
        atomicAdd(grad_sp + p, gate_grad * S * z);
    }
}

// Compute per-row stats (r_mean, m_mean, masks)
__global__ void row_stats_kernel(const int* F, const float* Z, const float* SP,
                                 int B, int P, float z_ref, float eps_z,
                                 int mask_w_mode, int mask_f_mode,
                                 float* r_mean, float* m_mean,
                                 float* mask_w, float* mask_f) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    float rsum = 0.f;
    float msum = 0.f;
    for (int b = 0; b < B; ++b) {
        int f = F[b * P + p];
        rsum += (float)f;
        if (f) {
            float margin = (Z[b * P + p] - SP[p]) / (z_ref + eps_z);
            if (margin > 0.f) msum += margin;
        }
    }
    float rmean = rsum / (float)B;
    float mmean = msum / (float)B;
    r_mean[p] = rmean;
    m_mean[p] = mmean;
    float mw = 1.f;
    float mf = 1.f;
    if (mask_w_mode == 1) mw = (rmean > 0.f) ? 1.f : 0.f;
    else if (mask_w_mode == 2) mw = (rmean == 0.f) ? 1.f : 0.f;
    if (mask_f_mode == 1) mf = (rmean > 0.f) ? 1.f : 0.f;
    else if (mask_f_mode == 2) mf = (rmean == 0.f) ? 1.f : 0.f;
    mask_w[p] = mw;
    mask_f[p] = mf;
}

// Update index weights with BP + Oja and clipping
__global__ void update_w_index_kernel(const float* grad_widx, const int* tok, const float* Z, const int* F,
                                      float* Widx, const float* mask_w,
                                      int B, int P, int vocab,
                                      float eta_bp_w, float eta_w, float w_min, float w_max,
                                      float z_ref, float eps_z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = P * vocab;
    if (idx >= total) return;
    int p = idx / vocab;
    int t = idx - p * vocab;
    float mw = mask_w[p];
    float grad = grad_widx[idx];
    float z_ref2 = z_ref * z_ref + eps_z;
    float oja = 0.f;
    for (int b = 0; b < B; ++b) {
        if (tok[b] != t) continue;
        int f = F[b * P + p];
        if (!f) continue;
        float z = Z[b * P + p];
        float w_old = Widx[idx];
        oja += eta_w * (z - (z * z / z_ref2) * w_old);
    }
    float delta = mw * (-eta_bp_w * grad) + oja;
    float new_w = clipf(Widx[idx] + delta, w_min, w_max);
    Widx[idx] = new_w;
}

// Update f with BP + Oja and clipping
__global__ void update_f_kernel(const float* grad_f, const float* Z, const float* Y, const int* F,
                                float* Fw, const float* mask_f,
                                int B, int P, int alpha,
                                float eta_bp_f, float eta_f, float f_min, float f_max,
                                float y_ref, float eps_y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = P * alpha;
    if (idx >= total) return;
    int p = idx / alpha;
    int g = idx - p * alpha;
    float mf = mask_f[p];
    float y_ref2 = y_ref * y_ref + eps_y;
    float bp = -eta_bp_f * grad_f[idx];
    float oja = 0.f;
    for (int b = 0; b < B; ++b) {
        int f = F[b * P + p];
        if (!f) continue;
        float z = Z[b * P + p];
        float y = Y[(b * P + p) * alpha + g];
        float w_old = Fw[p * alpha + g];
        oja += eta_f * (y * z - (y * y / y_ref2) * w_old);
    }
    float new_f = Fw[p * alpha + g] + mf * bp + oja;
    new_f = clipf(new_f, f_min, f_max);
    Fw[p * alpha + g] = new_f;
}

// Update SP with EMA and homeostatic term
__global__ void update_sp_kernel(float* SP, float* r_bar, float* m_bar,
                                 const float* r_mean, const float* m_mean,
                                 int P, float beta_r, float beta_m,
                                 float eta_sp, float lambda_r, float lambda_m,
                                 float r_target, float m_target,
                                 float sp_min, float sp_max,
                                 const float* grad_sp) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    float r_new = (1.f - beta_r) * r_bar[p] + beta_r * r_mean[p];
    float m_new = (1.f - beta_m) * m_bar[p] + beta_m * m_mean[p];
    r_bar[p] = r_new;
    m_bar[p] = m_new;
    float delta = eta_sp * (lambda_r * (r_new - r_target) + lambda_m * (m_new - m_target));
    delta += grad_sp ? grad_sp[p] : 0.f;
    float sp_val = clipf(SP[p] + delta, sp_min, sp_max);
    SP[p] = sp_val;
}

// Dense weight update kernel (completed with Oja term)
__global__ void update_w_dense_complete(const float* grad_w, const float* X, const float* Z, const int* F,
                                        float* W, const float* mask_w,
                                        int B, int P, int d,
                                        float eta_bp_w, float eta_w, float w_min, float w_max,
                                        float z_ref, float eps_z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = P * d;
    if (idx >= total) return;
    int p = idx / d;
    int k = idx - p * d;
    float mw = mask_w[p];
    float w_old = W[idx];
    float z_ref2 = z_ref * z_ref + eps_z;
    float oja = 0.f;
    for (int b = 0; b < B; ++b) {
        int f = F[b * P + p];
        if (!f) continue;
        float z = Z[b * P + p];
        float x = X[b * d + k];
        oja += eta_w * (z * x - (z * z / z_ref2) * w_old);
    }
    float bp = -eta_bp_w * grad_w[idx];
    float new_w = w_old + mw * bp + oja;
    new_w = clipf(new_w, w_min, w_max);
    W[idx] = new_w;
}

// helper to zero buffers
__global__ void zero_buffer(float* buf, int n) {
    GRID_STRIDE_LOOP(i, n) {
        buf[i] = 0.f;
    }
}

// --- Host side API ---

DLL_EXPORT CeptaPerceptronState* cepta_perceptron_create(int P, int d, int alpha, int vocab) {
    CeptaPerceptronState* st = (CeptaPerceptronState*)malloc(sizeof(CeptaPerceptronState));
    memset(st, 0, sizeof(CeptaPerceptronState));
    st->P = P;
    st->d = d;
    st->alpha = alpha;
    st->vocab = vocab;
    // default hyperparameters
    st->eta_bp_w = 1e-3f;
    st->eta_bp_f = 1e-3f;
    st->eta_w = 1e-3f;
    st->eta_f = 1e-3f;
    st->eta_sp = 1e-3f;
    st->w_min = -1.f;
    st->w_max = 1.f;
    st->f_min = -1.f;
    st->f_max = 1.f;
    st->sp_min = -5.f;
    st->sp_max = 5.f;
    st->z_ref = 1.f;
    st->y_ref = 1.f;
    st->eps_z = 1e-6f;
    st->eps_y = 1e-6f;
    st->beta_r = 0.1f;
    st->beta_m = 0.1f;
    st->r_target = 0.2f;
    st->m_target = 0.2f;
    st->lambda_r = 1.f;
    st->lambda_m = 1.f;
    st->mask_w_mode = 0;
    st->mask_f_mode = 0;
    st->use_ste = 0;
    st->ste_slope = 1.f;

    CUDA_CHECK(cudaMalloc(&st->w_dense, sizeof(float) * P * d));
    CUDA_CHECK(cudaMalloc(&st->w_index, sizeof(float) * P * vocab));
    CUDA_CHECK(cudaMalloc(&st->f_out, sizeof(float) * P * alpha));
    CUDA_CHECK(cudaMalloc(&st->sp, sizeof(float) * P));
    CUDA_CHECK(cudaMalloc(&st->r_bar, sizeof(float) * P));
    CUDA_CHECK(cudaMalloc(&st->m_bar, sizeof(float) * P));
    CUDA_CHECK(cudaMalloc(&st->buf_grad_w, sizeof(float) * P * d));
    CUDA_CHECK(cudaMalloc(&st->buf_grad_w_index, sizeof(float) * P * vocab));
    CUDA_CHECK(cudaMalloc(&st->buf_grad_f, sizeof(float) * P * alpha));
    CUDA_CHECK(cudaMalloc(&st->buf_grad_sp, sizeof(float) * P));
    CUDA_CHECK(cudaMalloc(&st->buf_row_mask_w, sizeof(float) * P));
    CUDA_CHECK(cudaMalloc(&st->buf_row_mask_f, sizeof(float) * P));
    CUDA_CHECK(cudaMalloc(&st->buf_r_mean, sizeof(float) * P));
    CUDA_CHECK(cudaMalloc(&st->buf_m_mean, sizeof(float) * P));

    dim3 block(256);
    dim3 grid_w((P * d + block.x - 1) / block.x);
    dim3 grid_wi((P * vocab + block.x - 1) / block.x);
    dim3 grid_f((P * alpha + block.x - 1) / block.x);
    dim3 grid_p((P + block.x - 1) / block.x);
    init_params_kernel<<<grid_w, block>>>(st->w_dense, P * d, 1234, 0.05f, 0.f);
    init_params_kernel<<<grid_wi, block>>>(st->w_index, P * vocab, 5678, 0.05f, 0.f);
    init_params_kernel<<<grid_f, block>>>(st->f_out, P * alpha, 91011, 0.05f, 0.f);
    init_params_kernel<<<grid_p, block>>>(st->sp, P, 4242, 0.1f, 0.5f);
    fill_kernel<<<grid_p, block>>>(st->r_bar, P, 0.f);
    fill_kernel<<<grid_p, block>>>(st->m_bar, P, 0.f);
    CUDA_CHECK(cudaDeviceSynchronize());
    return st;
}

DLL_EXPORT void cepta_perceptron_set_hyperparams(CeptaPerceptronState* st,
                                                 float eta_bp_w, float eta_bp_f,
                                                 float eta_w, float eta_f, float eta_sp,
                                                 float w_min, float w_max,
                                                 float f_min, float f_max,
                                                 float sp_min, float sp_max,
                                                 float z_ref, float y_ref,
                                                 float eps_z, float eps_y,
                                                 float beta_r, float beta_m,
                                                 float r_target, float m_target,
                                                 float lambda_r, float lambda_m,
                                                 int mask_w_mode, int mask_f_mode,
                                                 int use_ste, float ste_slope) {
    st->eta_bp_w = eta_bp_w;
    st->eta_bp_f = eta_bp_f;
    st->eta_w = eta_w;
    st->eta_f = eta_f;
    st->eta_sp = eta_sp;
    st->w_min = w_min;
    st->w_max = w_max;
    st->f_min = f_min;
    st->f_max = f_max;
    st->sp_min = sp_min;
    st->sp_max = sp_max;
    st->z_ref = z_ref;
    st->y_ref = y_ref;
    st->eps_z = eps_z;
    st->eps_y = eps_y;
    st->beta_r = beta_r;
    st->beta_m = beta_m;
    st->r_target = r_target;
    st->m_target = m_target;
    st->lambda_r = lambda_r;
    st->lambda_m = lambda_m;
    st->mask_w_mode = mask_w_mode;
    st->mask_f_mode = mask_f_mode;
    st->use_ste = use_ste;
    st->ste_slope = ste_slope;
}

static void ensure_capacity(CeptaPerceptronState* st, int B) {
    if (B <= st->capacity_B) return;
    st->capacity_B = B;
    size_t sz_bp = sizeof(float) * B * st->P;
    size_t sz_int = sizeof(int) * B * st->P;
    size_t sz_y = sizeof(float) * B * st->P * st->alpha;
    size_t sz_x = sizeof(float) * B * st->d;
    size_t sz_tok = sizeof(int) * B;
    alloc_or_resize(&st->buf_Z, sz_bp);
    alloc_or_resize_int(&st->buf_F, sz_int);
    alloc_or_resize(&st->buf_Y, sz_y);
    alloc_or_resize(&st->buf_t, sz_bp);
    alloc_or_resize(&st->buf_deltaY, sz_y);
    alloc_or_resize(&st->buf_grad_X, sz_x);
    alloc_or_resize(&st->buf_X, sz_x);
    alloc_or_resize_int(&st->buf_tok, sz_tok);
    // zero grad buffers for new allocations
    CUDA_CHECK(cudaMemset(st->buf_grad_w, 0, sizeof(float) * st->P * st->d));
    CUDA_CHECK(cudaMemset(st->buf_grad_w_index, 0, sizeof(float) * st->P * st->vocab));
    CUDA_CHECK(cudaMemset(st->buf_grad_f, 0, sizeof(float) * st->P * st->alpha));
    CUDA_CHECK(cudaMemset(st->buf_grad_sp, 0, sizeof(float) * st->P));
}

// Forward dense: copies X_host to device, computes, stores buffers, optionally copies outputs back
DLL_EXPORT void cepta_forward_dense(CeptaPerceptronState* st, const float* X_host, int B,
                                    float* Z_out, int* F_out, float* Y_out, float* t_out) {
    ensure_capacity(st, B);
    size_t sz_x = sizeof(float) * B * st->d;
    CUDA_CHECK(cudaMemcpy(st->buf_X, X_host, sz_x, cudaMemcpyHostToDevice));
    dim3 block(256);
    dim3 grid((B * st->P + block.x - 1) / block.x);
    forward_dense_kernel<<<grid, block>>>(st->buf_X, st->w_dense, st->f_out, st->sp,
                                          B, st->P, st->d, st->alpha,
                                          st->buf_Z, st->buf_F, st->buf_Y, st->buf_t);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (Z_out) CUDA_CHECK(cudaMemcpy(Z_out, st->buf_Z, sizeof(float) * B * st->P, cudaMemcpyDeviceToHost));
    if (F_out) CUDA_CHECK(cudaMemcpy(F_out, st->buf_F, sizeof(int) * B * st->P, cudaMemcpyDeviceToHost));
    if (Y_out) CUDA_CHECK(cudaMemcpy(Y_out, st->buf_Y, sizeof(float) * B * st->P * st->alpha, cudaMemcpyDeviceToHost));
    if (t_out) CUDA_CHECK(cudaMemcpy(t_out, st->buf_t, sizeof(float) * B * st->P, cudaMemcpyDeviceToHost));
    st->last_B = B;
    st->last_mode = 1;
}

DLL_EXPORT void cepta_forward_index(CeptaPerceptronState* st, const int* tok_host, int B,
                                    float* Z_out, int* F_out, float* Y_out, float* t_out) {
    ensure_capacity(st, B);
    size_t sz_tok = sizeof(int) * B;
    CUDA_CHECK(cudaMemcpy(st->buf_tok, tok_host, sz_tok, cudaMemcpyHostToDevice));
    dim3 block(256);
    dim3 grid((B * st->P + block.x - 1) / block.x);
    forward_index_kernel<<<grid, block>>>(st->buf_tok, st->w_index, st->f_out, st->sp,
                                          B, st->P, st->vocab, st->alpha,
                                          st->buf_Z, st->buf_F, st->buf_Y, st->buf_t);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (Z_out) CUDA_CHECK(cudaMemcpy(Z_out, st->buf_Z, sizeof(float) * B * st->P, cudaMemcpyDeviceToHost));
    if (F_out) CUDA_CHECK(cudaMemcpy(F_out, st->buf_F, sizeof(int) * B * st->P, cudaMemcpyDeviceToHost));
    if (Y_out) CUDA_CHECK(cudaMemcpy(Y_out, st->buf_Y, sizeof(float) * B * st->P * st->alpha, cudaMemcpyDeviceToHost));
    if (t_out) CUDA_CHECK(cudaMemcpy(t_out, st->buf_t, sizeof(float) * B * st->P, cudaMemcpyDeviceToHost));
    st->last_B = B;
    st->last_mode = 2;
}

DLL_EXPORT void cepta_backward_dense(CeptaPerceptronState* st, const float* deltaY_host, const float* delta_t_host, int B, float* grad_X_out) {
    if (st->last_mode != 1 || B != st->last_B) return;
    size_t sz_y = sizeof(float) * B * st->P * st->alpha;
    if (deltaY_host) CUDA_CHECK(cudaMemcpy(st->buf_deltaY, deltaY_host, sz_y, cudaMemcpyHostToDevice));
    float* delta_t_dev = nullptr;
    if (delta_t_host) {
        CUDA_CHECK(cudaMalloc(&delta_t_dev, sizeof(float) * B * st->P));
        CUDA_CHECK(cudaMemcpy(delta_t_dev, delta_t_host, sizeof(float) * B * st->P, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemset(st->buf_grad_w, 0, sizeof(float) * st->P * st->d));
    CUDA_CHECK(cudaMemset(st->buf_grad_f, 0, sizeof(float) * st->P * st->alpha));
    CUDA_CHECK(cudaMemset(st->buf_grad_sp, 0, sizeof(float) * st->P));
    CUDA_CHECK(cudaMemset(st->buf_grad_X, 0, sizeof(float) * B * st->d));
    dim3 block(256);
    dim3 grid((B * st->P + block.x - 1) / block.x);
    backward_dense_kernel<<<grid, block>>>(deltaY_host ? st->buf_deltaY : nullptr, delta_t_dev,
                                           st->buf_X, st->w_dense, st->f_out,
                                           st->buf_Z, st->buf_F, B, st->P, st->d, st->alpha,
                                           st->buf_grad_w, st->buf_grad_f, st->buf_grad_sp,
                                           st->buf_grad_X, st->use_ste, st->ste_slope);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (grad_X_out) CUDA_CHECK(cudaMemcpy(grad_X_out, st->buf_grad_X, sizeof(float) * B * st->d, cudaMemcpyDeviceToHost));
    if (delta_t_dev) cudaFree(delta_t_dev);
}

DLL_EXPORT void cepta_backward_index(CeptaPerceptronState* st, const float* deltaY_host, const float* delta_t_host, int B) {
    if (st->last_mode != 2 || B != st->last_B) return;
    size_t sz_y = sizeof(float) * B * st->P * st->alpha;
    if (deltaY_host) CUDA_CHECK(cudaMemcpy(st->buf_deltaY, deltaY_host, sz_y, cudaMemcpyHostToDevice));
    float* delta_t_dev = nullptr;
    if (delta_t_host) {
        CUDA_CHECK(cudaMalloc(&delta_t_dev, sizeof(float) * B * st->P));
        CUDA_CHECK(cudaMemcpy(delta_t_dev, delta_t_host, sizeof(float) * B * st->P, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemset(st->buf_grad_w_index, 0, sizeof(float) * st->P * st->vocab));
    CUDA_CHECK(cudaMemset(st->buf_grad_f, 0, sizeof(float) * st->P * st->alpha));
    CUDA_CHECK(cudaMemset(st->buf_grad_sp, 0, sizeof(float) * st->P));
    dim3 block(256);
    dim3 grid((B * st->P + block.x - 1) / block.x);
    backward_index_kernel<<<grid, block>>>(deltaY_host ? st->buf_deltaY : nullptr, delta_t_dev,
                                           st->buf_tok, st->w_index, st->f_out,
                                           st->buf_Z, st->buf_F, B, st->P, st->vocab, st->alpha,
                                           st->buf_grad_w_index, st->buf_grad_f, st->buf_grad_sp,
                                           st->use_ste, st->ste_slope);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (delta_t_dev) cudaFree(delta_t_dev);
}

DLL_EXPORT void cepta_update_params(CeptaPerceptronState* st) {
    int B = st->last_B;
    dim3 block_row(256);
    dim3 grid_row((st->P + block_row.x - 1) / block_row.x);
    // compute stats and masks
    row_stats_kernel<<<grid_row, block_row>>>(st->buf_F, st->buf_Z, st->sp,
                                              B, st->P, st->z_ref, st->eps_z,
                                              st->mask_w_mode, st->mask_f_mode,
                                              st->buf_r_mean, st->buf_m_mean,
                                              st->buf_row_mask_w, st->buf_row_mask_f);
    CUDA_CHECK(cudaDeviceSynchronize());

    dim3 block_w(256);
    dim3 grid_w((st->P * st->d + block_w.x - 1) / block_w.x);
    dim3 grid_wi((st->P * st->vocab + block_w.x - 1) / block_w.x);
    dim3 grid_f((st->P * st->alpha + block_w.x - 1) / block_w.x);
    // Dense path update
    if (st->last_mode == 1) {
        update_w_dense_complete<<<grid_w, block_w>>>(st->buf_grad_w, st->buf_X, st->buf_Z, st->buf_F,
                                                     st->w_dense, st->buf_row_mask_w,
                                                     B, st->P, st->d,
                                                     st->eta_bp_w, st->eta_w, st->w_min, st->w_max,
                                                     st->z_ref, st->eps_z);
    } else if (st->last_mode == 2) {
        // Index update
        update_w_index_kernel<<<grid_wi, block_w>>>(st->buf_grad_w_index, st->buf_tok, st->buf_Z, st->buf_F,
                                                    st->w_index, st->buf_row_mask_w,
                                                    B, st->P, st->vocab,
                                                    st->eta_bp_w, st->eta_w, st->w_min, st->w_max,
                                                    st->z_ref, st->eps_z);
    }
    // f update
    update_f_kernel<<<grid_f, block_w>>>(st->buf_grad_f, st->buf_Z, st->buf_Y, st->buf_F,
                                         st->f_out, st->buf_row_mask_f,
                                         B, st->P, st->alpha,
                                         st->eta_bp_f, st->eta_f, st->f_min, st->f_max,
                                         st->y_ref, st->eps_y);
    // SP update
    update_sp_kernel<<<grid_row, block_row>>>(st->sp, st->r_bar, st->m_bar,
                                              st->buf_r_mean, st->buf_m_mean,
                                              st->P, st->beta_r, st->beta_m,
                                              st->eta_sp, st->lambda_r, st->lambda_m,
                                              st->r_target, st->m_target,
                                              st->sp_min, st->sp_max,
                                              st->buf_grad_sp);
    CUDA_CHECK(cudaDeviceSynchronize());
}

DLL_EXPORT void cepta_perceptron_destroy(CeptaPerceptronState* st) {
    if (!st) return;
    cudaFree(st->w_dense);
    cudaFree(st->w_index);
    cudaFree(st->f_out);
    cudaFree(st->sp);
    cudaFree(st->r_bar);
    cudaFree(st->m_bar);
    cudaFree(st->buf_X);
    cudaFree(st->buf_tok);
    cudaFree(st->buf_Z);
    cudaFree(st->buf_F);
    cudaFree(st->buf_Y);
    cudaFree(st->buf_t);
    cudaFree(st->buf_deltaY);
    cudaFree(st->buf_grad_w);
    cudaFree(st->buf_grad_w_index);
    cudaFree(st->buf_grad_f);
    cudaFree(st->buf_grad_sp);
    cudaFree(st->buf_grad_X);
    cudaFree(st->buf_row_mask_w);
    cudaFree(st->buf_row_mask_f);
    cudaFree(st->buf_r_mean);
    cudaFree(st->buf_m_mean);
    free(st);
}

} // extern "C"
