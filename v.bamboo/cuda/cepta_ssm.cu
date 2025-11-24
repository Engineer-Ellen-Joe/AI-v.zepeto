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

struct CeptaSSMState {
    int P;   // number of paths
    int Pr;  // low rank dimension

    // Parameters
    float* V_r;   // [P, Pr]
    float* V_b;   // [P, Pr]
    float* V_o;   // [Pr, P]
    float* W_l;   // [Pr, Pr]
    float* b_l;   // [Pr]

    // Hyperparameters
    float lr_Vr;
    float lr_Vb;
    float lr_Vo;
    float lr_Wl;
    float lr_bl;
    float a_min;
    float a_max;
    float eps_norm;
    int use_norm;
    int scale_vb;

    // Buffers (device)
    float* t_seq;     // [T, P]
    int* F_seq;       // [T, P]
    float* gate_seq;  // [T, P]
    float* r_seq;     // [T, Pr]
    float* a_seq;     // [T, Pr]
    float* s_seq;     // [T, Pr]
    float* s_scales;  // [T]
    float* ttilde_seq;// [T, P]
    float* delta_seq; // [T, P] upstream grad
    float* grad_t_seq;// [T, P] gradient wrt t input

    float* grad_Vr;   // [P, Pr]
    float* grad_Vb;   // [P, Pr]
    float* grad_Vo;   // [Pr, P]
    float* grad_Wl;   // [Pr, Pr]
    float* grad_bl;   // [Pr]
    float* grad_state0; // [Pr]
    float* grad_s;     // [Pr] running gradient for state
    float* grad_tmp_pr; // [Pr] workspace
    float* grad_pre_a;  // [Pr] workspace
    float* grad_r;      // [Pr] workspace
    float* grad_gate;   // [P] workspace
    float* grad_a;      // [Pr] workspace

    float* state0;    // [Pr] cache of provided initial state
    float* state_tmp; // [Pr] workspace
    float* tmp_scale; // [1] workspace for rms

    int T_cap;
    int last_T;
};

__device__ __forceinline__ uint32_t hash32_s(uint32_t x) {
    x ^= x >> 17;
    x *= 0xed5ad4bbU;
    x ^= x >> 11;
    x *= 0xac4c1b51U;
    x ^= x >> 15;
    x *= 0x31848babU;
    x ^= x >> 14;
    return x;
}

__global__ void init_small_kernel(float* arr, int n, uint32_t seed, float scale) {
    GRID_STRIDE_LOOP(i, n) {
        uint32_t h = hash32_s(seed ^ (uint32_t)i);
        float r = (h & 0xFFFF) / 65535.0f;
        arr[i] = (r * 2.f - 1.f) * scale;
    }
}

__global__ void gate_kernel(const float* t, const int* F, int T, int P, float* gate_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * P;
    if (idx >= total) return;
    gate_out[idx] = (float)F[idx] * t[idx];
}

// Matrix multiply: (T x P) @ (P x R) -> (T x R)
__global__ void matmul_TPR_kernel(const float* A, const float* B, int T, int P, int R, float* C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * R;
    if (idx >= total) return;
    int t = idx / R;
    int r = idx - t * R;
    float acc = 0.f;
    const float* arow = A + t * P;
    for (int p = 0; p < P; ++p) {
        acc += arow[p] * B[p * R + r];
    }
    C[idx] = acc;
}

// Matrix multiply: (T x Pr) @ (Pr x Pr) -> (T x Pr)
__global__ void matmul_square_kernel(const float* A, const float* B, int T, int R, float* C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * R;
    if (idx >= total) return;
    int t = idx / R;
    int r = idx - t * R;
    float acc = 0.f;
    const float* arow = A + t * R;
    for (int k = 0; k < R; ++k) {
        acc += arow[k] * B[k * R + r];
    }
    C[idx] = acc;
}

__global__ void add_bias_sigmoid_clamp_kernel(float* A, const float* b, int T, int R, float a_min, float a_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * R;
    if (idx >= total) return;
    int r = idx % R;
    float v = A[idx] + b[r];
    float s = sigmoidf(v);
    s = clipf(s, a_min, a_max);
    A[idx] = s;
}

// Compute s_t for a single time step
__global__ void state_step_kernel(const float* a_t, const float* gate_t, const float* Vb,
                                  const float* s_prev, float* s_out,
                                  int P, int Pr, int scale_vb, float inv_sqrt_pr) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= Pr) return;
    float contrib = 0.f;
    for (int p = 0; p < P; ++p) {
        contrib += gate_t[p] * Vb[p * Pr + r];
    }
    if (scale_vb) contrib *= inv_sqrt_pr;
    float val = a_t[r] * s_prev[r] + contrib;
    s_out[r] = val;
}

template <unsigned int blockSize>
__global__ void rms_scale_kernel(const float* v, int n, float eps, float* scale_out) {
    float sum = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float t = v[i];
        sum += t * t;
    }
    sum = block_reduce_sum<blockSize>(sum);
    if (threadIdx.x == 0) {
        float mean = sum / (float)n;
        scale_out[0] = rsqrtf(mean + eps);
    }
}

__global__ void apply_scale_kernel(float* v, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) v[idx] *= scale;
}

__global__ void copy_state_kernel(const float* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = src[idx];
}

__global__ void matmul_state_vo_kernel(const float* S, const float* Vo, int T, int Pr, int P, float* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * P;
    if (idx >= total) return;
    int t = idx / P;
    int p = idx - t * P;
    const float* srow = S + t * Pr;
    float acc = 0.f;
    for (int r = 0; r < Pr; ++r) {
        acc += srow[r] * Vo[r * P + p];
    }
    out[idx] = acc;
}

// Backward helpers
__global__ void grad_vo_kernel(const float* s_t, const float* delta_t, int Pr, int P, float* grad_Vo) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Pr * P;
    if (idx >= total) return;
    int r = idx / P;
    int p = idx - r * P;
    atomicAdd(grad_Vo + idx, s_t[r] * delta_t[p]);
}

__global__ void ds_from_vo_kernel(const float* delta_t, const float* Vo, int Pr, int P, float* ds_out) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= Pr) return;
    float acc = 0.f;
    for (int p = 0; p < P; ++p) {
        acc += delta_t[p] * Vo[r * P + p];
    }
    ds_out[r] += acc;
}

__global__ void norm_backward_kernel(const float* s_t, const float* ds, float scale, int Pr, float eps, float* ds_out) {
    __shared__ float dot_shared;
    float dot = 0.f;
    for (int r = threadIdx.x; r < Pr; r += blockDim.x) {
        dot += s_t[r] * ds[r];
    }
    dot = block_reduce_sum<256>(dot);
    if (threadIdx.x == 0) dot_shared = dot;
    __syncthreads();
    float norm_cubed = scale * scale * scale;
    for (int r = threadIdx.x; r < Pr; r += blockDim.x) {
        float val = ds[r] * scale - s_t[r] * dot_shared * norm_cubed / (float)Pr;
        ds_out[r] = val;
    }
}

__global__ void grad_a_kernel(const float* ds_pre, const float* s_prev, const float* a_t,
                              int Pr, float* grad_a, float* ds_prev_out) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= Pr) return;
    grad_a[r] = ds_pre[r] * s_prev[r];
    ds_prev_out[r] += ds_pre[r] * a_t[r];
}

__global__ void grad_vb_kernel(const float* ds_pre, const float* gate_t, const float* Vb,
                               int P, int Pr, int scale_vb, float inv_sqrt_pr,
                               float* grad_Vb, float* grad_gate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = P * Pr;
    if (idx >= total) return;
    int p = idx / Pr;
    int r = idx - p * Pr;
    float scale = scale_vb ? inv_sqrt_pr : 1.f;
    float contrib = gate_t[p] * ds_pre[r] * scale;
    atomicAdd(grad_Vb + idx, contrib);
    atomicAdd(grad_gate + p, ds_pre[r] * Vb[p * Pr + r] * scale);
}

__global__ void grad_pre_a_kernel(const float* grad_a, const float* a_t, const float* r_t,
                                  int Pr, float a_min, float a_max,
                                  float* grad_Wl, float* grad_bl, float* grad_r_out) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= Pr) return;
    float a = a_t[r];
    float mask = (a > a_min && a < a_max) ? 1.f : 0.f;
    float gpa = grad_a[r] * a * (1.f - a) * mask;
    grad_bl[r] += gpa;
    for (int k = 0; k < Pr; ++k) {
        atomicAdd(grad_Wl + k * Pr + r, r_t[k] * gpa);
    }
    grad_r_out[r] = gpa;
}

__global__ void grad_r_finalize_kernel(const float* grad_pre_a, const float* Wl, int Pr, float* grad_r_out) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= Pr) return;
    float acc = 0.f;
    for (int r = 0; r < Pr; ++r) {
        acc += grad_pre_a[r] * Wl[k * Pr + r];
    }
    grad_r_out[k] += acc;
}

__global__ void grad_vr_kernel(const float* grad_r, const float* t_vec, const float* Vr, int P, int Pr, float* grad_Vr, float* grad_t_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = P * Pr;
    if (idx >= total) return;
    int p = idx / Pr;
    int r = idx - p * Pr;
    float tval = t_vec[p];
    atomicAdd(grad_Vr + idx, tval * grad_r[r]);
    atomicAdd(grad_t_out + p, grad_r[r] * Vr[idx]);
}

__global__ void apply_gate_to_grad_t(float* grad_t_out, const float* grad_gate, const int* F_t, int P) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    grad_t_out[p] += grad_gate[p] * (float)F_t[p];
}

__global__ void sgd_update_kernel(float* param, const float* grad, int n, float lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        param[idx] -= lr * grad[idx];
    }
}

// RMSNorm forward/backward
__global__ void rmsnorm_forward_kernel(const float* X, const float* w, float eps, int B, int D,
                                       float* Y, float* rms_cache) {
    int b = blockIdx.x;
    float sum = 0.f;
    for (int k = threadIdx.x; k < D; k += blockDim.x) {
        float v = X[b * D + k];
        sum += v * v;
    }
    sum = block_reduce_sum<256>(sum);
    __shared__ float rms_shared;
    if (threadIdx.x == 0) {
        float rms = rsqrtf(sum / (float)D + eps);
        rms_shared = rms;
        rms_cache[b] = rms;
    }
    __syncthreads();
    float rms = rms_shared;
    for (int k = threadIdx.x; k < D; k += blockDim.x) {
        float scale = w ? w[k] : 1.f;
        Y[b * D + k] = X[b * D + k] * rms * scale;
    }
}

__global__ void rmsnorm_backward_kernel(const float* X, const float* w, const float* delta,
                                        const float* rms_cache, float eps, int B, int D,
                                        float* grad_X, float* grad_w) {
    int b = blockIdx.x;
    float rms = rms_cache[b];
    __shared__ float common_sum;
    float acc = 0.f;
    for (int k = threadIdx.x; k < D; k += blockDim.x) {
        float wv = w ? w[k] : 1.f;
        acc += delta[b * D + k] * X[b * D + k] * wv;
    }
    acc = block_reduce_sum<256>(acc);
    if (threadIdx.x == 0) common_sum = acc;
    __syncthreads();
    float common = -common_sum * rms * rms * rms / (float)D;
    for (int k = threadIdx.x; k < D; k += blockDim.x) {
        float wv = w ? w[k] : 1.f;
        float dx = delta[b * D + k] * rms * wv + common * X[b * D + k];
        grad_X[b * D + k] = dx;
        if (w) atomicAdd(grad_w + k, delta[b * D + k] * X[b * D + k] * rms);
    }
}

// Linear layer forward/backward (row-major W: [out_dim, in_dim])
__global__ void linear_forward_kernel(const float* X, const float* W, const float* b,
                                      int B, int in_dim, int out_dim, float* Y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * out_dim;
    if (idx >= total) return;
    int bidx = idx / out_dim;
    int o = idx - bidx * out_dim;
    float acc = 0.f;
    for (int k = 0; k < in_dim; ++k) {
        acc += X[bidx * in_dim + k] * W[o * in_dim + k];
    }
    if (b) acc += b[o];
    Y[idx] = acc;
}

__global__ void linear_backward_kernel(const float* X, const float* W, const float* delta,
                                       int B, int in_dim, int out_dim,
                                       float* grad_X, float* grad_W, float* grad_b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * out_dim;
    if (idx >= total) return;
    int bidx = idx / out_dim;
    int o = idx - bidx * out_dim;
    float d = delta[idx];
    // grad_X
    for (int k = 0; k < in_dim; ++k) {
        atomicAdd(grad_X + bidx * in_dim + k, d * W[o * in_dim + k]);
        atomicAdd(grad_W + o * in_dim + k, d * X[bidx * in_dim + k]);
    }
    atomicAdd(grad_b + o, d);
}

extern "C" DLL_EXPORT CeptaSSMState* cepta_ssm_create(int P, int Pr) {
    CeptaSSMState* st = (CeptaSSMState*)malloc(sizeof(CeptaSSMState));
    memset(st, 0, sizeof(CeptaSSMState));
    st->P = P;
    st->Pr = Pr;
    st->lr_Vr = st->lr_Vb = st->lr_Vo = st->lr_Wl = st->lr_bl = 1e-3f;
    st->a_min = 0.01f;
    st->a_max = 0.99f;
    st->eps_norm = 1e-6f;
    st->use_norm = 1;
    st->scale_vb = 1;

    CUDA_CHECK(cudaMalloc(&st->V_r, sizeof(float) * P * Pr));
    CUDA_CHECK(cudaMalloc(&st->V_b, sizeof(float) * P * Pr));
    CUDA_CHECK(cudaMalloc(&st->V_o, sizeof(float) * Pr * P));
    CUDA_CHECK(cudaMalloc(&st->W_l, sizeof(float) * Pr * Pr));
    CUDA_CHECK(cudaMalloc(&st->b_l, sizeof(float) * Pr));

    CUDA_CHECK(cudaMalloc(&st->grad_Vr, sizeof(float) * P * Pr));
    CUDA_CHECK(cudaMalloc(&st->grad_Vb, sizeof(float) * P * Pr));
    CUDA_CHECK(cudaMalloc(&st->grad_Vo, sizeof(float) * Pr * P));
    CUDA_CHECK(cudaMalloc(&st->grad_Wl, sizeof(float) * Pr * Pr));
    CUDA_CHECK(cudaMalloc(&st->grad_bl, sizeof(float) * Pr));
    CUDA_CHECK(cudaMalloc(&st->grad_state0, sizeof(float) * Pr));
    CUDA_CHECK(cudaMalloc(&st->grad_s, sizeof(float) * Pr));
    CUDA_CHECK(cudaMalloc(&st->grad_tmp_pr, sizeof(float) * Pr));
    CUDA_CHECK(cudaMalloc(&st->grad_pre_a, sizeof(float) * Pr));
    CUDA_CHECK(cudaMalloc(&st->grad_r, sizeof(float) * Pr));
    CUDA_CHECK(cudaMalloc(&st->grad_gate, sizeof(float) * P));
    CUDA_CHECK(cudaMalloc(&st->grad_a, sizeof(float) * Pr));

    CUDA_CHECK(cudaMalloc(&st->state0, sizeof(float) * Pr));
    CUDA_CHECK(cudaMalloc(&st->state_tmp, sizeof(float) * Pr));
    CUDA_CHECK(cudaMalloc(&st->tmp_scale, sizeof(float)));

    dim3 block(256);
    dim3 grid_v((P * Pr + block.x - 1) / block.x);
    dim3 grid_p((Pr * Pr + block.x - 1) / block.x);
    dim3 grid_b((Pr + block.x - 1) / block.x);
    init_small_kernel<<<grid_v, block>>>(st->V_r, P * Pr, 1111, 0.05f);
    init_small_kernel<<<grid_v, block>>>(st->V_b, P * Pr, 2222, 0.05f);
    init_small_kernel<<<grid_v, block>>>(st->V_o, Pr * P, 3333, 0.05f);
    init_small_kernel<<<grid_p, block>>>(st->W_l, Pr * Pr, 4444, 0.05f);
    init_small_kernel<<<grid_b, block>>>(st->b_l, Pr, 5555, 0.0f);
    CUDA_CHECK(cudaDeviceSynchronize());
    return st;
}

extern "C" DLL_EXPORT void cepta_ssm_set_hyperparams(CeptaSSMState* st,
                                                     float lr_Vr, float lr_Vb, float lr_Vo,
                                                     float lr_Wl, float lr_bl,
                                                     float a_min, float a_max,
                                                     float eps_norm, int use_norm, int scale_vb) {
    st->lr_Vr = lr_Vr;
    st->lr_Vb = lr_Vb;
    st->lr_Vo = lr_Vo;
    st->lr_Wl = lr_Wl;
    st->lr_bl = lr_bl;
    st->a_min = a_min;
    st->a_max = a_max;
    st->eps_norm = eps_norm;
    st->use_norm = use_norm;
    st->scale_vb = scale_vb;
}

static void ensure_capacity_ssm(CeptaSSMState* st, int T) {
    if (T <= st->T_cap) return;
    st->T_cap = T;
    size_t tp = sizeof(float) * T * st->P;
    size_t tpr = sizeof(float) * T * st->Pr;
    size_t ti = sizeof(int) * T * st->P;
    if (st->t_seq) cudaFree(st->t_seq);
    if (st->F_seq) cudaFree(st->F_seq);
    if (st->gate_seq) cudaFree(st->gate_seq);
    if (st->r_seq) cudaFree(st->r_seq);
    if (st->a_seq) cudaFree(st->a_seq);
    if (st->s_seq) cudaFree(st->s_seq);
    if (st->s_scales) cudaFree(st->s_scales);
    if (st->ttilde_seq) cudaFree(st->ttilde_seq);
    if (st->delta_seq) cudaFree(st->delta_seq);
    if (st->grad_t_seq) cudaFree(st->grad_t_seq);
    CUDA_CHECK(cudaMalloc(&st->t_seq, tp));
    CUDA_CHECK(cudaMalloc(&st->F_seq, ti));
    CUDA_CHECK(cudaMalloc(&st->gate_seq, tp));
    CUDA_CHECK(cudaMalloc(&st->r_seq, tpr));
    CUDA_CHECK(cudaMalloc(&st->a_seq, tpr));
    CUDA_CHECK(cudaMalloc(&st->s_seq, tpr));
    CUDA_CHECK(cudaMalloc(&st->s_scales, sizeof(float) * T));
    CUDA_CHECK(cudaMalloc(&st->ttilde_seq, tp));
    CUDA_CHECK(cudaMalloc(&st->delta_seq, tp));
    CUDA_CHECK(cudaMalloc(&st->grad_t_seq, tp));
}

extern "C" DLL_EXPORT void cepta_ssm_forward(CeptaSSMState* st, const float* t_host, const int* F_host,
                                             int T, const float* state0_host,
                                             float* ttilde_out_host, float* state_out_host) {
    ensure_capacity_ssm(st, T);
    size_t tp = sizeof(float) * T * st->P;
    size_t ti = sizeof(int) * T * st->P;
    CUDA_CHECK(cudaMemcpy(st->t_seq, t_host, tp, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(st->F_seq, F_host, ti, cudaMemcpyHostToDevice));
    if (state0_host) {
        CUDA_CHECK(cudaMemcpy(st->state0, state0_host, sizeof(float) * st->Pr, cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemset(st->state0, 0, sizeof(float) * st->Pr));
    }
    // gate = F * t
    dim3 block(256);
    dim3 grid_gate((T * st->P + block.x - 1) / block.x);
    gate_kernel<<<grid_gate, block>>>(st->t_seq, st->F_seq, T, st->P, st->gate_seq);
    // r_seq = t_seq @ V_r
    dim3 grid_r((T * st->Pr + block.x - 1) / block.x);
    matmul_TPR_kernel<<<grid_r, block>>>(st->t_seq, st->V_r, T, st->P, st->Pr, st->r_seq);
    // pre_a = r_seq @ W_l + b -> a_seq
    dim3 grid_a((T * st->Pr + block.x - 1) / block.x);
    matmul_square_kernel<<<grid_a, block>>>(st->r_seq, st->W_l, T, st->Pr, st->a_seq);
    add_bias_sigmoid_clamp_kernel<<<grid_a, block>>>(st->a_seq, st->b_l, T, st->Pr, st->a_min, st->a_max);
    // sequential state update
    float inv_sqrt_pr = rsqrtf((float)st->Pr);
    float* s_prev = st->state0;
    for (int t = 0; t < T; ++t) {
        const float* a_t = st->a_seq + t * st->Pr;
        const float* gate_t = st->gate_seq + t * st->P;
        float* s_t = st->s_seq + t * st->Pr;
        state_step_kernel<<<(st->Pr + 255) / 256, 256>>>(a_t, gate_t, st->V_b, s_prev, st->state_tmp,
                                                         st->P, st->Pr, st->scale_vb, inv_sqrt_pr);
        if (st->use_norm) {
            rms_scale_kernel<256><<<1, 256>>>(st->state_tmp, st->Pr, st->eps_norm, st->tmp_scale);
            float scale_host = 1.f;
            CUDA_CHECK(cudaMemcpy(&scale_host, st->tmp_scale, sizeof(float), cudaMemcpyDeviceToHost));
            apply_scale_kernel<<<(st->Pr + 255) / 256, 256>>>(st->state_tmp, st->Pr, scale_host);
            CUDA_CHECK(cudaMemcpy(st->s_scales + t, &scale_host, sizeof(float), cudaMemcpyHostToDevice));
        } else {
            float scale_host = 1.f;
            CUDA_CHECK(cudaMemcpy(st->s_scales + t, &scale_host, sizeof(float), cudaMemcpyHostToDevice));
        }
        // copy to s_t
        copy_state_kernel<<<(st->Pr + 255) / 256, 256>>>(st->state_tmp, s_t, st->Pr);
        s_prev = s_t;
    }
    // ttilde = s_seq @ V_o
    dim3 grid_o((T * st->P + block.x - 1) / block.x);
    matmul_state_vo_kernel<<<grid_o, block>>>(st->s_seq, st->V_o, T, st->Pr, st->P, st->ttilde_seq);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (ttilde_out_host) CUDA_CHECK(cudaMemcpy(ttilde_out_host, st->ttilde_seq, tp, cudaMemcpyDeviceToHost));
    if (state_out_host && T > 0) CUDA_CHECK(cudaMemcpy(state_out_host, st->s_seq + (T - 1) * st->Pr, sizeof(float) * st->Pr, cudaMemcpyDeviceToHost));
    st->last_T = T;
}

extern "C" DLL_EXPORT void cepta_ssm_backward(CeptaSSMState* st, const float* delta_ttilde_host, int T, float* grad_state0_out, float* grad_t_out_host) {
    if (T != st->last_T) return;
    size_t tp = sizeof(float) * T * st->P;
    CUDA_CHECK(cudaMemcpy(st->delta_seq, delta_ttilde_host, tp, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(st->grad_Vr, 0, sizeof(float) * st->P * st->Pr));
    CUDA_CHECK(cudaMemset(st->grad_Vb, 0, sizeof(float) * st->P * st->Pr));
    CUDA_CHECK(cudaMemset(st->grad_Vo, 0, sizeof(float) * st->Pr * st->P));
    CUDA_CHECK(cudaMemset(st->grad_Wl, 0, sizeof(float) * st->Pr * st->Pr));
    CUDA_CHECK(cudaMemset(st->grad_bl, 0, sizeof(float) * st->Pr));
    CUDA_CHECK(cudaMemset(st->grad_t_seq, 0, tp));
    CUDA_CHECK(cudaMemset(st->grad_s, 0, sizeof(float) * st->Pr));

    float inv_sqrt_pr = rsqrtf((float)st->Pr);

    for (int t = T - 1; t >= 0; --t) {
        const float* s_t = st->s_seq + t * st->Pr;
        const float* a_t = st->a_seq + t * st->Pr;
        const float* r_t = st->r_seq + t * st->Pr;
        const float* gate_t = st->gate_seq + t * st->P;
        const float* t_vec = st->t_seq + t * st->P;
        const int* F_t = st->F_seq + t * st->P;
        const float* delta_t = st->delta_seq + t * st->P;
        float scale_t = 1.f;
        CUDA_CHECK(cudaMemcpy(&scale_t, st->s_scales + t, sizeof(float), cudaMemcpyDeviceToHost));

        // ds accumulator starts from grad_s (next timestep)
        CUDA_CHECK(cudaMemcpy(st->grad_tmp_pr, st->grad_s, sizeof(float) * st->Pr, cudaMemcpyDeviceToDevice));

        // add Vo^T * delta_t
        ds_from_vo_kernel<<<(st->Pr + 255) / 256, 256>>>(delta_t, st->V_o, st->Pr, st->P, st->grad_tmp_pr);
        grad_vo_kernel<<<(st->Pr * st->P + 255) / 256, 256>>>(s_t, delta_t, st->Pr, st->P, st->grad_Vo);

        // normalization backward
        if (st->use_norm) {
            norm_backward_kernel<<<1, 256>>>(s_t, st->grad_tmp_pr, scale_t, st->Pr, st->eps_norm, st->grad_tmp_pr);
        }

        // grad_Vb and grad_gate
        CUDA_CHECK(cudaMemset(st->grad_gate, 0, sizeof(float) * st->P));
        grad_vb_kernel<<<(st->P * st->Pr + 255) / 256, 256>>>(st->grad_tmp_pr, gate_t, st->V_b,
                                                               st->P, st->Pr, st->scale_vb, inv_sqrt_pr,
                                                               st->grad_Vb, st->grad_gate);

        // grad_a and ds_prev
        CUDA_CHECK(cudaMemset(st->grad_a, 0, sizeof(float) * st->Pr));
        CUDA_CHECK(cudaMemset(st->grad_s, 0, sizeof(float) * st->Pr));
        const float* s_prev = (t == 0) ? st->state0 : (st->s_seq + (t - 1) * st->Pr);
        grad_a_kernel<<<(st->Pr + 255) / 256, 256>>>(st->grad_tmp_pr, s_prev, a_t, st->Pr, st->grad_a, st->grad_s);

        // grad_pre_a -> grad_Wl/bl and grad_r
        CUDA_CHECK(cudaMemset(st->grad_pre_a, 0, sizeof(float) * st->Pr));
        CUDA_CHECK(cudaMemset(st->grad_r, 0, sizeof(float) * st->Pr));
        grad_pre_a_kernel<<<(st->Pr + 255) / 256, 256>>>(st->grad_a, a_t, r_t,
                                                         st->Pr, st->a_min, st->a_max,
                                                         st->grad_Wl, st->grad_bl, st->grad_pre_a);
        grad_r_finalize_kernel<<<(st->Pr + 255) / 256, 256>>>(st->grad_pre_a, st->W_l, st->Pr, st->grad_r);

        // grad_Vr and grad_t via r path
        grad_vr_kernel<<<(st->P * st->Pr + 255) / 256, 256>>>(st->grad_r, t_vec, st->V_r, st->P, st->Pr, st->grad_Vr, st->grad_t_seq + t * st->P);
        // add gate contribution to grad_t
        apply_gate_to_grad_t<<<(st->P + 255) / 256, 256>>>(st->grad_t_seq + t * st->P, st->grad_gate, F_t, st->P);

        // prepare grad_s for previous timestep already set in grad_s by grad_a_kernel (ds_prev)
    }
    if (grad_state0_out) CUDA_CHECK(cudaMemcpy(grad_state0_out, st->grad_s, sizeof(float) * st->Pr, cudaMemcpyDeviceToHost));
    if (grad_t_out_host) CUDA_CHECK(cudaMemcpy(grad_t_out_host, st->grad_t_seq, sizeof(float) * T * st->P, cudaMemcpyDeviceToHost));
}

extern "C" DLL_EXPORT void cepta_ssm_update(CeptaSSMState* st) {
    // Simple SGD update on device
    int total_V = st->P * st->Pr;
    int total_Vo = st->Pr * st->P;
    int total_W = st->Pr * st->Pr;
    sgd_update_kernel<<<(total_V + 255) / 256, 256>>>(st->V_r, st->grad_Vr, total_V, st->lr_Vr);
    sgd_update_kernel<<<(total_V + 255) / 256, 256>>>(st->V_b, st->grad_Vb, total_V, st->lr_Vb);
    sgd_update_kernel<<<(total_Vo + 255) / 256, 256>>>(st->V_o, st->grad_Vo, total_Vo, st->lr_Vo);
    sgd_update_kernel<<<(total_W + 255) / 256, 256>>>(st->W_l, st->grad_Wl, total_W, st->lr_Wl);
    sgd_update_kernel<<<(st->Pr + 255) / 256, 256>>>(st->b_l, st->grad_bl, st->Pr, st->lr_bl);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// RMSNorm host wrappers
extern "C" DLL_EXPORT void cepta_rmsnorm_forward(const float* X_host, const float* w_host,
                                                 int B, int D, float eps,
                                                 float* Y_out_host, float* rms_out_host) {
    float* dX = nullptr; float* dw = nullptr; float* dY = nullptr; float* dR = nullptr;
    size_t sz_x = sizeof(float) * B * D;
    size_t sz_w = sizeof(float) * D;
    CUDA_CHECK(cudaMalloc(&dX, sz_x));
    CUDA_CHECK(cudaMemcpy(dX, X_host, sz_x, cudaMemcpyHostToDevice));
    if (w_host) {
        CUDA_CHECK(cudaMalloc(&dw, sz_w));
        CUDA_CHECK(cudaMemcpy(dw, w_host, sz_w, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&dY, sz_x));
    CUDA_CHECK(cudaMalloc(&dR, sizeof(float) * B));
    rmsnorm_forward_kernel<<<B, 256>>>(dX, dw, eps, B, D, dY, dR);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (Y_out_host) CUDA_CHECK(cudaMemcpy(Y_out_host, dY, sz_x, cudaMemcpyDeviceToHost));
    if (rms_out_host) CUDA_CHECK(cudaMemcpy(rms_out_host, dR, sizeof(float) * B, cudaMemcpyDeviceToHost));
    cudaFree(dX); cudaFree(dw); cudaFree(dY); cudaFree(dR);
}

extern "C" DLL_EXPORT void cepta_rmsnorm_backward(const float* X_host, const float* w_host,
                                                  const float* delta_host, const float* rms_host,
                                                  int B, int D, float eps,
                                                  float* grad_X_out_host, float* grad_w_out_host) {
    float* dX = nullptr; float* dw = nullptr; float* dDelta = nullptr; float* dRms = nullptr;
    float* dGradX = nullptr; float* dGradW = nullptr;
    size_t sz_x = sizeof(float) * B * D;
    size_t sz_w = sizeof(float) * D;
    CUDA_CHECK(cudaMalloc(&dX, sz_x));
    CUDA_CHECK(cudaMemcpy(dX, X_host, sz_x, cudaMemcpyHostToDevice));
    if (w_host) {
        CUDA_CHECK(cudaMalloc(&dw, sz_w));
        CUDA_CHECK(cudaMemcpy(dw, w_host, sz_w, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&dDelta, sz_x));
    CUDA_CHECK(cudaMemcpy(dDelta, delta_host, sz_x, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dRms, sizeof(float) * B));
    CUDA_CHECK(cudaMemcpy(dRms, rms_host, sizeof(float) * B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dGradX, sz_x));
    CUDA_CHECK(cudaMemset(dGradX, 0, sz_x));
    CUDA_CHECK(cudaMalloc(&dGradW, sz_w));
    CUDA_CHECK(cudaMemset(dGradW, 0, sz_w));
    rmsnorm_backward_kernel<<<B, 256>>>(dX, dw, dDelta, dRms, eps, B, D, dGradX, dGradW);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (grad_X_out_host) CUDA_CHECK(cudaMemcpy(grad_X_out_host, dGradX, sz_x, cudaMemcpyDeviceToHost));
    if (grad_w_out_host) CUDA_CHECK(cudaMemcpy(grad_w_out_host, dGradW, sz_w, cudaMemcpyDeviceToHost));
    cudaFree(dX); cudaFree(dw); cudaFree(dDelta); cudaFree(dRms); cudaFree(dGradX); cudaFree(dGradW);
}

// Linear host wrappers
extern "C" DLL_EXPORT void cepta_linear_forward(const float* X_host, const float* W_host, const float* b_host,
                                                int B, int in_dim, int out_dim, float* Y_out_host) {
    float* dX = nullptr; float* dW = nullptr; float* dB = nullptr; float* dY = nullptr;
    size_t sz_x = sizeof(float) * B * in_dim;
    size_t sz_w = sizeof(float) * out_dim * in_dim;
    size_t sz_b = sizeof(float) * out_dim;
    CUDA_CHECK(cudaMalloc(&dX, sz_x));
    CUDA_CHECK(cudaMemcpy(dX, X_host, sz_x, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dW, sz_w));
    CUDA_CHECK(cudaMemcpy(dW, W_host, sz_w, cudaMemcpyHostToDevice));
    if (b_host) {
        CUDA_CHECK(cudaMalloc(&dB, sz_b));
        CUDA_CHECK(cudaMemcpy(dB, b_host, sz_b, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&dY, sizeof(float) * B * out_dim));
    dim3 block(256);
    dim3 grid((B * out_dim + block.x - 1) / block.x);
    linear_forward_kernel<<<grid, block>>>(dX, dW, dB, B, in_dim, out_dim, dY);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (Y_out_host) CUDA_CHECK(cudaMemcpy(Y_out_host, dY, sizeof(float) * B * out_dim, cudaMemcpyDeviceToHost));
    cudaFree(dX); cudaFree(dW); cudaFree(dB); cudaFree(dY);
}

extern "C" DLL_EXPORT void cepta_linear_backward(const float* X_host, const float* W_host, const float* delta_host,
                                                 int B, int in_dim, int out_dim,
                                                 float* grad_X_out_host, float* grad_W_out_host, float* grad_b_out_host) {
    float* dX = nullptr; float* dW = nullptr; float* dDelta = nullptr;
    float* dGradX = nullptr; float* dGradW = nullptr; float* dGradB = nullptr;
    size_t sz_x = sizeof(float) * B * in_dim;
    size_t sz_w = sizeof(float) * out_dim * in_dim;
    size_t sz_delta = sizeof(float) * B * out_dim;
    CUDA_CHECK(cudaMalloc(&dX, sz_x));
    CUDA_CHECK(cudaMemcpy(dX, X_host, sz_x, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dW, sz_w));
    CUDA_CHECK(cudaMemcpy(dW, W_host, sz_w, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dDelta, sz_delta));
    CUDA_CHECK(cudaMemcpy(dDelta, delta_host, sz_delta, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dGradX, sz_x));
    CUDA_CHECK(cudaMalloc(&dGradW, sz_w));
    CUDA_CHECK(cudaMalloc(&dGradB, sizeof(float) * out_dim));
    CUDA_CHECK(cudaMemset(dGradX, 0, sz_x));
    CUDA_CHECK(cudaMemset(dGradW, 0, sz_w));
    CUDA_CHECK(cudaMemset(dGradB, 0, sizeof(float) * out_dim));
    dim3 block(256);
    dim3 grid((B * out_dim + block.x - 1) / block.x);
    linear_backward_kernel<<<grid, block>>>(dX, dW, dDelta, B, in_dim, out_dim, dGradX, dGradW, dGradB);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (grad_X_out_host) CUDA_CHECK(cudaMemcpy(grad_X_out_host, dGradX, sz_x, cudaMemcpyDeviceToHost));
    if (grad_W_out_host) CUDA_CHECK(cudaMemcpy(grad_W_out_host, dGradW, sz_w, cudaMemcpyDeviceToHost));
    if (grad_b_out_host) CUDA_CHECK(cudaMemcpy(grad_b_out_host, dGradB, sizeof(float) * out_dim, cudaMemcpyDeviceToHost));
    cudaFree(dX); cudaFree(dW); cudaFree(dDelta); cudaFree(dGradX); cudaFree(dGradW); cudaFree(dGradB);
}

extern "C" DLL_EXPORT void cepta_ssm_destroy(CeptaSSMState* st) {
    if (!st) return;
    cudaFree(st->V_r);
    cudaFree(st->V_b);
    cudaFree(st->V_o);
    cudaFree(st->W_l);
    cudaFree(st->b_l);
    cudaFree(st->t_seq);
    cudaFree(st->F_seq);
    cudaFree(st->gate_seq);
    cudaFree(st->r_seq);
    cudaFree(st->a_seq);
    cudaFree(st->s_seq);
    cudaFree(st->s_scales);
    cudaFree(st->ttilde_seq);
    cudaFree(st->delta_seq);
    cudaFree(st->grad_t_seq);
    cudaFree(st->grad_Vr);
    cudaFree(st->grad_Vb);
    cudaFree(st->grad_Vo);
    cudaFree(st->grad_Wl);
    cudaFree(st->grad_bl);
    cudaFree(st->grad_state0);
    cudaFree(st->grad_s);
    cudaFree(st->grad_tmp_pr);
    cudaFree(st->grad_pre_a);
    cudaFree(st->grad_r);
    cudaFree(st->grad_gate);
    cudaFree(st->grad_a);
    cudaFree(st->state0);
    cudaFree(st->state_tmp);
    cudaFree(st->tmp_scale);
    free(st);
}
