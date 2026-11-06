// ==============================================================================
// TIG c006 NeuralNet Optimizer: hyper_optim_v1 CUDA Kernels
// Architecture: NVIDIA Ada Lovelace (sm_89), Ampere (sm_86), Hopper (sm_90)
// Features: Shared-Memory Cached Fused Tensor Dispatch, Register-Resident Adan,
//           Vectorized Striding, Lookahead Proposal, DC Bias Steering.
// ==============================================================================

#include <cuda_runtime.h>

// First-layer kink initialization: spreads initial ReLU transition points evenly
// across [-spread, spread] to ensure rich functional representation from step 0.
extern "C" __global__ void sk_kink(
    float* __restrict__ delta,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float spread,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float t = spread * (-1.0f + 2.0f * ((float)i + 0.5f) / (float)n);
    delta[i] = -w[i] * t - b[i];
}

// Vector scaling kernel: delta = factor * src
extern "C" __global__ void sk_scale(
    float* __restrict__ delta,
    const float* __restrict__ src,
    float factor,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    delta[i] = factor * src[i];
}

// In-place vector offset: delta += g * v
extern "C" __global__ void sk_offset_vec(
    float* __restrict__ delta,
    const float* __restrict__ v,
    float g,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    delta[i] += g * v[i];
}

// Lookahead proposal kernel: evaluates gradients at projected weights theta + alpha * m
extern "C" __global__ void sk_lookahead_proposal(
    float* const* __restrict__ proposed,
    const float* const* __restrict__ param,
    const float* const* __restrict__ m,
    const int* __restrict__ prefix,
    float alpha,
    int n_tensors,
    int total
) {
    __shared__ int s_prefix[64];
    if (threadIdx.x < n_tensors + 1) {
        s_prefix[threadIdx.x] = prefix[threadIdx.x];
    }
    __syncthreads();

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    int lo = 0, hi = n_tensors - 1, t = 0;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (s_prefix[mid] <= gid) {
            t = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    int i = gid - s_prefix[t];
    proposed[t][i] = param[t][i] + alpha * m[t][i];
}

// Ultra-fast fused tensor update kernel:
// 1. Shared-memory cached tensor metadata (prefix, lrs, wds)
// 2. Fused first/second moment Adan update + cautious descent
// 3. Decoupled weight decay and delta generation in registers
extern "C" __global__ void sk_hyper_adan_fused(
    const float* const* __restrict__ grad,
    const float* const* __restrict__ param,
    float* const* __restrict__ m,
    float* const* __restrict__ v,
    float* const* __restrict__ nsq,
    float* const* __restrict__ gprev,
    float* const* __restrict__ delta,
    const int*   __restrict__ prefix,
    const float* __restrict__ lrs,
    const float* __restrict__ wds,
    int   n_tensors,
    float b1, float b2, float b3, float eps,
    float bc1, float bc2, float bc3,
    int   first_step,
    float cautious,
    float ab,
    int   wd_gate,
    int   total
) {
    __shared__ int   s_prefix[64];
    __shared__ float s_lrs[64];
    __shared__ float s_wds[64];

    if (threadIdx.x < n_tensors + 1) {
        s_prefix[threadIdx.x] = prefix[threadIdx.x];
    }
    if (threadIdx.x < n_tensors) {
        s_lrs[threadIdx.x] = lrs[threadIdx.x];
        s_wds[threadIdx.x] = wds[threadIdx.x];
    }
    __syncthreads();

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    // Fast search in shared memory
    int lo = 0, hi = n_tensors - 1, t = 0;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (s_prefix[mid] <= gid) {
            t = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    int i = gid - s_prefix[t];

    float lr = s_lrs[t];
    float wd = s_wds[t];
    float g  = grad[t][i];
    float p  = param[t][i];
    float gp = gprev[t][i];
    float gd = (first_step != 0) ? 0.0f : (g - gp);
    gprev[t][i] = g;

    // Adan momentum & variance updates
    float mi = (1.0f - b1) * g + b1 * m[t][i];
    float vi = (1.0f - b2) * gd + b2 * v[t][i];
    float comb = g + b2 * gd;
    float dev = comb - mi;
    float sq = ab * comb * comb + (1.0f - ab) * (dev * dev + 1e-16f);
    float ni = (1.0f - b3) * sq + b3 * nsq[t][i];

    m[t][i] = mi;
    v[t][i] = vi;
    nsq[t][i] = ni;

    // Numerically stable normalized step
    float step = (mi / bc1 + b2 * vi / bc2) / (sqrtf(ni / bc3) + eps);
    float upd = -lr * step;

    // Cautious update: dampen steps that point against instantaneous gradient
    if (cautious < 1.0f && upd * g > 0.0f) {
        upd *= cautious;
    }

    // Decoupled weight decay
    float np;
    if (wd_gate != 0) {
        float d = (upd * p <= 0.0f) ? (wd * p) : 0.0f;
        np = p + upd - lr * d;
    } else {
        np = (p + upd) / (1.0f + lr * wd);
    }

    delta[t][i] = np - p;
}
