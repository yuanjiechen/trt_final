#include "reorder_rsmlayernorm.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

template <typename Tf, typename T>
__inline__ __device__ Tf compute_layernorm(Tf val, float s_mean, float s_variance, const T* gamma, int i)
{
    Tf ret = (val) * s_variance * cuda_cast<Tf>(gamma[i]);

    return ret;
}


template <typename T, bool USE_DIFF_OF_SQUARES = false>
__global__ void layer_norm_kernel_fp16(const T* input, const T* gamma, T* normed_output, const float eps,const int32_t *dst_index,
    int tokens, int hidden_dim, bool use_shmem)
{
    constexpr auto num_elems_T = num_elems<T>::value;
    using int8_packed_t = typename packed_as<int8_t, num_elems_T>::type;
    using float_packed_t = typename packed_as<float, num_elems_T>::type;
    using T_scalar = typename packed_as<T, 1>::type;

    extern __shared__ __align__(sizeof(float)) char _shmem[];
    T* shmem = reinterpret_cast<T*>(_shmem);
    __shared__ float s_mean;
    __shared__ float s_variance;

    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;

    float mean = 0.0f;
    float variance = 0.0f;
    float local_sum = 0.0f;
    float local_var_sum = 0.0f;

    const int n_elems = hidden_dim / num_elems_T;
    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        const T val = input[bidx * n_elems + i];
        if (use_shmem)
        {
            shmem[i] = val;
        }

        const float_packed_t val_f = cuda_cast<float_packed_t>(val);
        local_sum += cuda_sum<float>(val_f);
        if (USE_DIFF_OF_SQUARES)
        {
            local_var_sum += cuda_sum<float>(val_f * val_f);
        }
    }

    if (USE_DIFF_OF_SQUARES)
    {
        float packed[2] = {local_sum, local_var_sum};
        blockReduceSumV2<float, 2>(packed);
        mean = packed[0];
        variance = packed[1];
    }
    else
    {
        mean = blockReduceSum(local_sum);
    }

    if (threadIdx.x == 0)
    {
        mean = mean / hidden_dim;
        s_mean = mean;
        if (USE_DIFF_OF_SQUARES)
        {
            variance = (variance / hidden_dim) - (mean * mean); // Var[x] = E[x²] - E[x]²
            s_variance = rsqrtf(variance + eps);
        }
    }
    __syncthreads();

    if (!USE_DIFF_OF_SQUARES)
    {
        for (int i = tidx; i < n_elems; i += blockDim.x)
        {
            const T val = use_shmem ? shmem[i] : input[bidx * n_elems + i];
            float_packed_t diff = cuda_cast<float_packed_t>(val) - s_mean;
            local_var_sum += cuda_sum<float>(diff * diff);
        }
        variance = blockReduceSum(local_var_sum);

        if (threadIdx.x == 0)
        {
            s_variance = rsqrtf(variance / hidden_dim + eps);
        }
        __syncthreads();
    }

    T_scalar amax = 1e-6f;

    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        const int index = bidx * n_elems + i;
        const float_packed_t val_f = cuda_cast<float_packed_t>(use_shmem ? shmem[i] : input[index]);
        const T val = cuda_cast<T>(compute_layernorm(val_f, s_mean, s_variance, gamma, i));
        normed_output[index] = val; 
        /*
        reorder rms norm with rptq
        */
        //normed_output[dst_index[i] + bidx * n_elems] = val;  
    }

}


template <typename T>
void reorder_rsm_norm_fp16(const T* input,T* output,const T *gamma, const int32_t *dst_index,  
      float eps, int b, int c) {
    // int b = input.size(0);
    // int c = input.size(1);

    dim3 gridDim(b);
    dim3 blockDim(min(c, 1024));
    blockDim.x = 32 * ((blockDim.x + 31) / 32);

    layer_norm_kernel_fp16<T><<<gridDim, blockDim>>>(
        input,
        gamma,
        output,
        eps,
        dst_index,
        b, c, false
        );
    cudaDeviceSynchronize();
}

#define INSTANTIATE_GENERAL_RSMNORM(T)                                                                         \
    template void reorder_rsm_norm_fp16(const T* input,T* output,const T *gamma, const int32_t *dst_index,  \
     float eps, int b, int c)

INSTANTIATE_GENERAL_RSMNORM(float);
INSTANTIATE_GENERAL_RSMNORM(half);

#ifdef ENABLE_BF16
// INSTANTIATE_GENERAL_RSMNORM(__nv_bfloat16);
#endif

} // namespace kernels
} // namespace tensorrt_llm