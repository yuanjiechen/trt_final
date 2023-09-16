#include "reorder_rsmlayernorm.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

template <typename T>
__global__ void layer_norm_kernel_fp16(const T *input,T *output,const T *gamma,const  T *scale,const  T *zero_point, const long *dst_index, half* out_quant,const float eps, int b, int c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    constexpr auto num_elems_T = num_elems<T>::value;
    using int8_packed_t = typename packed_as<int8_t, num_elems_T>::type;
    if (idx < b && idy < c) {
        // Calculate mean using parallel reduction
        __shared__ half shared_mean[32][33];
        half thread_sum = __float2half(0.0f);
        for (int i = threadIdx.y; i < c; i += blockDim.y) {
            int linear_idx = idx * c + i;
            thread_sum = __hadd(thread_sum, (half)input[linear_idx]);
        }
        shared_mean[threadIdx.x][threadIdx.y] = thread_sum;
        __syncthreads();


        for (int stride = blockDim.y / 2; stride > 0; stride >>= 1) {
            if (threadIdx.y < stride) {
                shared_mean[threadIdx.x][threadIdx.y] = __hadd(shared_mean[threadIdx.x][threadIdx.y], shared_mean[threadIdx.x][threadIdx.y + stride]);
            }
            __syncthreads();
        }

        if (threadIdx.y == 0) {
            shared_mean[threadIdx.x][0] = __hdiv(shared_mean[threadIdx.x][0], __int2half_rn(c));
        }
        __syncthreads();

        half mean = shared_mean[threadIdx.x][0];

        // Calculate variance using parallel reduction
        __shared__ half shared_var[32][33];
        half thread_var_sum = __float2half(0.0f);
        for (int i = threadIdx.y; i < c; i += blockDim.y) {
            int linear_idx = idx * c + i;
            half diff = __hsub(input[linear_idx], mean);
            thread_var_sum = __hadd(thread_var_sum, __hmul(diff, diff));
        }
        shared_var[threadIdx.x][threadIdx.y] = thread_var_sum;
        __syncthreads();

        for (int stride = blockDim.y / 2; stride > 0; stride >>= 1) {
            if (threadIdx.y < stride) {
                shared_var[threadIdx.x][threadIdx.y] = __hadd(shared_var[threadIdx.x][threadIdx.y], shared_var[threadIdx.x][threadIdx.y + stride]);
            }
            __syncthreads();
        }

        if (threadIdx.y == 0) {
            shared_var[threadIdx.x][0] = __hdiv(shared_var[threadIdx.x][0], __int2half_rn(c));
        }
        __syncthreads();

        half var = shared_var[threadIdx.x][0];

        const bool with_per_token_scaling = scale != nullptr;
        const bool with_zero_point        = zero_point != nullptr;

        // Normalize input
        // int linear_idx = idx * c + idy;
        // half normalized_value = __hdiv(__hsub(input[linear_idx], mean), hsqrt(__hadd(var, __float2half(1e-5f))));
        // int dst_linear_idy = idx * c + dst_index[idy];
        // output[dst_linear_idy] = __hfma(normalized_value, gamma[idy], scale[idy]);
        // RSM Normalize input
        int linear_idx = idx * c + idy;
        half normalized_value = __hdiv(input[linear_idx], hsqrt(__hadd(var, __float2half(eps))));
        int dst_linear_idy = idx * c + dst_index[idy];
        // output INT8
        // reinterpret_cast<int8_packed_t*>(normed_output_quant)[index]
        //         = cuda_cast<int8_packed_t>(cuda_cast<float_packed_t>(val) * scale_orig_quant);
        if(with_per_token_scaling){
            out_quant[dst_linear_idy] 
                = __hmul(normalized_value, (half)gamma[idy]);//__hdiv(__hmul(normalized_value, (half)gamma[idy]), (half)scale[idy]);
            //out_quant[idy] = __hdiv(output[idy], (half)scale[idy]);
            // out_quant[dst_linear_idy] 
            //     = cuda_cast<int8_packed_t>(__hdiv(__hmul(normalized_value, (half)gamma[idy]), (half)scale[idy]));
        }
        else{
            output[dst_linear_idy] = __hmul(normalized_value, gamma[idy]);
        }

    }
}

template <typename T>
void reorder_rsm_norm_fp16(const T* input, T* output,const T *gamma,const T* scale,const T *zero_point, const long *dst_index, half* out_quant ,const float eps,int b ,int c) {
    // int b = input.size(0);
    // int c = input.size(1);

    dim3 blockDim(32, 32);
    dim3 gridDim((b + blockDim.x - 1) / blockDim.x, (c + blockDim.y - 1) / blockDim.y);

    layer_norm_kernel_fp16<T><<<gridDim, blockDim>>>(
        input,
        output,
        gamma,
        scale,
        zero_point,
        dst_index,
        out_quant,
        eps,
        b, c
        );
    cudaDeviceSynchronize();
}

#define INSTANTIATE_GENERAL_RSMNORM(T)                                                                                                        \
    template void reorder_rsm_norm_fp16(const T* input,T* output,const T *gamma,const T* scale ,const T *zero_point , const long *dst_index,  \
    half* out_quant,const float eps, int b, int c)

//INSTANTIATE_GENERAL_RSMNORM(float);
INSTANTIATE_GENERAL_RSMNORM(half);

#ifdef ENABLE_BF16
//INSTANTIATE_GENERAL_RSMNORM(__nv_bfloat16);
#endif

} // namespace kernels
} // namespace tensorrt_llm