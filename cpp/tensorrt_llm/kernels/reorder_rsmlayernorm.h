#pragma once

#include "tensorrt_llm/common/cudaUtils.h"
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
/*
Utilize shared memory for input data:
Loading input data into shared memory can help reduce global memory accesses, as shared memory has much lower latency than global memory.

Use 2D grid and 2D block:
The current blockDim and gridDim configuration might not efficiently utilize the GPU hardware. Consider using a 2D grid and 2D block for better occupancy.

Minimize shared memory bank conflicts:
To avoid shared memory bank conflicts, pad shared memory to ensure that consecutive threads access different banks.

NOTE: optmize not ok
*/


namespace tensorrt_llm
{
namespace kernels
{

template <typename T>
void reorder_rsm_norm_fp16(const T* input,T* output,const T *gamma, const int32_t *dst_index,  \
     float eps, int b, int c);


} // namespace kernels
} // namespace tensorrt_llm
