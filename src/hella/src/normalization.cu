/***************************************************************************
 *
 *   Copyright (C) 2025 TBA
 *   Copyright (C) 2025 Fourier Space
 *   Authors: TBA, Andrew Jameson
 *
 ***************************************************************************/

#include "hella/alloc.h"
#include "hella/macros.h"
#include "hella/normalization.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {

  // Compute the per-row means of an image. Launch with 256 threads per threadblock, and a threadblock per row
  template <typename T>
  __global__ void computeRowMeans(const T *data, float *row_means, int width, int stride)
  {
    static constexpr unsigned cols_per_block{256};
    assert(cols_per_block == blockDim.x);

    unsigned idx = blockIdx.x * stride + threadIdx.x;

    __shared__ float row_vals[cols_per_block];
    row_vals[threadIdx.x] = 0;
    __syncthreads();

    while(idx < blockIdx.x * stride + width)
    {
      const float val = static_cast<float>(data[idx]);
      row_vals[threadIdx.x] += val;

      idx += cols_per_block;
    }

    __syncthreads();

    // Sum within each warp
    const int lane_id = threadIdx.x & (warpSize - 1);

    float partial_sum = row_vals[threadIdx.x];
    
    for (int i = warpSize/2; i >= 1; i /= 2)
      partial_sum += __shfl_down_sync(0xffffffff, partial_sum, i, warpSize);

    __syncthreads();

    // Write each warp sum back to shared mem
    const int warp_id = threadIdx.x / warpSize;

    if(lane_id == 0)
    {
      row_vals[warp_id] = partial_sum;
    }

    __syncthreads();

    // Perform the final sum on the first thread
    if (threadIdx.x == 0)
    {
      const int n_warp = cols_per_block / warpSize;
      for (int i = 1; i < n_warp; i++)
        partial_sum += row_vals[i];
      row_means[blockIdx.x] = partial_sum / width;
    }
  }

  // Compute the per-row variance of an image. Launch with 256 threads per threadblock, and a threadblock per row
  template <typename T>
  __global__ void computeRowVariances(const T *data, float *row_vars, float mean, int width, int stride)
  {
    static constexpr unsigned cols_per_block{256};
    assert(cols_per_block == blockDim.x);

    unsigned idx = blockIdx.x * stride + threadIdx.x;

    __shared__ float row_vals[cols_per_block];
    row_vals[threadIdx.x] = 0;
    __syncthreads();

    while(idx < blockIdx.x * stride + width)
    {
      const float val = static_cast<float>(data[idx]) - mean;
      row_vals[threadIdx.x] += val * val;

      idx += cols_per_block;
    }

    __syncthreads();

    // Sum within each warp
    const int lane_id = threadIdx.x & (warpSize - 1);

    float partial_sum = row_vals[threadIdx.x];
    
    for (int i = warpSize/2; i >= 1; i /= 2)
      partial_sum += __shfl_down_sync(0xffffffff, partial_sum, i, warpSize);

    __syncthreads();

    // Write each warp sum back to shared mem
    const int warp_id = threadIdx.x / warpSize;

    if(lane_id == 0)
    {
      row_vals[warp_id] = partial_sum;
    }

    __syncthreads();

    // Perform the final sum on the first thread
    if (threadIdx.x == 0)
    {
      const int n_warp = cols_per_block / warpSize;
      for (int i = 1; i < n_warp; i++)
        partial_sum += row_vals[i];
      row_vars[blockIdx.x] = partial_sum / width;
    }
  }
} // namespace anonymous

template <typename T>
float hella::calculate_stddev(hella::pinfo_t *p, T * data, int width, int height, int stride, float *mean_ret)
{
  static constexpr unsigned cols_per_block{256};
  const int new_width = static_cast<int>(cols_per_block * floor(width / cols_per_block));
  const int nblocks = height;

  const size_t row_out_size = sizeof(float) * height;
  hella::lock_d_scratch(p, row_out_size);
  hella::lock_h_scratch(p, row_out_size);
  computeRowMeans<<<nblocks, cols_per_block>>>(data, reinterpret_cast<float *>(p->d_scratch), new_width, stride);
  checkCuda(cudaMemcpy(p->h_scratch, p->d_scratch, row_out_size, cudaMemcpyDeviceToHost));

  auto row_means = reinterpret_cast<float *>(p->h_scratch);
  float mean = 0;
  for (int i = 0; i < height; i++)
    mean += row_means[i];
  mean /= height;

  computeRowVariances<<<nblocks, cols_per_block>>>(data, reinterpret_cast<float *>(p->d_scratch), mean, new_width, stride);
  checkCuda(cudaMemcpy(p->h_scratch, p->d_scratch, row_out_size, cudaMemcpyDeviceToHost));

  auto row_vars = reinterpret_cast<float *>(p->h_scratch);
  float var = 0;
  for (int i = 0; i < height; i++)
    var += row_vars[i];
  var /= height;

  const float stddev = sqrtf(var);

  hella::unlock_d_scratch(p);
  hella::unlock_h_scratch(p);

  if (mean_ret)
    *mean_ret = mean;
  
  return stddev;
}

template float hella::calculate_stddev<half>(hella::pinfo_t *p, half * data, int width, int height, int stride, float *mean_ret);
template float hella::calculate_stddev<float>(hella::pinfo_t *p, float * data, int width, int height, int stride, float *mean_ret);
template float hella::calculate_stddev<uint8_t>(hella::pinfo_t *p, uint8_t * data, int width, int height, int stride, float *mean_ret);

