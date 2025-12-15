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

  // kernel to warp reduce float
  __device__ void warpReduce(volatile float *sdata, unsigned int tid)
  {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
  }

  // each block will compute the sums and qsums for a row
  __global__ void sumArray(const half * data, float * sums, float * qsums, int width, int stride)
  {
    unsigned idx = blockIdx.x * stride + threadIdx.x;
    __shared__ float2 sdata[512];

    // each thread computes a stride sum and qsum
    __half2 s = make_half2(0.0f, 0.0f);
    for (unsigned i=threadIdx.x; i<width; i+=blockDim.x, idx+=blockDim.x)
    {
      const half val = data[idx];
      s.x += val;
      s.y += val * val;
    }

    // assign the stride sum to shared memory, converting to a float
    sdata[threadIdx.x] = __half22float2(s);
    __syncthreads();

    if (threadIdx.x < 256) {
      sdata[threadIdx.x].x += sdata[threadIdx.x + 256].x;
      sdata[threadIdx.x].y += sdata[threadIdx.x + 256].y;
    }
    __syncthreads();

    if (threadIdx.x < 128) {
      sdata[threadIdx.x].x += sdata[threadIdx.x + 128].x;
      sdata[threadIdx.x].y += sdata[threadIdx.x + 128].y;
    }
    __syncthreads();

    if (threadIdx.x < 64) {
      sdata[threadIdx.x].x += sdata[threadIdx.x + 64].x;
      sdata[threadIdx.x].y += sdata[threadIdx.x + 64].y;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
      sdata[threadIdx.x].x += sdata[threadIdx.x + 32].x;
      sdata[threadIdx.x].y += sdata[threadIdx.x + 32].y;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
      // Ensure all threads in the warp are active for __shfl_down_sync
      unsigned int mask = __activemask();
      for (int offset = 16; offset > 0; offset >>= 1) {
        sdata[threadIdx.x].x += __shfl_down_sync(mask, sdata[threadIdx.x].x, offset);
        sdata[threadIdx.x].y += __shfl_down_sync(mask, sdata[threadIdx.x].y, offset);
      }
    }

    if (threadIdx.x == 0)
    {
      sums[blockIdx.x] = sdata[threadIdx.x].x;
      qsums[blockIdx.x] = sdata[threadIdx.x].y;
    }
  }

  // kernel to sum array and its squares
  __global__ void sumArrayFloat(float * data, float * sums, float * qsums, int width, int stride)
  {
    unsigned idx = blockIdx.x * stride + threadIdx.x;
    __shared__ float2 sdata[512];

    // each thread computes a stride sum and qsum
    float2 s = make_float2(0.0f, 0.0f);
    for (unsigned i=threadIdx.x; i<width; i+=blockDim.x, idx+=blockDim.x)
    {
      const float val = data[idx];
      s.x += val;
      s.y += val * val;
    }

    // assign the stride sum to shared memory, converting to a float
    sdata[threadIdx.x] = s;
    __syncthreads();

    if (threadIdx.x < 256) {
      sdata[threadIdx.x].x += sdata[threadIdx.x + 256].x;
      sdata[threadIdx.x].y += sdata[threadIdx.x + 256].y;
    }
    __syncthreads();

    if (threadIdx.x < 128) {
      sdata[threadIdx.x].x += sdata[threadIdx.x + 128].x;
      sdata[threadIdx.x].y += sdata[threadIdx.x + 128].y;
    }
    __syncthreads();

    if (threadIdx.x < 64) {
      sdata[threadIdx.x].x += sdata[threadIdx.x + 64].x;
      sdata[threadIdx.x].y += sdata[threadIdx.x + 64].y;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
      sdata[threadIdx.x].x += sdata[threadIdx.x + 32].x;
      sdata[threadIdx.x].y += sdata[threadIdx.x + 32].y;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
      // Ensure all threads in the warp are active for __shfl_down_sync
      unsigned int mask = __activemask();
      for (int offset = 16; offset > 0; offset >>= 1) {
        sdata[threadIdx.x].x += __shfl_down_sync(mask, sdata[threadIdx.x].x, offset);
        sdata[threadIdx.x].y += __shfl_down_sync(mask, sdata[threadIdx.x].y, offset);
      }
    }

    if (threadIdx.x == 0)
    {
      sums[blockIdx.x] = sdata[threadIdx.x].x;
      qsums[blockIdx.x] = sdata[threadIdx.x].y;
    }
  }

} // namespace anonymous

float hella::calculate_stddev(pinfo_t *p, half * data, int width, int height, int stride)
{
  int new_width = (int)(512*floor(width/512.));
  int nblocks = height;

  const size_t sums_size = sizeof(float) * height;
  hella::lock_d_scratch(p, sums_size * 2);
  auto d_sums = reinterpret_cast<float*>(p->d_scratch);
  auto d_qsums = d_sums + nblocks;

  sumArray<<<nblocks,512>>>(data,d_sums,d_qsums,new_width,stride);

  hella::lock_h_scratch(p, sums_size * 2);
  auto sums = reinterpret_cast<float*>(p->h_scratch);
  auto qsums = sums + nblocks;

  checkCuda(cudaMemcpy(sums, d_sums, sums_size, cudaMemcpyDeviceToHost));
  checkCuda(cudaMemcpy(qsums, d_qsums, sums_size, cudaMemcpyDeviceToHost));

  float sum=0., qsum=0.;
  for (int i=0;i<nblocks;i++) {
    sum += sums[i];
    qsum += qsums[i];
  }
  float mn = sum/(new_width*height*1.);

  float stdDev = qsum-2.*sum*mn+mn*mn*new_width*height*1.;
  stdDev /= 1.*new_width*height;
  stdDev = sqrt(stdDev);

  hella::unlock_d_scratch(p);
  hella::unlock_h_scratch(p);

  return stdDev;
}

float hella::calculate_stddev_float(pinfo_t *p, float * data, int width, int height, int stride)
{
  int new_width = static_cast<int>(512*floor(width/512.));
  int nblocks = height;

  const size_t sums_size = sizeof(float) * height;
  hella::lock_d_scratch(p, sums_size * 2);
  auto d_sums = reinterpret_cast<float*>(p->d_scratch);
  auto d_qsums = d_sums + nblocks;
  sumArrayFloat<<<nblocks,512>>>(data,d_sums,d_qsums,new_width,stride);

  hella::lock_h_scratch(p, sums_size * 2);
  auto sums = reinterpret_cast<float*>(p->h_scratch);
  auto qsums = sums + nblocks;
  checkCuda(cudaMemcpy(sums,d_sums,sums_size,cudaMemcpyDeviceToHost));
  checkCuda(cudaMemcpy(qsums,d_qsums,sums_size,cudaMemcpyDeviceToHost));

  float sum=0., qsum=0.;
  for (int i=0;i<nblocks;i++) {
    sum += sums[i];
    qsum += qsums[i];
  }
  float mn = sum/(new_width*height*1.);

  float stdDev = qsum-2.*sum*mn+mn*mn*new_width*height*1.;
  stdDev /= 1.*new_width*height;
  stdDev = sqrt(stdDev);

  hella::unlock_d_scratch(p);
  hella::unlock_h_scratch(p);

  return stdDev;
}
