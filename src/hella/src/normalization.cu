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

  // kernel to sum array and its squares
  __global__ void sumArray(half * data, float * sums, float * qsums, int width, int height, int stride)
  {
    __shared__ float sdata[512], qdata[512];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*512 + tid;
    int x = i % width;
    int y = i / width;
    int iidx = y*stride+x;

    sdata[tid] = __half2float(data[iidx]);
    qdata[tid] = __half2float(data[iidx]*data[iidx]);

    __syncthreads();

    if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
    if (tid < 32) warpReduce(sdata, tid);
    if (tid < 256) { qdata[tid] += qdata[tid + 256]; } __syncthreads();
    if (tid < 128) { qdata[tid] += qdata[tid + 128]; } __syncthreads();
    if (tid < 64) { qdata[tid] += qdata[tid + 64]; } __syncthreads();
    if (tid < 32) warpReduce(qdata, tid);

    if (tid == 0) sums[blockIdx.x] = sdata[0];
    if (tid == 0) qsums[blockIdx.x] = qdata[0];
  }

  // kernel to sum array and its squares
  __global__ void sumArrayFloat(float * data, float * sums, float * qsums, int width, int height, int stride)
  {
    __shared__ float sfdata[512], qfdata[512];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*512 + tid;
    int x = i % width;
    int y = i / width;
    int iidx = y*stride+x;

    sfdata[tid] = data[iidx];
    qfdata[tid] = data[iidx]*data[iidx];

    __syncthreads();

    if (tid < 256) { sfdata[tid] += sfdata[tid + 256]; } __syncthreads();
    if (tid < 128) { sfdata[tid] += sfdata[tid + 128]; } __syncthreads();
    if (tid < 64) { sfdata[tid] += sfdata[tid + 64]; } __syncthreads();
    if (tid < 32) warpReduce(sfdata, tid);
    if (tid < 256) { qfdata[tid] += qfdata[tid + 256]; } __syncthreads();
    if (tid < 128) { qfdata[tid] += qfdata[tid + 128]; } __syncthreads();
    if (tid < 64) { qfdata[tid] += qfdata[tid + 64]; } __syncthreads();
    if (tid < 32) warpReduce(qfdata, tid);

    if (tid == 0) sums[blockIdx.x] = sfdata[0];
    if (tid == 0) qsums[blockIdx.x] = qfdata[0];
  }

} // namespace anonymous

float hella::calculate_stddev(pinfo_t *p, half * data, int width, int height, int stride)
{
  int new_width = (int)(512*floor(width/512.));
  int nblocks = new_width*height / 512;

  const size_t sums_size = sizeof(float) * nblocks;
  hella::lock_d_scratch(p, sums_size * 2);
  auto d_sums = reinterpret_cast<float*>(p->d_scratch);
  auto d_qsums = d_sums + nblocks;

  sumArray<<<nblocks,512>>>(data,d_sums,d_qsums,new_width,height,stride);

  hella::lock_h_scratch(p, sums_size * 2);
  auto sums = reinterpret_cast<float*>(p->h_scratch);
  auto qsums = sums + nblocks;

  checkCuda(cudaMemcpy(sums,d_sums, sums_size,cudaMemcpyDeviceToHost));
  checkCuda(cudaMemcpy(qsums,d_qsums, sums_size,cudaMemcpyDeviceToHost));

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
  int nblocks = new_width*height / 512;

  const size_t sums_size = sizeof(float) * nblocks;
  hella::lock_d_scratch(p, sums_size * 2);
  auto d_sums = reinterpret_cast<float*>(p->d_scratch);
  auto d_qsums = d_sums + nblocks;
  sumArrayFloat<<<nblocks,512>>>(data,d_sums,d_qsums,new_width,height,stride);

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
