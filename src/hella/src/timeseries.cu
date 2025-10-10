/***************************************************************************
 *
 *   Copyright (C) 2025 TBA
 *   Copyright (C) 2025 Fourier Space
 *   Authors: TBA, Andrew Jameson
 *
 ***************************************************************************/

#include "hella/definitions.h"
#include "hella/macros.h"
#include "hella/median_filter.h"
#include "hella/timeseries.h"

#include <cuda_runtime.h>

namespace
{
  // cuda kernel to measure per-beam time baseline
  // run with NBATCH*width lots of 32 threads
  __global__ void measure_ts(half * data, float * ts, int width, int stride)
  {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    // int id = bid*32+tid;
    int iTime = (int)(bid % width);
    int iBatch = (int)(bid / width);

    int npartials = (int)(NCHAN / 32); // number partial sums
    half fac = (half)((32.*npartials));
    __shared__ half cpsum[32];

    // calculate partial sums

    cpsum[tid] = 0.;
    int idx0 = iBatch*NCHAN*stride + tid*npartials*stride + iTime;
    for (int i=idx0;i<idx0+npartials*stride;i+=stride)
      cpsum[tid] += data[i];
    cpsum[tid] /= fac;

    __syncthreads();

    // sum over shared memory
    if (tid < 16) { cpsum[tid] += cpsum[tid + 16]; } __syncthreads();
    if (tid < 8) { cpsum[tid] += cpsum[tid + 8]; } __syncthreads();
    if (tid < 4) { cpsum[tid] += cpsum[tid + 4]; } __syncthreads();
    if (tid < 2) { cpsum[tid] += cpsum[tid + 2]; } __syncthreads();
    if (tid < 1) { cpsum[tid] += cpsum[tid + 1]; } __syncthreads();
    //if (tid < 32) warpReduce(psum, tid);
    __syncthreads();

    if (tid==0) ts[bid] = __half2float(cpsum[0]);
  }

  // cuda kernel to divide data by time series
  // run with NBATCH*NCHAN*width/32 blocks of 32 threads
  __global__ void divide_by_ts(half * data, float * ts, int width, int stride, int flag_ts) {

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int idx = bid*32+tid;
    int y = (int)(idx / width);
    int x = (int)(idx % width);
    int b = (int)(y / NCHAN);
    int tsidx = b*width+x;
    int iidx = y*stride+x;

    data[iidx] /= __float2half(ts[tsidx]);

    if (flag_ts==1) {
      if (ts[tsidx]>TIME_SERIES_HIGH_THRESHOLD) data[iidx] = __float2half(1.);
      if (ts[tsidx]<TIME_SERIES_LOW_THRESHOLD) data[iidx] = __float2half(1.);
    }
  }

} // namespace anonymous

void hella::ts_correct(half * data, float * d_ts, int width, int stride)
{
  // calculate ts
  measure_ts<<<NBATCH*width,32>>>(data, d_ts, width, stride);

  checkCuda(cudaDeviceSynchronize());

  // median filter ts
  // hella::med_filter_ts(d_ts,width);

  // correct ts in data
  divide_by_ts<<<NBATCH*NCHAN*width/32,32>>>(data,d_ts,width,stride,0);
  checkCuda(cudaDeviceSynchronize());
}

void hella::med_filter_ts(float * d_ts, int width)
{
  float * hts = (float *)malloc(sizeof(float)*NBATCH*width);
  float * mhts = (float *)malloc(sizeof(float)*NBATCH*width);
  checkCuda(cudaMemcpy(hts,d_ts,sizeof(float)*NBATCH*width,cudaMemcpyDeviceToHost));
  float mn_ts = hella::median_filter(hts,mhts,NBATCH*width,NTSMED);
  checkCuda(cudaMemcpy(d_ts,mhts,sizeof(float)*NBATCH*width,cudaMemcpyHostToDevice));

  free(hts);
  free(mhts);
}
