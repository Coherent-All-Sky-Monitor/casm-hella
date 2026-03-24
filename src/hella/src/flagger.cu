/***************************************************************************
 *
 *   Copyright (C) 2025 Vikram Ravi
 *   Copyright (C) 2025 Fourier Space
 *   Authors: Vikram Ravi, Andrew Jameson
 *
 ***************************************************************************/

#include "hella/alloc.h"
#include "hella/convolution.h"
#include "hella/definitions.h"
#include "hella/flagger.h"
#include "hella/macros.h"
#include "hella/median_filter.h"
#include "hella/normalization.h"
#include "hella/transpose.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

// use anonymous namespace to force internal linkage
namespace
{
  // cuda kernel to add number to data
  // run with NBATCH*NCHAN*width/32 blocks of 32 threads
  __global__ void add_number(half * data, half num, int width, int stride)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y = (int)(idx / width);
    int x = (int)(idx % width);
    int iidx = y*stride+x;

    data[iidx] += num;
  }

  __global__ void add_then_multiply_by_number(half * data, half a, half c, int width, int stride)
  {
    int isamp = blockIdx.x * blockDim.x + threadIdx.x;
    if (isamp >= width)
      return;

    int idx = (blockIdx.y * stride) + isamp;
    data[idx] = (data[idx] + a) * c;
  }

  // kernel to calculate bandpass
  // launch with NCHAN*NBATCH blocks of 256 threads
  // will ignore last (width % 256) times
  __global__ void calc_bandpass(half * data, float * bandpass, int width, int stride)
  {
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int npartials = (int)(width / 256); // number partial sums
    half fac = (half)((256.*npartials));
    __shared__ half psum[256];

    // calculate partial sums
    int idx0 = bid*stride + tid*npartials;
    psum[tid] = 0.;
    for (int i=idx0;i<npartials+idx0;i++)
      psum[tid] += data[i];
    psum[tid] /= fac;

    __syncthreads();

    // sum over shared memory
    if (tid < 128) { psum[tid] += psum[tid + 128]; } __syncthreads();
    if (tid < 64) { psum[tid] += psum[tid + 64]; } __syncthreads();
    if (tid < 32) { psum[tid] += psum[tid + 32]; } __syncthreads();
    if (tid < 16) { psum[tid] += psum[tid + 16]; } __syncthreads();
    if (tid < 8) { psum[tid] += psum[tid + 8]; } __syncthreads();
    if (tid < 4) { psum[tid] += psum[tid + 4]; } __syncthreads();
    if (tid < 2) { psum[tid] += psum[tid + 2]; } __syncthreads();
    if (tid < 1) { psum[tid] += psum[tid + 1]; } __syncthreads();
    //if (tid < 32) warpReduce(psum, tid);
    __syncthreads();

    if (tid==0) bandpass[bid] = __half2float(psum[0]);
  }

  // in each block, warps will compute the bandpass for a single channel
  // each block will write out 256/32 channels
  // grid handles the rest
  __global__ void calc_bandpass_new(const __restrict__ half* input_data, float * bandpass, int width, int stride)
  {
    const unsigned warp_idx = threadIdx.x & 0x1F; // % 32;
    const unsigned warp_num = threadIdx.x / warpSize;
    const unsigned channelbeam = (blockIdx.x * 8) + warp_num;
    unsigned idx = (channelbeam * stride) + warp_idx;

    const int npartials = width / 32; // number partial sums
    float sum = 0;
    for (unsigned i=0; i<npartials; i++)
    {
      const float raw = __half2float(input_data[idx]);
      idx += warpSize;
      sum += raw;
    }

    // warp level reduction
    for (int offset = 16; offset > 0; offset /= 2)
    {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (warp_idx == 0)
    {
      bandpass[channelbeam] = sum / float(width);
    }
  }

  // add bandpasses
  // add_bandpass<<<NCHAN*NBATCH/32,32>>>(d_mask,d_flagSpec);
  __global__ void add_bandpass(float * d_mask, float * d_flagSpec)
  {

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int idx = bid*32 + tid;

    d_flagSpec[idx] += d_mask[idx];
  }

  // cuda kernel to replace masked values
  // run with NBATCH*NCHAN*width/32 blocks of 32 threads
  __global__ void replace_data(half * data, half * mask, float repval, int width, int stride, int flag1, int flag2) {

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int idx = bid*32+tid;
    int y = (int)(idx / width);
    int x = (int)(idx % width);
    int ch = (int)(y % NCHAN);
    int iidx = y*stride+x;

    if (mask[iidx]>(half)(0.))
      data[iidx] = repval;

    if (ch>=flag1 && ch<flag2)
      data[iidx] = repval;
  }

  __global__ void replace_data_and_add_bandpass(half * data, const float * bp, int width, int stride, float t1, float t2)
  {
    const int ibeamchan = (blockIdx.y * blockDim.y) + threadIdx.y;
    const float bp_val = bp[ibeamchan];
    const int idx = ibeamchan * stride + (blockIdx.x * blockDim.x) + threadIdx.x;
    if (bp_val < t1 || bp_val > t2)
    {
      data[idx] = __half(1);
    }
    else {
      data[idx] += __half(1);
    }
  }

  // cuda kernel to find data above threshold and add into mask
  // run with NBATCH*NCHAN*width/32 blocks of 32 threads
  __global__ void threshold_data(half * data, half * mask, float threshold, int width, int stride) {

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int idx = bid*32+tid;
    int y = (int)(idx / width);
    int x = (int)(idx % width);
    int iidx = y*stride+x;

    if (data[iidx]>__float2half(threshold))
      mask[iidx] = 1.;
    if (data[iidx]<__float2half(-1.*threshold))
      mask[iidx] = 1.;
  }

  // cuda kernel to divide data by bandpass
  // run with NBATCH*NCHAN*width/64 blocks of 64 threads
  __global__ void divide_by_bp(half * data, float * bandpass, int width, int stride)
  {
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int idx = bid*blockDim.x+tid;
    int y = (int)(idx / width);
    int x = (int)(idx % width);
    int iidx = y*stride+x;

    if (bandpass[y] > 0)
      data[iidx] /= __float2half(bandpass[y]);
    else
      data[iidx] = 0;
  }

} // namespace anonymous

void hella::normalize_data(hella::pinfo_t* p, half * data, int width, int stride)
{
  // note: uses h and d scratch
  float mean{};
  float stdDev = hella::calculate_stddev(p, data, width, NBATCH*NCHAN, stride, &mean);

  int nt = 512;
  half c = __float2half(-mean);
  half m = __float2half(1.f/stdDev);

  dim3 gridDim(width/nt, NBATCH*NCHAN, 1);
  if (width % nt != 0)
    gridDim.x++;

  add_then_multiply_by_number<<<gridDim,nt>>>(data,c,m,width,stride);
  checkCuda(cudaDeviceSynchronize());
}

float hella::bandpass_flag(hella::pinfo_t * p, half * data, int width)
{
  // bandpass correct [uses h and d scratch]
  spdlog::trace("hella::bandpass_flag bandpass_correct(p, data, {}, {})", width, p->batch_stride);
  float mn_bp = bandpass_correct(p, data, width, p->batch_stride);

  // normalize data [uses h and d_scratch]
  spdlog::trace("hella::bandpass_flag normalize_data(p, data, {}, {})", width, p->batch_stride);
  normalize_data(p, data, width, p->batch_stride);

  // calculate bandpass
  const size_t bp_size = sizeof(float) * NBATCH * NCHAN;
  hella::lock_d_scratch(p, bp_size);
  auto bandpass = reinterpret_cast<float*>(p->d_scratch);
  spdlog::trace("hella::bandpass_flag calc_bandpass_new(data, bandpass, {}, {})", width, p->batch_stride);
  calc_bandpass_new<<<NCHAN*NBATCH/8,256>>>(data, bandpass, width, p->batch_stride);

  dim3 blockDim(32 ,16 ,1);
  dim3 gridDim(width/blockDim.x, NBATCH*NCHAN/blockDim.y, 1);
  if (width % blockDim.x != 0)
    gridDim.x++;
  spdlog::trace("hella::bandpass_flag replace_data_and_add_bandpass(data, bandpass, {}, {}, {}, {})", width, p->batch_stride, p->spec_min, p->spec_max);
  replace_data_and_add_bandpass<<<gridDim,blockDim>>>(data, bandpass, width, p->batch_stride, p->spec_min, p->spec_max);
  hella::unlock_d_scratch(p);

  return mn_bp;
}

float hella::bandpass_correct(hella::pinfo_t* p, half * input_data, int width, int stride)
{
  const size_t nval = NBATCH * NCHAN;
  const size_t bp_size = sizeof(float) * nval;
  hella::lock_d_scratch(p, bp_size);

  // allocate bandpass
  auto bandpass = reinterpret_cast<float *>(p->d_scratch);

  // calculate bandpass
  calc_bandpass_new<<<NCHAN*NBATCH/8,256>>>(input_data, bandpass, width, stride);

  checkCuda(cudaDeviceSynchronize());

  // median filter bandpass [uses h_scratch]
  float mn_bp = med_filter_bandpass(p, bandpass);

  // correct bandpass in data
  const unsigned nt = 512;
  divide_by_bp<<<NBATCH*NCHAN*width/nt,nt>>>(input_data,bandpass,width,stride);

  checkCuda(cudaDeviceSynchronize());
  hella::unlock_d_scratch(p);

  return mn_bp;
}

float hella::apply_scrunch(
  hella::pinfo_t * p, half * data, half * mask, half * d_smooth, float * d_ts,
  int width, int stride, int tscrunch, int fscrunch, float thresh,
  int flag, int ts, float * d_flagSpec, int flag1, int flag2)
{
  float begin{}, end{};

  const size_t mask_nval = NBATCH * NCHAN;
  const size_t mask_size = mask_nval * sizeof(float);

  // bandpass
  begin = clock();
  float mn_bp = bandpass_correct(p,data,width,stride);
  checkCuda(cudaDeviceSynchronize());
  end = clock();
  p->t1 += static_cast<float>(end - begin) / CLOCKS_PER_SEC;

  // baseline
  if (ts==1) {
    begin = clock();
    ts_correct(data,d_ts,width,stride);
    end = clock();
    p->t2 += static_cast<float>(end - begin) / CLOCKS_PER_SEC;
  }

  // normalize
  begin = clock();
  normalize_data(p, data, width, stride); // uses h and d scratch
  end = clock();
  p->t3 += static_cast<float>(end - begin) / CLOCKS_PER_SEC;

  if (flag==1) {

    // derive smoothed data
    begin = clock();
    //smooth_data<<<NBATCH*NCHAN*width/32,32>>>(data, d_smooth, 1./tscrunch/fscrunch, tscrunch, fscrunch, width, stride);

    // uses h and d scratch
    hella::npp_convolve_handler(p, data, d_smooth, 1./tscrunch/fscrunch, tscrunch, fscrunch, width, stride);
    checkCuda(cudaDeviceSynchronize());
    end = clock();
    p->t4 += static_cast<float>(end - begin) / CLOCKS_PER_SEC;

    // threshold data
    begin = clock();
    checkCuda(cudaMemset(mask,0,NBATCH*NCHAN*stride*sizeof(half)));
    threshold_data<<<NBATCH*NCHAN*width/32,32>>>(d_smooth,mask,thresh/sqrt(1.*tscrunch*fscrunch),width,stride);
    checkCuda(cudaDeviceSynchronize());
    end = clock();
    p->t5 += static_cast<float>(end - begin) / CLOCKS_PER_SEC;

    // replace data after growing mask
    begin = clock();
    //smooth_data<<<NBATCH*NCHAN*width/32,32>>>(mask, d_smooth, 1./tscrunch/fscrunch, tscrunch, fscrunch, width, stride);

    // uses h and d scratch
    npp_convolve_handler(p, mask, d_smooth, 1./tscrunch/fscrunch, tscrunch, fscrunch, width, stride);

    hella::lock_d_scratch(p, mask_size);
    auto d_mask = reinterpret_cast<float*>(p->d_scratch);
    replace_data<<<NBATCH*NCHAN*width/32,32>>>(data, d_smooth, 0., width, stride, flag1, flag2);
    add_number<<<NBATCH*NCHAN*width/32,32>>>(data,1.,width,stride);
    calc_bandpass<<<NCHAN*NBATCH,256>>>(d_smooth, d_mask, width, stride);
    add_bandpass<<<NCHAN*NBATCH/32,32>>>(d_mask, d_flagSpec);
    hella::unlock_d_scratch(p);

    checkCuda(cudaDeviceSynchronize());
    end = clock();
    p->t6 += static_cast<float>(end - begin) / CLOCKS_PER_SEC;
  }

  return mn_bp;
}

void hella::fast_flagger(hella::pinfo_t *p)
{
  float begin{}, end{};

  // setup
  int nBatches = static_cast<int>(p->nbeam / NBATCH);
  spdlog::debug("hella::fast_flagger have nbatches {}", nBatches);
  std::vector<float> mn_bp;
  mn_bp.resize(nBatches, 0);

  for (uint64_t batch = 0; batch < nBatches; batch++)
  {
    begin = clock();
    spdlog::trace("hella::fast_flagger transpose_input_handler");
    transpose_input_handler(p, p->d_gulp + p->gulp_nbyte * batch * NBATCH * NCHAN * p->gulp, p->batch, p->gulp, p->batch_stride);
    end = clock();
    p->t7 += static_cast<float>(end - begin) / CLOCKS_PER_SEC; 

    mn_bp[batch] = bandpass_flag(p,p->batch, p->gulp);
    // loop over scrunches
    for (int scrnch=0;scrnch<p->nscrunches;scrnch++)
    {
      apply_scrunch(
        p, 
        p->batch, 
        p->mask, 
        p->d_smooth, 
        p->d_ts, 
        p->gulp, 
        p->batch_stride, 
        p->scrunches[scrnch].tscrunch,
        p->scrunches[scrnch].fscrunch, 
        p->scrunches[scrnch].thresh,
        1, // flag
        0, // ts
        p->d_flagSpec,
        p->flag1,
        p->flag2
      );
    }
    apply_scrunch(
      p,
      p->batch,
      p->mask,
      p->d_smooth,
      p->d_ts,
      p->gulp,
      p->batch_stride,
      8,    // tscrunch
      8,    // fscrunch
      100,  // thresh
      0,    // flag
      1,    // ts
      p->d_flagSpec,
      p->flag1,
      p->flag2
    );

    begin = clock();
    spdlog::trace("hella::fast_flagger transpose_output_handler");
    transpose_output_handler(p, p->d_data+batch*NBATCH*NCHAN*p->NTIME,p->batch);
    checkCuda(cudaMemcpy(p->h_flagSpec,p->d_flagSpec,sizeof(float)*NBATCH*NCHAN,cudaMemcpyDeviceToHost));
    end = clock();
    p->t7 += static_cast<float>(end - begin) / CLOCKS_PER_SEC;
  }
  
}
