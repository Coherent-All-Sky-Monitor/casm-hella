/***************************************************************************
 *
 *   Copyright (C) 2025 TBA
 *   Copyright (C) 2025 Fourier Space
 *   Authors: TBA, Andrew Jameson
 *
 ***************************************************************************/

#include "hella/convolution.h"
#include "hella/definitions.h"
#include "hella/flagger.h"
#include "hella/macros.h"
#include "hella/median_filter.h"
#include "hella/transpose.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// use anonymous namespace to force internal linkage
namespace
{
  #ifdef ORIGINAL
  // cuda kernel to add number to data
  // run with NBATCH*NCHAN*width/32 blocks of 32 threads
  __global__ void add_number(half * data, float num, int width, int stride) {

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int idx = bid*32+tid;
    int y = (int)(idx / width);
    int x = (int)(idx % width);
    int iidx = y*stride+x;

    data[iidx] += __float2half(num);
  }
  #else
  // cuda kernel to add number to data
  // run with NBATCH*NCHAN*width/32 blocks of 32 threads
  __global__ void add_number(half * data, float num, int width, int stride)
  {

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int idx = bid*32+tid;
    int y = (int)(idx / width);
    int x = (int)(idx % width);
    int iidx = y*stride+x;

    data[iidx] += __float2half(num);
  }
  #endif


  // cuda kernel to multiply data by number
  // run with NBATCH*NCHAN*width/32 blocks of 32 threads
  __global__ void multiply_by_number(half * data, float num, int width, int stride) {

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int idx = bid*32+tid;
    int y = (int)(idx / width);
    int x = (int)(idx % width);
    int iidx = y*stride+x;

    data[iidx] *= __float2half(num);
  }

  // cuda kernel to transpose and scale single beam data for loader
  // beam is [width, NCHAN], data is [NCHAN, width]
  // assume breakdown into tiles of 32x32, and run with 32x8 threads per block
  // launch with dim3 dimBlock(32, 8) and dim3 dimGrid(NCHAN/32, width/32)
  __global__ void transpose_input(unsigned char * beam, half * data, int width) {

    __shared__ half tile[32][33];

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    int mywidth = gridDim.x * 32;

    for (int j = 0; j < 32; j += 8)
      tile[threadIdx.y+j][threadIdx.x] = __float2half((float)(beam[(y+j)*mywidth + x]));

    __syncthreads();

    x = blockIdx.y * 32 + threadIdx.x;  // transpose block offset
    y = blockIdx.x * 32 + threadIdx.y;
    mywidth = gridDim.y * 32;

    for (int j = 0; j < 32; j += 8)
      data[(y+j)*mywidth + x] = tile[threadIdx.x][threadIdx.y + j];

  }

  // kernel to calculate bandpass
  // launch with NCHAN*NBATCH blocks of 256 threads
  // will ignore last (width % 256) times
  __global__ void calc_bandpass(half * data, float * bandpass, int width, int stride) {

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
  // run with NBATCH*NCHAN*width/32 blocks of 32 threads
  __global__ void divide_by_bp(half * data, float * bp, int width, int stride) {

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int idx = bid*32+tid;
    int y = (int)(idx / width);
    int x = (int)(idx % width);
    int iidx = y*stride+x;

    data[iidx] /= __float2half(bp[y]);
  }

  // cuda kernel to replace masked values
  // run with NBATCH*NCHAN*width/32 blocks of 32 threads
  __global__ void replace_data_bandpass(half * data, float * bp, float repval, int width, int stride, float t1, float t2)
  {
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int idx = bid*32+tid;
    int y = (int)(idx / width);
    int x = (int)(idx % width);
    int iidx = y*stride+x;

    if (bp[y]<t1 || bp[y]>t2)
      data[iidx] = repval;
  }

} // namespace anonymous

void hella::normalize_data(half * data, int width, int stride)
{
  float stdDev = calculate_stddev(data,width,NBATCH*NCHAN,stride);

  add_number<<<NBATCH*NCHAN*width/32,32>>>(data,-1.,width,stride);

  multiply_by_number<<<NBATCH*NCHAN*width/32,32>>>(data,1./stdDev,width,stride);

  cudaDeviceSynchronize();
}

float hella::bandpass_flag(pinfo * p, half * data)
{
  // bandpass correct
  float mn_bp = bandpass_correct(data,p->NTIME, p->batch_stride);

  // normalize data
  normalize_data(data,p->NTIME, p->batch_stride);

  // calculate bandpass
  calc_bandpass<<<NCHAN*NBATCH,256>>>(data, p->d_bpout, p->NTIME, p->batch_stride);

  // flag data
  replace_data_bandpass<<<NBATCH*NCHAN*p->NTIME/32,32>>>(data, p->d_bpout, 0., p->NTIME, p->batch_stride, p->spec_min, p->spec_max);

  // finish up
  add_number<<<NBATCH*NCHAN*p->NTIME/32,32>>>(data,1.,p->NTIME, p->batch_stride);

  return mn_bp;
}

float hella::bandpass_correct(half * data, int width, int stride) {

  // allocate bandpass
  float * d_bandpass{nullptr};
  checkCuda(cudaMalloc(&d_bandpass, NBATCH * NCHAN * sizeof(float)));

  // calculate bandpass
  calc_bandpass<<<NCHAN*NBATCH,256>>>(data, d_bandpass, width, stride);

  checkCuda(cudaDeviceSynchronize());

  // median filter bandpass
  float mn_bp = med_filter_bandpass(d_bandpass);

  // correct bandpass in data
  divide_by_bp<<<NBATCH*NCHAN*width/32,32>>>(data,d_bandpass,width,stride);

  checkCuda(cudaDeviceSynchronize());
  checkCuda(cudaFree(d_bandpass));

  return mn_bp;
}

float hella::apply_scrunch(
  pinfo * p, half * data, half * mask, half * d_smooth, float * d_ts,
  int width, int stride, int tscrunch, int fscrunch, float thresh,
  int flag, int ts, float * d_flagSpec, int flag1, int flag2)
{

  float begin, end;
  float * d_mask;
  checkCuda(cudaMalloc(&d_mask, NBATCH * NCHAN * sizeof(float)));

  // bandpass
  //printf("bandpass\n");
  begin = clock();
  float mn_bp = bandpass_correct(data,width,stride);
  checkCuda(cudaDeviceSynchronize());
  end = clock();
  p->t1 += (float)(end - begin) / CLOCKS_PER_SEC;

  // baseline
  //printf("baseline\n");
  if (ts==1) {
    begin = clock();
    ts_correct(data,d_ts,width,stride);
    end = clock();
    p->t2 += (float)(end - begin) / CLOCKS_PER_SEC;
  }

  // normalize
  //printf("normalize\n");
  begin = clock();
  normalize_data(data,width,stride);
  end = clock();
  p->t3 += (float)(end - begin) / CLOCKS_PER_SEC;

  if (flag==1) {

    // derive smoothed data
    //printf("smooth\n");
    begin = clock();
    //smooth_data<<<NBATCH*NCHAN*width/32,32>>>(data, d_smooth, 1./tscrunch/fscrunch, tscrunch, fscrunch, width, stride);
    hella::npp_convolve_handler(data, d_smooth, 1./tscrunch/fscrunch, tscrunch, fscrunch, width, stride);
    checkCuda(cudaDeviceSynchronize());
    end = clock();
    p->t4 += (float)(end - begin) / CLOCKS_PER_SEC;

    // threshold data
    //printf("threshold\n");
    begin = clock();
    checkCuda(cudaMemset(mask,0,NBATCH*NCHAN*stride*sizeof(half)));
    threshold_data<<<NBATCH*NCHAN*width/32,32>>>(d_smooth,mask,thresh/sqrt(1.*tscrunch*fscrunch),width,stride);
    checkCuda(cudaDeviceSynchronize());
    end = clock();
    p->t5 += (float)(end - begin) / CLOCKS_PER_SEC;

    // replace data after growing mask
    //printf("replace\n");
    begin = clock();
    //smooth_data<<<NBATCH*NCHAN*width/32,32>>>(mask, d_smooth, 1./tscrunch/fscrunch, tscrunch, fscrunch, width, stride);
    npp_convolve_handler(mask, d_smooth, 1./tscrunch/fscrunch, tscrunch, fscrunch, width, stride);
    replace_data<<<NBATCH*NCHAN*width/32,32>>>(data, d_smooth, 0., width, stride, flag1, flag2);
    add_number<<<NBATCH*NCHAN*width/32,32>>>(data,1.,width,stride);
    calc_bandpass<<<NCHAN*NBATCH,256>>>(d_smooth, d_mask, width, stride);
    add_bandpass<<<NCHAN*NBATCH/32,32>>>(d_mask,d_flagSpec);
    checkCuda(cudaDeviceSynchronize());
    end = clock();
    p->t6 += (float)(end - begin) / CLOCKS_PER_SEC;

  }

  checkCuda(cudaFree(d_mask));

  return mn_bp;
}

// flagger
// load a batch beam by beam
// apply all scrunches to the batch
// unload the batch
void hella::fast_flagger(pinfo * p)
{

  float begin, end;

  // setup
  int nBatches = (int)(NBEAMS / NBATCH);
  checkCuda(cudaMemset(p->d_flagSpec,0,4*NBATCH*NCHAN));
  syslog(LOG_INFO,"have nbatches %d",nBatches);
  float mn_bp[nBatches], tmp;

  // output bandpass
  FILE *fout{nullptr};
  float *h_bpout{nullptr};
  char fnam[200];
  if (p->output_bandpass>0) {
    h_bpout = (float *)malloc(sizeof(float)*NBATCH*NCHAN);
    sprintf(fnam,"/home/ubuntu/data/bpout_%d.tmp",p->output_bandpass);
    fout=fopen(fnam,"w");
  }

  //printf("fast_flagger ");

  // loop over batches
  for (uint64_t batch = 0; batch < nBatches; batch++) {

    // load a batch
    //printf("transpose input %d of %d\n",batch+1,nBatches);
    begin = clock();
    transpose_input_handler(p->d_data+batch*NBATCH*NCHAN*p->NTIME,p->batch,p->NTIME,p->batch_stride);
    end = clock();
    p->t7 += (float)(end - begin) / CLOCKS_PER_SEC;

    // output init bandpass
    if (p->output_bandpass>0) {
      begin = clock();
      calc_bandpass<<<NCHAN*NBATCH,256>>>(p->batch, p->d_bpout, p->NTIME, p->batch_stride);
      checkCuda(cudaMemcpy(h_bpout,p->d_bpout,NBATCH*NCHAN*4,cudaMemcpyDeviceToHost));
      for (int i=0;i<NBATCH*NCHAN;i++)
        fprintf(fout,"%g\n",h_bpout[i]);
      end = clock();
      p->t8 += (float)(end - begin) / CLOCKS_PER_SEC;
    }

    // bandpass flag / correct
    mn_bp[batch] = bandpass_flag(p,p->batch);

    // loop over scrunches
    for (int scrnch=0;scrnch<p->nscrunches;scrnch++) {
      //printf("scrunch %d...",scrnch);
      tmp = apply_scrunch(p, p->batch, p->mask, p->d_smooth, p->d_ts, p->NTIME, p->batch_stride, p->scrunches[scrnch].tscrunch,p->scrunches[scrnch].fscrunch, p->scrunches[scrnch].thresh,1,0,p->d_flagSpec,p->flag1,p->flag2);
      checkCuda(cudaDeviceSynchronize());
    }
    tmp = apply_scrunch(p, p->batch, p->mask, p->d_smooth, p->d_ts, p->NTIME, p->batch_stride, 8, 8, 100., 0, 1, p->d_flagSpec,p->flag1,p->flag2);
    //    printf("\n");

    checkCuda(cudaDeviceSynchronize());

    // output final bandpass
    if (p->output_bandpass>0) {
      begin = clock();
      calc_bandpass<<<NCHAN*NBATCH,256>>>(p->batch, p->d_bpout, p->NTIME, p->batch_stride);
      cudaMemcpy(h_bpout,p->d_bpout,NBATCH*NCHAN*4,cudaMemcpyDeviceToHost);
      for (int i=0;i<NBATCH*NCHAN;i++)
        fprintf(fout,"%g\n",h_bpout[i]);
      end = clock();
      p->t8 += (float)(end - begin) / CLOCKS_PER_SEC;
    }

    // unload the batch
    begin = clock();
    transpose_output_handler(p->d_data+batch*NBATCH*NCHAN*p->NTIME,p->batch,p->NTIME,p->batch_stride);
    cudaMemcpy(p->h_flagSpec,p->d_flagSpec,4*NBATCH*NCHAN,cudaMemcpyDeviceToHost);
    end = clock();
    p->t7 += (float)(end - begin) / CLOCKS_PER_SEC;

    //printf("done\n");
    //printf("%g ",mn_bp);

  }
  //printf("\n");

  syslog(LOG_INFO,"fast_flagger %g %g %g %g",mn_bp[0],mn_bp[1],mn_bp[2],mn_bp[3]);

  if (p->output_bandpass>0) {
    sprintf(fnam,"mv /home/ubuntu/data/bpout_%d.tmp /home/ubuntu/data/bpout_%d.out",p->output_bandpass,p->output_bandpass);
    system(fnam);
    fclose(fout);
    free(h_bpout);
  }

}

