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
#include "hella/unused.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace
{
  // cuda kernel to divide data by array
  // run with NBATCH*NCHAN*width/32 blocks of 32 threads
  __global__ void divide_by_array(half * data, half * arr, int width, int stride) {

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int idx = bid*32+tid;
    int y = (int)(idx / width);
    int x = (int)(idx % width);
    int iidx = y*stride+x;

    data[iidx] /= arr[iidx];
  }

  // kernel to sort out time series
  // run with width*NBATCH/32 blocks of 32 threads
  __global__ void fix_ts(half * temp_ts, float * ts, int width, int stride)
  {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int idx = bid*32+tid;
    int bat = (int)(idx/NBATCH);
    int tim = (int)(idx % NBATCH);

    ts[idx] = __half2float(temp_ts[bat*stride+tim]);
  }

  // TEST cuda kernel to load test data
  // run with NBATCH*NCHAN*width/32 blocks of 32 threads
  __global__ void load_test(float * input, half * data, int width, int stride) {

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int idx = bid*32+tid;
    int y = (int)(idx / width);
    int x = (int)(idx % width);
    int iidx = y*stride+x;

    data[iidx] = __float2half(input[idx]);

  }

  // TEST cuda kernel to unload test data
  // run with NBATCH*NCHAN*width/32 blocks of 32 threads
  __global__ void unload_test(float * output, half * data, int width, int stride) {

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int idx = bid*32+tid;
    int y = (int)(idx / width);
    int x = (int)(idx % width);
    int iidx = y*stride+x;

    output[idx] = __half2float(data[iidx]);

  }

} // namespace anonymous

// function to remove time-frequency baseline
void hella::remove_tf_baseline(half * data, int width, int stride) {

  // allocate smooth array
  half * d_smooth;
  checkCuda(cudaMalloc(&d_smooth, NBATCH * NCHAN * stride * sizeof(half)));

  // smooth data
  //smooth_data<<<NBATCH*NCHAN*width/32,32>>>(data, d_smooth, (float)(1./NTIME_BOX/NCHAN_BOX), NTIME_BOX, NCHAN_BOX, width, stride);
  hella::npp_convolve_handler(data, d_smooth, (float)(1./NTIME_BOX/NCHAN_BOX), NTIME_BOX, NCHAN_BOX, width, stride);

  // divide by smoothed data
  divide_by_array<<<NBATCH*NCHAN*width/32,32>>>(data,d_smooth,width,stride);

  checkCuda(cudaDeviceSynchronize());
  checkCuda(cudaFree(d_smooth));

}

// function to measure ts using cublas calls
void hella::blas_ts(half * data, half * unity, half * temp_ts, float * ts, int width, int stride) {

  // set up for gemm
  cublasHandle_t cublasH = NULL;
  cudaStream_t stream = NULL;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cublasCreate(&cublasH);
  cublasSetStream(cublasH, stream);

  // gemm settings
  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;
  const int m = stride;
  const int n = 1;
  const int k = NCHAN;
  const half alpha = 1./NCHAN;
  const int lda = m;
  const int ldb = k;
  const half beta = 0.;
  const int ldc = m;
  const long long int strideA = NCHAN*stride;
  const long long int strideB = NCHAN;
  const long long int strideC = stride;
  const int batchCount = NBATCH;

  // run strided batched gemm
  cublasHgemmStridedBatched(cublasH,transa,transb,m,n,k,
                            &alpha,data,lda,strideA,
                            unity,ldb,strideB,&beta,
                            temp_ts,ldc,strideC,
                            batchCount);

  checkCuda(cudaDeviceSynchronize());

  // run kernel to place in output
  fix_ts<<<width*NBATCH/32,32>>>(temp_ts,ts,width,stride);
  checkCuda(cudaDeviceSynchronize());

}

void hella::apply_batch_test(float * input, float * output, int width, int stride)
{
  half * batch, * mask;
  //int stride = 16384;
  //int width = 14912;
  cudaMalloc(&batch, NBATCH * NCHAN * stride * sizeof(half));
  cudaMalloc(&mask, NBATCH * NCHAN * stride * sizeof(half));
  float * d_data;
  cudaMalloc(&d_data, NBATCH * NCHAN * width * sizeof(float));

  /// APPLY TEST HERE

  //transpose_input_handler(d_input,batch,width,stride);

  float * d_ts, begin, end;
  cudaMalloc(&d_ts, NBATCH * width * sizeof(float));
  cudaMemcpy(d_data,input,NBATCH * NCHAN * width * sizeof(float),cudaMemcpyHostToDevice);
  load_test<<<NBATCH*NCHAN*width/32,32>>>(d_data,batch,width,stride);

  begin = clock();
  //measure_ts<<<NBATCH*width,32>>>(batch, d_ts, width, stride);
  ts_correct(batch, d_ts, width, stride);
  end = clock();
  printf("Time %g\n",(float)(end - begin) / CLOCKS_PER_SEC);

  cudaMemcpy(output,d_ts,NBATCH * width * sizeof(float),cudaMemcpyDeviceToHost);


  //transpose_output_handler(d_input,batch,width,stride);

  //unload_test<<<NBATCH*NCHAN*width/32,32>>>(d_data,batch,width,stride);
  //cudaMemcpy(output,d_data,sizeof(float)*NBATCH*NCHAN*width,cudaMemcpyDeviceToHost);
  //cudaMemcpy(output,d_input,NBATCH * NCHAN * width,cudaMemcpyDeviceToHost);
}