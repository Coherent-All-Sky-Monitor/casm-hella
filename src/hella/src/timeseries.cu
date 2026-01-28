/***************************************************************************
 *
 *   Copyright (C) 2025 Vikram Ravi
 *   Copyright (C) 2025 Fourier Space
 *   Authors: Vikram Ravi, Andrew Jameson
 *
 ***************************************************************************/

#include "hella/alloc.h"
#include "hella/definitions.h"
#include "hella/macros.h"
#include "hella/median_filter.h"
#include "hella/timeseries.h"

#include <cuda_runtime.h>

namespace
{
  __global__ void measure_ts(half * data, float * ts, int width, int stride)
  {
    // beam == blockIdx.y
    // time == blockIdx.x * blockDim.x + threadIdx.x

    int isamp = blockIdx.x * blockDim.x + threadIdx.x;
    if (isamp >= width)
      return;

    int idx = (blockIdx.y * NCHAN * stride) + isamp;
    half sum = 0;
    for (int ichan=0; ichan<NCHAN; ichan++)
    {
      sum += data[idx];
      idx += stride;
    }
    ts[blockIdx.y * width + isamp] = __half2float(sum) / float(NCHAN);
  }

  __global__ void divide_by_ts(half * data, const float * ts, int width, int stride, int flag_ts)
  {
    // beam == blockIdx.y
    // chan == blockIdx.z
    // time == blockIdx.x * blockDim.x + threadIdx.x

    int isamp = blockIdx.x * blockDim.x + threadIdx.x;
    if (isamp >= width)
      return;

    // common scale factor by which all channels are divided
    const float facf = ts[blockIdx.y * stride + isamp];

    const half fac = __float2half(facf);

    const int idx = ((blockIdx.y * NCHAN + blockIdx.z) * stride) + isamp;
    data[idx] = __hdiv(data[idx], fac);
  }

} // namespace anonymous

void hella::ts_correct(half * data, float * d_ts, int width, int stride)
{
  // calculate ts
  dim3 blockDim(256, 1, 1);
  dim3 gridDim(width/blockDim.x, NBATCH, 1);
  if (width % blockDim.x != 0)
    gridDim.x++;
  measure_ts<<<gridDim, blockDim>>>(data, d_ts, width, stride);

  // AJ this was disabled: median filter ts
  // hella::med_filter_ts(d_ts,width);

  gridDim.z = NCHAN;
  divide_by_ts<<<gridDim, blockDim>>>(data,d_ts,width,stride,0);
  checkCuda(cudaDeviceSynchronize());
}

void hella::med_filter_ts(hella::pinfo_t* p, float * d_ts, int width)
{
  const size_t nval = NBATCH * width;
  const size_t hts_size = sizeof(float) * nval;
  hella::lock_h_scratch(p, hts_size * 2);

  auto hts = reinterpret_cast<float*>(p->h_scratch);
  auto mhts = hts + nval;
  checkCuda(cudaMemcpy(hts,d_ts,hts_size,cudaMemcpyDeviceToHost));
  float mn_ts = hella::median_filter(hts,mhts,nval,NTSMED);
  checkCuda(cudaMemcpy(d_ts,mhts,hts_size,cudaMemcpyHostToDevice));

  hella::unlock_h_scratch(p);
}
