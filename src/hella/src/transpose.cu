/***************************************************************************
 *
 *   Copyright (C) 2025 TBA
 *   Copyright (C) 2025 Fourier Space
 *   Authors: TBA, Andrew Jameson
 *
 ***************************************************************************/

#include "hella/alloc.h"
#include "hella/definitions.h"
#include "hella/macros.h"
#include "hella/transpose.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace
{
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

  // cuda kernel to transpose and scale single beam data for output
  // beam is [width, NCHAN], data is [NCHAN, width]
  // assume breakdown into tiles of 32x32, and run with 32x8 threads per block
  // launch with dim3 dimBlock(32, 8) and dim3 dimGrid(width/32, NCHAN/32)
  __global__ void transpose_output(unsigned char * beam, half * data, int width)
  {
    __shared__ half tile[32][33];

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    int mywidth = gridDim.x * 32;

    for (int j = 0; j < 32; j += 8)
      tile[threadIdx.y+j][threadIdx.x] = data[(y+j)*mywidth + x];

    __syncthreads();

    x = blockIdx.y * 32 + threadIdx.x;  // transpose block offset
    y = blockIdx.x * 32 + threadIdx.y;
    mywidth = gridDim.y * 32;

    // to saturate uchar8 at -4. and 10.
    float v, scf;
    for (int j = 0; j < 32; j += 8)
    {
      v = __half2float(tile[threadIdx.x][threadIdx.y + j]);
      scf = 255.f/14.f;
      v = scf*(v+4.f);
      if (v<0.) v = 0;
      if (v>255.f) v = 255.f;
      beam[(y+j)*mywidth + x] = (unsigned char)(v);
    }
  }

} // namespace anonymous

// TODO: handle_transpose_input and handle_transpose_output to do transpose via memcpy2d from intermediate array
void hella::transpose_input_handler(hella::pinfo_t* p, unsigned char * d_data, half * batch, int width, int stride)
{
  spdlog::trace("hella::transpose_input_handler width={} stride={}", width, stride);

  dim3 dimBlockIn(32, 8), dimGridIn(NCHAN/32, width/32);

  const size_t required_size = sizeof(half) * NCHAN * width;
  hella::lock_d_scratch(p, required_size);

  // do transpose by beam
  for (uint64_t bm=0; bm<NBATCH; bm++)
  {
    transpose_input<<<dimGridIn,dimBlockIn>>>(
      d_data+bm*NCHAN*width,
      reinterpret_cast<half*>(p->d_scratch),
      width
    );
    checkCuda(cudaMemcpy2D(
      batch+bm*NCHAN*stride,
      stride*sizeof(half),
      reinterpret_cast<half*>(p->d_scratch),
      width*sizeof(half),
      width*sizeof(half),
      NCHAN,
      cudaMemcpyDeviceToDevice
    ));
  }
  hella::unlock_d_scratch(p);

  checkCuda(cudaDeviceSynchronize());
}

void hella::transpose_output_handler(hella::pinfo_t* p, unsigned char * d_data, half * batch, int width, int stride)
{
  dim3 dimBlockOut(32, 8), dimGridOut(width/32, NCHAN/32);
  const size_t nval = NCHAN * width;
  size_t required_size = sizeof(half) * nval;
  hella::lock_d_scratch(p, required_size);
  auto tmp = reinterpret_cast<half*>(p->d_scratch);

  // do transpose by beam
  for (int bm=0; bm<NBATCH; bm++)
  {
    checkCuda(cudaMemcpy2D(
      tmp,
      width*sizeof(half),
      batch+bm*NCHAN*stride,
      stride*sizeof(half),
      width*sizeof(half),
      NCHAN,
      cudaMemcpyDeviceToDevice
    ));

    transpose_output<<<dimGridOut,dimBlockOut>>>(
      d_data + (bm * nval),
      tmp,
      width
    );
  }
  hella::unlock_d_scratch(p);

  checkCuda(cudaDeviceSynchronize());
}
