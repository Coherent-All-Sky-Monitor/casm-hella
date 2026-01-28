/***************************************************************************
 *
 *   Copyright (C) 2025 Vikram Ravi
 *   Copyright (C) 2025 Fourier Space
 *   Authors: Vikram Ravi, Andrew Jameson
 *
 ***************************************************************************/

#include "hella/alloc.h"
#include "hella/convolution.h"
#include "hella/macros.h"

#include <cuda_fp16.h>
#include <npp.h>
#include <nppdefs.h>
#include <nppcore.h>
#include <nppi.h>
#include <npps.h>

void hella::npp_convolve_handler(hella::pinfo_t* p, half * data, half * output, float scfac, int xw, int yw, int width, int stride)
{
  NppiSize oSrcSize = {stride,NCHAN*NBATCH};
  NppiPoint oSrcOffset = {0,0};
  NppiSize pKernelSize = {xw,yw};
  NppiPoint oAnchor = {(int)(xw/2),(int)(yw/2)};

  const size_t nval = xw * yw;
  const size_t required_kernel_size = sizeof(float) * nval;
  hella::lock_h_scratch(p, required_kernel_size);
  hella::lock_d_scratch(p, required_kernel_size);

  auto h = reinterpret_cast<float*>(p->h_scratch);
  for (int i=0;i<nval;i++)
  {
    h[i] = scfac;
  }

  checkCuda(cudaMemcpy(p->d_scratch, p->h_scratch, required_kernel_size, cudaMemcpyHostToDevice));

  checkNpp(nppiFilterBorder32f_16f_C1R(
    reinterpret_cast<Npp16f *>(data),
    stride*2,
    oSrcSize,
    oSrcOffset,
    reinterpret_cast<Npp16f *>(output),
    stride*2,
    oSrcSize,
    reinterpret_cast<float *>(p->d_scratch),
    pKernelSize,
    oAnchor,
    NPP_BORDER_REPLICATE
  ));
  hella::unlock_h_scratch(p);
  hella::unlock_d_scratch(p);
}
