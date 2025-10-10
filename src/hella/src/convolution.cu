/***************************************************************************
 *
 *   Copyright (C) 2025 TBA
 *   Copyright (C) 2025 Fourier Space
 *   Authors: TBA, Andrew Jameson
 *
 ***************************************************************************/

#include "hella/convolution.h"
#include "hella/definitions.h"
#include "hella/macros.h"

#include <npp.h>
#include <nppdefs.h>
#include <nppcore.h>
#include <nppi.h>
#include <npps.h>

void hella::npp_convolve_handler(half * data, half * output, float scfac, int xw, int yw, int width, int stride) {

  NppiSize oSrcSize = {stride,NCHAN*NBATCH};
  NppiPoint oSrcOffset = {0,0};
  Npp32f * pKernel{nullptr};
  NppiSize pKernelSize = {xw,yw};
  NppiPoint oAnchor = {(int)(xw/2),(int)(yw/2)};
  float * h_kernel{nullptr};
  const size_t kernel_bytes = sizeof(float)*xw*yw;
  checkCuda(cudaMalloc((void **)(&pKernel), kernel_bytes));
  h_kernel  = (float *)malloc(kernel_bytes);
  for (int i=0;i<xw*yw;i++) h_kernel[i] = scfac;
  checkCuda(cudaMemcpy(pKernel,h_kernel, kernel_bytes,cudaMemcpyHostToDevice));

  checkNpp(nppiFilterBorder32f_16f_C1R((Npp16f *)data,stride*2,oSrcSize,oSrcOffset,(Npp16f *)output,stride*2,oSrcSize,pKernel,pKernelSize,oAnchor,NPP_BORDER_REPLICATE));

  checkCuda(cudaFree(pKernel));
  free(h_kernel);

}
