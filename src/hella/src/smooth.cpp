/***************************************************************************
 *
 *   Copyright (C) 2025 Vikram Ravi
 *   Copyright (C) 2025 Fourier Space
 *   Authors: Vikram Ravi, Andrew Jameson
 *
 ***************************************************************************/

#include "hella/alloc.h"
#include "hella/smooth.h"
#include "hella/normalization.h"
#include "hella/macros.h"

#include <npp.h>
#include <nppdefs.h>
#include <nppcore.h>
#include <nppi.h>
#include <npps.h>

void hella::smooth(hella::pinfo_t *p, int scale)
{
  NppiSize oSrcSize = {p->ntime_dd,p->ndms};
  NppiPoint oSrcOffset = {0,0};
  NppiSize oSizeROI = {p->ntime_dd,p->ndms};
  NppiSize oOutROI = {p->ntime_out,p->ndms-2};
  Npp32f w[3] = {0.3,1.,0.3};
  Npp32f v, filtSum;
  NppiSize pKernelSize;
  NppiPoint oAnchor;

  // perform a single, maximal configuration
  int sm = static_cast<int>(powf(p->maxWidth, 2));
  size_t required_kernel_size = sizeof(float)*(2*sm+1)*3;
  hella::lock_h_scratch(p, required_kernel_size);
  hella::lock_d_scratch(p, required_kernel_size);

  auto h_kernel = reinterpret_cast<float*>(p->h_scratch);
  auto d_kernel = reinterpret_cast<float*>(p->d_scratch);

  int smit = 0;
  for (int sm=(int)(p->minWidth); sm<(int)(p->maxWidth); sm *= 2) {

    // do smooth
    pKernelSize = {2*sm+1,3};
    filtSum = 0.;
    for (int i=0;i<3;i++) {
      for (int j=0;j<2*sm+1;j++) {
        v = 1.-0.5*((j-sm)/(sm/2.355))*((j-sm)/(sm/2.355))+0.25*((j-sm)/(sm/2.355))*((j-sm)/(sm/2.355))*0.25*((j-sm)/(sm/2.355))*((j-sm)/(sm/2.355))-0.083*((j-sm)/(sm/2.355))*((j-sm)/(sm/2.355))*0.083*((j-sm)/(sm/2.355))*((j-sm)/(sm/2.355))*0.083*((j-sm)/(sm/2.355))*((j-sm)/(sm/2.355));
        h_kernel[i*(2*sm+1)+j] = w[i]*v*v;
        filtSum += w[i]*v*v;
      }
    }
    oAnchor = {0,1};
    for (int i=0;i<(2*sm+1)*3;i++)
    {
      h_kernel[i] /= filtSum;
    }
    checkCuda(cudaMemcpy(d_kernel,h_kernel,4*(2*sm+1)*3,cudaMemcpyHostToDevice));

    checkNpp(nppiFilterBorder_32f_C1R(
      p->d_dedisp,
      p->d_dedisp_step,
      oSrcSize,
      oSrcOffset,
      p->imbox,
      p->imbox_step,
      oSizeROI,
      d_kernel,
      pKernelSize,
      oAnchor,
      NPP_BORDER_REPLICATE
    ));

    // get rid of first and last DM, and maxWidth/2 from each edge
    checkCuda(cudaMemcpy2D(
      p->boxes+smit*(p->ndms-2)*p->boxes_step/sizeof(float),
      p->boxes_step,
      p->imbox+p->imbox_step/sizeof(float)+(int)(p->maxWidth)/2,
      p->imbox_step,
      sizeof(float)*p->ntime_out,
      p->ndms-2,
      cudaMemcpyDeviceToDevice
    ));

    smit++;
  }

  // Unlock scratch before normalization (calculate_stddev needs it)
  hella::unlock_d_scratch(p);
  hella::unlock_h_scratch(p);

  // Normalize each width trial to zero mean, unit variance.
  // Compute mean and std from the data.
  if (scale==1)
  {
    for (smit=0;smit<p->nboxcar;smit++)
    {
      const int height = p->ndms - 2;
      const int stride = p->boxes_step / sizeof(float);
      float* data = p->boxes + smit * height * stride;
      float measured_mean{};
      float measured_std = hella::calculate_stddev(p, data, p->ntime_out, height, stride, &measured_mean);

      if (measured_std > 0) {
        checkNpp(nppiAddC_32f_C1IR(
          (const Npp32f)(-1.0f * measured_mean),
          data,
          p->boxes_step,
          oOutROI
        ));
        checkNpp(nppiMulC_32f_C1IR(
          (const Npp32f)(1.0f / measured_std),
          data,
          p->boxes_step,
          oOutROI
        ));
      }
    }
  }

}
