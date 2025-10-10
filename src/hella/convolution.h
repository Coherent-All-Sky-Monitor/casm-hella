/***************************************************************************
 *
 *   Copyright (C) 2025 TBA
 *   Copyright (C) 2025 Fourier Space
 *   Authors: TBA, Andrew Jameson
 *
 ***************************************************************************/

#include <cuda_fp16.h>

#ifndef HELLA_CONVOLUTION_H
#define HELLA_CONVOLUTION_H

namespace hella
{
  //! handler for half-precision boxcar convolution from npp
  void npp_convolve_handler(half * data, half * output, float scfac, int xw, int yw, int width, int stride);

} // namespace hella

#endif // HELLA_CONVOLUTION_H