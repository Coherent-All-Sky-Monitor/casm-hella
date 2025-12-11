/***************************************************************************
 *
 *   Copyright (C) 2025 TBA
 *   Copyright (C) 2025 Fourier Space
 *   Authors: TBA, Andrew Jameson
 *
 ***************************************************************************/

#include "hella/definitions.h"

#include <cuda_fp16.h>

#ifndef HELLA_CONVOLUTION_H
#define HELLA_CONVOLUTION_H

namespace hella
{
  /**
   * @brief handler for half-precision boxcar convolution from npp
   *
   * @note uses h and d scratch
   *
   * @param p
   * @param data
   * @param output
   * @param scfac
   * @param xw
   * @param yw
   * @param width
   * @param stride
   */
  void npp_convolve_handler(pinfo_t* p, half * data, half * output, float scfac, int xw, int yw, int width, int stride);

} // namespace hella

#endif // HELLA_CONVOLUTION_H
