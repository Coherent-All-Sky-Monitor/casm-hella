/***************************************************************************
 *
 *   Copyright (C) 2025 TBA
 *   Copyright (C) 2025 Fourier Space
 *   Authors: TBA, Andrew Jameson
 *
 ***************************************************************************/

#include <cuda_fp16.h>

#ifndef HELLA_NORMALIZATION_H
#define HELLA_NORMALIZATION_H

namespace hella
{
  // Host function to orchestrate the normalization process
  float calculate_stddev(half * d_data, int width, int height, int stride);

  // Host function to orchestrate the normalization process
  float calculate_stddev_float(float * d_data, int width, int height, int stride);

} // namespace hella

#endif // HELLA_NORMALIZATION_H