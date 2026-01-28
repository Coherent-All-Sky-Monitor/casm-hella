/***************************************************************************
 *
 *   Copyright (C) 2025 TBA
 *   Copyright (C) 2025 Fourier Space
 *   Authors: TBA, Andrew Jameson
 *
 ***************************************************************************/

#include "hella/definitions.h"

#include <cuda_fp16.h>

#ifndef HELLA_NORMALIZATION_H
#define HELLA_NORMALIZATION_H

namespace hella
{

  /**
   * @brief Host function to calculate the standard deviation of a 2D array of values in device memory. Accepts either fp16 or fp32 data
   *
   * @note uses h_scratch and d_scratch
   *
   * @param p pointer to pipeline info struct 
   * @param data pointer to device memory buffer
   * @param width the width of the array in elements
   * @param height the height of the array in elements
   * @param stride the stride between rows of the array in elements
   * @param[out] mean_ret pointer to storage for computed mean. May be null. 
   *
   * @return the standard deviation calculated over all values in the array
   */
  template <typename T>
  float calculate_stddev(hella::pinfo_t *p, T * data, int width, int height, int stride, float *mean_ret = nullptr);
} // namespace hella

#endif // HELLA_NORMALIZATION_H
