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
   * @brief Host function to orchestrate the normalization process on half input
   *
   * @note uses h_scratch and d_scratch
   *
   * @param p
   * @param data
   * @param width
   * @param height
   * @param stride
   * @return float
   */
  float calculate_stddev(pinfo_t *p, half * data, int width, int height, int stride);

  /**
   * @brief Host function to orchestrate the normalization process on float32 input
   *
   * @note uses h_scratch and d_scratch
   *
   * @param p
   * @param data
   * @param width
   * @param height
   * @param stride
   * @return float
   */
  float calculate_stddev_float(pinfo_t *p, float * data, int width, int height, int stride);

} // namespace hella

#endif // HELLA_NORMALIZATION_H