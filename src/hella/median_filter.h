/***************************************************************************
 *
 *   Copyright (C) 2025 Vikram Ravi
 *   Copyright (C) 2025 Fourier Space
 *   Authors: Vikram Ravi, Andrew Jameson
 *
 ***************************************************************************/

#include "hella/definitions.h"

#ifndef HELLA_MEDIAN_FILTER_H
#define HELLA_MEDIAN_FILTER_H

namespace hella
{
  //! Function to swap two float values
  void swap(float *a, float *b);

  //! Function to find the median of an array
  float find_median(float arr[], int n);

  //! Function to apply median filter
  float median_filter(float *input, float *output, int size, int windowSize);

  /**
   * @brief Orchestrate host median filtering of bandpass.
   *
   * @note uses h_scratch
   *
   * @param d_bandpass bandpass in device memory with size NBATCH * NCHAN
   * @return float mean of the median filtered bandpass
   */
  float med_filter_bandpass(hella::pinfo_t* p, float * d_bandpass);

} // namespace hella

#endif // HELLA_MEDIAN_FILTER_H
