/***************************************************************************
 *
 *   Copyright (C) 2025 TBA
 *   Copyright (C) 2025 Fourier Space
 *   Authors: TBA, Andrew Jameson
 *
 ***************************************************************************/

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

  // function to orchestrate host median filtering of bandpass
  float med_filter_bandpass(float * d_bandpass);

} // namespace hella

#endif // HELLA_MEDIAN_FILTER_H