/***************************************************************************
 *
 *   Copyright (C) 2025 TBA
 *   Copyright (C) 2025 Fourier Space
 *   Authors: TBA, Andrew Jameson
 *
 ***************************************************************************/

#include "hella/alloc.h"
#include "hella/definitions.h"
#include "hella/macros.h"
#include "hella/median_filter.h"

void hella::swap(float *a, float *b)
{
  float temp = *a;
  *a = *b;
  *b = temp;
}

float hella::find_median(float arr[], int n)
{
  // Sort the array using bubble sort (you can use faster algorithms)
  for (int i = 0; i < n - 1; i++) {
    for (int j = 0; j < n - i - 1; j++) {
      if (arr[j] > arr[j + 1]) {
        swap(&arr[j], &arr[j + 1]);
      }
    }
  }

  // Return the middle element if odd, or average of two middle elements if even
  if (n % 2 != 0)
    return (float)arr[n / 2];
  return (float)(arr[(n - 1) / 2] + arr[n / 2]) / 2.0;
}

float hella::median_filter(float *input, float *output, int size, int windowSize)
{
  if (windowSize % 2 == 0)
  {
    printf("Error: Window size must be odd.\n");
    return 0.;
  }

  int halfWindowSize = windowSize / 2;

  for (int i = halfWindowSize; i < size-halfWindowSize; i++)
  {
    float window[windowSize];
    int windowIndex = 0;

    // Populate the window array
    for (int j = i - halfWindowSize; j <= i + halfWindowSize; j++) {
      if (j >= 0 && j < size) {
        window[windowIndex] = input[j];
      } else {
        // Handle boundary cases (you can choose different strategies)
        window[windowIndex] = input[i]; // Repeat edge value
      }
      windowIndex++;
    }

    // Calculate and store the median for the current window
    output[i] = hella::find_median(window, windowSize);
  }

  // edge values
  for (int i=0;i<halfWindowSize;i++)
    output[i] = output[halfWindowSize];
  for (int i=size-halfWindowSize;i<size;i++)
    output[i] = output[size-halfWindowSize-1];

  // return mean
  float mn = 0.;
  for (int i=0;i<size;i++)
  {
    mn += output[i];
  }
  mn /= 1.*size;

  return mn;
}

float hella::med_filter_bandpass(hella::pinfo_t* p, float * d_bandpass)
{
  // acquire the host scratch space
  const size_t nval = NBATCH * NCHAN;
  const size_t bp_size = sizeof(float) * nval;
  hella::lock_h_scratch(p, bp_size * 2);

  auto hbp = reinterpret_cast<float*>(p->h_scratch);
  auto mhbp = hbp + nval;

  // copy the device bandpass to the host
  checkCuda(cudaMemcpy(hbp,d_bandpass,bp_size, cudaMemcpyDeviceToHost));

  // median filter the host bandpass with a window of NMEDFILT
  // AJ: I wonder what happens at the edges of each beam?
  float mn_bp = hella::median_filter(hbp,mhbp,nval,NMEDFILT);

  // copy the median filtered host bandpass back to the input device bandpass
  checkCuda(cudaMemcpy(d_bandpass,mhbp,bp_size,cudaMemcpyHostToDevice));
  hella::unlock_h_scratch(p);

  return mn_bp;
}
