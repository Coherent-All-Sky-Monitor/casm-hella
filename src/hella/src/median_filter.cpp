/***************************************************************************
 *
 *   Copyright (C) 2025 TBA
 *   Copyright (C) 2025 Fourier Space
 *   Authors: TBA, Andrew Jameson
 *
 ***************************************************************************/

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

// function to orchestrate host median filtering of bandpass
float hella::med_filter_bandpass(float * d_bandpass)
{

  float * hbp = (float *)malloc(sizeof(float)*NBATCH*NCHAN);
  float * mhbp = (float *)malloc(sizeof(float)*NBATCH*NCHAN);
  checkCuda(cudaMemcpy(hbp,d_bandpass,sizeof(float)*NBATCH*NCHAN,cudaMemcpyDeviceToHost));
  float mn_bp = hella::median_filter(hbp,mhbp,NBATCH*NCHAN,NMEDFILT);
  checkCuda(cudaMemcpy(d_bandpass,mhbp,sizeof(float)*NBATCH*NCHAN,cudaMemcpyHostToDevice));

  free(hbp);
  free(mhbp);

  return mn_bp;

}