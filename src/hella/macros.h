/***************************************************************************
 *
 *   Copyright (C) 2025 TBA
 *   Copyright (C) 2025 Fourier Space
 *   Authors: TBA, Andrew Jameson
 *
 ***************************************************************************/

#ifndef HELLA_MACROS_H
#define HELLA_MACROS_H

#include <cstdio>

#define checkCuda(err) { \
  if (err != cudaSuccess) { \
    fprintf(stderr, "FATAL CUDA Error in %s at line %d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  } \
}

#define checkNpp(err) { \
  if (err != NPP_NO_ERROR) { \
    fprintf(stderr, "NPP Error in %s at line %d: code=%d\n", \
            __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE); \
  } \
}

#endif // HELLA_MACROS_H