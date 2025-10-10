/***************************************************************************
 *
 *   Copyright (C) 2025 TBA
 *   Copyright (C) 2025 Fourier Space
 *   Authors: TBA, Andrew Jameson
 *
 ***************************************************************************/

#ifndef HELLA_SIGPROC_H
#define HELLA_SIGPROC_H

#include <cstdio>

namespace hella {

  /**
   * @brief Simple read_header function for filterbank files
   *
   * This is a basic implementation - you may need to enhance it for your specific needs
   * @param fin
   * @return int
   */
  int read_header(FILE *fin);
}

#endif // HELLA_SIGPROC_H