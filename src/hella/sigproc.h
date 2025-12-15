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
   * @brief Open a sigproc filename file, read the header and return a file pointer starting at the data
   *
   * @param filename sigproc file name to open
   * @return FILE* pointer to the opened file, read to read data
   */
  FILE* open_filterbank_file(const char * filename);

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