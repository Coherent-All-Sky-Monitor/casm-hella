/***************************************************************************
 *
 *   Copyright (C) 2025 Vikram Ravi
 *   Copyright (C) 2025 Fourier Space
 *   Authors: Vikram Ravi, Andrew Jameson
 *
 ***************************************************************************/

#ifndef HELLA_SIGPROC_H
#define HELLA_SIGPROC_H

#include <cstdio>

// define the signedness for the 8-bit data type
#define OSIGN 1
#define SIGNED OSIGN < 0

namespace hella {

  //! convenience struct to hold the readable parameters from the filterbank header
  typedef struct filterbank_header
  {
    int machine_id;
    int telescope_id;
    int data_type;
    int nchans;
    int nbits;
    int nifs;
    int scan_number;
    int barycentric;
    int pulsarcentric;
    int ibeam;
    int nbeams;
    double tstart;
    double mjdobs;
    double tsamp;
    double fch1;
    double foff;
    double refdm;
    double az_start;
    double za_start;
    double src_raj;
    double src_dej;
    double period;
    char isign;
    long int npuls;
    char source_name[256];
    char raw_data_file[256];
    double frequency_table[4096];
  } filterbank_header_t;

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
   * @param fin input file handle from which to read the header
   * @param hdr structure into which header parameters are read
   * @return int number of bytes read from the file
   */
  int read_header(FILE *fin, filterbank_header_t *hdr);
}

#endif // HELLA_SIGPROC_H
