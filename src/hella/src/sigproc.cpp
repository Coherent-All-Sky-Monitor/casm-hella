/***************************************************************************
 *
 *   Copyright (C) 2025 TBA
 *   Copyright (C) 2025 Fourier Space
 *   Authors: TBA, Andrew Jameson
 *
 ***************************************************************************/

#include "hella/alloc.h"
#include "hella/definitions.h"
#include "hella/sigproc.h"

#include <cstring>
#include <cstdio>
#include <cstdlib>

#define LIAM_FILTERBANK_HACK
namespace
{
  /**
  * @brief read a string from the input which looks like nchars-char[1-nchars]
  *
  * @param inputfile
  * @param nbytes
  * @param string
  */
  void get_string(FILE *inputfile, int *nbytes, char string[]) /* includefile */
  {
    int nchar;
    strcpy(string,"ERROR");
    fread(&nchar, sizeof(int), 1, inputfile);
    spdlog::trace("get_string nchar={}", nchar);
    *nbytes=sizeof(int);
    if (feof(inputfile))
      exit(0);
    if (nchar>80 || nchar<1)
      return;
    fread(string, nchar, 1, inputfile);
    string[nchar]='\0';
    spdlog::trace("get_string string={}", string);
    *nbytes+=nchar;
  }

  int get_int(FILE *inputfile)
  {
    int val{0};
    fread(&val, sizeof(int), 1, inputfile);
    spdlog::trace("get_int val={}", val);
    return val;
  }

  double get_double(FILE *inputfile)
  {
    double val{0};
    fread(&val, sizeof(double), 1, inputfile);
    spdlog::trace("get_int val={}", val);
    return val;
  }

  /**
  * @brief
  *
  * @param string1
  * @param string2
  * @return int
  */
  int strings_equal (char *string1, char *string2)
  {
    if (!strcmp(string1,string2)) {
      return 1;
    } else {
      return 0;
    }
  }
}

FILE* hella::open_filterbank_file(const char * filename)
{
  const char* modes = "rb";
  FILE* fin = fopen(filename, modes);
  int nbytes_header = hella::read_header(fin);
  fclose(fin);

  char *header{nullptr};
  hella::alloc_cpu<char>(&header, nbytes_header);
  fin = fopen(filename, modes);
  fread(header, sizeof(char), nbytes_header, fin);
  hella::release_cpu(&header);

  spdlog::trace("Finished with header (nbytes {}) of input filFile {}", nbytes_header, filename);
  return fin;
}

/**
 * @brief attempt to read in the general header info from a pulsar data file
 *
 * @param inputfile
 * @return int
 */
int hella::read_header (FILE *inputfile) /* includefile */
{
  char string[80], message[80];
  int itmp,nbytes,expecting_rawdatafile=0,expecting_source_name=0;
  int expecting_frequency_table=0,channel_index=0;

  /* try to read in the first line of the header */
  get_string(inputfile,&nbytes,string);

  if (!strings_equal(string,"HEADER_START"))
  {
    /* the data file is not in standard format, rewind and return */
    return 0;
  }

  /* loop over and read remaining header lines until HEADER_END reached */
  while (1) {

    get_string(inputfile,&nbytes,string);
    spdlog::trace("string={}", string);
    if (strings_equal(string,"HEADER_END")) break;

    #ifdef LIAM_FILTERBANK_HACK
    if (
      (strcmp(string, "machine_id") == 0) ||
      (strcmp(string, "telescope_id") == 0) ||
      (strcmp(string, "data_type") == 0) ||
      (strcmp(string, "barycentric") == 0) ||
      (strcmp(string, "pulsarcentric") == 0) ||
      (strcmp(string, "nbits") == 0) ||
      (strcmp(string, "nifs") == 0) ||
      (strcmp(string, "nchans") == 0) ||
      (strcmp(string, "ibeam") == 0) ||
      (strcmp(string, "nbeams") == 0)
    )
      get_int(inputfile);
    if (
      (strcmp(string, "fch1") == 0) ||
      (strcmp(string, "foff") == 0) ||
      (strcmp(string, "tsamp") == 0) ||
      (strcmp(string, "tstart") == 0)
    )
      get_double(inputfile);
    #endif
  }
  /* return total number of bytes read */
  return ftell(inputfile);
}
