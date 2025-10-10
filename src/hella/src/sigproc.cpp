/***************************************************************************
 *
 *   Copyright (C) 2025 TBA
 *   Copyright (C) 2025 Fourier Space
 *   Authors: TBA, Andrew Jameson
 *
 ***************************************************************************/

#include "hella/sigproc.h"

#include <cstring>
#include <cstdlib>

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
    *nbytes=sizeof(int);
    if (feof(inputfile)) exit(0);
    if (nchar>80 || nchar<1) return;
    fread(string, nchar, 1, inputfile);
    string[nchar]='\0';
    *nbytes+=nchar;
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
    if (strings_equal(string,"HEADER_END")) break;
  }
  /* return total number of bytes read */
  return ftell(inputfile);
}
