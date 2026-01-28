/***************************************************************************
 *
 *   Copyright (C) 2025 Vikram Ravi
 *   Copyright (C) 2025 Fourier Space
 *   Authors: Vikram Ravi, Andrew Jameson
 *
 ***************************************************************************/

#include "hella/alloc.h"
#include "hella/definitions.h"
#include "hella/sigproc.h"

#include <cstring>
#include <cstdio>
#include <cstdlib>

namespace
{
  /**
  * @brief read a string from the input which looks like nchars-char[1-nchars]
  *
  * @param inputfile input file pointer
  * @param nbytes number of bytes read from the file pointer
  * @param string string that was read from the file pointer
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

  /**
   * @brief Compare two strings for equality
   *
   * @param string1 string being tested
   * @param string2 reference string be tested
   * @return int
   */
  int strings_equal(char *string1, const char *string2)
  {
    if (!strcmp(string1,string2)){
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
  filterbank_header_t hdr;
  if (!fin)
  {
    spdlog::error("Failed to open filterbank file: {}", filename);
    throw std::runtime_error("hella::open_filterbank_file failed to open input file");
  }
  int nbytes_header = hella::read_header(fin, &hdr);
  spdlog::info("Finished with header (nbytes {}) of input filterbank file: {}", nbytes_header, filename);
  return fin;
}

int hella::read_header(FILE *inputfile, filterbank_header_t *hdr)
{
  char string[80], message[80];
  int nbins{0}, itmp{0}, nbytes{0}, expecting_rawdatafile{0}, expecting_source_name{0};
  int expecting_frequency_table{0},channel_index{0};
  double period{0};

  spdlog::debug("hella::read_header inputfile={}", reinterpret_cast<void*>(inputfile));
  /* try to read in the first line of the header */
  get_string(inputfile,&nbytes, string);

  if (!strings_equal(string,"HEADER_START"))
  {
    /* the data file is not in standard format, rewind and return */
    return 0;
  }

  /* loop over and read remaining header lines until HEADER_END reached */
  while (1) {
    get_string(inputfile,&nbytes,string);
    if (strings_equal(string,"HEADER_END")) break;
    if (strings_equal(string,"rawdatafile")) {
      expecting_rawdatafile=1;
    } else if (strings_equal(string,"source_name")) {
      expecting_source_name=1;
    } else if (strings_equal(string,"FREQUENCY_START")) {
      expecting_frequency_table=1;
      channel_index=0;
    } else if (strings_equal(string,"FREQUENCY_END")) {
      expecting_frequency_table=0;
    } else if (strings_equal(string,"az_start")) {
      fread(&hdr->az_start,sizeof(hdr->az_start),1,inputfile);
    } else if (strings_equal(string,"za_start")) {
      fread(&hdr->za_start,sizeof(hdr->za_start),1,inputfile);
    } else if (strings_equal(string,"src_raj")) {
      fread(&hdr->src_raj,sizeof(hdr->src_raj),1,inputfile);
    } else if (strings_equal(string,"src_dej")) {
      fread(&hdr->src_dej,sizeof(hdr->src_dej),1,inputfile);
    } else if (strings_equal(string,"tstart")) {
      fread(&hdr->tstart,sizeof(hdr->tstart),1,inputfile);
    } else if (strings_equal(string,"tsamp")) {
      fread(&hdr->tsamp,sizeof(hdr->tsamp),1,inputfile);
    } else if (strings_equal(string,"period")) {
      fread(&hdr->period,sizeof(hdr->period),1,inputfile);
    } else if (strings_equal(string,"fch1")) {
      fread(&hdr->fch1,sizeof(hdr->fch1),1,inputfile);
    } else if (strings_equal(string,"fchannel")) {
      fread(&hdr->frequency_table[channel_index++],sizeof(double),1,inputfile);
      hdr->fch1=hdr->foff=0.0; /* set to 0.0 to signify that a table is in use */
    } else if (strings_equal(string,"foff")) {
      fread(&hdr->foff,sizeof(hdr->foff),1,inputfile);
    } else if (strings_equal(string,"nchans")) {
      fread(&hdr->nchans,sizeof(hdr->nchans),1,inputfile);
    } else if (strings_equal(string,"telescope_id")) {
      fread(&hdr->telescope_id,sizeof(hdr->telescope_id),1,inputfile);
    } else if (strings_equal(string,"machine_id")) {
      fread(&hdr->machine_id,sizeof(hdr->machine_id),1,inputfile);
    } else if (strings_equal(string,"data_type")) {
      fread(&hdr->data_type,sizeof(hdr->data_type),1,inputfile);
    } else if (strings_equal(string,"ibeam")) {
      fread(&hdr->ibeam,sizeof(hdr->ibeam),1,inputfile);
    } else if (strings_equal(string,"nbeams")) {
      fread(&hdr->nbeams,sizeof(hdr->nbeams),1,inputfile);
    } else if (strings_equal(string,"nbits")) {
      fread(&hdr->nbits,sizeof(hdr->nbits),1,inputfile);
    } else if (strings_equal(string,"barycentric")) {
      fread(&hdr->barycentric,sizeof(hdr->barycentric),1,inputfile);
    } else if (strings_equal(string,"pulsarcentric")) {
      fread(&hdr->pulsarcentric,sizeof(hdr->pulsarcentric),1,inputfile);
    } else if (strings_equal(string,"nbins")) {
      fread(&nbins,sizeof(nbins),1,inputfile);
    } else if (strings_equal(string,"nsamples")) {
      /* read this one only for backwards compatibility */
      fread(&itmp,sizeof(itmp),1,inputfile);
    } else if (strings_equal(string,"nifs")) {
      fread(&hdr->nifs,sizeof(hdr->nifs),1,inputfile);
    } else if (strings_equal(string,"npuls")) {
      fread(&hdr->npuls,sizeof(hdr->npuls),1,inputfile);
    } else if (strings_equal(string,"refdm")) {
      fread(&hdr->refdm,sizeof(hdr->refdm),1,inputfile);
    } else if (strings_equal(string,"signed")) {
      fread(&hdr->isign,sizeof(hdr->isign),1,inputfile);
    } else if (expecting_rawdatafile) {
      strcpy(hdr->raw_data_file,string);
      expecting_rawdatafile=0;
    } else if (expecting_source_name) {
      strcpy(hdr->source_name,string);
      expecting_source_name=0;
    } else {
      std::string message = fmt::format("read_header: - unknown parameter: {}", string);
      spdlog::error(message);
      throw std::runtime_error(message);
    }
  }

  if (hdr->isign < 0 && OSIGN > 0){
	  fprintf(stderr,"WARNING! You are reading unsigned numbers with a signed version of dspsr/sigproc\n");
  }
  if (hdr->isign > 0 && OSIGN < 0){
	  fprintf(stderr,"WARNING! You are reading signed numbers with a unsigned version of dspsr/sigproc\n");
  }

  spdlog::info("Parsed nchan={} nbeams={} nbits={} from filterbank header", hdr->nchans, hdr->nbeams, hdr->nbits);

  // add some additional checks
  if (hdr->nchans != NCHAN)
  {
    throw std::runtime_error("hella::read_header: mismatch between config and file: hdr->nchans=" + std::to_string(hdr->nchans) + " NCHAN=" + std::to_string(NCHAN));
  }
  if (hdr->nbeams != NBEAMS)
  {
    throw std::runtime_error("hella::read_header: mismatch between config and file: hdr->nbeams=" + std::to_string(hdr->nbeams) + " NBEAMS=" + std::to_string(NBEAMS));
  }
  if (hdr->nbits != 8)
  {
    throw std::runtime_error("hella::read_header: mismatch between config and file: hdr->nbits=" + std::to_string(hdr->nbits) + " hdr->nbits=8");
  }

  /* return total number of bytes read */
  return ftell(inputfile);
}
