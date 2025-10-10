/***************************************************************************
 *
 *   Copyright (C) 2025 TBA
 *   Copyright (C) 2025 Fourier Space
 *   Authors: TBA, Andrew Jameson
 *
 ***************************************************************************/

#include "hella/dedisperse.h"
#include "hella/definitions.h"
#include "hella/macros.h"

// function to do all dedispersion stuff
void hella::dedisperse(pinfo *p, int beam)
{
  const size_t beam_stride = NCHAN * p->NTIME;
  const size_t beam_offset = beam_stride * beam;
  checkCuda(cudaMemcpy(p->d_inputPacked,p->d_data + beam_offset, beam_stride,cudaMemcpyDeviceToDevice));
  //cudaMemcpy(p->indata,p->d_data+beam*NCHAN*p->NTIME,NCHAN*p->NTIME,cudaMemcpyDeviceToHost);

  dedisp_error       derror;
  //const dedisp_byte* in = &((unsigned char *)(p->indata))[0];
  //dedisp_byte*       out = &((unsigned char *)(p->h_dedisp))[0];
  const dedisp_byte* in = (unsigned char *)(p->d_inputPacked);
  dedisp_byte*       out = (unsigned char *)(p->d_dedispPacked);
  dedisp_size        in_nbits = 8;
  dedisp_size        in_stride = NCHAN;// p->d_datapreT_step; //NCHAN * in_nbits/8;
  dedisp_size        out_nbits = 32;
  dedisp_size        out_stride = p->ntime_dedisp * out_nbits/8;
  unsigned           flags = 1 << 2;
  //unsigned           flags = 0;
  derror = dedisp_execute_adv(p->dedispersion_plan, p->NTIME,
                              in, in_nbits, in_stride,
                              out, out_nbits, out_stride,
                              flags);

  if (derror!=0)
    std::cout << "DEDISP ERROR " << derror << std::endl;
  //cudaMemcpy2D(p->d_dedisp,p->d_dedisp_step,p->h_dedisp,4*p->ntime_dd,4*p->ntime_dd,p->ndms,cudaMemcpyHostToDevice);
  checkCuda(cudaMemcpy2D(p->d_dedisp,p->d_dedisp_step,p->d_dedispPacked,4*p->ntime_dedisp,4*p->ntime_dd,p->ndms,cudaMemcpyDeviceToDevice));
}