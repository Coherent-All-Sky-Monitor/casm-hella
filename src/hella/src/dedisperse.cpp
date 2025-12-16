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

#include <dedisp.h>
#include <spdlog/spdlog.h>
#include <sstream>

void hella::dedisperse(hella::pinfo_t *p, int beam)
{
  const size_t beam_stride = NCHAN * p->NTIME;
  const size_t beam_offset = beam_stride * beam;
  checkCuda(cudaMemcpy(p->d_inputPacked,p->d_data + beam_offset, beam_stride,cudaMemcpyDeviceToDevice));
  cudaDeviceSynchronize();

  dedisp_error       derror;
  const dedisp_byte* in = reinterpret_cast<unsigned char *>(p->d_inputPacked);
  dedisp_byte*       out = reinterpret_cast<unsigned char *>(p->d_dedispPacked);
  dedisp_size        in_nbits = 8;
  dedisp_size        in_stride = NCHAN;// p->d_datapreT_step; //NCHAN * in_nbits/8;
  dedisp_size        out_nbits = 32;
  dedisp_size        out_stride = p->ntime_dedisp * out_nbits/8;
  unsigned           flags = 1 << 2;
  derror = dedisp_execute_adv(
    p->dedispersion_plan,
    p->NTIME,
    in, in_nbits, in_stride,
    out, out_nbits, out_stride,
    flags
  );

  if (derror != DEDISP_NO_ERROR)
  {
    std::ostringstream oss;
    oss << "hella::dedisperse dedisp_execute_adv failed: " << get_dedisp_error(derror);
    spdlog::error(oss.str());
    throw std::runtime_error(oss.str());
  }
  checkCuda(cudaMemcpy2D(p->d_dedisp,p->d_dedisp_step,p->d_dedispPacked,4*p->ntime_dedisp,4*p->ntime_dd,p->ndms,cudaMemcpyDeviceToDevice));
}

std::string hella::get_dedisp_error(int dedisp_error_code)
{
  switch (dedisp_error_code)
  {
    case DEDISP_NO_ERROR:
      return "no error";
	  case DEDISP_MEM_ALLOC_FAILED:
        return "mem alloc failed";
	  case DEDISP_MEM_COPY_FAILED:
      return "mem copy failed";
	  case DEDISP_NCHANS_EXCEEDS_LIMIT:
      return "nchans exceeds limit";
	  case DEDISP_INVALID_PLAN:
      return "invalid plan";
	  case DEDISP_INVALID_POINTER:
      return "invalid pointer";
	  case DEDISP_INVALID_STRIDE:
      return "invalid stride";
	  case DEDISP_NO_DM_LIST_SET:
      return "no dm list set";
	  case DEDISP_TOO_FEW_NSAMPS:
      return "too few nsamps";
	  case DEDISP_INVALID_FLAG_COMBINATION:
      return "invalid flag combination";
	  case DEDISP_UNSUPPORTED_IN_NBITS:
      return "unsupported in nbits";
	  case DEDISP_UNSUPPORTED_OUT_NBITS:
      return "unsupported out nbits";
	  case DEDISP_INVALID_DEVICE_INDEX:
      return "invalid device index";
	  case DEDISP_DEVICE_ALREADY_SET:
      return "device already set";
	  case DEDISP_PRIOR_GPU_ERROR:
      return "prior gpu error";
	  case DEDISP_INTERNAL_GPU_ERROR:
      return "internal gpu error";
	  case DEDISP_UNKNOWN_ERROR:
      return "unknown error";
    default:
      return "unknown code";
  }
}
