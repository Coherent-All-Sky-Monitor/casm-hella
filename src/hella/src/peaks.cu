/***************************************************************************
 *
 *   Copyright (C) 2025 TBA
 *   Copyright (C) 2025 Fourier Space
 *   Authors: TBA, Andrew Jameson
 *
 ***************************************************************************/

#include "hella/macros.h"
#include "hella/normalization.h"
#include "hella/peaks.h"

#include <cstdlib>
#include <sstream>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <spdlog/spdlog.h>

#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/host_vector.h>

void hella::find_peaks(hella::pinfo_t *p, int bm)
{
  float * dmt_ptr = thrust::raw_pointer_cast(&p->dmt[0]);
  float * d_outputs = thrust::raw_pointer_cast(&p->output_values[0]);
  int * d_idxs = thrust::raw_pointer_cast(&p->output_indices[0]);
  int n_found;
  p->npeaks = 0;
  float myStd;

  for (int sm=0;sm<p->nboxcar;sm++) {

    // measure rms - should be 1
    //calculate_stddev_float(float * d_data, int width, int height, int stride)
    if (sm==0) {
      // note: calculate_stddev_float use h and d scratch
      const int height = p->ndms - 2;
      const int stride = p->boxes_step / sizeof(float);
      float* data = p->boxes + sm * height * stride;
      float mean{};
      myStd = hella::calculate_stddev(p, data, p->ntime_out, height, stride, &mean);
      spdlog::debug("hella::find_peaks bm: {} mean: {} std: {}", bm, mean, myStd);
      // TODO(ldunn) I don't really understand how these standard deviation checks are supposed to function. I have reversed the order of these first two checks, so that
      // if myStd is below STD_DEV_VERY_LOW_THRESHOLD we set myStd to a high value and skip over doing the peak-finding. But anything else
      // that's between VERY_LOW_THRESHOLD and LOW_THRESHOLD will have myStd set to 1, which seems wrong!
      if (myStd<STD_DEV_VERY_LOW_THRESHOLD) myStd = 2.;
      if (myStd<STD_DEV_LOW_THRESHOLD) myStd = 1.;
    }
    if ((myStd<STD_DEV_HIGH_THRESHOLD && myStd>STD_DEV_VERY_HIGH_THRESHOLD) || DEBUG_ALWAYS_FIND_PEAKS) {


      // copy to thrust vector
      //cudaMemcpy(dmt_ptr,p->boxes+sm*(p->ndms-2)*p->boxes_step/sizeof(float),(p->ndms-2)*p->boxes_step,cudaMemcpyDeviceToDevice);
      checkCuda(cudaMemcpy2D(dmt_ptr,p->ntime_out*4,p->boxes+sm*(p->ndms-2)*p->boxes_step/sizeof(float),p->boxes_step,p->ntime_out*4,p->ndms-2,cudaMemcpyDeviceToDevice));

      // Find indices and values of points greater than the threshold
      //  thrust::copy(p->dmt.begin(), p->dmt.begin() + 20, std::ostream_iterator<float>(std::cout, " "));
      thrust::device_vector<int>::iterator end = thrust::copy_if(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(p->ntime_out*(p->ndms-2)),
        p->dmt.begin(),
        p->output_indices.begin(),
        thrust::placeholders::_1 > p->snr*myStd
      );
      n_found = end-p->output_indices.begin();
      if (p->npeaks + n_found > MAX_GIANTS)
        n_found = MAX_GIANTS - p->npeaks;

      thrust::copy(thrust::make_permutation_iterator(p->dmt.begin(), p->output_indices.begin()),
                   thrust::make_permutation_iterator(p->dmt.end(), p->output_indices.begin()+n_found),
                   p->output_values.begin());

      // copy to host
      checkCuda(cudaMemcpy(p->peaks+p->npeaks, d_outputs, n_found*sizeof(float), cudaMemcpyDeviceToHost));
      thrust::for_each(p->output_indices.begin(), p->output_indices.begin()+n_found, thrust::placeholders::_1 += (p->ndms-2)*sm*p->ntime_out);
      checkCuda(cudaMemcpy(p->h_idxs+p->npeaks, d_idxs, n_found*sizeof(int), cudaMemcpyDeviceToHost));

      // iterate npeaks
      p->npeaks += n_found;
    }
    else {
      spdlog::debug("hella::find_peaks Skipping beam {}, std dev={}", bm, myStd);
    }
  }

  // sort out stuff on host
  int tmp;
  int imax;
  //std::cout << p->npeaks << std::endl;
  if (p->out_npeaks+p->npeaks>MAX_GIANTS)
    imax = MAX_GIANTS;
  else
    imax = p->out_npeaks+p->npeaks;
  for (int i=p->out_npeaks;i<imax;i++)
  {
    //spdlog::trace("find_peaks: {} {} {}", p->h_idxs[i], (int)(p->h_idxs[i] % p->ntime_dd), (int)(p->h_idxs[i] / p->ntime_dd));
    p->out_peaks[i] = p->peaks[i-p->out_npeaks];
    p->out_beam[i] = bm;
    p->out_samp[i] = (int)(p->h_idxs[i-p->out_npeaks] % p->ntime_out);
    tmp = (int)(p->h_idxs[i-p->out_npeaks] / p->ntime_out);
    p->out_width[i] = (int)(tmp / (p->ndms-2));
    p->out_dm_idx[i] = (int)(tmp % (p->ndms-2)) + 1;

  }
  p->out_npeaks = imax;
}

void hella::clear_peaks(hella::pinfo_t *p)
{
  memset(p->out_peaks,0,MAX_GIANTS*sizeof(float));
  memset(p->out_beam,0,MAX_GIANTS*sizeof(int));
  memset(p->out_samp,0,MAX_GIANTS*sizeof(int));
  memset(p->out_width,0,MAX_GIANTS*sizeof(int));
  memset(p->out_dm_idx,0,MAX_GIANTS*sizeof(int));
  p->out_npeaks = 0;
}

void hella::output_peaks(hella::pinfo_t *p, int samp, int restart_socket)
{
  // if output is file or both
  if (p->out_format != 1) {

    FILE *fout{nullptr};
    fout=fopen(p->out_path,"a");
    if (!fout)
    {
      spdlog::error("failed to open p->out_path={} for appending", p->out_path);
      throw std::runtime_error("failed to open output file for appending");
    }

    if (p->out_npeaks > 0)
      spdlog::debug("S/N SAMP TIME WIDTH DM_IDX DM BEAM");

    for (int i=0;i<p->out_npeaks;i++) {
      if (i < 10)
        spdlog::debug("{} {} {} {} {} {} {}",p->out_peaks[i],p->out_samp[i]+samp,TIME_RESOLUTION*(p->out_samp[i]+samp),p->out_width[i],p->out_dm_idx[i],p->DMs[p->out_dm_idx[i]],p->out_beam[i]+p->BEAM0);

      // if (p->samp[i]>p->maxWidth/2 && p->samp[i]<=p->ntime_dd-p->maxWidth/2)
      //   fprintf(fout,"A %g %d %g %d %d %g %d\n",p->peaks[i],p->samp[i]+samp,262.144e-6*(p->samp[i]+samp),p->width[i],p->dm_idx[i],p->DMs[p->dm_idx[i]],bm);
      // else
      //   fprintf(fout,"B %g %d %g %d %d %g %d\n",p->peaks[i],p->samp[i]+samp,262.144e-6*(p->samp[i]+samp),p->width[i],p->dm_idx[i],p->DMs[p->dm_idx[i]],bm);
      fprintf(fout,"%g %d %d %g %d %d %g %d\n",p->out_peaks[i],p->out_samp[i]+samp,p->out_samp[i]+samp,TIME_RESOLUTION*(p->out_samp[i]+samp)/SECONDS_PER_DAY,p->out_width[i],p->out_dm_idx[i],p->DMs[p->out_dm_idx[i]],p->out_beam[i]+p->BEAM0);

    }
    fclose(fout);
  }

  // if output is socket or both
  if (p->out_format != 0) {

    // socket output
    std::ostringstream oss;
    oss.flush();
    oss.str("");
    int sstat=1;
    sockaddr_in m_addr;
    int m_sock = -1;

    // open socket
    spdlog::info("opening socket");
    memset ( &m_addr, 0, sizeof ( m_addr ) );
    m_sock = socket ( AF_INET, SOCK_STREAM, 0 );
    if (m_sock==-1) {
      spdlog::error("Socket exception: could not create socket");
      return;
    }

    // connect stuff
    m_addr.sin_family = AF_INET;
    m_addr.sin_port = htons ( p->coincidencer_port );
    m_addr.sin_addr.s_addr = inet_addr (p->coincidencer_host.c_str());
    sstat = connect ( m_sock, ( sockaddr * ) &m_addr, sizeof ( m_addr )) ;

    if (sstat!=0) {
      spdlog::error("Socket exception: could not open socket: {}", sstat);
      return;
    }
    else
      sstat=1;

    if (sstat && (m_sock != -1)) {
      oss << (int)(samp/p->gulp)+1 << std::endl;

      // record output
      for( int i=0; i<p->out_npeaks; i++ ) {
        oss << p->out_peaks[i] << " "
            << p->out_samp[i]+samp << " "
            << p->out_samp[i]+samp << " "
            << TIME_RESOLUTION*(p->out_samp[i]+samp)/SECONDS_PER_DAY << " "
            << p->out_width[i] << " "
            << p->out_dm_idx[i] << " "
            << p->DMs[p->out_dm_idx[i]] << " "
            << p->out_beam[i]+p->BEAM0 << std::endl;

      }

      std::string s = oss.str();
      spdlog::info("sending data");
      sstat = send ( m_sock, s.c_str(), s.size(), MSG_NOSIGNAL );
      if (sstat==-1) {
        spdlog::error("Socket exception: could not send cand");
        return;
      }

      oss.flush();
      oss.str("");

    }

    // close socket
    if (m_sock != -1) {
      spdlog::info("closing socket AFTER");
      sstat = ::close( m_sock );
      if (sstat!=0) {
        spdlog::error("Socket exception: could not close socket: {}", sstat);
        return;
      }
      m_sock = -1;
    }
  }
}
