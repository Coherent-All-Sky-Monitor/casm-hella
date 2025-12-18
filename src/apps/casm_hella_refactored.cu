/***************************************************************************
 *
 *   Copyright (C) 2025 TBA
 *   Copyright (C) 2025 Fourier Space
 *   Authors: TBA, Andrew Jameson
 *
 ***************************************************************************/

#include "hella/alloc.h"
#include "hella/dada.h"
#include "hella/definitions.h"
#include "hella/dedisperse.h"
#include "hella/flagger.h"
#include "hella/peaks.h"
#include "hella/sigproc.h"
#include "hella/smooth.h"
#include "hella/transpose.h"


#include <time.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <dedisp.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/select.h>

#include "ipcio.h"
#include "dada_affinity.h"
#include "ascii_header.h"

#include <cuda.h>
#include "cuda_fp16.h"
#include <cublas_v2.h>

#include <cstdio>
#include <spdlog/spdlog.h>

#include "hella/definitions.h"
#include "hella/macros.h"

int finished = 0;

//! function to initialize everything based on a config file
void initialize(FILE *fconf, hella::pinfo_t* p) {

  // read input file line by line
  p->nscrunches = 0;
  p->BEAM_OFFSET = 0;
  char * line = NULL;
  ssize_t read;
  size_t len = 0;
  char c1[20], c2[500];
  p->flag1 = -1;
  p->flag2 = -1;
  p->spec_min = -0.05;
  p->spec_max = 0.15;
  p->output_bandpass = 0;
  int gpu_id = 0;

  while (!feof(fconf)) {

    read = getline(&line, &len, fconf);
    sscanf(line,"%s %s",c1,c2);

    if (strcmp(c1,"INPUT")==0) {
      if (strcmp(c2,"DADA")==0) p->inp_format=0;
      if (strcmp(c2,"FILE")==0) p->inp_format=1;
      if (strcmp(c2,"FILTERBANK")==0) p->inp_format=2;
      if (strcmp(c2,"CANDIDATE")==0) p->inp_format=3;
      spdlog::info("Using input format {}",p->inp_format);
    }
    if (strcmp(c1,"OUTPUT")==0) {
      if (strcmp(c2,"FILE")==0) p->out_format=0;
      if (strcmp(c2,"SOCKET")==0) p->out_format=1;
      if (strcmp(c2,"BOTH")==0) p->out_format=2;
      spdlog::info("Using output format {}",p->out_format);
    }
    if (strcmp(c1,"HOST")==0) {
      p->coincidencer_host = c2;
    }
    if (strcmp(c1,"BEAM0")==0) {
      p->BEAM0 = atoi(c2);
    }
    if (strcmp(c1,"PORT")==0) {
      p->coincidencer_port = atoi(c2);
    }
    if (strcmp(c1,"OUTPUTPATH")==0) {
      strcpy(p->out_path,c2);
    }
    if (strcmp(c1,"BEAM_OFFSET")==0) {
      p->BEAM_OFFSET = atoi(c2);
    }

    if (strcmp(c1,"BEAMFLAGS")==0) {
      strcpy(p->beamflags,c2);
    }
    if (strcmp(c1,"SPECFLAGS")==0) {
      strcpy(p->specflags,c2);
    }

    if (strcmp(c1,"INPUT_PATH")==0) {
      strcpy(p->inp_path,c2);
      spdlog::info("Input path: {}",p->inp_path);
    }

    if (strcmp(c1,"DADA_OUT")==0) {
      strcpy(p->dada_out,c2);
      spdlog::info("DADA out: {}",p->dada_out);
    }

    if (strcmp(c1,"DM_MIN")==0)
      p->minDM=atof(c2);
    if (strcmp(c1,"DM_MAX")==0)
      p->maxDM=atof(c2);
    if (strcmp(c1,"WIDTH_MIN")==0) {
      p->minWidth=atoi(c2);
    }
    if (strcmp(c1,"WIDTH_MAX")==0) {
      p->maxWidth=atoi(c2);
    }
    if (strcmp(c1,"SNR")==0) {
      p->snr=atof(c2);
    }
    if (strcmp(c1,"GULP")==0)
      p->gulp=atoi(c2);
    if (strcmp(c1,"FLAG1")==0)
      p->flag1=atoi(c2);
    if (strcmp(c1,"FLAG2")==0)
      p->flag2=atoi(c2);
    if (strcmp(c1,"SPEC_MIN")==0)
      p->spec_min=atof(c2);
    if (strcmp(c1,"SPEC_MAX")==0)
      p->spec_max=atof(c2);
    if (strcmp(c1,"OUTPUT_BANDPASS")==0)
      p->output_bandpass=atoi(c2);
    if (strcmp(c1,"GPU")==0)
      gpu_id = atoi(c2);

    if (strcmp(c1,"SCRUNCH")==0) {

      p->nscrunches = atoi(c2);
      p->scrunches = reinterpret_cast<hella::scrunch *>(malloc(p->nscrunches*sizeof(hella::scrunch)));

      for (int i=0;i<p->nscrunches;i++) {
        read = getline(&line, &len, fconf);
        sscanf(line,"%d %d %f %d",&(p->scrunches[i].tscrunch),&(p->scrunches[i].fscrunch),&(p->scrunches[i].thresh),&(p->scrunches[i].nits));
        spdlog::info("Have a scrunch with {} {} {} {}",p->scrunches[i].tscrunch,p->scrunches[i].fscrunch,p->scrunches[i].thresh,p->scrunches[i].nits);
      }

    }
  }
  fclose(fconf);

  // set GPU ID
  cudaSetDevice(gpu_id);
  int current_device{-1};
  cudaGetDevice(&current_device);
  if (current_device != gpu_id)
  {
    spdlog::error("failed to select GPU ID {}", gpu_id);
    exit(EXIT_FAILURE);
  }
  spdlog::info("Using GPU ID {}",current_device);

  // derived parameters
  p->NTIME=p->gulp;
  p->rewind=0;
  int i=(int)(p->minWidth), j=0;
  while (i<(int)(p->maxWidth))  {
    i *= 2;
    j += 1;
  }
  p->nboxcar=j;
  spdlog::info("Search parameters: DM range {} to {}, WIDTHS {} to {} ({} trials), SNR {}", p->minDM,p->maxDM,p->minWidth,p->maxWidth,p->nboxcar,p->snr);
  if (p->out_format != 0)
    spdlog::info("Outputting to socket {}:{}",p->coincidencer_host.c_str(),p->coincidencer_port);
  if (p->out_format != 1)
    spdlog::info("Outputting to text file {}",p->out_path);

  if (DEBUG_ALWAYS_FIND_PEAKS)
  {
    spdlog::warn("Standard deviation thresholds will be ignored during peak-finding");
  }

  // set up DM plan
  spdlog::info("Creating a dedispersion plan nchans={} dt={} f0={} df={}\n", NCHAN, TIME_RESOLUTION, CENTER_FREQ/1e6,FREQ_CHANNEL_WIDTH);
  dedisp_create_plan(&p->dedispersion_plan,NCHAN,TIME_RESOLUTION,CENTER_FREQ/1e6,FREQ_CHANNEL_WIDTH);
  // generate DM list
  dedisp_generate_dm_list(p->dedispersion_plan,p->minDM,p->maxDM,40,TOL);
  p->DMs = dedisp_get_dm_list(p->dedispersion_plan);
  p->ndms = dedisp_get_dm_count(p->dedispersion_plan);
  p->ntime_dd = p->NTIME - dedisp_get_max_delay(p->dedispersion_plan);
  p->ntime_out = p->ntime_dd - p->maxWidth;
  p->ntime_dedisp = p->ntime_dd;
  // modify NTIME and ntime_dd in case of non-text input
  int oo;
  if (p->inp_format == 0 || p->inp_format == 2)
  {
    p->NTIME = p->gulp + dedisp_get_max_delay(p->dedispersion_plan) + p->maxWidth;
    oo = 32*((int)(p->NTIME/32)+1);
    p->NTIME = oo;
    p->ntime_dedisp = oo-dedisp_get_max_delay(p->dedispersion_plan);
    p->ntime_dd = p->gulp + p->maxWidth;
    p->ntime_out = p->gulp;
    spdlog::debug("dedisp_get_max_delay(p->dedispersion_plan)={}", dedisp_get_max_delay(p->dedispersion_plan));
  }

  spdlog::info("sizes NCHAN={} NBATCH={} p->NTIME={} p->gulp={} NBEAMS={} ndms={} ntime_dedisp={} ntime_dd={}",
   NCHAN, NBATCH, p->NTIME, p->gulp, NBEAMS, p->ndms, p->ntime_dedisp, p->ntime_dd);

  // allocate everything
  const size_t nrewind = static_cast<size_t>(p->NTIME) - p->gulp;
  const size_t nrewind_chan_beams = nrewind * NCHAN * NBEAMS;
  const size_t ntime_chan_beams = static_cast<size_t>(p->NTIME) * NCHAN * NBEAMS;

  hella::alloc_cpu<float>(&p->h_flagSpec, NCHAN * NBATCH);
  hella::alloc_gpu<float>(&p->d_flagSpec, NCHAN * NBATCH);
  hella::alloc_cpu<unsigned char>(&p->rewinds, nrewind * NCHAN * NBEAMS);
  memset(p->rewinds, 0, nrewind * NCHAN * NBEAMS);

  hella::alloc_cpu_host<unsigned char>(&p->data, p->NTIME * NBEAMS * NCHAN);

  hella::alloc_cpu<float>(&p->h_dataF, p->NTIME * NCHAN);
  hella::alloc_gpu<float>(&p->d_dedispPacked, p->ndms * p->ntime_dedisp);
  hella::alloc_gpu<unsigned char>(&p->d_inputPacked, p->NTIME * NCHAN);
  hella::alloc_gpu<unsigned char>(&p->d_data, p->NTIME * NBEAMS * NCHAN);
  if (p->inp_format == 0) // DADA input, gulps are fp16 FT ordered
  {
    p->gulp_nbyte = sizeof(half);
    hella::alloc_gpu<half>(reinterpret_cast<half **>(&p->d_gulp), p->gulp * NBEAMS * NCHAN);
  }
  else // Otherwise gulps are u8 TF ordered
  {
    p->gulp_nbyte = sizeof(unsigned char);
    hella::alloc_gpu<unsigned char>(&p->d_gulp, p->gulp * NBEAMS * NCHAN); 
  }

  hella::alloc_gpu_pitch<half>(&p->batch, reinterpret_cast<size_t*>(&p->batch_stride), p->gulp, NBATCH * NCHAN);
  hella::alloc_gpu_pitch<half>(&p->mask, reinterpret_cast<size_t*>(&p->batch_stride), p->gulp, NBATCH * NCHAN);
  p->batch_stride = p->batch_stride / sizeof(half);

  hella::alloc_gpu<half>(&p->d_smooth, p->batch_stride * NBATCH * NCHAN);
  p->d_dedisp = nppiMalloc_32f_C1(p->ntime_dd,p->ndms,&(p->d_dedisp_step));
  if (!p->d_dedisp)
  {
    fprintf(stderr, "nppiMalloc_32f_C1 failed to allocate width=%d height=%d\n", p->ntime_dd,p->ndms);
    exit(1);
  }

  hella::alloc_gpu<float>(&p->d_ts, p->NTIME * NBATCH);
  hella::alloc_gpu<float>(&p->d_bpout, NBATCH * NCHAN);

  spdlog::info("Will use {} DM trials, output {} times, process {} times with stride {}",p->ndms,p->ntime_dd,p->NTIME,p->batch_stride);

  // boxcars
  p->boxes = nppiMalloc_32f_C1(p->ntime_out,(p->ndms-2)*p->nboxcar,&(p->boxes_step));
  p->imbox = nppiMalloc_32f_C1(p->ntime_dd,p->ndms,&(p->imbox_step));
  hella::alloc_cpu<float>(&p->stds, p->nboxcar);
  p->mean = BOXCAR_MEAN;
  p->stds[0] = BOXCAR_STD_0;
  p->stds[1] = BOXCAR_STD_1;
  p->stds[2] = BOXCAR_STD_2;
  p->stds[3] = BOXCAR_STD_3;
  p->stds[4] = BOXCAR_STD_4;
  p->stds[5] = BOXCAR_STD_5;
  p->stds[6] = BOXCAR_STD_6;

  // peak finding
  p->dmt.resize((p->ndms-2)*p->ntime_out);
  p->output_indices.resize((p->ndms-2)*p->ntime_out);
  p->output_values.resize((p->ndms-2)*p->ntime_out);

  hella::alloc_cpu<int>(&p->h_idxs, (p->ndms-2)*p->ntime_out);
  hella::alloc_cpu<int>(&p->beam, MAX_GIANTS);
  hella::alloc_cpu<int>(&p->width, MAX_GIANTS);
  hella::alloc_cpu<int>(&p->dm_idx, MAX_GIANTS);
  hella::alloc_cpu<int>(&p->samp, MAX_GIANTS);
  hella::alloc_cpu<float>(&p->peaks, MAX_GIANTS);

  hella::alloc_cpu<int>(&p->out_beam, MAX_GIANTS);
  hella::alloc_cpu<int>(&p->out_samp, MAX_GIANTS);
  hella::alloc_cpu<int>(&p->out_width, MAX_GIANTS);
  hella::alloc_cpu<int>(&p->out_dm_idx, MAX_GIANTS);
  hella::alloc_cpu<float>(&p->out_peaks, MAX_GIANTS);

  // flag timing
  p->fcpy=0.;
  p->fprep=0.;
  p->fflag=0.;
  p->fapply=0.;
  p->t1=0.;
  p->t2=0.;
  p->t3=0.;
  p->t4=0.;
  p->t5=0.;
  p->t6=0.;
  p->t7=0.;
  p->t8=0.;
  p->t9=0.;

}

// deallocate everything
void deallocator(hella::pinfo_t * p)
{
  spdlog::info("deallocating pinfo_t struct");
  hella::release_cpu_host(&p->data);
  hella::release_cpu(&p->h_dataF);
  hella::release_gpu(&p->d_bpout);
  hella::release_gpu(&p->d_data);
  hella::release_gpu(&p->batch);
  hella::release_gpu(&p->mask);
  hella::release_gpu(&p->d_dedisp);
  hella::release_gpu(&p->boxes);
  hella::release_gpu(&p->d_dedispPacked);
  hella::release_gpu(&p->d_inputPacked);
  p->dmt.clear();
  p->output_indices.clear();
  p->output_values.clear();

  hella::release_cpu(&p->h_idxs);
  hella::release_cpu(&p->beam);
  hella::release_cpu(&p->width);
  hella::release_cpu(&p->dm_idx);
  hella::release_cpu(&p->samp);
  hella::release_cpu(&p->peaks);
  hella::release_cpu(&p->stds);
  hella::release_cpu(&p->rewinds);
}

void help() {

  spdlog::info("Usage: pipeline [options] <config file>");
  spdlog::info("options:");
  spdlog::info("   -i core     bind processing to the specified cpu core");
  spdlog::info("   -v          increase the logging verbosity");
  spdlog::info("");
  spdlog::info("Everything is in the config file. Specific parameters include: ");
  spdlog::info("INPUT <DADA or FILE or FILTERBANK>");
  spdlog::info("INPUT_PATH <dada buffer or full path to filterbank file>");
  spdlog::info("DADA_OUT <dada buffer>");
  spdlog::info("BEAM_OFFSET <offset in number of beams in input dada buffer>");
  spdlog::info("DM_MIN <min DM of search>");
  spdlog::info("DM_MAX <max DM of search>");
  spdlog::info("WIDTH_MIN <min width of search>");
  spdlog::info("WIDTH_MAX <max width of search>");
  spdlog::info("SNR <SNR threshold for search>");
  spdlog::info("GULP <base gulp size>");
  spdlog::info("BEAMFLAGS <full path to beam flags output>");
  spdlog::info("SPECFLAGS <full path to spec flags output>");
  spdlog::info("OUTPUT <FILE or SOCKET of BOTH>");
  spdlog::info("OUTPUTPATH <path to output file>");
  spdlog::info("HOST <ip of T2 host>");
  spdlog::info("PORT <T2 port>");
  spdlog::info("GPU <GPU ID 0 or 1>");
  spdlog::info("BEAM0 <first beam in output>");
  spdlog::info("OUTPUT_BANDPASS <0 or 1 or 2>");
  spdlog::info("SPEC_MAX <max thresh in spec flagging>");
  spdlog::info("SPEC_MAX <max thresh in spec flagging>");

  spdlog::info("SCRUNCH <number of scrunches>");
  spdlog::info("<time scrunch> <frequency scrunch> <flagging threshold> <number of iterations>");
  spdlog::info("repeat the above as many times as you like for different parameters");

}

/************ END *************/



// code to empirically measure thresholds
/*
// curand stuff
__global__ void setup_kernel(curandState* state, uint64_t seed)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, tid, 0, &state[tid]);
}
__global__ void generate_randoms(curandState* globalState, float* randoms)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curandState localState = globalState[tid];
    randoms[tid * 2 + 0] = curand_normal(&localState);
    randoms[tid * 2 + 1] = curand_normal(&localState);
}

void measure_thresholds(hella::pinfo_t *p) {

  // generate data
  spdlog::info("THRESHOLD: Generating random values");
  int threads = 256;
  int blocks = (NCHAN/256)*p->NTIME / 2;
  int threadCount = blocks * threads;
  int N = blocks * threads * 2;
  curandState* dev_curand_states;
  float* randomValues;
  cudaMalloc(&dev_curand_states, threadCount * sizeof(curandState));
  cudaMalloc(&randomValues, N * sizeof(float));
  setup_kernel<<<blocks, threads>>>(dev_curand_states, time(NULL));
  generate_randoms<<<blocks, threads>>>(dev_curand_states, randomValues);

  // prepare for dedispersion
  spdlog::info("THRESHOLD: dedisperse and smooth");
  NppiSize preROI = {NCHAN,p->NTIME};
  cudaMemcpy(p->dataFT,randomValues,4*N,cudaMemcpyDeviceToDevice);
  nppiScale_32f8u_C1R(p->dataFT,p->dataFT_step,p->d_datapreT,p->d_datapreT_step,preROI,-4.,10.);
  cudaMemcpy2D(p->data,NCHAN,p->d_datapreT,p->d_datapreT_step,NCHAN,p->NTIME,cudaMemcpyDeviceToHost);

  // dedisperse
  hella::dedisperse(p);

  // smooth
  smooth(p,0);

  // measure stats
  spdlog::info("THRESHOLD: measure stats");
  Npp64f *pMean, *pStd;
  NppiSize oSizeROI = {p->ntime_dd,p->ndms};
  int nBufferSize;
  Npp8u * pDeviceBuffer;
  nppiMeanStdDevGetBufferHostSize_32f_C1R(oSizeROI, &nBufferSize);
  cudaMalloc((void **)(&pDeviceBuffer), nBufferSize);
  cudaMalloc((void **)(&pMean), p->nboxcar*8);
  cudaMalloc((void **)(&pStd), p->nboxcar*8);

  for (int i=0;i<p->nboxcar;i++)
    nppiMean_StdDev_32f_C1R(p->boxes+i*p->ntime_dd*p->ndms,p->boxes_step,oSizeROI,pDeviceBuffer,pMean+i,pStd+i);

  double *hMean, *hStd;
  hMean = (double *)malloc(sizeof(double)*p->nboxcar);
  hStd = (double *)malloc(sizeof(double)*p->nboxcar);
  cudaMemcpy(hMean,pMean,sizeof(double)*p->nboxcar,cudaMemcpyDeviceToHost);
  cudaMemcpy(hStd,pStd,sizeof(double)*p->nboxcar,cudaMemcpyDeviceToHost);

  spdlog::info("(Boxcar) Mean Std");
  for (int i=0;i<p->nboxcar;i++)
    spdlog::info("(%d) %g %g\n",i,hMean[i],hStd[i]);

  // scale sigmas by 0.95 to accommodate reduction at increased DM due to more co-added data.

  cudaFree(pDeviceBuffer);
  cudaFree(pMean);
  cudaFree(pStd);
  cudaFree(dev_curand_states);
  cudaFree(randomValues);
  free(hMean);
  free(hStd);

}
*/

// deals with data IO
int main(int argc, char *argv[]) try
{
  // parse command line
  FILE *fconf{nullptr};
  int core = -1;
  int verbosity = 0;

  opterr = 0;
  int c{};

  while ((c = getopt(argc, argv, "c:i:hv")) != EOF)
  {
    switch (c)
    {
      case 'c':
        spdlog::info("Getting config from {}", optarg);
        fconf = fopen(optarg, "r");
        break;

      case 'h':
        help();
        exit(EXIT_SUCCESS);

      case 'i':
        core = atoi(optarg);
        break;

      case 'v':
        verbosity++;
        spdlog::info("verbosity now {}", verbosity);
        if (verbosity == 1)
        {
          spdlog::info("set_level(spdlog::level::debug)");
          spdlog::set_level(spdlog::level::debug);
        }
        else if (verbosity >= 2)
        {
          spdlog::info("set_level(spdlog::level::trace)");
          spdlog::set_level(spdlog::level::trace);
        }
        else
        {
          spdlog::info("set_level(spdlog::level::info)");
          // spdlog::set_level(spdlog::level::info);
        }
        break;

      default:
        spdlog::error("Unrecognised option: {}", static_cast<char>(optopt));
        help();
        return EXIT_FAILURE;
    }
  }

  // Bind to cpu core
  if (core >= 0)
  {
    spdlog::debug("binding to core {}", core);
    if (dada_bind_thread_to_core(core) < 0)
    {
      spdlog::error("failed to bind to core {}", core);
    }
  }

  // set up pipeline, allocate appropriate mem
  hella::pinfo_t p{};
  float tflags = 0.;

  initialize(fconf,&p);
  FILE *fin{nullptr}, *ftest{nullptr};

  // allocate these arrays after initialize is called since p.NTIME can be adjusted there
  unsigned char * hodata{nullptr};
  hella::alloc_cpu<unsigned char>(&hodata, p.NTIME * NCHAN);
  float * h_ts = (float *)malloc(sizeof(float)*p.NTIME*NBATCH);

  if (p.inp_format==3) {

    // read header
    fin=fopen(p.inp_path,"rb");
    hella::filterbank_header_t hdr;
    int nbytes_header = hella::read_header(fin, &hdr);
    fclose(fin);
    char * header{nullptr};
    hella::alloc_cpu<char>(&header, nbytes_header);
    fin = fopen(p.inp_path,"rb");
    fread(header, sizeof(char), nbytes_header, fin);
    hella::release_cpu(&header);
    spdlog::info("Finished with header (nbytes {}) of input filFile {}",nbytes_header,p.inp_path);

    // read data
    fread(p.data,sizeof(char),p.NTIME*NCHAN,fin);
    if (NBEAMS>1) {
      for (int i=1;i<NBEAMS;i++)
        memcpy(p.data+i*p.NTIME*NCHAN,p.data,p.NTIME*NCHAN);
    }
    cudaMemcpy(p.d_data,p.data,NBEAMS*p.NTIME*NCHAN,cudaMemcpyHostToDevice);
    fclose(fin);

    // flag it
    hella::fast_flagger(&p);

    // output data
    cudaMemcpy(hodata,p.d_data+2*NCHAN*p.NTIME,NCHAN*p.NTIME,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ts,p.d_ts,4*NBATCH*p.NTIME,cudaMemcpyDeviceToHost);
    ftest = fopen("image.out","w");
    for (int i=0;i<NCHAN*p.NTIME;i++)
      fprintf(ftest,"%f\n",(float)(hodata[i]));
    fclose(ftest);
    ftest = fopen("ts.out","w");
    for (int i=0;i<NBATCH*p.NTIME;i++)
      fprintf(ftest,"%f\n",h_ts[i]);
    fclose(ftest);
    ftest = fopen("flags.out","w");
    for (int i=0;i<NBATCH*NCHAN;i++)
      fprintf(ftest,"%f\n",p.h_flagSpec[i]);
    fclose(ftest);


    for (int i=0;i<NCHAN*NBATCH;i++) {
      tflags += (1.*p.NTIME*p.h_flagSpec[i]);
    }

    spdlog::info("TOT FLAGS {}", tflags);

    exit(1);

  }

  // begin read of data
  float v;
  unsigned char * sigproc_buf{nullptr};
  if (p.inp_format == 2)
  {
    spdlog::info("allocating sigproc_buf");
    hella::alloc_cpu<unsigned char>(&sigproc_buf, p.gulp * NCHAN * NBEAMS * 2);
  }

  // DADA Header plus Data Unit
  dada_hdu_t* hdu_in{nullptr};

  switch (p.inp_format)
  {
    case 0: // dada input
      hdu_in = hella::hdu_connect_read(&p);
      break;

    case 1: // text file input
      fin=fopen(p.inp_path,"r");
      for (int i=0;i<p.NTIME*NCHAN;i++) {
        fscanf(fin,"%f\n",&v);
        p.data[i] = (unsigned char)(v);
      }
      fclose(fin);
      break;

    case 2: // filterbank input
      fin = hella::open_filterbank_file(p.inp_path);
      break;

    default:
      spdlog::error("unsupported input format: {}", p.inp_format);
      return EXIT_FAILURE;
  }

  spdlog::info("Starting...");
  int samp = 0;
  if (p.inp_format!=1)
    samp = -(p.NTIME-p.gulp) + (int)(p.maxWidth)/2;
  int gulp = 0;

  //measure_thresholds(&p);

  // set up output
  // FILE *fout{nullptr}, *fbeam{nullptr};
  FILE *fspec{nullptr};
  int beamflags[NBEAMS], specflags[NCHAN];
  int bm{0};

  // loop over data gulps, figuring out at the end if we're finished

  // dada stuff
  char * block{nullptr};
  uint64_t bytes_read = 0;
  uint64_t block_id{0}, written{0};

  // timer stuff
  float readt = 0., flagt = 0., dedispt = 0., smootht = 0., peakt = 0., outputt = 0.;
  float tot_time;
  clock_t begin, end;

  // outputs
  float tot_flags = 0.;
  int socket_count = 0;

  while (finished==0) {

    // dada input
    if (p.inp_format==0)
    {
      block = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);
      if (!block)
      {
        finished = 1;
        continue;
      }
    }

    if (p.inp_format==2 && fin != NULL)
      fread(sigproc_buf, sizeof(unsigned char), NBEAMS*p.gulp*NCHAN, fin);

    // set up logging and reset output
    //for (int i=0;i<NBEAMS;i++) beamflags[i] = 0;
    for (int i=0;i<NCHAN;i++) specflags[i] = 0;
    hella::clear_peaks(&p);

    spdlog::debug("Starting gulp {}",gulp);

    const size_t beam_time_stride = p.NTIME * NCHAN;
    const size_t beam_gulp_stride = p.gulp * NCHAN;
    const size_t beam_rewind_stride = (p.NTIME - p.gulp) * NCHAN;
    // read in data
    begin = clock();

    if (p.inp_format==0) // DADA input - fp16
    {
      checkCuda(cudaMemcpy(p.d_gulp, block, NBEAMS * beam_gulp_stride * sizeof(half), cudaMemcpyHostToDevice));
    }
    else if (p.inp_format==2) // filterbank input - u8
    {
      checkCuda(cudaMemcpy(p.d_gulp, sigproc_buf, NBEAMS * beam_gulp_stride, cudaMemcpyHostToDevice));
    }
    end = clock();
    readt += static_cast<float>(end - begin) / CLOCKS_PER_SEC;

    begin = clock();
    // flag the input gulp, storing the result in d_data
    hella::fast_flagger(&p);
    end = clock();
    flagt += static_cast<float>(end - begin) / CLOCKS_PER_SEC;

    if ((gulp>0 && p.inp_format!=1) || (p.inp_format==1))
    {
      // deal with flags
      for (int j=0;j<NBATCH;j++)
      {
        for (int i=0;i<NCHAN;i++)
        {
          //beamflags[bm] += (int)(p.h_flagSpec[j*NCHAN+i]);
          specflags[i] += (int)(1.*p.NTIME*p.h_flagSpec[j*NCHAN+i]);
          tot_flags += (1.*p.NTIME*p.h_flagSpec[j*NCHAN+i])/FLAG_NORMALIZATION_FACTOR;
        }
      }

      // loop over beams to dedisperse and search
      // check time, out_npeaks
      bm = 0;
      tot_time = readt+flagt;

      while ((bm<NBEAMS) && (tot_time<PROCESSING_TIME_LIMIT) && (p.out_npeaks < MAX_GIANTS))
      {
        begin = clock();
        hella::dedisperse(&p,bm);
        end = clock();
        dedispt += static_cast<float>(end - begin) / CLOCKS_PER_SEC;

        begin = clock();
        hella::smooth(&p,1);
        end = clock();
        smootht += static_cast<float>(end - begin) / CLOCKS_PER_SEC;
        
        begin = clock();
        hella::find_peaks(&p,bm);
        end = clock();
        peakt += static_cast<float>(end - begin) / CLOCKS_PER_SEC;

        tot_time = readt+flagt+dedispt+smootht+peakt;
        bm += 1;
      }

      begin = clock();
      // output peaks
      if (socket_count==0)
        hella::output_peaks(&p,samp,1);
      else
        hella::output_peaks(&p,samp,0);
      // output flags
      //fbeam = fopen(p.beamflags,"a");
      fspec = fopen(p.specflags,"a");
      //for (int i=0;i<NBEAMS;i++) fprintf(fbeam,"%d\n",beamflags[i]);
      for (int i=0;i<NCHAN;i++) fprintf(fspec,"%d\n",specflags[i]);
      //fclose(fbeam);
      fclose(fspec);
      end = clock();
      outputt += static_cast<float>(end - begin) / CLOCKS_PER_SEC;

      // increment socket_count
      socket_count++;
      if (socket_count==SOCKET_CADENCE)
        socket_count=0;
    }

    // Rewind d_data in gulp-sized chunks

    // Need to be careful at the boundary, since NTIME % gulp != 0 in general, so there will be one gulp hich would be rewound to before
    // the start of our buffer. The first `last_dat` samples in the newly rewound buffer will come from that gulp.
    long last_dat = p.NTIME;
    while (last_dat - static_cast<int>(p.gulp) > 0)
      last_dat -= p.gulp;
    spdlog::debug("First rewind chunk will fill up to last_dat={}", last_dat);

    for (int ibeam = 0; ibeam < NBEAMS; ibeam++)
    {
      unsigned char *beam_data = p.d_data + ibeam * p.NTIME * NCHAN;

      size_t load_dat = p.gulp;
      size_t store_dat = 0;
      checkCuda(cudaMemcpyAsync(beam_data + store_dat * NCHAN, beam_data + (load_dat * NCHAN), last_dat * NCHAN, cudaMemcpyDeviceToDevice));

      store_dat = last_dat;
      load_dat = store_dat + p.gulp;
      while (load_dat <= p.NTIME-p.gulp)
      {
        checkCuda(cudaMemcpyAsync(beam_data + store_dat * NCHAN, beam_data + load_dat * NCHAN, p.gulp * NCHAN, cudaMemcpyDeviceToDevice));
        store_dat += p.gulp;
        load_dat = store_dat + p.gulp;
      }
    }
    checkCuda(cudaDeviceSynchronize());

    // increment sample
    samp += p.gulp;
    gulp += 1;

    // assume only one gulp for text file input
    if (p.inp_format==1)
      finished = 1;

    // look for eof for fil input
    if (p.inp_format==2)
      if (feof(fin)) finished = 1;
    //std::cout << "Finished: " << finished << std::endl;

    // close off dada block
    if (p.inp_format==0)
      ipcio_close_block_read (hdu_in->data_block, bytes_read);

    spdlog::debug("Beamstats {} giants {} {}",bm,p.out_npeaks,tot_flags);
    tot_flags = 0.;
    spdlog::info("processed {} s in read {} flag {} dedisp {} smooth {} peak {} output {} [{}]",(p.ntime_dd)*TIME_CONVERSION_FACTOR,readt,flagt,dedispt,smootht,peakt,outputt,readt+flagt+dedispt+smootht+peakt+outputt);
    // fprintf(stderr, "%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\n",(p.ntime_dd)*TIME_CONVERSION_FACTOR,readt,flagt,dedispt,smootht,peakt,outputt,readt+flagt+dedispt+smootht+peakt+outputt);
    spdlog::debug("Flagging: {} {} {} {} {} {} {} {}",p.t1,p.t2,p.t3,p.t4,p.t5,p.t6,p.t7,p.t8);
    readt = 0.;
    flagt = 0.;
    dedispt = 0.;
    smootht = 0.;
    peakt = 0.;
    outputt = 0.;
    p.t1 = 0.;
    p.t2 = 0.;
    p.t3 = 0.;
    p.t4 = 0.;
    p.t5 = 0.;
    p.t6 = 0.;
    p.t7 = 0.;
    p.t8 = 0.;
  }

  // deallocate stuff
  if (p.inp_format==0)
  {
    hella::hdu_cleanup(hdu_in);
  }

  hella::release_cpu(&hodata);
  hella::release_cpu(&sigproc_buf);

  deallocator(&p);
}
catch (std::exception& exc)
{
  spdlog::error("Runtime error: {}", exc.what());
}
