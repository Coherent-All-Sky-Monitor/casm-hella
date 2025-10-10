// -*- c++ -*-

#include <sys/cdefs.h>
#include <time.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <math.h>
#include <cuda_runtime.h>
#include <npp.h>
#include <nppdefs.h>
#include <nppcore.h>
#include <nppi.h>
#include <npps.h>
//#include <helper_cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <dedisp.h>
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
// #include <sigproc.h>  // Commented out due to missing dependencies
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/select.h>
#include <syslog.h>

#include "sock.h"
#include "tmutil.h"
#include "dada_client.h"
#include "dada_def.h"
#include "dada_hdu.h"
#include "ipcio.h"
#include "ipcbuf.h"
#include "dada_affinity.h"
#include "ascii_header.h"

#include <cuda.h>
#include "cuda_fp16.h"
#include <cublas_v2.h>

//using namespace std;

const int MAXHOSTNAME = 200;
const int MAXCONNECTIONS = 5;
const int MAXRECV = 500;

#define NMEDFILT 13
#define NTSMED 7
#define NBATCH 4
#define NCHAN 3072
#define NBEAMS 64
#define NCHAN_BOX 48
#define NTIME_BOX 500
#define MAX_DM 2000
#define TOL 1.3
#define MAX_BOX 15
#define MAX_GIANTS 10000
#define DADA_BLOCK_KEY 0x0000dada // for capture program.
#define SOCKET_CADENCE 1

// CASM frequency and timing parameters
// #define FREQ_CHANNEL_WIDTH 0.03075  // 0.03075 MHz
// #define CENTER_FREQ 450.0e6         // 450 MHz in Hz
#define FREQ_CHANNEL_WIDTH -0.030518
#define CENTER_FREQ 500.0e6         // 450 MHz in Hz
#define TIME_RESOLUTION 1.0e-3      // 1 ms in seconds

// Other configurable constants
#define SECONDS_PER_DAY 86400.0     // seconds in a day
#define TIME_CONVERSION_FACTOR 1.048576e-3 // conversion factor for time calculations
#define PROCESSING_TIME_LIMIT 100.1   // time limit for processing in seconds, causes dedispersion to be skipped
#define FLAG_NORMALIZATION_FACTOR 805306368.0  // normalization factor for flagging statistics

// Statistical parameters for boxcar smoothing
#define BOXCAR_MEAN 0.21368
// Individual standard deviation values for boxcar smoothing
#define BOXCAR_STD_0 0.001309
#define BOXCAR_STD_1 0.00124735
#define BOXCAR_STD_2 0.00103835
#define BOXCAR_STD_3 0.00081225
#define BOXCAR_STD_4 0.00062605
#define BOXCAR_STD_5 0.00047785
#define BOXCAR_STD_6 0.00036005

// Threshold parameters
#define TIME_SERIES_HIGH_THRESHOLD 1.05
#define TIME_SERIES_LOW_THRESHOLD 0.95
#define STD_DEV_LOW_THRESHOLD 1.2
#define STD_DEV_VERY_LOW_THRESHOLD 0.96
#define STD_DEV_HIGH_THRESHOLD 1.5
#define STD_DEV_VERY_HIGH_THRESHOLD 0.92

#define checkCuda(err) { \
  if (err != cudaSuccess) { \
    fprintf(stderr, "FATAL CUDA Error in %s at line %d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  } \
}

#define checkNpp(err) { \
  if (err != NPP_NO_ERROR) { \
    fprintf(stderr, "NPP Error in %s at line %d: code=%d\n", \
            __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE); \
  } \
}

int finished = 0;

void dsaX_dbgpu_cleanup (dada_hdu_t * in, dada_hdu_t * out);

void dsaX_dbgpu_cleanup (dada_hdu_t * in, dada_hdu_t * out)
{
  if (dada_hdu_unlock_read (in) < 0)
  {
    syslog(LOG_ERR, "could not unlock read on hdu_in");
  }
  dada_hdu_destroy (in);

  if (dada_hdu_unlock_write (out) < 0)
  {
    syslog(LOG_ERR, "could not unlock write on hdu_out");
  }
  dada_hdu_destroy (out);
}


// define a SCRUNCH structure
typedef struct scrunch {

  int tscrunch, fscrunch, nits;
  float thresh;

} scrunch;

// define a structure to carry all info and pre-allocated arrays, which can be passed between functions
typedef struct pinfo {

  // input params
  int inp_format; // 0 for dada, 1 for file, 2 for filterbank
  char inp_path[500];
  char dada_out[100];
  float minDM, maxDM, snr;
  int minWidth, maxWidth;
  int gulp;
  scrunch * scrunches; // array of scrunches
  int nscrunches;
  char beamflags[500], specflags[500];
  int out_format; // 0 for file, 1 for socket, 2 for both
  int coincidencer_port;
  std::string coincidencer_host;
  char out_path[500]; // path or IP
  int BEAM_OFFSET;
  int BEAM0;
  int flag1, flag2; // flag ranges
  int output_bandpass;
  float spec_min, spec_max; // thresholds for spectrum flagging

  // derived params
  int NTIME; // gulp that includes rewind
  int rewind; // samples to rewind by
  int nchan; // number of resampled channels
  int ndms; // number of DM trials
  int ntime_dd; // dedisp number of times
  int ntime_out; // final output number of times
  int ntime_dedisp; // number of dedispersed times (must be >= NTIME-max_delay)
  int nboxcar;

  // pre-allocates - host
  unsigned char * data{nullptr}, * rewinds{nullptr};
  //unsigned char * indata{nullptr};
  float * h_dedisp{nullptr};
  const float * DMs{nullptr};
  dedisp_plan dedispersion_plan;
  float * h_dataF{nullptr};
  float * h_flagSpec{nullptr};

  // pre-allocates - GPU
  float * d_flagSpec{nullptr};
  Npp32f * d_dedisp{nullptr}; // dedispersion output
  int d_dedisp_step;
  float * d_dedispPacked{nullptr}; // dedisp output
  unsigned char * d_inputPacked{nullptr}; // dedisp input
  unsigned char * d_data{nullptr}; // all the input
  half * batch{nullptr}, * mask{nullptr}, * d_smooth{nullptr};
  float * d_ts{nullptr};
  int batch_stride;

  bool have_bp{false};        // flag to prevent recalculation of the bandpass
  float * d_bp{nullptr};      // bandpass
  float * d_bp_sq{nullptr};   // bandpass squared
  float * d_bpout{nullptr};   // bandpass normalized

  // boxcars
  Npp32f * boxes{nullptr};
  int boxes_step;
  Npp32f * imbox{nullptr};
  int imbox_step;
  float * stds{nullptr};
  float mean;

  // peak finding
  thrust::device_vector<float> dmt;
  thrust::device_vector<int> output_indices;
  thrust::device_vector<float> output_values;
  int * h_idxs{nullptr};
  int * beam{nullptr}, * out_beam{nullptr};
  int * width{nullptr}, * out_width{nullptr};
  int * dm_idx{nullptr}, * out_dm_idx{nullptr};
  int * samp{nullptr}, * out_samp{nullptr};
  float * peaks{nullptr}, * out_peaks{nullptr};
  int npeaks, out_npeaks;

  // flag timing
  float fcpy, fprep, fflag, fapply;
  float t1, t2, t3, t4, t5, t6, t7, t8, t9;

} pinfo;

//! get the GPU id from the configuration file, idenitifed by the GPU keyword
int get_gpu_id(FILE *fconf) {

  char * line = NULL;
  ssize_t read;
  size_t len = 0;
  char c1[20], c2[500];

  while (!feof(fconf)) {

    read = getline(&line, &len, fconf);
    sscanf(line,"%s %s",c1,c2);
    if (strcmp(c1,"GPU")==0) {
      rewind(fconf);
      return atoi(c2);
    }

  }
  return 0;
}

// function to initialize everything based on a config file
void initialize(FILE *fconf, pinfo * p) {

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
  while (!feof(fconf)) {

    read = getline(&line, &len, fconf);
    sscanf(line,"%s %s",c1,c2);

    if (strcmp(c1,"INPUT")==0) {
      if (strcmp(c2,"DADA")==0) p->inp_format=0;
      if (strcmp(c2,"FILE")==0) p->inp_format=1;
      if (strcmp(c2,"FILTERBANK")==0) p->inp_format=2;
      if (strcmp(c2,"CANDIDATE")==0) p->inp_format=3;
      printf("Using input format %d\n",p->inp_format);
    }
    if (strcmp(c1,"OUTPUT")==0) {
      if (strcmp(c2,"FILE")==0) p->out_format=0;
      if (strcmp(c2,"SOCKET")==0) p->out_format=1;
      if (strcmp(c2,"BOTH")==0) p->out_format=2;
      printf("Using output format %d\n",p->out_format);
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
      printf("Input path: %s\n",p->inp_path);
    }

    if (strcmp(c1,"DADA_OUT")==0) {
      strcpy(p->dada_out,c2);
      printf("DADA out: %s\n",p->dada_out);
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

    if (strcmp(c1,"SCRUNCH")==0) {

      p->nscrunches = atoi(c2);
      p->scrunches = (scrunch *)malloc(p->nscrunches*sizeof(scrunch));

      for (int i=0;i<p->nscrunches;i++) {
        read = getline(&line, &len, fconf);
        sscanf(line,"%d %d %f %d",&(p->scrunches[i].tscrunch),&(p->scrunches[i].fscrunch),&(p->scrunches[i].thresh),&(p->scrunches[i].nits));
        printf("Have a scrunch with %d %d %g %d\n",p->scrunches[i].tscrunch,p->scrunches[i].fscrunch,p->scrunches[i].thresh,p->scrunches[i].nits);
      }

    }
  }
  fclose(fconf);

  // derived parameters
  p->NTIME=p->gulp;
  p->rewind=0;
  int i=(int)(p->minWidth), j=0;
  while (i<(int)(p->maxWidth))  {
    i *= 2;
    j += 1;
  }
  p->nboxcar=j;
  printf("Search parameters: DM range %g to %g, WIDTHS %d to %d (%d trials), SNR %g\n",p->minDM,p->maxDM,p->minWidth,p->maxWidth,p->nboxcar,p->snr);
  if (p->out_format != 0)
    printf("Outputting to socket %s:%d\n",p->coincidencer_host.c_str(),p->coincidencer_port);
  if (p->out_format != 1)
    printf("Outputting to text file %s\n",p->out_path);


  // set up DM plan
  fprintf(stderr, "Creating a dedispersion plan nchans=%d dt=%f f0=%f df=%f\n", NCHAN, TIME_RESOLUTION, CENTER_FREQ/1e6,FREQ_CHANNEL_WIDTH);
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
  if (p->inp_format == 0 || p->inp_format == 2) {
    p->NTIME = p->gulp + dedisp_get_max_delay(p->dedispersion_plan) + p->maxWidth;
    oo = 32*((int)(p->NTIME/32)+1);
    p->NTIME = oo;
    p->ntime_dedisp = oo-dedisp_get_max_delay(p->dedispersion_plan);
    p->ntime_dd = p->gulp + p->maxWidth;
    p->ntime_out = p->gulp;
  }


  // allocate everything

  p->h_flagSpec = (float *)malloc(sizeof(float)*NCHAN*NBATCH);
  cudaMalloc((void **)(&p->d_flagSpec), sizeof(float)*NCHAN*NBATCH);
  p->rewinds = (unsigned char *)malloc(sizeof(unsigned char)*NCHAN*(p->NTIME-p->gulp)*NBEAMS);
  memset(p->rewinds,0,NCHAN*(p->NTIME-p->gulp)*NBEAMS);
  p->data = (unsigned char *)malloc(sizeof(unsigned char)*NBEAMS*NCHAN*p->NTIME);
  //p->h_dedisp = (float *)malloc(sizeof(float)*p->ndms*p->ntime_dd);
  //p->indata = (unsigned char *)malloc(sizeof(unsigned char)*NCHAN*p->NTIME);
  p->h_dataF = (float *)malloc(sizeof(float)*NCHAN*p->NTIME);
  printf("allocating %ld bytes ndms=%d ntime_dedisp=%d\n", sizeof(float)*p->ndms*p->ntime_dedisp, p->ndms, p->ntime_dedisp);
  checkCuda(cudaMalloc((void **)(&p->d_dedispPacked), sizeof(float)*p->ndms*p->ntime_dedisp));
  checkCuda(cudaMalloc((void **)(&p->d_inputPacked), sizeof(unsigned char)*NCHAN*p->NTIME));
  checkCuda(cudaMalloc((void **)(&p->d_data), sizeof(unsigned char)*NBEAMS*NCHAN*p->NTIME));
  checkCuda(cudaMallocPitch((void **)(&p->batch), (size_t *)(&p->batch_stride), (unsigned long)(p->NTIME*sizeof(half)), NBATCH*NCHAN));
  checkCuda(cudaMallocPitch((void **)(&p->mask), (size_t *)(&p->batch_stride), (unsigned long)(p->NTIME*sizeof(half)), NBATCH*NCHAN));
  p->batch_stride = p->batch_stride / sizeof(half);
  checkCuda(cudaMalloc((void **)(&p->d_smooth), sizeof(half) * NBATCH * NCHAN * p->batch_stride));
  p->d_dedisp = nppiMalloc_32f_C1(p->ntime_dd,p->ndms,&(p->d_dedisp_step));
  if (!p->d_dedisp)
  {
    fprintf(stderr, "nppiMalloc_32f_C1 failed to allocate width=%d height=%d\n", p->ntime_dd,p->ndms);
    exit(1);
  }

  checkCuda(cudaMalloc((void **)(&p->d_ts), sizeof(float) * NBATCH * p->NTIME));
  checkCuda(cudaMalloc((&p->d_bp), sizeof(float) * NBATCH * NCHAN));
  checkCuda(cudaMalloc((&p->d_bp_sq), sizeof(float) * NBATCH * NCHAN));
  checkCuda(cudaMalloc((&p->d_bpout), sizeof(float) * NBATCH * NCHAN));

  printf("Will use %d DM trials, output %d times, process %d times with stride %d\n",p->ndms,p->ntime_dd,p->NTIME,p->batch_stride);


  // boxcars
  p->boxes = nppiMalloc_32f_C1(p->ntime_out,(p->ndms-2)*p->nboxcar,&(p->boxes_step));
  p->imbox = nppiMalloc_32f_C1(p->ntime_dd,p->ndms,&(p->imbox_step));
  p->stds = (float *)malloc(sizeof(float)*p->nboxcar);
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
  p->h_idxs = (int *)malloc(sizeof(int)*(p->ndms-2)*p->ntime_out);
  p->beam = (int *)malloc(sizeof(int)*MAX_GIANTS);
  p->width =  (int *)malloc(sizeof(int)*MAX_GIANTS);
  p->dm_idx = (int *)malloc(sizeof(int)*MAX_GIANTS);
  p->samp = (int *)malloc(sizeof(int)*MAX_GIANTS);
  p->peaks = (float *)malloc(sizeof(float)*MAX_GIANTS);

  p->out_beam = (int *)malloc(sizeof(int)*MAX_GIANTS);
  p->out_samp = (int *)malloc(sizeof(int)*MAX_GIANTS);
  p->out_width =  (int *)malloc(sizeof(int)*MAX_GIANTS);
  p->out_dm_idx = (int *)malloc(sizeof(int)*MAX_GIANTS);
  p->out_peaks = (float *)malloc(sizeof(float)*MAX_GIANTS);

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
void deallocator(pinfo * p) {

  printf("deallocating pinfo struct\n");
  if (p->data)
    free(p->data);
  p->data = nullptr;
  if (p->h_dataF)
    free(p->h_dataF);
  checkCuda(cudaFree(p->d_bp));
  checkCuda(cudaFree(p->d_bp_sq));
  checkCuda(cudaFree(p->d_bpout));
  checkCuda(cudaFree(p->d_data));
  checkCuda(cudaFree(p->batch));
  checkCuda(cudaFree(p->mask));
  checkCuda(cudaFree(p->d_dedisp));
  checkCuda(cudaFree(p->boxes));
  checkCuda(cudaFree(p->d_dedispPacked));
  checkCuda(cudaFree(p->d_inputPacked));
  p->dmt.clear();
  p->output_indices.clear();
  p->output_values.clear();
  if (p->h_idxs)
    free(p->h_idxs);
  if (p->beam)
    free(p->beam);
  if (p->width)
    free(p->width);
  if (p->dm_idx)
    free(p->dm_idx);
  if (p->samp)
    free(p->samp);
  if (p->peaks)
    free(p->peaks);
  if (p->stds)
    free(p->stds);
  if (p->rewinds)
    free(p->rewinds);

}

void help() {

  printf("Usage: pipeline -c <config file>\n");
  printf("Everything is in the config file. Specific parameters include: \n");
  printf("INPUT <DADA or FILE or FILTERBANK>\n");
  printf("INPUT_PATH <dada buffer or full path to filterbank file>\n");
  printf("DADA_OUT <dada buffer>\n");
  printf("BEAM_OFFSET <offset in number of beams in input dada buffer>\n");
  printf("DM_MIN <min DM of search>\n");
  printf("DM_MAX <max DM of search>\n");
  printf("WIDTH_MIN <min width of search>\n");
  printf("WIDTH_MAX <max width of search>\n");
  printf("SNR <SNR threshold for search>\n");
  printf("GULP <base gulp size>\n");
  printf("BEAMFLAGS <full path to beam flags output>\n");
  printf("SPECFLAGS <full path to spec flags output>\n");
  printf("OUTPUT <FILE or SOCKET of BOTH>\n");
  printf("OUTPUTPATH <path to output file>\n");
  printf("HOST <ip of T2 host>\n");
  printf("PORT <T2 port>\n");
  printf("GPU <GPU ID 0 or 1>\n");
  printf("BEAM0 <first beam in output>\n");
  printf("OUTPUT_BANDPASS <0 or 1 or 2>\n");
  printf("SPEC_MAX <max thresh in spec flagging>\n");
  printf("SPEC_MAX <max thresh in spec flagging>\n");

  printf("SCRUNCH <number of scrunches>\n");
  printf("<time scrunch> <frequency scrunch> <flagging threshold> <number of iterations>\n");
  printf("repeat the above as many times as you like for different parameters\n");

}


/************ FUNCTIONS FOR NEW FLAGGER ***********/

// kernel to warp reduce float
__device__ void warpReduce(volatile float *sdata, unsigned int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

// kernel to sum array and its squares
__global__ void sumArray(half * data, float * sums, float * qsums, int width, int height, int stride)
{
  __shared__ float sdata[512], qdata[512];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*512 + tid;
  int x = i % width;
  int y = i / width;
  int iidx = y*stride+x;

  sdata[tid] = __half2float(data[iidx]);
  qdata[tid] = __half2float(data[iidx]*data[iidx]);

  __syncthreads();

  if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
  if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
  if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
  if (tid < 32) warpReduce(sdata, tid);
  if (tid < 256) { qdata[tid] += qdata[tid + 256]; } __syncthreads();
  if (tid < 128) { qdata[tid] += qdata[tid + 128]; } __syncthreads();
  if (tid < 64) { qdata[tid] += qdata[tid + 64]; } __syncthreads();
  if (tid < 32) warpReduce(qdata, tid);

  if (tid == 0) sums[blockIdx.x] = sdata[0];
  if (tid == 0) qsums[blockIdx.x] = qdata[0];


}
// kernel to sum array and its squares
__global__ void sumArrayFloat(float * data, float * sums, float * qsums, int width, int height, int stride) {

  extern __shared__ float sfdata[512], qfdata[512];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*512 + tid;
  int x = i % width;
  int y = i / width;
  int iidx = y*stride+x;

  sfdata[tid] = data[iidx];
  qfdata[tid] = data[iidx]*data[iidx];

  __syncthreads();

  if (tid < 256) { sfdata[tid] += sfdata[tid + 256]; } __syncthreads();
  if (tid < 128) { sfdata[tid] += sfdata[tid + 128]; } __syncthreads();
  if (tid < 64) { sfdata[tid] += sfdata[tid + 64]; } __syncthreads();
  if (tid < 32) warpReduce(sfdata, tid);
  if (tid < 256) { qfdata[tid] += qfdata[tid + 256]; } __syncthreads();
  if (tid < 128) { qfdata[tid] += qfdata[tid + 128]; } __syncthreads();
  if (tid < 64) { qfdata[tid] += qfdata[tid + 64]; } __syncthreads();
  if (tid < 32) warpReduce(qfdata, tid);

  if (tid == 0) sums[blockIdx.x] = sfdata[0];
  if (tid == 0) qsums[blockIdx.x] = qfdata[0];
}


// Host function to orchestrate the normalization process
// width = p->NTIME
// height = NBATCH * NCHAN
// stride = p->batch_stride
float calculateStdDev(pinfo *p, half * d_data, int width, int height, int stride) {

  #ifdef OLD
  float *d_sums, *d_qsums;
  int new_width = (int)(512*floor(width/512.));
  int nblocks = new_width*height / 512;

  checkCuda(cudaMalloc(&d_sums, nblocks * sizeof(float)));
  checkCuda(cudaMalloc(&d_qsums, nblocks * sizeof(float)));
  sumArray<<<nblocks,512>>>(d_data,d_sums,d_qsums,new_width,height,stride);
  #else
  int new_width = 1;
  int nblocks = height;
  float *d_sums = p->d_bp;
  float *d_qsums = p->d_bp_sq;
  #endif

  float *sums = (float *)malloc(sizeof(float)*nblocks);
  float *qsums = (float *)malloc(sizeof(float)*nblocks);
  checkCuda(cudaMemcpy(sums,d_sums,nblocks*sizeof(float),cudaMemcpyDeviceToHost));
  checkCuda(cudaMemcpy(qsums,d_qsums,nblocks*sizeof(float),cudaMemcpyDeviceToHost));

  float sum=0., qsum=0.;
  for (int i=0;i<nblocks;i++) {
    sum += sums[i];
    qsum += qsums[i];
  }
  float mn = sum/(new_width*height*1.);

  float stdDev = qsum-2.*sum*mn+mn*mn*new_width*height*1.;
  stdDev /= 1.*new_width*height;
  stdDev = sqrt(stdDev);

  #ifdef OLD
  checkCuda(cudaFree(d_sums));
  checkCuda(cudaFree(d_qsums));
  #endif
  free(sums);
  free(qsums);

  return stdDev;
}


// Host function to orchestrate the normalization process
float calculateStdDevFloat(float * d_data, int width, int height, int stride) {

  float *d_sums{nullptr}, *d_qsums{nullptr};
  int new_width = (int)(512*floor(width/512.));
  int nblocks = new_width*height / 512;

  checkCuda(cudaMalloc(&d_sums, sizeof(float) * nblocks));
  checkCuda(cudaMalloc(&d_qsums, sizeof(float) * nblocks));
  sumArrayFloat<<<nblocks,512>>>(d_data,d_sums,d_qsums,new_width,height,stride);

  float *sums{nullptr}, *qsums{nullptr};
  sums = (float *)malloc(sizeof(float)*nblocks);
  qsums = (float *)malloc(sizeof(float)*nblocks);
  checkCuda(cudaMemcpy(sums,d_sums,nblocks*sizeof(float),cudaMemcpyDeviceToHost));
  checkCuda(cudaMemcpy(qsums,d_qsums,nblocks*sizeof(float),cudaMemcpyDeviceToHost));

  float sum=0., qsum=0.;
  for (int i=0;i<nblocks;i++) {
    sum += sums[i];
    qsum += qsums[i];
  }
  float mn = sum/(new_width*height*1.);

  float stdDev = qsum-2.*sum*mn+mn*mn*new_width*height*1.;
  stdDev /= 1.*new_width*height;
  stdDev = sqrt(stdDev);

  checkCuda(cudaFree(d_sums));
  checkCuda(cudaFree(d_qsums));
  free(sums);
  free(qsums);

  return stdDev;
}


/*
// warp reduce half
__device__ void warpReduceHalf(volatile half *data, int tid) {
  data[tid] = data[tid] + data[tid + 32];
  data[tid] = data[tid] + data[tid + 16];
  data[tid] = data[tid] + data[tid + 8];
  data[tid] = data[tid] + data[tid + 4];
  data[tid] = data[tid] + data[tid + 2];
  data[tid] = data[tid] + data[tid + 1];
}*/

// add bandpasses
// add_bandpass<<<NCHAN*NBATCH/32,32>>>(d_mask,d_flagSpec);
__global__ void add_bandpass(float * d_mask, float * d_flagSpec) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int idx = bid*32 + tid;

  d_flagSpec[idx] += d_mask[idx];

}

// in each block, warps will compute the bandpass for a single channel
// each block will write out 256/32 channels
// grid handles the rest
__global__ void calc_bandpass_new(const __restrict__ half * data, float * bandpass, float * bandpass_sq, float * bandpass_norm, int width, int stride)
{
  const unsigned warp_idx = threadIdx.x & 0x1F; // % 32;
  const unsigned warp_num = threadIdx.x / warpSize;
  const unsigned channelbeam = (blockIdx.x * 8) + warp_num;
  unsigned idx = (channelbeam * stride) + warp_idx;

  const int npartials = width / 32; // number partial sums
  float sum = 0;
  float sum_sq = 0;
  for (unsigned i=0; i<npartials; i++)
  {
    const float raw = __half2float(data[idx]);
    idx += warpSize;
    sum += raw;
    sum_sq += raw * raw;
  }

  // warp level reduction
  for (int offset = 16; offset > 0; offset /= 2)
  {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
    sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
  }

  if (warp_idx == 0)
  {
    bandpass[channelbeam] = sum;
    bandpass_sq[channelbeam] = sum_sq;
    bandpass_norm[channelbeam] = sum / float(width);
  }
}

// kernel to calculate bandpass
// launch with NCHAN*NBATCH blocks of 256 threads
// will ignore last (width % 256) times
__global__ void calc_bandpass(const __restrict__ half * data, float * bandpass, int width, int stride) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  int npartials = (int)(width / 256); // number partial sums
  half fac = (half)((256.*npartials));
  __shared__ half psum[256];


  // calculate partial sums
  int idx0 = bid*stride + tid*npartials;
  psum[tid] = 0.;
  int count = 0;
  for (int i=idx0;i<npartials+idx0;i++)
  {
    psum[tid] += data[i];
  }
  psum[tid] /= fac;
  __syncthreads();

  // sum over shared memory
  if (tid < 128) { psum[tid] += psum[tid + 128]; } __syncthreads();
  if (tid < 64) { psum[tid] += psum[tid + 64]; } __syncthreads();
  if (tid < 32) { psum[tid] += psum[tid + 32]; } __syncthreads();
  if (tid < 16) { psum[tid] += psum[tid + 16]; } __syncthreads();
  if (tid < 8) { psum[tid] += psum[tid + 8]; } __syncthreads();
  if (tid < 4) { psum[tid] += psum[tid + 4]; } __syncthreads();
  if (tid < 2) { psum[tid] += psum[tid + 2]; } __syncthreads();
  if (tid < 1) { psum[tid] += psum[tid + 1]; } __syncthreads();
  //if (tid < 32) warpReduce(psum, tid);
  __syncthreads();

  if (tid==0) bandpass[bid] = __half2float(psum[0]);
}

// Function to swap two float values
void swap(float *a, float *b) {
    float temp = *a;
    *a = *b;
    *b = temp;
}

// Function to find the median of an array
float findMedian(float arr[], int n) {
    // Sort the array using bubble sort (you can use faster algorithms)
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(&arr[j], &arr[j + 1]);
            }
        }
    }

    // Return the middle element if odd, or average of two middle elements if even
    if (n % 2 != 0)
        return (float)arr[n / 2];
    return (float)(arr[(n - 1) / 2] + arr[n / 2]) / 2.0;
}

// Function to apply median filter
float medianFilter(float *input, float *output, int size, int windowSize) {

    if (windowSize % 2 == 0) {
        printf("Error: Window size must be odd.\n");
        return 0.;
    }

    int halfWindowSize = windowSize / 2;

    for (int i = halfWindowSize; i < size-halfWindowSize; i++) {
        float window[windowSize];
        int windowIndex = 0;

        // Populate the window array
        for (int j = i - halfWindowSize; j <= i + halfWindowSize; j++) {
            if (j >= 0 && j < size) {
                window[windowIndex] = input[j];
            } else {
                // Handle boundary cases (you can choose different strategies)
                window[windowIndex] = input[i]; // Repeat edge value
            }
            windowIndex++;
        }

        // Calculate and store the median for the current window
        output[i] = findMedian(window, windowSize);
    }

    // edge values
    for (int i=0;i<halfWindowSize;i++) output[i] = output[halfWindowSize];
    for (int i=size-halfWindowSize;i<size;i++) output[i] = output[size-halfWindowSize-1];

    // return mean
    float mn = 0.;
    for (int i=0;i<size;i++)
      mn += output[i];
    mn /= 1.*size;

    return mn;

}

// function to orchestrate host median filtering of bandpass
float medFilterBandpass(float * d_bandpass) {

  float * hbp = (float *)malloc(sizeof(float)*NBATCH*NCHAN);
  float * mhbp = (float *)malloc(sizeof(float)*NBATCH*NCHAN);
  cudaMemcpy(hbp,d_bandpass,sizeof(float)*NBATCH*NCHAN,cudaMemcpyDeviceToHost);
  float mn_bp = medianFilter(hbp,mhbp,NBATCH*NCHAN,NMEDFILT);
  cudaMemcpy(d_bandpass,mhbp,sizeof(float)*NBATCH*NCHAN,cudaMemcpyHostToDevice);

  free(hbp);
  free(mhbp);

  return mn_bp;

}

// function to orchestrate host median filtering of time series
void medFilterTs(float * d_ts, int width) {

  float * hts = (float *)malloc(sizeof(float)*NBATCH*width);
  float * mhts = (float *)malloc(sizeof(float)*NBATCH*width);
  cudaMemcpy(hts,d_ts,sizeof(float)*NBATCH*width,cudaMemcpyDeviceToHost);
  float mn_ts = medianFilter(hts,mhts,NBATCH*width,NTSMED);
  cudaMemcpy(d_ts,mhts,sizeof(float)*NBATCH*width,cudaMemcpyHostToDevice);

  free(hts);
  free(mhts);

}

// cuda kernel to divide data by bandpass
// run with NBATCH*NCHAN*width/64 blocks of 64 threads
__global__ void divide_by_bp(half * data, float * bp, int width, int stride) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  int idx = bid*blockDim.x+tid;
  int y = (int)(idx / width);
  int x = (int)(idx % width);
  int iidx = y*stride+x;

  data[iidx] /= __float2half(bp[y]);

}

// cuda kernel to divide data by time series
// run with NBATCH*NCHAN*width/32 blocks of 32 threads
__global__ void divide_by_ts(half * data, float * ts, int width, int stride, int flag_ts) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  int idx = bid*32+tid;
  int y = (int)(idx / width);   // ibeamchan
  int x = (int)(idx % width);   // isamp
  int b = (int)(y / NCHAN);     // ibeam
  int tsidx = b*width+x;
  int iidx = y*stride+x;

  data[iidx] /= __float2half(ts[tsidx]);

  if (flag_ts==1) {
    if (ts[tsidx]>TIME_SERIES_HIGH_THRESHOLD) data[iidx] = __float2half(1.);
    if (ts[tsidx]<TIME_SERIES_LOW_THRESHOLD) data[iidx] = __float2half(1.);
  }
}

__global__ void divide_by_ts_new(half * data, const float * ts, int width, int stride, int flag_ts)
{
  // beam == blockIdx.y
  // chan == blockIdx.z
  // time == blockIdx.x * blockDim.x + threadIdx.x

  int isamp = blockIdx.x * blockDim.x + threadIdx.x;
  if (isamp >= width)
    return;

  // common scale factor by which all channels are divided
  const float facf = ts[blockIdx.y * stride + isamp];

  // special case to flagging all channels
  // bool flag = (flag_ts == 1 && (facf > TIME_SERIES_HIGH_THRESHOLD || facf < TIME_SERIES_LOW_THRESHOLD));
  // if (flag)
  // {
  //   int idx = (blockIdx.y * NCHAN * stride) + isamp;
  //   for (int ichan=0; ichan<NCHAN; ichan++)
  //   {
  //     data[idx] = 1;
  //     idx += stride;
  //   }
  // }
  // else
  {
    const half fac = __float2half(facf);

    const int idx = ((blockIdx.y * NCHAN + blockIdx.z) * stride) + isamp;
    data[idx] = __hdiv(data[idx], fac);
    // for (int ichan=0; ichan<NCHAN; ichan++)
    // {
      // data[idx] = __hdiv(data[idx], fac);
      // idx += stride;
    // }
  }
}

// handler for half-precision boxcar convolution from npp
void npp_convolve_handler(half * data, half * output, float scfac, int xw, int yw, int width, int stride) {

  NppiSize oSrcSize = {stride,NCHAN*NBATCH};
  NppiPoint oSrcOffset = {0,0};
  Npp32f * pKernel{nullptr};
  NppiSize pKernelSize = {xw,yw};
  NppiPoint oAnchor = {(int)(xw/2),(int)(yw/2)};
  float * h_kernel{nullptr};
  const size_t kernel_bytes = sizeof(float)*xw*yw;
  checkCuda(cudaMalloc((void **)(&pKernel), kernel_bytes));
  h_kernel  = (float *)malloc(kernel_bytes);
  for (int i=0;i<xw*yw;i++) h_kernel[i] = scfac;
  checkCuda(cudaMemcpy(pKernel,h_kernel, kernel_bytes,cudaMemcpyHostToDevice));

  checkNpp(nppiFilterBorder32f_16f_C1R((Npp16f *)data,stride*2,oSrcSize,oSrcOffset,(Npp16f *)output,stride*2,oSrcSize,pKernel,pKernelSize,oAnchor,NPP_BORDER_REPLICATE));

  checkCuda(cudaFree(pKernel));
  free(h_kernel);

}


// cuda kernel to smooth 2d array and output into different array
// run with NBATCH*NCHAN*width/32 blocks of 32 threads
__global__ void smooth_data(half * data, half * output, float scfac, int xw, int yw, int width, int stride) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;
  half mysc = __float2half(scfac);

  int idx = bid*32+tid;
  int y = (int)(idx / width);
  int x = (int)(idx % width);
  int iidx = y*stride+x;
  int ch = (int)(y % NCHAN);
  int bt = (int)(y / NCHAN);

  int xs = x-xw/2;
  int xe = x+xw/2+1;
  if (xs<0) {
    xe -= xs;
    xs = 0;
  }
  else if (xe>width) {
    xs -= xe-width;
    xe = width;
  }

  int ys = ch - yw/2;
  int ye = ch + yw/2 + 1;
  if (ys<0) {
    ye -= ys;
    ys = 0;
  }
  else if (ye>NCHAN) {
    ys -= ye-NCHAN;
    ye = NCHAN;
  }

  output[iidx] = 0.;
  for (int yi=ys; yi<ye; yi++) {
    for (int xi=xs; xi<xe; xi++)
      output[iidx] += data[bt*NCHAN*stride + yi*stride + xi]*mysc;
  }
  //output[iidx] *= mysc;

}


// cuda kernel to divide data by array
// run with NBATCH*NCHAN*width/32 blocks of 32 threads
__global__ void divide_by_array(half * data, half * arr, int width, int stride) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  int idx = bid*32+tid;
  int y = (int)(idx / width);
  int x = (int)(idx % width);
  int iidx = y*stride+x;

  data[iidx] /= arr[iidx];

}

// cuda kernel to multiply data by number
// run with NBATCH*NCHAN*width/32 blocks of 32 threads
__global__ void multiply_by_number(half * data, half num, int width, int stride) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  int idx = bid*blockDim.x+tid;
  int y = (int)(idx / width);
  int x = (int)(idx % width);
  int iidx = y*stride+x;

  data[iidx] *= num;
}

__global__ void add_then_multiply_by_number(half * data, half a, half c, int width, int stride)
{
  int isamp = blockIdx.x * blockDim.x + threadIdx.x;
  if (isamp >= width)
    return;

  int idx = (blockIdx.y * stride) + isamp;
  data[idx] = (data[idx] + a) * c;
}

// cuda kernel to add number to data
// run with NBATCH*NCHAN*width/32 blocks of 32 threads
__global__ void add_number(half * data, half num, int width, int stride)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int y = (int)(idx / width);
  int x = (int)(idx % width);
  int iidx = y*stride+x;

  data[iidx] += num;

}

// cuda kernel to find data above threshold and add into mask
// run with NBATCH*NCHAN*width/32 blocks of 32 threads
__global__ void threshold_data(half * data, half * mask, float threshold, int width, int stride) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  int idx = bid*32+tid;
  int y = (int)(idx / width);
  int x = (int)(idx % width);
  int iidx = y*stride+x;

  if (data[iidx]>__float2half(threshold))
    mask[iidx] = 1.;
  if (data[iidx]<__float2half(-1.*threshold))
    mask[iidx] = 1.;

}

// cuda kernel to replace masked values
// run with NBATCH*NCHAN*width/32 blocks of 32 threads
__global__ void replace_data_bandpass(half * data, float * bp, float repval, int width, int stride, float t1, float t2) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  int idx = bid*blockDim.x+tid;
  int y = (int)(idx / width);
  int x = (int)(idx % width);
  int iidx = y*stride+x;

  if (bp[y]<t1 || bp[y]>t2)
    data[iidx] = repval;
}

__global__ void replace_data_and_add_bandpass(half * data, const float * bp, int width, int stride, float t1, float t2)
{
  // __shared__ float bps[16];

  // load 16 bandpass values into shm
  // if (threadIdx.x < 16)
    // bps[threadIdx.x] = bp[blockIdx.y * 16 + threadIdx.x];
  // __syncthreads();

  const int ibeamchan = (blockIdx.y * blockDim.y) + threadIdx.y;
  const float bp_val = bp[ibeamchan];
  const int idx = ibeamchan * stride + (blockIdx.x * blockDim.x) + threadIdx.x;
  if (bp_val < t1 || bp_val > t2)
  {
    data[idx] = __half(1);
  }
  else {
    data[idx] += __half(1);
  }
}

// cuda kernel to replace masked values
// run with NBATCH*NCHAN*width/32 blocks of 32 threads
__global__ void replace_data(half * data, half * mask, float repval, int width, int stride, int flag1, int flag2) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  int idx = bid*32+tid;
  int y = (int)(idx / width);
  int x = (int)(idx % width);
  int ch = (int)(y % NCHAN);
  int iidx = y*stride+x;

  if (mask[iidx]>(half)(0.))
    data[iidx] = repval;

  if (ch>=flag1 && ch<flag2)
    data[iidx] = repval;

}

// cuda kernel to transpose and scale single beam data for loader
// beam is [width, NCHAN], data is [NCHAN, width]
// assume breakdown into tiles of 32x32, and run with 32x8 threads per block
// launch with dim3 dimBlock(32, 8) and dim3 dimGrid(NCHAN/32, width/32)
__global__ void transpose_input(unsigned char * beam, half * data, int width) {

  __shared__ half tile[32][33];

  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;
  int mywidth = gridDim.x * 32;

  for (int j = 0; j < 32; j += 8)
     tile[threadIdx.y+j][threadIdx.x] = __float2half((float)(beam[(y+j)*mywidth + x]));

  __syncthreads();

  x = blockIdx.y * 32 + threadIdx.x;  // transpose block offset
  y = blockIdx.x * 32 + threadIdx.y;
  mywidth = gridDim.y * 32;

  for (int j = 0; j < 32; j += 8)
    data[(y+j)*mywidth + x] = tile[threadIdx.x][threadIdx.y + j];

}

// cuda kernel to transpose and scale single beam data for output
// beam is [width, NCHAN], data is [NCHAN, width]
// assume breakdown into tiles of 32x32, and run with 32x8 threads per block
// launch with dim3 dimBlock(32, 8) and dim3 dimGrid(width/32, NCHAN/32)
__global__ void transpose_output(unsigned char * beam, half * data, int width) {

  __shared__ half tile[32][33];

  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;
  int mywidth = gridDim.x * 32;

  for (int j = 0; j < 32; j += 8)
     tile[threadIdx.y+j][threadIdx.x] = data[(y+j)*mywidth + x];

  __syncthreads();

  x = blockIdx.y * 32 + threadIdx.x;  // transpose block offset
  y = blockIdx.x * 32 + threadIdx.y;
  mywidth = gridDim.y * 32;

  // to saturate uchar8 at -4. and 10.
  float v, scf;
  for (int j = 0; j < 32; j += 8) {
    v = __half2float(tile[threadIdx.x][threadIdx.y + j]);
    scf = 255.f/14.f;
    v = scf*(v+4.f);
    if (v<0.f) v = 0.f;
    if (v>255.f) v = 255.f;
    beam[(y+j)*mywidth + x] = (unsigned char)(v);
  }

}

// TODO: handle_transpose_input and handle_transpose_output to do transpose via memcpy2d from intermediate array

void transpose_input_handler(unsigned char * d_data, half * batch, int width, int stride) {

  dim3 dimBlockIn(32, 8), dimGridIn(NCHAN/32, width/32);
  half * tmpBuffer{nullptr};
  checkCuda(cudaMalloc(&tmpBuffer, sizeof(half) * NCHAN * width));

  // do transpose by beam
  for (uint64_t bm=0; bm<NBATCH; bm++) {
    transpose_input<<<dimGridIn,dimBlockIn>>>(d_data+bm*NCHAN*width,tmpBuffer,width);
    checkCuda(cudaMemcpy2D(batch+bm*NCHAN*stride,stride*sizeof(half),tmpBuffer,width*sizeof(half),width*sizeof(half),NCHAN,cudaMemcpyDeviceToDevice));
  }

  checkCuda(cudaDeviceSynchronize());
  checkCuda(cudaFree(tmpBuffer));
}

void transpose_output_handler(unsigned char * d_data, half * batch, int width, int stride) {

  dim3 dimBlockOut(32, 8), dimGridOut(width/32, NCHAN/32);
  half * tmpBuffer;
  checkCuda(cudaMalloc(&tmpBuffer, NCHAN * width * sizeof(half)));

  // do transpose by beam
  for (int bm=0; bm<NBATCH; bm++) {
    checkCuda(cudaMemcpy2D(tmpBuffer,width*sizeof(half),batch+bm*NCHAN*stride,stride*sizeof(half),width*sizeof(half),NCHAN,cudaMemcpyDeviceToDevice));
    transpose_output<<<dimGridOut,dimBlockOut>>>(d_data+bm*NCHAN*width,tmpBuffer,width);
  }

  checkCuda(cudaDeviceSynchronize());
  checkCuda(cudaFree(tmpBuffer));
}

// kernel to sort out time series
// run with width*NBATCH/32 blocks of 32 threads
__global__ void fix_ts(half * temp_ts, float * ts, int width, int stride) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int idx = bid*32+tid;
  int bat = (int)(idx/NBATCH);
  int tim = (int)(idx % NBATCH);

  ts[idx] = __half2float(temp_ts[bat*stride+tim]);

}


// cuda kernel to measure per-beam time baseline
// run with NBATCH*width lots of 32 threads
__global__ void measure_ts(half * data, float * ts, int width, int stride) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int id = bid*32+tid;
  int iTime = (int)(bid % width);
  int iBatch = (int)(bid / width);

  int npartials = (int)(NCHAN / 32); // number partial sums
  half fac = (half)((32.*npartials));
  extern __shared__ half cpsum[32];

  // calculate partial sums

  cpsum[tid] = 0.;
  int idx0 = iBatch*NCHAN*stride + tid*npartials*stride + iTime;
  for (int i=idx0;i<idx0+npartials*stride;i+=stride)
    cpsum[tid] += data[i];
  cpsum[tid] /= fac;

  __syncthreads();

  // sum over shared memory
  if (tid < 16) { cpsum[tid] += cpsum[tid + 16]; } __syncthreads();
  if (tid < 8) { cpsum[tid] += cpsum[tid + 8]; } __syncthreads();
  if (tid < 4) { cpsum[tid] += cpsum[tid + 4]; } __syncthreads();
  if (tid < 2) { cpsum[tid] += cpsum[tid + 2]; } __syncthreads();
  if (tid < 1) { cpsum[tid] += cpsum[tid + 1]; } __syncthreads();
  //if (tid < 32) warpReduce(psum, tid);
  __syncthreads();

  if (tid==0) ts[bid] = __half2float(cpsum[0]);
}

__global__ void measure_ts_new(half * data, float * ts, int width, int stride)
{
  // beam == blockIdx.y
  // time == blockIdx.x * blockDim.x + threadIdx.x

  int isamp = blockIdx.x * blockDim.x + threadIdx.x;
  if (isamp >= width)
    return;

  int idx = (blockIdx.y * NCHAN * stride) + isamp;
  half sum = 0;
  for (int ichan=0; ichan<NCHAN; ichan++)
  {
    sum += data[idx];
    idx += stride;
  }
  ts[blockIdx.y * width + isamp] = __half2float(sum) / float(NCHAN);
}

// function to bandpass-correct data
float bandpass_correct(pinfo * p, half * data, int width, int stride) {

  // allocate bandpass
  // float * d_bandpass{nullptr};
  // checkCuda(cudaMalloc(&d_bandpass, NBATCH * NCHAN * sizeof(float)));

  if (!p->have_bp)
  {
    // calculate bandpass
    calc_bandpass_new<<<NCHAN*NBATCH/8,256>>>(data, p->d_bp, p->d_bp_sq, p->d_bpout, width, stride);
    p->have_bp = true;
  }

  checkCuda(cudaDeviceSynchronize());

  // median filter bandpass
  float mn_bp = medFilterBandpass(p->d_bpout);

  // correct bandpass in data
  unsigned nt = 512;
  divide_by_bp<<<NBATCH*NCHAN*width/nt,nt>>>(data, p->d_bpout, width, stride);

  checkCuda(cudaDeviceSynchronize());
  // checkCuda(cudaFree(d_bandpass));

  return mn_bp;

}

// function to ts-correct data
void ts_correct(half * data, float * d_ts, int width, int stride) {

  // calculate ts
  // measure_ts<<<NBATCH*width,32>>>(data, d_ts, width, stride);

  // calculate ts in the new way
  dim3 blockDim(256, 1, 1);
  dim3 gridDim(width/blockDim.x, NBATCH, 1);
  if (width % blockDim.x != 0)
    gridDim.x++;
  measure_ts_new<<<gridDim, blockDim>>>(data, d_ts, width, stride);

  // checkCuda(cudaDeviceSynchronize());

  // median filter ts
  //medFilterTs(d_ts,width);

  // correct ts in data
  // divide_by_ts<<<NBATCH*NCHAN*width/32,32>>>(data,d_ts,width,stride,0);

  gridDim.z = NCHAN;
  divide_by_ts_new<<<gridDim, blockDim>>>(data,d_ts,width,stride,0);
  checkCuda(cudaDeviceSynchronize());
}


// function to remove time-frequency baseline
void remove_tf_baseline(half * data, int width, int stride) {

  // allocate smooth array
  half * d_smooth;
  checkCuda(cudaMalloc(&d_smooth, NBATCH * NCHAN * stride * sizeof(half)));

  // smooth data
  //smooth_data<<<NBATCH*NCHAN*width/32,32>>>(data, d_smooth, (float)(1./NTIME_BOX/NCHAN_BOX), NTIME_BOX, NCHAN_BOX, width, stride);
  npp_convolve_handler(data, d_smooth, (float)(1./NTIME_BOX/NCHAN_BOX), NTIME_BOX, NCHAN_BOX, width, stride);

  // divide by smoothed data
  divide_by_array<<<NBATCH*NCHAN*width/32,32>>>(data,d_smooth,width,stride);

  checkCuda(cudaDeviceSynchronize());
  checkCuda(cudaFree(d_smooth));

}

// function to normalize data
// assume initial mean is 1
void normalize_data(pinfo *p, half * data, int width, int stride) {

  float stdDev = calculateStdDev(p, data,width,NBATCH*NCHAN,stride);

  int nt = 512;

  half c = __float2half(-1.f);
  half m = __float2half(1.f/stdDev);
  // add_number<<<NBATCH*NCHAN*width/nt,nt>>>(data,c,width,stride);
  // multiply_by_number<<<NBATCH*NCHAN*width/nt,nt>>>(data,m,width,stride);

  dim3 gridDim(width/nt, NBATCH*NCHAN, 1);
  if (width % nt != 0)
    gridDim.x++;
  add_then_multiply_by_number<<<gridDim,nt>>>(data,c,m,width,stride);
  checkCuda(cudaDeviceSynchronize());
}

// function to implement bandpass flagging on data
float bandpass_flag(pinfo * p, half * data) {

  // bandpass correct
  float mn_bp = bandpass_correct(p, data,p->NTIME, p->batch_stride);

  // normalize data
  normalize_data(p, data,p->NTIME, p->batch_stride);

  if (!p->have_bp)
  {
    // calculate bandpass
    calc_bandpass_new<<<NCHAN*NBATCH/8,256>>>(data, p->d_bp, p->d_bp_sq, p->d_bpout, p->NTIME, p->batch_stride);
    p->have_bp = true;
  }

  // flag data
  // replace_data_bandpass<<<NBATCH*NCHAN*p->NTIME/32,32>>>(data, p->d_bpout, 0., p->NTIME, p->batch_stride, p->spec_min, p->spec_max);
  // finish up
  // add_number<<<NBATCH*NCHAN*p->NTIME/32,32>>>(data,__float2half(1.f),p->NTIME, p->batch_stride);

  // fuse the replace_data_bandpass and add_number kernels
  dim3 blockDim(32 ,16 ,1);
  dim3 gridDim(p->NTIME/blockDim.x, NBATCH*NCHAN/blockDim.y, 1);
  if (p->NTIME % blockDim.x != 0)
    gridDim.x++;
  replace_data_and_add_bandpass<<<gridDim,blockDim>>>(data, p->d_bpout, p->NTIME, p->batch_stride, p->spec_min, p->spec_max);

  return mn_bp;
}

// function to apply a single scrunch to the data
float apply_scrunch(pinfo * p, half * data, half * mask, half * d_smooth, float * d_ts, int width, int stride, int tscrunch, int fscrunch, float thresh, int flag, int ts, float * d_flagSpec, int flag1, int flag2) {

  float begin, end;
  float * d_mask;
  checkCuda(cudaMalloc(&d_mask, NBATCH * NCHAN * sizeof(float)));

  // bandpass
  //printf("bandpass\n");
  begin = clock();
  float mn_bp = bandpass_correct(p, data,width,stride);
  checkCuda(cudaDeviceSynchronize());
  end = clock();
  p->t1 += (float)(end - begin) / CLOCKS_PER_SEC;

  // baseline
  //printf("baseline\n");
  if (ts==1) {
    begin = clock();
    ts_correct(data,d_ts,width,stride);
    end = clock();
    p->t2 += (float)(end - begin) / CLOCKS_PER_SEC;
  }

  // normalize
  //printf("normalize\n");
  begin = clock();
  normalize_data(p, data,width,stride);
  end = clock();
  p->t3 += (float)(end - begin) / CLOCKS_PER_SEC;

  if (flag==1) {

    // derive smoothed data
    //printf("smooth\n");
    begin = clock();
    //smooth_data<<<NBATCH*NCHAN*width/32,32>>>(data, d_smooth, 1./tscrunch/fscrunch, tscrunch, fscrunch, width, stride);
    npp_convolve_handler(data, d_smooth, 1./tscrunch/fscrunch, tscrunch, fscrunch, width, stride);
    checkCuda(cudaDeviceSynchronize());
    end = clock();
    p->t4 += (float)(end - begin) / CLOCKS_PER_SEC;

    // threshold data
    //printf("threshold\n");
    begin = clock();
    checkCuda(cudaMemset(mask,0,NBATCH*NCHAN*stride*sizeof(half)));
    threshold_data<<<NBATCH*NCHAN*width/32,32>>>(d_smooth,mask,thresh/sqrt(1.*tscrunch*fscrunch),width,stride);
    checkCuda(cudaDeviceSynchronize());
    end = clock();
    p->t5 += (float)(end - begin) / CLOCKS_PER_SEC;

    // replace data after growing mask
    //printf("replace\n");
    begin = clock();
    //smooth_data<<<NBATCH*NCHAN*width/32,32>>>(mask, d_smooth, 1./tscrunch/fscrunch, tscrunch, fscrunch, width, stride);
    npp_convolve_handler(mask, d_smooth, 1./tscrunch/fscrunch, tscrunch, fscrunch, width, stride);
    replace_data<<<NBATCH*NCHAN*width/32,32>>>(data, d_smooth, 0., width, stride, flag1, flag2);
    add_number<<<NBATCH*NCHAN*width/512,512>>>(data,__float2half(1.f),width,stride);
    calc_bandpass<<<NCHAN*NBATCH,256>>>(d_smooth, d_mask, width, stride);
    add_bandpass<<<NCHAN*NBATCH/32,32>>>(d_mask,d_flagSpec);
    checkCuda(cudaDeviceSynchronize());
    end = clock();
    p->t6 += (float)(end - begin) / CLOCKS_PER_SEC;

  }

  checkCuda(cudaFree(d_mask));

  return mn_bp;

}

// flagger
// load a batch beam by beam
// apply all scrunches to the batch
// unload the batch
void fastflagger(pinfo * p) {

  float begin, end;

  // setup
  int nBatches = (int)(NBEAMS / NBATCH);
  checkCuda(cudaMemset(p->d_flagSpec,0,4*NBATCH*NCHAN));
  syslog(LOG_INFO,"have nbatches %d",nBatches);
  float mn_bp[nBatches], tmp;

  // output bandpass
  FILE *fout{nullptr};
  float *h_bpout{nullptr};
  char fnam[200];
  if (p->output_bandpass>0) {
    h_bpout = (float *)malloc(sizeof(float)*NBATCH*NCHAN);
    sprintf(fnam,"/home/ubuntu/data/bpout_%d.tmp",p->output_bandpass);
    fout=fopen(fnam,"w");
  }

  // loop over batches
  for (uint64_t batch = 0; batch < nBatches; batch++) {

    // load a batch
    begin = clock();
    transpose_input_handler(p->d_data+batch*NBATCH*NCHAN*p->NTIME,p->batch,p->NTIME,p->batch_stride);
    p->have_bp = false;
    end = clock();
    p->t7 += (float)(end - begin) / CLOCKS_PER_SEC;

    // output init bandpass
    if (p->output_bandpass>0) {
      begin = clock();
      if (!p->have_bp)
      {
        calc_bandpass_new<<<NCHAN*NBATCH/8,256>>>(p->batch, p->d_bp, p->d_bp_sq, p->d_bpout, p->NTIME, p->batch_stride);
        p->have_bp = true;
      }
      checkCuda(cudaMemcpy(h_bpout, p->d_bpout,NBATCH*NCHAN*4,cudaMemcpyDeviceToHost));
      for (int i=0;i<NBATCH*NCHAN;i++)
        fprintf(fout,"%g\n",h_bpout[i]);
      end = clock();
      p->t8 += (float)(end - begin) / CLOCKS_PER_SEC;
    }

    // bandpass flag / correct
    mn_bp[batch] = bandpass_flag(p,p->batch);

    // loop over scrunches
    for (int scrnch=0;scrnch<p->nscrunches;scrnch++) {
      printf("scrunch %d...",scrnch);
      tmp = apply_scrunch(p, p->batch, p->mask, p->d_smooth, p->d_ts, p->NTIME, p->batch_stride, p->scrunches[scrnch].tscrunch,p->scrunches[scrnch].fscrunch, p->scrunches[scrnch].thresh,1,0,p->d_flagSpec,p->flag1,p->flag2);
      checkCuda(cudaDeviceSynchronize());
    }
    tmp = apply_scrunch(p, p->batch, p->mask, p->d_smooth, p->d_ts, p->NTIME, p->batch_stride, 8, 8, 100., 0, 1, p->d_flagSpec,p->flag1,p->flag2);
    // printf("\n");

    checkCuda(cudaDeviceSynchronize());

    // output final bandpass
    if (p->output_bandpass>0) {
      begin = clock();
      calc_bandpass<<<NCHAN*NBATCH,256>>>(p->batch, p->d_bpout, p->NTIME, p->batch_stride);
      cudaMemcpy(h_bpout,p->d_bpout,NBATCH*NCHAN*4,cudaMemcpyDeviceToHost);
      for (int i=0;i<NBATCH*NCHAN;i++)
        fprintf(fout,"%g\n",h_bpout[i]);
      end = clock();
      p->t8 += (float)(end - begin) / CLOCKS_PER_SEC;
    }

    // unload the batch
    begin = clock();
    transpose_output_handler(p->d_data+batch*NBATCH*NCHAN*p->NTIME,p->batch,p->NTIME,p->batch_stride);
    cudaMemcpy(p->h_flagSpec,p->d_flagSpec,4*NBATCH*NCHAN,cudaMemcpyDeviceToHost);
    end = clock();
    p->t7 += (float)(end - begin) / CLOCKS_PER_SEC;

    //printf("done\n");
    //printf("%g ",mn_bp);

  }
  //printf("\n");

  syslog(LOG_INFO,"fastflagger %g %g %g %g",mn_bp[0],mn_bp[1],mn_bp[2],mn_bp[3]);

  if (p->output_bandpass>0) {
    sprintf(fnam,"mv /home/ubuntu/data/bpout_%d.tmp /home/ubuntu/data/bpout_%d.out",p->output_bandpass,p->output_bandpass);
    system(fnam);
    fclose(fout);
    free(h_bpout);
  }

}

// TEST cuda kernel to load test data
// run with NBATCH*NCHAN*width/32 blocks of 32 threads
__global__ void load_test(float * input, half * data, int width, int stride) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  int idx = bid*32+tid;
  int y = (int)(idx / width);
  int x = (int)(idx % width);
  int iidx = y*stride+x;

  data[iidx] = __float2half(input[idx]);

}

// TEST cuda kernel to unload test data
// run with NBATCH*NCHAN*width/32 blocks of 32 threads
__global__ void unload_test(float * output, half * data, int width, int stride) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  int idx = bid*32+tid;
  int y = (int)(idx / width);
  int x = (int)(idx % width);
  int iidx = y*stride+x;

  output[idx] = __half2float(data[iidx]);

}

// input has shape [NBATCH, NCHAN, 15000]
void apply_batch_test(float * input, float * output, int width, int stride) {

  half * batch, * mask;
  //int stride = 16384;
  //int width = 14912;
  cudaMalloc(&batch, NBATCH * NCHAN * stride * sizeof(half));
  cudaMalloc(&mask, NBATCH * NCHAN * stride * sizeof(half));
  float * d_data;
  cudaMalloc(&d_data, NBATCH * NCHAN * width * sizeof(float));


  /// APPLY TEST HERE

  //transpose_input_handler(d_input,batch,width,stride);

  float * d_ts, begin, end;
  cudaMalloc(&d_ts, NBATCH * width * sizeof(float));
  cudaMemcpy(d_data,input,NBATCH * NCHAN * width * sizeof(float),cudaMemcpyHostToDevice);
  load_test<<<NBATCH*NCHAN*width/32,32>>>(d_data,batch,width,stride);

  begin = clock();
  //measure_ts<<<NBATCH*width,32>>>(batch, d_ts, width, stride);
  ts_correct(batch, d_ts, width, stride);
  end = clock();
  printf("Time %g\n",(float)(end - begin) / CLOCKS_PER_SEC);

  cudaMemcpy(output,d_ts,NBATCH * width * sizeof(float),cudaMemcpyDeviceToHost);


  //transpose_output_handler(d_input,batch,width,stride);

  //unload_test<<<NBATCH*NCHAN*width/32,32>>>(d_data,batch,width,stride);
  //cudaMemcpy(output,d_data,sizeof(float)*NBATCH*NCHAN*width,cudaMemcpyDeviceToHost);
  //cudaMemcpy(output,d_input,NBATCH * NCHAN * width,cudaMemcpyDeviceToHost);


}

/************ END *************/


// function to do all dedispersion stuff
void dedisperse(pinfo *p, int beam) {

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

void smooth(pinfo *p, int scale) {

  NppiSize oSrcSize = {p->ntime_dd,p->ndms};
  NppiPoint oSrcOffset = {0,0};
  NppiSize oSizeROI = {p->ntime_dd,p->ndms};
  NppiSize oOutROI = {p->ntime_out,p->ndms-2};
  Npp32f * pKernel;
  Npp32f w[3] = {0.3,1.,0.3};
  Npp32f v, filtSum;
  NppiSize pKernelSize;
  NppiPoint oAnchor;

  /*int zeros_step;
  NppiSize oROI = {p->ntime_dd,1};
  Npp32f * zeros = nppiMalloc_32f_C1(p->ntime_dd,1,&(zeros_step));
  nppiSet_32f_C1R(0.,zeros,zeros_step,oROI);*/

  float * h_kernel;

  int smit = 0;
  for (int sm=(int)(p->minWidth); sm<(int)(p->maxWidth); sm *= 2) {

    // do smooth
    pKernelSize = {2*sm+1,3};
    cudaMalloc((void **)(&pKernel), 4*(2*sm+1)*3);
    h_kernel  = (float *)malloc(sizeof(float)*(2*sm+1)*3);
    filtSum = 0.;
    for (int i=0;i<3;i++) {
      for (int j=0;j<2*sm+1;j++) {
        v = 1.-0.5*((j-sm*2.)/(sm/2.355))*((j-sm*2.)/(sm/2.355))+0.25*((j-sm*2.)/(sm/2.355))*((j-sm*2.)/(sm/2.355))*0.25*((j-sm*2.)/(sm/2.355))*((j-sm*2.)/(sm/2.355))-0.083*((j-sm*2.)/(sm/2.355))*((j-sm*2.)/(sm/2.355))*0.083*((j-sm*2.)/(sm/2.355))*((j-sm*2.)/(sm/2.355))*0.083*((j-sm*2.)/(sm/2.355))*((j-sm*2.)/(sm/2.355));
        h_kernel[i*(2*sm+1)+j] = w[i]*v*v;
        filtSum += w[i]*v*v;
      }
    }
    oAnchor = {0,1};
    for (int i=0;i<(2*sm+1)*3;i++) h_kernel[i] /= filtSum;
    checkCuda(cudaMemcpy(pKernel,h_kernel,4*(2*sm+1)*3,cudaMemcpyHostToDevice));

    //nppiFilterBorder_32f_C1R(p->d_dedisp,p->d_dedisp_step,oSrcSize,oSrcOffset,p->boxes+smit*p->ndms*p->boxes_step/sizeof(float),p->boxes_step,oSizeROI,pKernel,pKernelSize,oAnchor,NPP_BORDER_REPLICATE);
    checkNpp(nppiFilterBorder_32f_C1R(p->d_dedisp,p->d_dedisp_step,oSrcSize,oSrcOffset,p->imbox,p->imbox_step,oSizeROI,pKernel,pKernelSize,oAnchor,NPP_BORDER_REPLICATE));

    // get rid of first and last DM, and maxWidth/2 from each edge
    checkCuda(cudaMemcpy2D(p->boxes+smit*(p->ndms-2)*p->boxes_step/sizeof(float),p->boxes_step,p->imbox+p->imbox_step/sizeof(float)+(int)(p->maxWidth)/2,p->imbox_step,sizeof(float)*p->ntime_out,p->ndms-2,cudaMemcpyDeviceToDevice));

    smit++;
    checkCuda(cudaFree(pKernel));
    free(h_kernel);

  }

  // zero mean, std 1
  const Npp32f npm = -1.*p->mean;
  if (scale==1) {
    for (smit=0;smit<p->nboxcar;smit++) {

      checkNpp(nppiAddC_32f_C1IR(npm,p->boxes+smit*(p->ndms-2)*p->boxes_step/sizeof(float),p->boxes_step,oOutROI));
      checkNpp(nppiMulC_32f_C1IR((const Npp32f)(1./p->stds[smit]),p->boxes+smit*(p->ndms-2)*p->boxes_step/sizeof(float),p->boxes_step,oOutROI));

      // flag first DM trial
      //cudaMemcpy(p->boxes+smit*p->ndms*p->boxes_step/sizeof(float),zeros,p->ntime_dd*sizeof(float),cudaMemcpyDeviceToDevice);
      //cudaMemcpy(p->boxes+smit*p->ndms*p->boxes_step/sizeof(float)+(p->ndms-1)*p->ntime_dd,zeros,p->ntime_dd*sizeof(float),cudaMemcpyDeviceToDevice);


    }
  }

  //cudaFree(pKernel);
  //free(h_kernel);

}

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

void measure_thresholds(pinfo *p) {

  // generate data
  printf("THRESHOLD: Generating random values\n");
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
  printf("THRESHOLD: dedisperse and smooth\n");
  NppiSize preROI = {NCHAN,p->NTIME};
  cudaMemcpy(p->dataFT,randomValues,4*N,cudaMemcpyDeviceToDevice);
  nppiScale_32f8u_C1R(p->dataFT,p->dataFT_step,p->d_datapreT,p->d_datapreT_step,preROI,-4.,10.);
  cudaMemcpy2D(p->data,NCHAN,p->d_datapreT,p->d_datapreT_step,NCHAN,p->NTIME,cudaMemcpyDeviceToHost);

  // dedisperse
  dedisperse(p);

  // smooth
  smooth(p,0);

  // measure stats
  printf("THRESHOLD: measure stats\n");
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

  printf("(Boxcar) Mean Std\n");
  for (int i=0;i<p->nboxcar;i++)
    printf("(%d) %g %g\n",i,hMean[i],hStd[i]);

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

// peak finding

void find_peaks(pinfo *p, int bm) {

  float * dmt_ptr = thrust::raw_pointer_cast(&p->dmt[0]);
  float * d_outputs = thrust::raw_pointer_cast(&p->output_values[0]);
  int * d_idxs = thrust::raw_pointer_cast(&p->output_indices[0]);
  int n_found;
  p->npeaks = 0;
  float myStd;

  for (int sm=0;sm<p->nboxcar;sm++) {

    // measure rms - should be 1
    //calculateStdDevFloat(float * d_data, int width, int height, int stride) {
    if (sm==0) {
      myStd = calculateStdDevFloat(p->boxes+sm*(p->ndms-2)*p->boxes_step/sizeof(float),p->ntime_out,p->ndms-2,p->boxes_step/sizeof(float));
      if (bm==32) syslog(LOG_INFO,"STDDEV %g",myStd);
      if (myStd<STD_DEV_LOW_THRESHOLD) myStd = 1.;
      if (myStd<STD_DEV_VERY_LOW_THRESHOLD) myStd = 2.;
    }
    if (myStd<STD_DEV_HIGH_THRESHOLD && myStd>STD_DEV_VERY_HIGH_THRESHOLD) {

      // copy to thrust vector
      //cudaMemcpy(dmt_ptr,p->boxes+sm*(p->ndms-2)*p->boxes_step/sizeof(float),(p->ndms-2)*p->boxes_step,cudaMemcpyDeviceToDevice);
      checkCuda(cudaMemcpy2D(dmt_ptr,p->ntime_out*4,p->boxes+sm*(p->ndms-2)*p->boxes_step/sizeof(float),p->boxes_step,p->ntime_out*4,p->ndms-2,cudaMemcpyDeviceToDevice));

      // Find indices and values of points greater than the threshold
      //  thrust::copy(p->dmt.begin(), p->dmt.begin() + 20, std::ostream_iterator<float>(std::cout, " "));
      thrust::device_vector<int>::iterator end = thrust::copy_if(thrust::make_counting_iterator(0),
                                                                 thrust::make_counting_iterator(p->ntime_out*(p->ndms-2)),
                                                                 p->dmt.begin(),
                                                                 p->output_indices.begin(),
                                                                 thrust::placeholders::_1 > p->snr*myStd);
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

  }

  // sort out stuff on host
  int tmp;
  int imax;
  //std::cout << p->npeaks << std::endl;
  if (p->out_npeaks+p->npeaks>MAX_GIANTS)
    imax = MAX_GIANTS;
  else
    imax = p->out_npeaks+p->npeaks;
  for (int i=p->out_npeaks;i<imax;i++) {

    //printf("%d %d %d\n",p->h_idxs[i],(int)(p->h_idxs[i] % p->ntime_dd),(int)(p->h_idxs[i] / p->ntime_dd));
    p->out_peaks[i] = p->peaks[i-p->out_npeaks];
    p->out_beam[i] = bm;
    p->out_samp[i] = (int)(p->h_idxs[i-p->out_npeaks] % p->ntime_out);
    tmp = (int)(p->h_idxs[i-p->out_npeaks] / p->ntime_out);
    p->out_width[i] = (int)(tmp / (p->ndms-2));
    p->out_dm_idx[i] = (int)(tmp % (p->ndms-2)) + 1;

  }
  p->out_npeaks = imax;

}

// to clear all peaks
void clear_peaks(pinfo *p) {

  memset(p->out_peaks,0,MAX_GIANTS*sizeof(float));
  memset(p->out_beam,0,MAX_GIANTS*sizeof(int));
  memset(p->out_samp,0,MAX_GIANTS*sizeof(int));
  memset(p->out_width,0,MAX_GIANTS*sizeof(int));
  memset(p->out_dm_idx,0,MAX_GIANTS*sizeof(int));
  p->out_npeaks = 0;
}

// output peaks
void output_peaks(pinfo *p, int samp, int restart_socket) {

  // if output is file or both
  if (p->out_format != 1) {

    FILE *fout{nullptr};
    fout=fopen(p->out_path,"a");

    for (int i=0;i<p->out_npeaks;i++) {
      // if (p->samp[i]>p->maxWidth/2 && p->samp[i]<=p->ntime_dd-p->maxWidth/2)
      //   fprintf(stderr,"A %g %d %g %d %d %g %d\n",p->peaks[i],p->samp[i]+samp,TIME_CONVERSION_FACTOR*(p->samp[i]+samp),p->width[i],p->dm_idx[i],p->DMs[p->dm_idx[i]]);
      // else
      //   fprintf(stderr,"B %g %d %g %d %d %g %d\n",p->peaks[i],p->samp[i]+samp,TIME_CONVERSION_FACTOR*(p->samp[i]+samp),p->width[i],p->dm_idx[i],p->DMs[p->dm_idx[i]]);
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
    syslog(LOG_INFO,"opening socket");
    memset ( &m_addr, 0, sizeof ( m_addr ) );
    m_sock = socket ( AF_INET, SOCK_STREAM, 0 );
    if (m_sock==-1) {
      syslog(LOG_ERR,"Socket exception: could not create socket");
      return;
    }

    // connect stuff
    m_addr.sin_family = AF_INET;
    m_addr.sin_port = htons ( p->coincidencer_port );
    m_addr.sin_addr.s_addr = inet_addr (p->coincidencer_host.c_str());
    sstat = connect ( m_sock, ( sockaddr * ) &m_addr, sizeof ( m_addr )) ;

    if (sstat!=0) {
      syslog(LOG_ERR,"Socket exception: could not open socket: %d",sstat);
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
      syslog(LOG_INFO,"sending data");
      sstat = send ( m_sock, s.c_str(), s.size(), MSG_NOSIGNAL );
      if (sstat==-1) {
        syslog(LOG_ERR,"Socket exception: could not send cand");
        return;
      }

      oss.flush();
      oss.str("");

    }

    // close socket
    if (m_sock != -1) {
      syslog(LOG_INFO,"closing socket AFTER");
      sstat = close( m_sock );
      if (sstat!=0) {
        syslog(LOG_ERR,"Socket exception: could not close socket: %d",sstat);
        return;
      }
      m_sock = -1;
    }

  }

}

/* read a string from the input which looks like nchars-char[1-nchars] */
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

int strings_equal (char *string1, char *string2) /* includefile */
{
  if (!strcmp(string1,string2)) {
    return 1;
  } else {
    return 0;
  }
}

/* attempt to read in the general header info from a pulsar data file */
int read_header (FILE *inputfile) /* includefile */
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

// deals with data IO
int main(int argc, char *argv[]) {

  // parse command line
  FILE *fconf{nullptr};
  int core = -1;
  for (int i=1;i<argc;i++) {

    // configuration
    if (strcmp(argv[i],"-c")==0) {
      fconf=fopen(argv[i+1],"r");
      syslog(LOG_INFO,"Getting config from %s\n",argv[i+1]);
      fprintf(stderr, "Getting config from %s\n",argv[i+1]);
    }
    if (strcmp(argv[i],"-i")==0) {
      core = atoi(argv[i+1]);
    }
    // help
    if (strcmp(argv[i],"-h")==0) {
      help();
      exit(1);
    }

  }

  // set GPU ID
  cudaSetDevice(get_gpu_id(fconf));
  int currentDevice;
  cudaGetDevice(&currentDevice);

  // startup syslog message
  // using LOG_LOCAL0
  if (currentDevice==0)
    openlog ("dsaX_hella0", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  else
    openlog ("dsaX_hella1", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  syslog (LOG_NOTICE, "Program started by User %d", getuid ());


  syslog(LOG_INFO,"Using GPU ID %d\n",currentDevice);

  // Bind to cpu core
  if (core >= 0)
    {
      syslog(LOG_INFO,"binding to core %d\n", core);
      if (dada_bind_thread_to_core(core) < 0)
        syslog(LOG_ERR,"failed to bind to core %d\n", core);
    }


  // set up pipeline, allocate appropriate mem
  pinfo p;
  float tflags = 0.;

  initialize(fconf,&p);
  FILE *fin{nullptr}, *ftest{nullptr};

  // allocate these arrays after initialize is claled since p.NTIME can be adjusted there
  unsigned char * hodata = (unsigned char *)malloc(sizeof(unsigned char)*p.NTIME*NCHAN);
  float * h_ts = (float *)malloc(sizeof(float)*p.NTIME*NBATCH);

  if (p.inp_format==3) {

    // read header
    fin=fopen(p.inp_path,"rb");
    int nbytes_header = read_header(fin);
    fclose(fin);
    char * heade = (char *)malloc(sizeof(char)*nbytes_header);
    fin=fopen(p.inp_path,"rb");
    fread(heade, sizeof(char), nbytes_header, fin);
    free(heade);
    syslog(LOG_INFO,"Finished with header (nbytes %d) of input filFile %s\n",nbytes_header,p.inp_path);

    // read data
    fread(p.data,sizeof(char),p.NTIME*NCHAN,fin);
    if (NBEAMS>1) {
      for (int i=1;i<NBEAMS;i++)
        memcpy(p.data+i*p.NTIME*NCHAN,p.data,p.NTIME*NCHAN);
    }
    cudaMemcpy(p.d_data,p.data,NBEAMS*p.NTIME*NCHAN,cudaMemcpyHostToDevice);
    fclose(fin);

    // flag it
    fastflagger(&p);

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

    printf("TOT FLAGS %g\n",tflags);

    exit(1);

  }





  // begin read of data
  float v;
  unsigned char * tmpbuf = (unsigned char *)malloc(sizeof(unsigned char)*NCHAN*p.gulp*NBEAMS*2);

  // DADA Header plus Data Unit
  dada_hdu_t* hdu_in = 0;
  dada_hdu_t* hdu_out = 0;
  key_t in_key = DADA_BLOCK_KEY;
  key_t out_key = DADA_BLOCK_KEY;
  char * header_in, * header_out;
  uint64_t header_size = 0;
  uint64_t block_size, block_out;

  // dada input
  if (p.inp_format==0) {

    sscanf(p.inp_path, "%x", &in_key);
    multilog_t* log = multilog_open("casm_hella", LOG_INFO);
    hdu_in  = dada_hdu_create (log);
    dada_hdu_set_key (hdu_in, in_key);
    dada_hdu_connect (hdu_in);
    dada_hdu_lock_read (hdu_in);
    header_in = ipcbuf_get_next_read (hdu_in->header_block, &header_size);
    ipcbuf_mark_cleared (hdu_in->header_block);
    block_size = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_in->data_block);
    syslog(LOG_INFO,"Connected to dada buffer\n");

    sscanf(p.dada_out, "%x", &out_key);
    hdu_out  = dada_hdu_create (log);
    dada_hdu_set_key (hdu_out, out_key);
    dada_hdu_connect (hdu_out);
    dada_hdu_lock_write(hdu_out);
    header_out = ipcbuf_get_next_write (hdu_out->header_block);
    memcpy (header_out, header_in, header_size);
    ipcbuf_mark_filled (hdu_out->header_block, header_size);
    block_out = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_out->data_block);
    syslog(LOG_INFO,"Ready for output buffer\n");

  }

  // text file input
  if (p.inp_format==1) {
    fin=fopen(p.inp_path,"r");
    for (int i=0;i<p.NTIME*NCHAN;i++) {
      fscanf(fin,"%f\n",&v);
      p.data[i] = (unsigned char)(v);
    }
    fclose(fin);
  }

  // filterbank input
  if (p.inp_format==2) {
    fin=fopen(p.inp_path,"rb");

    int nbytes_header = read_header(fin);
    fclose(fin);
    char * heade = (char *)malloc(sizeof(char)*nbytes_header);
    fin=fopen(p.inp_path,"rb");
    fread(heade, sizeof(char), nbytes_header, fin);
    free(heade);
    syslog(LOG_INFO,"Finished with header (nbytes %d) of input filFile %s\n",nbytes_header,p.inp_path);
  }

  syslog(LOG_INFO,"Starting...\n");
  int samp = 0;
  if (p.inp_format!=1)
    samp = -(p.NTIME-p.gulp) + (int)(p.maxWidth)/2;
  int gulp = 0;

  //measure_thresholds(&p);

  // set up output
  FILE *fout, *fspec, *fbeam;
  int beamflags[NBEAMS], specflags[NCHAN];
  int bm{0};

  // loop over data gulps, figuring out at the end if we're finished

  // dada stuff
  char * block;
  uint64_t  bytes_read = 0;
  uint64_t block_id, written;

  // timer stuff
  float readt = 0., flagt = 0., dedispt = 0., smootht = 0., peakt = 0., outputt = 0.;
  float tot_time;
  clock_t begin, end;

  // outputs
  //float * hodata = (float *)malloc(sizeof(float)*p.ntime_out*(p.ndms-2));
  float tot_flags = 0.;
  int socket_count = 0;

  while (finished==0)
  {
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
    {
      fread(tmpbuf, sizeof(unsigned char), NBEAMS*p.gulp*NCHAN, fin);
    }

    // set up logging and reset output
    //for (int i=0;i<NBEAMS;i++) beamflags[i] = 0;
    for (int i=0;i<NCHAN;i++) specflags[i] = 0;
    clear_peaks(&p);

    syslog(LOG_INFO,"Starting gulp %d\n",gulp);

    const size_t beam_time_stride = p.NTIME * NCHAN;
    const size_t beam_gulp_stride = p.gulp * NCHAN;
    const size_t beam_rewind_stride = (p.NTIME - p.gulp) * NCHAN;
    // loop over beams to read in data
    begin = clock();
    for (int bmm=0;bmm<NBEAMS;bmm++) {

      // get data from reader

      // dada input
      if (p.inp_format==0) {
        memcpy(p.data + beam_time_stride*bmm + beam_rewind_stride, block + beam_gulp_stride*(bmm+p.BEAM_OFFSET), beam_gulp_stride);
        memcpy(p.data + beam_time_stride*bmm, p.rewinds + beam_rewind_stride*bmm, beam_rewind_stride);
        memcpy(p.rewinds + beam_rewind_stride*bmm, p.data + beam_time_stride*bmm + beam_gulp_stride, beam_rewind_stride);
      }

      // filterbank input
      if (p.inp_format==2) {
        memcpy(p.data + bmm*p.NTIME*NCHAN + NCHAN*(p.NTIME-p.gulp),tmpbuf+(bmm+p.BEAM_OFFSET)*p.gulp*NCHAN,p.gulp*NCHAN);
        memcpy(p.data + bmm*p.NTIME*NCHAN, p.rewinds + bmm*NCHAN*(p.NTIME-p.gulp), NCHAN*(p.NTIME-p.gulp));
        memcpy(p.rewinds + bmm*NCHAN*(p.NTIME-p.gulp), p.data + bmm*p.NTIME*NCHAN + NCHAN*p.gulp, NCHAN*(p.NTIME-p.gulp));
      }

    }

    // copy to device
    cudaMemcpy(p.d_data,p.data,NBEAMS*p.NTIME*NCHAN,cudaMemcpyHostToDevice);
    end = clock();
    readt += (float)(end - begin) / CLOCKS_PER_SEC;

    // if gulp is zero
    if (p.inp_format==0 && gulp==0) {

      for (int bmm=0;bmm<NBEAMS;bmm++)
        checkCuda(cudaMemcpy(p.data + beam_gulp_stride*bmm, p.d_data + beam_time_stride*bmm + beam_rewind_stride, beam_gulp_stride, cudaMemcpyDeviceToHost));
      written = ipcio_write (hdu_out->data_block, (char *)(p.data), block_out);
    }

    if ((gulp>0 && p.inp_format!=1) || (p.inp_format==1)) {

      begin = clock();
      // printf("fastflagger\n");
      fastflagger(&p);

      // deal with flags
      for (int j=0;j<NBATCH;j++) {
        for (int i=0;i<NCHAN;i++) {
          //beamflags[bm] += (int)(p.h_flagSpec[j*NCHAN+i]);
          specflags[i] += (int)(1.*p.NTIME*p.h_flagSpec[j*NCHAN+i]);
                tot_flags += (1.*p.NTIME*p.h_flagSpec[j*NCHAN+i])/FLAG_NORMALIZATION_FACTOR;
        }
      }
      end = clock();
      flagt += (float)(end - begin) / CLOCKS_PER_SEC;

      // write to dada
      begin = clock();
      if (p.inp_format==0) {
        for (uint64_t bmm=0;bmm<NBEAMS;bmm++)
          checkCuda(cudaMemcpy(p.data + bmm*p.gulp*NCHAN, p.d_data + bmm*p.NTIME*NCHAN + NCHAN*(p.NTIME-p.gulp),p.gulp*NCHAN, cudaMemcpyDeviceToHost));
        written = ipcio_write (hdu_out->data_block, (char *)(p.data), block_out);
      }
      end = clock();
      readt += (float)(end - begin) / CLOCKS_PER_SEC;

      // write out to disk
      /*cudaMemcpy(hodata,p.d_data,NCHAN*p.NTIME,cudaMemcpyDeviceToHost);
      ftest = fopen("image.out","w");
      for (int i=0;i<NCHAN*p.NTIME;i++)
        fprintf(ftest,"%f\n",(float)(hodata[i]));
        fclose(ftest);*/


      // loop over beams to dedisperse and search
      // check time, out_npeaks
      //printf("Looping over beams...\n");
      bm = 0;
      tot_time = readt+flagt;
      // while ((bm<NBEAMS) && (tot_time<PROCESSING_TIME_LIMIT) && (p.out_npeaks < MAX_GIANTS)) {
      while ((bm<NBEAMS) && (p.out_npeaks < MAX_GIANTS)) {
        //while ((bm<NBEAMS) && (p.out_npeaks < MAX_GIANTS))

        // printf("dedisperse\n");
        begin =        clock();
        dedisperse(&p,bm);
        end = clock();
        dedispt += (float)(end - begin) / CLOCKS_PER_SEC;

        // printf("begin smooth\n");
        begin = clock();
        smooth(&p,1);
        end = clock();
        smootht += (float)(end - begin) / CLOCKS_PER_SEC;


        // printf("rest\n");
        begin = clock();
        find_peaks(&p,bm);
        end = clock();
        peakt += (float)(end - begin) / CLOCKS_PER_SEC;
        // printf("END   find peaks bm=%d\n", bm);


        tot_time = readt+flagt+dedispt+smootht+peakt;
        bm += 1;

      }

      begin = clock();
      // output peaks
      if (socket_count==0)
        output_peaks(&p,samp,1);
      else
        output_peaks(&p,samp,0);
      // output flags
      //fbeam = fopen(p.beamflags,"a");
      fspec = fopen(p.specflags,"a");
      //for (int i=0;i<NBEAMS;i++) fprintf(fbeam,"%d\n",beamflags[i]);
      for (int i=0;i<NCHAN;i++) fprintf(fspec,"%d\n",specflags[i]);
      //fclose(fbeam);
      fclose(fspec);
      end = clock();
      outputt += (float)(end - begin) / CLOCKS_PER_SEC;

      // increment socket_count
      socket_count++;
      if (socket_count==SOCKET_CADENCE)
        socket_count=0;

    }

    // increment sample
    samp += p.gulp;
    gulp += 1;

    // assume only one gulp for text file input
    if (p.inp_format==1)
      finished = 1;

    // look for eof for fil input
    if (p.inp_format==2)
      if (feof(fin)) finished = 1;

    // close off dada block
    if (p.inp_format==0)
      ipcio_close_block_read (hdu_in->data_block, bytes_read);

    syslog(LOG_INFO,"Beamstats %d giants %d %g\n",bm,p.out_npeaks,tot_flags);
    tot_flags = 0.;
    syslog(LOG_INFO,"processed %g s in read %g flag %g dedisp %g smooth %g peak %g output %g [%g]\n",(p.ntime_dd)*TIME_CONVERSION_FACTOR,readt,flagt,dedispt,smootht,peakt,outputt,readt+flagt+dedispt+smootht+peakt+outputt);
    // fprintf(stderr, "%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\n",(p.ntime_dd)*TIME_CONVERSION_FACTOR,readt,flagt,dedispt,smootht,peakt,outputt,readt+flagt+dedispt+smootht+peakt+outputt);
    syslog(LOG_INFO,"Flagging: %g %g %g %g %g %g %g %g\n",p.t1,p.t2,p.t3,p.t4,p.t5,p.t6,p.t7,p.t8);
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
    dsaX_dbgpu_cleanup (hdu_in, hdu_out);
  deallocator(&p);

}
