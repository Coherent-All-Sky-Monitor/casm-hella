/***************************************************************************
 *
 *   Copyright (C) 2025 TBA
 *   Copyright (C) 2025 Fourier Space
 *   Authors: TBA, Andrew Jameson
 *
 ***************************************************************************/

#include "hella/dada.h"
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

#include <cstdio>

//using namespace std;

#include "hella/definitions.h"
#include "hella/macros.h"

int finished = 0;

//! get the GPU id from the configuration file, identified by the GPU keyword
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

  printf("NBEAMS=%d NCHAN=%d NTIME=%d NBATCH=%d\n", NBEAMS, NCHAN, p->NTIME, NBATCH);
  // allocate everything

  p->h_flagSpec = (float *)malloc(sizeof(float)*NCHAN*NBATCH);
  cudaMalloc((void **)(&p->d_flagSpec), sizeof(float)*NCHAN*NBATCH);
  p->rewinds = (unsigned char *)malloc(sizeof(unsigned char)*NCHAN*(p->NTIME-p->gulp)*NBEAMS);
  memset(p->rewinds,0,NCHAN*(p->NTIME-p->gulp)*NBEAMS);
  p->data = (unsigned char *)malloc(sizeof(unsigned char)*NBEAMS*NCHAN*p->NTIME);
  //p->h_dedisp = (float *)malloc(sizeof(float)*p->ndms*p->ntime_dd);
  //p->indata = (unsigned char *)malloc(sizeof(unsigned char)*NCHAN*p->NTIME);
  p->h_dataF = (float *)malloc(sizeof(float)*NCHAN*p->NTIME);
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
  hella::dedisperse(p);

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

  // allocate these arrays after initialize is called since p.NTIME can be adjusted there
  unsigned char * hodata = (unsigned char *)malloc(sizeof(unsigned char)*p.NTIME*NCHAN);
  float * h_ts = (float *)malloc(sizeof(float)*p.NTIME*NBATCH);

  if (p.inp_format==3) {

    // read header
    fin=fopen(p.inp_path,"rb");
    int nbytes_header = hella::read_header(fin);
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

    int nbytes_header = hella::read_header(fin);
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
      fread(tmpbuf, sizeof(unsigned char), NBEAMS*p.gulp*NCHAN, fin);

    // set up logging and reset output
    //for (int i=0;i<NBEAMS;i++) beamflags[i] = 0;
    for (int i=0;i<NCHAN;i++) specflags[i] = 0;
    hella::clear_peaks(&p);

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
      hella::fast_flagger(&p);

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
      while ((bm<NBEAMS) && (tot_time<PROCESSING_TIME_LIMIT) && (p.out_npeaks < MAX_GIANTS)) {
        //while ((bm<NBEAMS) && (p.out_npeaks < MAX_GIANTS)) {

        // printf("dedisperse\n");
        begin =        clock();
        hella::dedisperse(&p,bm);
        end = clock();
        dedispt += (float)(end - begin) / CLOCKS_PER_SEC;

        // printf("begin smooth\n");
        begin = clock();
        hella::smooth(&p,1);
        end = clock();
        smootht += (float)(end - begin) / CLOCKS_PER_SEC;


        // printf("rest\n");
        begin = clock();
        hella::find_peaks(&p,bm);
        end = clock();
        peakt += (float)(end - begin) / CLOCKS_PER_SEC;
        // printf("END   find peaks bm=%d\n", bm);


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
    //std::cout << "Finished: " << finished << std::endl;

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
    hella::hdu_cleanup(hdu_in, hdu_out);

  deallocator(&p);

}
