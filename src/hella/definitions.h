/***************************************************************************
 *
 *   Copyright (C) 2025 TBA
 *   Copyright (C) 2025 Fourier Space
 *   Authors: TBA, Andrew Jameson
 *
 ***************************************************************************/

#include <dedisp.h>

#include <cuda_fp16.h>
#include <npp.h>
#include <nppdefs.h>
#include <nppcore.h>
#include <nppi.h>
#include <npps.h>
#include <string>
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
#include <syslog.h>

#ifndef HELLA_DEFINITIONS_H
#define HELLA_DEFINITIONS_H

const int MAXHOSTNAME = 200;
const int MAXCONNECTIONS = 5;
const int MAXRECV = 500;

#define NMEDFILT 13
#define NTSMED 7
#define NBATCH 4
#define NCHAN 3072
#define NBEAMS 128
#define NCHAN_BOX 48
#define NTIME_BOX 500
#define MAX_DM 2000
#define TOL 1.3
#define MAX_BOX 15
#define MAX_GIANTS 10000
#define DADA_BLOCK_KEY 0x0000dada // for capture program.
#define SOCKET_CADENCE 1

// CASM frequency and timing parameters
#define FREQ_CHANNEL_WIDTH -0.03075  // 0.03075 MHz
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
  unsigned long NTIME; // gulp that includes rewind
  unsigned long rewind; // samples to rewind by
  unsigned long nchan; // number of resampled channels
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
  float * d_bpout{nullptr};

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

  void * h_scratch{nullptr};
  size_t h_scratch_size{0};
  bool h_scratch_locked{false};

  void * d_scratch{nullptr};
  size_t d_scratch_size{0};
  bool d_scratch_locked{false};

} pinfo_t;

#endif // HELLA_DEFINITIONS_H