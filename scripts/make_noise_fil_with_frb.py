# make_noise_fil_with_frb.py
# Creates a SIGPROC .fil with Gaussian noise (uint8) + one fake FRB (DM=50)
# Compatible with your current setup (no writer dependency needed).

import numpy as np
import struct
from pathlib import Path

# --------- SIGPROC header helpers ---------
def _w_string(fh, s: str):
    b = s.encode("ascii")
    fh.write(struct.pack("<i", len(b)))
    fh.write(b)

def _w_int(fh, key: str, val: int):
    _w_string(fh, key)
    fh.write(struct.pack("<i", int(val)))

def _w_double(fh, key: str, val: float):
    _w_string(fh, key)
    fh.write(struct.pack("<d", float(val)))

def _w_strval(fh, key: str, val: str):
    _w_string(fh, key)
    _w_string(fh, val)

def write_sigproc_header(fh, hdr: dict):
    _w_string(fh, "HEADER_START")
    _w_strval(fh, "rawdatafile", hdr.get("rawdatafile", ""))
    _w_strval(fh, "source_name", hdr.get("source_name", ""))
    _w_int(fh, "machine_id", hdr.get("machine_id", 0))
    _w_int(fh, "telescope_id", hdr.get("telescope_id", 0))
    _w_int(fh, "data_type", hdr.get("data_type", 1))
    _w_int(fh, "barycentric", hdr.get("barycentric", 0))
    _w_int(fh, "pulsarcentric", hdr.get("pulsarcentric", 0))
    _w_int(fh, "nbits", hdr["nbits"])
    _w_int(fh, "nifs", hdr.get("nifs", 1))
    _w_int(fh, "nchans", hdr["nchans"])
    _w_int(fh, "ibeam", hdr.get("ibeam", 1))
    _w_int(fh, "nbeams", hdr.get("nbeams", 1))
    _w_double(fh, "fch1", hdr["fch1"])
    _w_double(fh, "foff", hdr["foff"])
    _w_double(fh, "tsamp", hdr["tsamp"])
    _w_double(fh, "tstart", hdr.get("tstart", 0.0))
    _w_string(fh, "HEADER_END")

# --------- Constants ---------
nchans = 3072
chan_bw_MHz = 0.03051757812
fch1_MHz = 468.75                     # top-of-band frequency (channel 0), MHz
foff_MHz = -chan_bw_MHz              # negative => descending frequency
fref_MHz = fch1_MHz                  # reference (arrives at middle here)
# Dispersion constant: t_delay(ms) = 4.148808 * DM * (1/nu^2 - 1/nu_ref^2), with nu in GHz
K_ms = 4.148808
fref_GHz = fref_MHz / 1000.0

nbits = 8
assert nbits == 8, "This example writes uint8 only"
tsamp_s = 0.001                      # 1 ms

# Channel center frequencies (GHz)
# ch 0 = fch1, ch 1 = fch1 + foff, ...
ch_idx = np.arange(nchans)
freq_MHz = fch1_MHz + ch_idx * foff_MHz
freq_GHz = freq_MHz / 1000.0
# Noise parameters
noise_mean = 128.0
pulse_sigma_ms = 2.0                 # intrinsic Gaussian width (std dev) in ms

def get_header(outfile: Path):
    return {
        "rawdatafile": outfile.name,
        "source_name": "gaussian_noise_plus_frb",
        "machine_id": 0,
        "telescope_id": 0,
        "data_type": 1,
        "fch1": fch1_MHz,
        "foff": foff_MHz,
        "nchans": nchans,
        "nbits": nbits,
        "tsamp": tsamp_s,
        "nifs": 1,
        "barycentric": 0,
        "pulsarcentric": 0,
        "tstart": 0.0,
        "ibeam": 1,
        "nbeams": 1,
    }

def make_config(input_type, input_fname, output_fname, base_config_fname, output_file, nbeam, injected_dm, injected_pulse_amp):

    with open(base_config_fname, 'r') as f:
        for line in f.readlines():
            print(line.strip(), file=output_file)

    print(f"INPUT {input_type}", file=output_file)
    print(f"INPUT_PATH {input_fname}", file=output_file)
    print(f"OUTPUTPATH {output_fname}", file=output_file)
    print(f"NBEAM {nbeam}", file=output_file)
    print(f"INJECTED_DM {injected_dm}", file=output_file)
    print(f"INJECTED_PULSE_AMP {injected_pulse_amp}", file=output_file)
    output_file.flush()

def gen_filterbank(
        duration_samps=8192, 
        noise_std=20.0, 
        DM=100.0, # pc cm^-3
        pulse_amp_counts=8.0, # additive amplitude (rough per-sample SNR ~ pulse_amp / noise_std)
        output_fname_base="./",
        with_noise=True
    ) -> Path:

    duration_s = duration_samps * tsamp_s                    # total length
    out_fname = f"{output_fname_base}noise+frb_DM{DM:.2f}_{nchans}x{chan_bw_MHz:5f}MHz_fch1_{fch1_MHz:.1f}MHz_amp_{pulse_amp_counts:.1f}_{nbits}bit.fil"
    outfile = Path(out_fname)

    header = get_header(outfile)

    # --------- Derived ---------
    nsamp = int(round(duration_s / tsamp_s))

    # Per-channel arrival delay (ms) relative to reference (fref = fch1)
    delay_ms = K_ms * DM * (1.0 / (freq_GHz**2) - 1.0 / (fref_GHz**2))
    delay_samples = np.round(delay_ms / (tsamp_s * 1000.0)).astype(np.int64)  # integer sample offsets

    # Pulse time center (sample index) for each channel
    mid_sample = nsamp // 2
    t0_ch = mid_sample + delay_samples    # shape: (nchans,)

    # Convert pulse sigma to samples; we’ll only draw within ±4σ for speed
    sigma_samp = max(1.0, pulse_sigma_ms / (1000.0 * tsamp_s))  # ensure >= 1 sample
    halfwin = int(np.ceil(4.0 * sigma_samp))

    with open(outfile, "wb") as fh:
        write_sigproc_header(fh, header)

        rng = np.random.default_rng()
        chunk_nsamp = 4096
        for i0 in range(0, nsamp, chunk_nsamp):
            print(i0)
            n = min(chunk_nsamp, nsamp - i0)
            i1 = i0 + n - 1

            # 1) draw noise
            if with_noise:
                block = rng.normal(noise_mean, noise_std, size=(n, nchans)).round().astype(np.float32)
            else:
                block = np.zeros((n, nchans), dtype=np.float32)

            # 2) add FRB where the pulse overlaps this chunk
            # channels whose pulse center falls within [i0-halfwin, i1+halfwin]
            mask = (t0_ch >= (i0 - halfwin)) & (t0_ch <= (i1 + halfwin))
            if np.any(mask):
                ch_sel = np.where(mask)[0]
                t0_sel = t0_ch[mask]  # centers for selected channels

                # For each selected channel, compute the time indices inside the chunk to “paint” the Gaussian
                # Vectorized approach: build a (time, channel) index grid only for the window around each t0
                for j, ch in enumerate(ch_sel):
                    c0 = int(t0_sel[j])
                    # overlap window in absolute samples
                    abs_start = max(i0, c0 - halfwin)
                    abs_end   = min(i1, c0 + halfwin)
                    if abs_start > abs_end:
                        continue

                    # convert to local [0..n-1] indices
                    t_idx = np.arange(abs_start, abs_end + 1) - i0
                    # Gaussian amplitude over these samples
                    # A * exp(-(t - c0)^2 / 2sigma^2)
                    g = pulse_amp_counts * np.exp(-0.5 * ((i0 + t_idx - c0) / sigma_samp)**2, dtype=np.float64)
                    # Add to the block (time, channel)
                    block[t_idx, ch] += g.astype(np.float32)

            # 3) clip/quantize to uint8 and write
            block = np.clip(block, 0, 255).astype(np.uint8, copy=False)
            fh.write(block.tobytes(order="C"))

    print(f"Wrote: {outfile} | nsamp={nsamp}, nchans={nchans}, tsamp={tsamp_s}s")
    print(f"FRB DM={DM} pc cm^-3, referenced to {fref_MHz:.3f} MHz, sigma={pulse_sigma_ms} ms, amp={pulse_amp_counts} counts")

    return outfile

def parse_range_str(range):
    if not ":" in range:
        return [float(range)]

    split = range.split(":")
    assert len(split) == 3

    start = float(split[0])
    step = float(split[1])
    end = float(split[2])

    return np.arange(start, end, step)

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser("make_noise_fil_with_frb")
    p.add_argument("--DM", type=str, help="Dispersion measure of the injected pulse. Can either by specified by a single value or a range of the form start:step:end", required=True)
    p.add_argument("--pulse_amp", type=str, help="Pulse amplitude. Can either by specified by a single value or a range of the form start:step:end", required=True)
    p.add_argument("--dada_nbeam", type=int, help="NBEAM value in the output .dada_cfg file", required=True)
    p.add_argument("--output", type=str, help="Output base filename", required=True)
    p.add_argument("--base_config", type=str, help="Template hella configuration", required=False)
    p.add_argument("--no_noise", action='store_false', dest='with_noise', help="Do not include noise", required=False)
    p.add_argument("--no_config", action='store_false', dest='write_configs', help="Do not write hella configuration files", required=False)

    args = p.parse_args()

    for DM in parse_range_str(args.DM):
        for pulse_amp in parse_range_str(args.pulse_amp):
            fil_path = gen_filterbank(DM=DM, pulse_amp_counts=pulse_amp,with_noise=args.with_noise)

            if args.write_configs:
                with open(fil_path.with_suffix(".fil_cfg"), 'w') as cfg_file:
                    make_config("FILTERBANK", fil_path.absolute(), fil_path.with_suffix(".fil_candidates").absolute(), args.base_config, cfg_file, 1, DM, pulse_amp)

                with open(fil_path.with_suffix(".dada_cfg"), 'w') as cfg_file:
                    make_config("DADA", "dada", fil_path.with_suffix(".dada_candidates").absolute(), args.base_config, cfg_file, args.dada_nbeam, DM, pulse_amp)
                

