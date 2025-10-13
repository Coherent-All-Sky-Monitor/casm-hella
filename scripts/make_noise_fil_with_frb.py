# make_noise_fil_with_frb.py
# Creates a SIGPROC .fil with Gaussian noise (uint8) + one fake FRB (DM=50)
# Compatible with your current setup (no writer dependency needed).

import numpy as np
import struct
from pathlib import Path

# --------- Spec ---------
nchans = 3072
chan_bw_MHz = 0.03051757812
fch1_MHz = 500.0                     # top-of-band frequency (channel 0), MHz
foff_MHz = -chan_bw_MHz              # negative => descending frequency
nbits = 8
tsamp_s = 0.001                      # 1 ms
duration_s = 262144 * tsamp_s                    # total length
outfile = Path("noise+frb_DM50_3072x0.0305176MHz_fch1_500MHz_8bit.fil")

# Noise parameters
noise_mean = 128.0
noise_std = 20.0

# FRB parameters
DM = 50.0                            # pc cm^-3
fref_MHz = fch1_MHz                  # reference (arrives at middle here)
pulse_sigma_ms = 2.0                 # intrinsic Gaussian width (std dev) in ms
pulse_amp_counts = 8.0               # additive amplitude (rough per-sample SNR ~ pulse_amp / noise_std)

# --------- Derived ---------
nsamp = int(round(duration_s / tsamp_s))
assert nbits == 8, "This example writes uint8 only"

# Dispersion constant: t_delay(ms) = 4.148808 * DM * (1/nu^2 - 1/nu_ref^2), with nu in GHz
K_ms = 4.148808
fref_GHz = fref_MHz / 1000.0

# Channel center frequencies (GHz)
# ch 0 = fch1, ch 1 = fch1 + foff, ...
ch_idx = np.arange(nchans)
freq_MHz = fch1_MHz + ch_idx * foff_MHz
freq_GHz = freq_MHz / 1000.0

# Per-channel arrival delay (ms) relative to reference (fref = fch1)
delay_ms = K_ms * DM * (1.0 / (freq_GHz**2) - 1.0 / (fref_GHz**2))
delay_samples = np.round(delay_ms / (tsamp_s * 1000.0)).astype(np.int64)  # integer sample offsets

# Pulse time center (sample index) for each channel
mid_sample = nsamp // 2
t0_ch = mid_sample + delay_samples    # shape: (nchans,)

# Convert pulse sigma to samples; we’ll only draw within ±4σ for speed
sigma_samp = max(1.0, pulse_sigma_ms / (1000.0 * tsamp_s))  # ensure >= 1 sample
halfwin = int(np.ceil(4.0 * sigma_samp))

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

header = {
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

with open(outfile, "wb") as fh:
    write_sigproc_header(fh, header)

    rng = np.random.default_rng()
    chunk_nsamp = 4096
    for i0 in range(0, nsamp, chunk_nsamp):
        print(i0)
        n = min(chunk_nsamp, nsamp - i0)
        i1 = i0 + n - 1

        # 1) draw noise
        block = rng.normal(noise_mean, noise_std, size=(n, nchans)).round().astype(np.float32)

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

