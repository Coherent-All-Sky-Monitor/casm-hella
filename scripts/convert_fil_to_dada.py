from __future__ import annotations
import numpy as np

def load_fil_data(input: str):

    with open(input, 'rb') as f:
        data = f.read()

    data_start = data.find(b"HEADER_END") + len("HEADER_END")
    return data[data_start:]

def get_dada_header(data_nbytes: int):
    hdr_size = 4096

    file_size = data_nbytes

    hdr =   f"HDR_SIZE {hdr_size}\n" + \
            f"FILE_SIZE {file_size}\n" + \
            "UTC_START 2020-01-01-00:00:00\n" + \
            "OBS_OFFSET 0\n"
    hdr += "\0" * (hdr_size - len(hdr))

    return hdr

def convert_fil_to_dada(input, output, nchan, nbeam, ibeam, gulp_samps, transpose, fp16): 
    raw_data = load_fil_data(input)
    ndat = len(raw_data) // nchan
    print(f"Inferred {ndat} samples")

    # Assume uint8 input in TF order
    data = np.frombuffer(raw_data, dtype=np.uint8).reshape(ndat, nchan)

    print(f"Mean: {np.mean(data.flatten())}, Std dev: {np.std(data.flatten())}")

    out_type = np.float16 if fp16 else np.uint8
    output_data = data.astype(out_type)

    if transpose:
        full_output = np.full((nbeam, nchan, ndat), 128, dtype=out_type)
        full_output[ibeam, :, :] = output_data.T
    else:
        full_output = np.full((nbeam, ndat, nchan), 128, dtype=out_type)
        full_output[ibeam, :, :] = output_data

    with open(output, 'wb') as f:
        idat = 0
        data_size = len(full_output.flatten()) * 2
        f.write(get_dada_header(data_size).encode())

        while (idat + gulp_samps <= ndat):
            print(f"Writing samples {idat}->{idat + gulp_samps}")
            if transpose:
                full_output[:, :, idat:(idat + gulp_samps)].tofile(f)
            else:
                full_output[:, idat:(idat + gulp_samps), :].tofile(f)

            idat += gulp_samps

    return idat

if __name__ == "__main__":

    import argparse

    p = argparse.ArgumentParser("convert_fil_to_dada")
    p.add_argument("--input", type=str, help="Input .fil file", required=True)
    p.add_argument("--output", type=str, help="Output .dada file", required=True)
    p.add_argument("--nchan", type=int, help="Number of frequency channels in filterbank", required=True)
    p.add_argument("--nbeam", type=int, help="Total number of beams in .dada file", required=True)
    p.add_argument("--ibeam", type=int, help="Beam to insert filterbank data into", required=True)
    p.add_argument("--gulp_samps", type=int, help="Number of time samples per gulp (i.e. ring buffer element)", required=True)
    p.add_argument("--transpose", type=bool, default=True, help="Transpose each beam from TF to FT order")
    p.add_argument("--fp16", type=bool, default=True, help="Write output as fp16 instead of uint8")
    args = p.parse_args()

    convert_fil_to_dada(args.input, args.output, args.nchan, args.nbeam, args.ibeam, args.gulp_samps, args.transpose, args.fp16)

