import subprocess
import numpy as np
import tempfile
import pathlib
import os
import glob
import sys
from convert_fil_to_dada import convert_fil_to_dada
from verify_injection_fil import parse_cfg,run_hella

def make_db(key, nbytes, nelement):
    subprocess.run(f"dada_db -k {key} -b {nbytes} -n {nelement} -l -p", shell=True)
def destroy_db(key):
    subprocess.run(f"dada_db -d -k {key}", shell=True)
def fill_db_from_disk(input, key):
    subprocess.run(f"dada_diskdb -f {input} -k {key}", shell=True)

DM_TOL = 2

def verify_inj(hella_path, cfg, nchan, nbeam, transpose, fp16):
    cfg = pathlib.Path(cfg)
    cfg_dict = parse_cfg(cfg)
    dada_key = cfg_dict["INPUT_PATH"]
    gulp = int(cfg_dict["GULP"])
    
    dada_file = tempfile.NamedTemporaryFile(delete=False)

    fil = cfg.with_suffix(".fil").absolute()
    print(fil)
    ndat = convert_fil_to_dada(fil, dada_file.name, nchan, nbeam, nbeam//2, gulp, transpose, fp16)

    db_size = nbeam * nchan * gulp * (2 if fp16 else 1)
    make_db(dada_key, db_size, ndat // gulp)
    fill_db_from_disk(dada_file.name, dada_key)
    os.unlink(dada_file.name)
    snr, recovered_dm = run_hella(hella_path, cfg_dict["OUTPUTPATH"], cfg.absolute())
    destroy_db("dada")
    if snr is None or recovered_dm is None:
        return False
    

    inj_dm = float(cfg_dict["INJECTED_DM"])
    print(f"Max SNR: {snr} Max DM: {recovered_dm} Expected DM: {inj_dm}")
    return np.abs(inj_dm - recovered_dm) < DM_TOL

if __name__ == "__main__":

    import argparse
    p = argparse.ArgumentParser("verify_injection_dada")
    p.add_argument("--cfg_glob", type=str, help="Glob matching input configurations for hella")
    p.add_argument("--hella", type=str, default="casm_hella", help="Path to hella binary (in case it's not in $PATH)")
    p.add_argument("--nchan", type=int, default=3072, help="Number of channels per beam")
    p.add_argument("--nbeam", type=int, default=8, help="Number of beams to produce")
    p.add_argument("--transpose", type=bool, default=True, help="Transpose the freq-time dimensions when converting from filterbank to DADA (i.e. TF -> FT)")
    p.add_argument("--fp16", type=bool, default=True, help="Save data as fp16 (otherwise uint8)")
    args = p.parse_args()

    any_bad = False

    for cfg in sorted(glob.glob(args.cfg_glob)):

        print(f"Doing {cfg}")

        verified = verify_inj(args.hella, cfg, args.nchan, args.nbeam, args.transpose, args.fp16)

        if verified:
            print(f"[OK] {cfg}")
        else:
            print(f"[BAD] {cfg}")
            any_bad = True

    if any_bad:
        sys.exit(1)
    else:
        sys.exit(0)

