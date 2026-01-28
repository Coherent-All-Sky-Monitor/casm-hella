import subprocess
import numpy as np
import glob
import sys
import os

def parse_cfg(cfg):
    cfg_dict = {}
    with open(cfg, 'r') as f:
        for line in f.readlines():
            split = line.split()
            assert len(split) == 2
            cfg_dict[split[0]] = split[1]
    return cfg_dict

def run_hella(hella_path, out, cfg_path):
    if os.path.isfile(out):
        os.unlink(out)

    subprocess.run(f"{hella_path} -c {cfg_path}", shell=True, stdout=subprocess.DEVNULL)

    results = np.atleast_2d(np.loadtxt(out))
    if len(results) == 1:
        return None, None
    by_snr = results[results[:,0].argsort()]
    loudest = by_snr[-1, :]

    return loudest[0], loudest[-2]

DM_TOL = 2

def verify_inj(hella_path, cfg):
    cfg_dict = parse_cfg(cfg)
    snr, recovered_dm = run_hella(hella_path, cfg_dict["OUTPUTPATH"], cfg)
    if snr is None or recovered_dm is None:
        return False
    
    inj_dm = float(cfg_dict["INJECTED_DM"])
    print(f"Max SNR: {snr} Max DM: {recovered_dm} Expected DM: {inj_dm}")
    return np.abs(inj_dm - recovered_dm) < DM_TOL

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser("verify_injection_fil")
    p.add_argument("--cfg_glob", type=str, help="Glob matching configuration files for hella", required=True)
    p.add_argument("--hella", type=str, default="casm_hella", help="Path to hella binary (in case it's not in $PATH)")
    args = p.parse_args()

    any_bad = False

    for cfg in sorted(glob.glob(args.cfg_glob)):

        print(f"Doing {cfg}")

        verified = verify_inj(args.hella, cfg)

        if verified:
            print(f"[OK] {cfg}")
        else:
            print(f"[BAD] {cfg}")
            any_bad = True

    if any_bad:
        sys.exit(1)
    else:
        sys.exit(0)

