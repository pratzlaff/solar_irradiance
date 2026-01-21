import argparse
import h5py
import numpy as np
import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt

srcdir = os.path.dirname(os.path.realpath(__file__))

def write_dates(dates, dates_file):
    dates_path = Path(dates_file)
    if not dates_path.exists():
        sys.stderr.write(f"writing dates to '{dates_file}'\n")
        dates = np.array([l.decode() for l in dates])
        np.savetxt(dates_file, dates, fmt='%s')

def sum_wavs(args):
    with h5py.File(args.infile, 'r') as f:
        keys = list(f.keys())
        for k in f.keys():
            continue
            sys.stderr.write(f'{k}:\t{f[k].shape}\n')
        write_dates(f['date'], args.datesfile)

        wav = f['wavelength'][:]
        irr = f['irradiance'][:,:]
        err = f['uncertainty'][:,:]

        i = np.where((wav>=args.wmin) & (wav<args.wmax))
        wavstr = [f'{w:.2f}' for w in wav[i].tolist()]
        hdr='wavelengths='+','.join(wavstr)

        irr_sum = irr[:,i].sum(axis=2)
        err_sum = np.sqrt((err[:,i]*err[:,i]).sum(axis=2))

        np.savetxt(sys.stdout, irr_sum[:10], fmt='%5g', header=hdr)

        if args.plot:
            plt.plot(irr_sum)
            plt.xlabel('Days since 1947')
            plt.ylabel(f'Irradiance: λ(Å)=[{args.wmin:g}, {args.wmax:g})')
            plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='Read solar irradiance data, sum those in a wavelength range.'
    )
    parser.add_argument('--infile', default=srcdir+'/../data/daily_data.nc')
    parser.add_argument('--datesfile', default='./dates.txt')
    parser.add_argument('-p', '--plot', action='store_true', help='Display plot')
    parser.add_argument('wmin', type=float, help='Minimum wavelength')
    parser.add_argument('wmax', type=float)
    args = parser.parse_args()

    sum_wavs(args)

if __name__ == '__main__':
    main()
