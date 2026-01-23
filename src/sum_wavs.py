import argparse
import h5py
import numpy as np
import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt

srcdir = os.path.dirname(os.path.realpath(__file__))

class BinUniformityException(ValueError):
    pass

def write_dates(dates, dates_file):
    dates_path = Path(dates_file)
    if not dates_path.exists():
        sys.stderr.write(f"writing dates to '{dates_file}'\n")
        dates = np.array([l.decode() for l in dates])
        np.savetxt(dates_file, dates, fmt='%s', header='Date')

def sum_wavs(args):
    with h5py.File(args.infile, 'r') as f:
        keys = list(f.keys())
        for k in f.keys():
            continue
            sys.stderr.write(f'{k}:\t{f[k].shape}\n')
        write_dates(f['date'], args.datesfile)

        wav = f['wavelength'][:]
        binsize = wav[1]-wav[0]
        if np.sum(np.abs((wav[1:]-wav[:-1])-binsize)>1e-5):
            raise BinUniformityException()

        i, = np.where((wav>=args.wmin) & (wav<args.wmax))

        irr = f['irradiance'][:,i]
        irr_sum = irr.sum(axis=1)*binsize

        if False:
            err = f['uncertainty'][:,i] * irr
            err_sum = np.sqrt((err*err).sum(axis=1))*binsize

        # outer limits of the wavelength bins, rather than centers
        wmin = wav[i[0]] - binsize/2
        wmax = wav[i[-1]] + binsize/2
        hdr=f'sum(wav=[{wmin:g},{wmax:g}])'

        try:
            np.savetxt(sys.stdout, irr_sum, fmt='%5g', header=hdr)
        except BrokenPipeError:
            pass

        if args.plot:
            years = np.arange(irr_sum.shape[0])/365.2425 + 1947 + 44.5/365
            plt.scatter(years, irr_sum, s=0.1, linewidths=0)
            plt.xlabel('Year')
            plt.ylabel(f'Irradiance: λ(nm)=[{wmin:g}, {wmax:g}]')
            plt.tight_layout()
            if args.plot_file is not None:
                plt.savefig(args.plot_file)
            else:
                plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='Read solar irradiance data, sum those in a wavelength range.'
    )
    infile_default = srcdir+'/../data/daily_data.nc'
    parser.add_argument('--infile', default=infile_default, help=f'Input HDF5 file containing irradiance data. Default is "{infile_default}"' )
    datesfile_default = './dates.txt'
    parser.add_argument('--datesfile', default=datesfile_default, help=f'Output dates filename. Default is "{datesfile_default}"')
    parser.add_argument('-p', '--plot', action='store_true', help='Display plot of data.')
    parser.add_argument('--plot_file', help='Write plot to the given filename. Automatically enables --plot.')
    parser.add_argument('wmin', type=float, help='Minimum wavelength (nm, bin center)')
    parser.add_argument('wmax', type=float, help='Maximum wavelength (nm, bin center)')
    args = parser.parse_args()

    if args.plot_file is not None:
        args.plot = True

    sum_wavs(args)

if __name__ == '__main__':
    main()
