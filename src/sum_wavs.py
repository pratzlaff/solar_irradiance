import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import sys

srcdir = os.path.dirname(os.path.realpath(__file__))

class BinUniformityError(ValueError):
    pass

def is_leap_year(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

def dates2years(dates):
    date0 = dates[0].decode()
    year0 = int(date0[:4])
    day0 = int(date0[4:])
    return np.arange(dates.shape[0])/365.2425 + year0 + (day0-0.5)/(365+is_leap_year(year0))

def write_dates(dates, dates_file):
    dates_path = Path(dates_file)
    if not dates_path.exists():
        sys.stderr.write(f"writing dates to '{dates_file}'\n")
        dates = np.array([l.decode() for l in dates])
        np.savetxt(dates_file, dates, fmt='%s', header='Date')

def sum_wavs(args):

    with h5py.File(args.infile, 'r') as f:
        if False:
            for k in f.keys():
                sys.stderr.write(f'{k}:\t{f[k].shape}\n')
        wav = f['wavelength'][:]
        date = f['date'][:]
        i, = np.where((wav>=args.wmin) & (wav<args.wmax))
        irr = f['irradiance'][:,i]
        if args.errs:
            err = f['uncertainty'][:,i] * irr

    write_dates(date, args.datesfile)

    irr_sum = irr.sum(axis=1)
    if args.errs:
        err_sum = np.sqrt((err*err).sum(axis=1))

    binsize = wav[1]-wav[0]
    if np.sum(np.abs((wav[1:]-wav[:-1])-binsize)>1e-5):
        raise BinUniformityError()

    if args.wavs:
        wavstr = 'Wavelengths = [' + ', '.join([f'{w:.2f}' for w in wav[i].tolist()]) + ']\n'
        sys.stderr.write(wavstr)

    # outer limits of the wavelength bins, rather than centers
    wmin = wav[i[0]] - binsize/2
    wmax = wav[i[-1]] + binsize/2

    units = r'$\text{W}\;\text{m}^{-2}\;\text{nm}^{-1}$'

    if args.mean:
        prefix = 'Mean '
        irr_sum /= i.shape[0]
        hdr=f'mean(wav=[{wmin:.1f},{wmax:.1f}])'
        if args.errs:
            err_sum /= i.shape[0]

    elif args.sum:
        prefix = 'Sum '
        hdr=f'sum(wav=[{wmin:.1f},{wmax:.1f}])'

    else:
        prefix=''
        irr_sum *= binsize
        hdr=f'sum(wav=[{wmin:.1f},{wmax:.1f}])*binsize'
        units = r'$\text{W}\;\text{m}^{-2}$'
        if args.errs:
            err_sum *= binsize

    try:
        toprint = irr_sum
        if args.errs:
            toprint = np.column_stack((irr_sum, err_sum))
            hdr += '\terr'
        np.savetxt(sys.stdout, toprint, fmt='%5g', delimiter='\t', header=hdr)
    except BrokenPipeError:
        pass

    if args.plot:
        plt.rcParams.update({ 'font.size':16  })
        fig = plt.figure(figsize=(11, 8.5))
        years = dates2years(date)
        if args.xmin is not None:
            i = years > args.xmin
            years = years[i]
            irr_sum = irr_sum[i]
        if args.xmax is not None:
            i = years < args.xmax
            years = years[i]
            irr_sum = irr_sum[i]
        plt.scatter(years, irr_sum, s=1, linewidths=1)
        plt.xlabel('Year')
        ylabel = f'{prefix}Irradiance ({units}): λ(nm)=[{wmin:.1f}, {wmax:.1f}]'
        plt.ylabel(ylabel)
        plt.tight_layout()
        if args.plot_file is not None:
            plt.savefig(args.plot_file)
        else:
            plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='Read solar irradiance data, sum those in a wavelength range, multiply by bin width.'
    )
    infile_default = srcdir+'/../data/daily_data.nc'
    parser.add_argument('--infile', default=infile_default, help=f'Input HDF5 file containing irradiance data. Default is "{infile_default}"' )
    datesfile_default = './dates.txt'
    parser.add_argument('--datesfile', default=datesfile_default, help=f'Output dates filename. Default is "{datesfile_default}"')
    parser.add_argument('-e', '--errs', action='store_true', help='Also print uncertainties.')
    parser.add_argument('-w', '--wavs', action='store_true', help='Print wavelength bin centers to stderr.')
    parser.add_argument('-m', '--mean', action='store_true', help='Calculate mean irradiance.')
    parser.add_argument('-s', '--sum', action='store_true', help='Calculate irradiance sums.')
    parser.add_argument('-p', '--plot', action='store_true', help='Display plot of data.')
    parser.add_argument('--plot_file', help='Write plot to the given filename. Automatically enables --plot.')
    parser.add_argument('--xmin', type=float, help='Earliest plotted year')
    parser.add_argument('--xmax', type=float, help='Latest plotted year')
    parser.add_argument('wmin', type=float, help='Minimum wavelength (nm, bin center)')
    parser.add_argument('wmax', type=float, help='Maximum wavelength (nm, bin center)')
    args = parser.parse_args()

    if args.plot_file is not None:
        args.plot = True

    sum_wavs(args)

if __name__ == '__main__':
    main()
