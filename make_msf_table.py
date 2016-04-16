import os
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from glob import glob


# If MS filter table has not been made, fetch pcad0{pcad8eng} data
# into /data/aca/archive/pcad8 with arc5gl.  Logic in the table making
# assumes that all data has been fetched from 2016:001

if 'mstable' not in globals():
    if os.path.exists('mstable.hd5'):
        mstable = Table.read('mstable.hd5', format='hdf5')
    else:
        pcadfiles = glob("/data/aca/archive/pcad8/pcad*")
        # cols = ['TIME', 'TLM_FMT', 'MJF', 'MNF', 'QUALITY', 'AOACIMSS']
        cols = ['TIME', 'AOACIMSS']

        hdus = fits.open(pcadfiles[0], uint=True)
        mstable = Table(hdus[1].data)[cols]

        for f in pcadfiles[1:]:
            print f
            hdus = fits.open(f, uint=True)
            data = Table(hdus[1].data)[cols]
            mstable = vstack([mstable, data])

        mstable.write('mstable.hd5', format='hdf5', path='mstable')
