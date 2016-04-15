import os
from Chandra.Time import DateTime
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from glob import glob
from kadi import events
from mica.starcheck import get_starcheck_catalog
from Ska.engarchive import fetch


if 'dwells' not in globals():
    dwells = events.dwells.filter(start='2015:287:12:22:55.309')

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


if 't' not in globals():

    if os.path.exists('obs_ms_table.dat'):
        t = Table.read('obs_ms_table.dat', format='ascii')
    else:
        obs_data = []
        for d in dwells:
            obs = {}
            obsid = d.get_obsid()
            if obsid is None:
                continue
            if DateTime(d.stop).secs > mstable['TIME'][-1]:
                continue
            # Get one shot and maneuver to for *next* obsid
            n = d.get_next()
            if not n:
                continue
            starcheck = get_starcheck_catalog(n.get_obsid())
            if not starcheck or not len(starcheck['manvr']):
                continue
            obs['manvr_angle'] = starcheck['manvr'][-1]['angle_deg']
            obs['manvr_slew_err'] =  starcheck['manvr'][-1]['slew_err_arcsec']
            obs['next_obsid'] = n.get_obsid()

            obs['date'] = d.manvr.kalman_start
            obs['datestop'] = d.stop
            obs['time'] = DateTime(d.manvr.kalman_start).secs
            obs['timestop'] = d.tstop
            obs['one_shot'] = n.manvr.one_shot
            obs['obsid'] = obsid

            # for data earlier than the chunk of pcad data I have
            # just set the status manuall
            if d.manvr.kalman_start < '2016:001':
                if (obsid == 17198) or (obsid == 18718):
                    obs['gui_ms'] = 'DISA'
                else:
                    obs['gui_ms'] = 'ENAB'
            else:
                mid_kalman = ((DateTime(d.manvr.kalman_start).secs
                               + DateTime(d.stop).secs) / 2)
                kal_idx = (np.searchsorted(mstable['TIME'],
                                           mid_kalman))
                obs['gui_ms'] = mstable['AOACIMSS'][kal_idx]

            for err_name, err_msid in zip(['roll_err', 'pitch_err', 'yaw_err'],
                                          ['AOATTER1', 'AOATTER2', 'AOATTER3']):
                err = fetch.Msid(err_msid, d.tstart + 500, d.stop)
                if len(err.vals):
                    obs[err_name] = np.degrees(np.percentile(np.abs(err.vals), 90)) * 3600
                else:
                    obs[err_name] = np.nan

            obs_data.append(obs)


        t = Table(obs_data)['obsid', 'next_obsid', 'time', 'timestop', 'date', 'datestop',
                            'gui_ms', 'manvr_angle', 'manvr_slew_err', 'one_shot',
                            'roll_err', 'pitch_err', 'yaw_err']
        t.write('obs_ms_table.dat', format='ascii')

