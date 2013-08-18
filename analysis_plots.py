import cPickle as pickle
import os
import copy

import matplotlib.pyplot as plt
import numpy as np
from Chandra.Time import DateTime
import update_flags_archive

import pyyaks.logger

logger = pyyaks.logger.get_logger(format='%(asctime)s: %(message)s')


def time_slice_dat(dat, start, stop):
    """
    Return a copy of dat which is filtered to contain only times between
    start and stop.
    """
    out = copy.deepcopy(dat)
    slots = out['slots']
    tstart = DateTime(start).secs
    tstop = DateTime(stop).secs

    times = out['times']['aoattqt1']
    ok = (times > tstart) & (times < tstop)
    for msid in update_flags_archive.ATT_MSIDS:
        out['times'][msid] = out['times'][msid][ok]
        out['vals'][msid] = out['vals'][msid][ok]

    for slot in slots:
        times = out['times']['aoacyan'][slot]
        ok = (times > tstart) & (times < tstop)
        for pcad_msid in update_flags_archive.PCAD_MSIDS:
            out['times'][pcad_msid][slot] = out['times'][pcad_msid][slot][ok]
            out['vals'][pcad_msid][slot] = out['vals'][pcad_msid][slot][ok]

        out['dyag'][slot] = out['dyag'][slot][ok]
        out['dzag'][slot] = out['dzag'][slot][ok]

    return out


def get_flags_match(dat, slot, sp, dp, ir, ms):
    ok = np.ones(len(dat['dyag'][slot]), dtype=bool)
    flag_vals = {'sp': sp, 'ir': ir, 'ms': ms, 'dp': dp}
    for flag in flag_vals:
        if flag_vals[flag] is not None:
            msid = 'aoaci{}'.format(flag)
            match_val = (1 if flag_vals[flag] else 0)
            ok &= dat['vals'][msid][slot] == match_val

    return ok


def get_obsid_data(obsid):
    filename = os.path.join('data', str(obsid) + '.pkl')
    if os.path.exists(filename):
        dat = pickle.load(open(filename, 'r'))
    else:
        import update_flags_archive
        dat, telems = update_flags_archive.get_obsid(obsid)
        pickle.dump(dat, open(filename, 'w'), protocol=-1)
        logger.info('Wrote data for {}'.format(obsid))

    return dat


def get_stats(val):
    p16, p84 = np.percentile(val, [15.87, 84.13])
    sig = (p84 - p16) / 2
    std = np.std(val)
    mean = np.mean(val)
    return mean, std, sig


def plot_centroids(dat, sp=None, dp=None, ir=None, ms=None, slots=None):
    """
    The value of 3.0 was semi-empirically derived as the value which minimizes
    the centroid spreads for a few obsids.  It also corresponds roughly to
    2.05 + (2.05 - 1.7 / 2) which could be the center of the ACA integration.
    Some obsids seem to prefer 2.0, others 3.0.
    """
    plt.clf()
    for slot in slots or dat['slots']:
        dyag = dat['dyag'][slot]
        dzag = dat['dzag'][slot]
        ok = get_flags_match(dat, slot, sp, dp, ir, ms)

        if np.any(ok):
            times = dat['times']['aoacyan'][slot][ok]
            dyag = dyag[ok]
            dzag = dzag[ok]
            plt.plot(times, dzag, '.', ms=2.0)
            y_mean, y_std, y_sig = get_stats(dyag)
            z_mean, z_std, z_sig = get_stats(dzag)
            logger.info('Slot {}: {} values: y_sig={:.2f} y_std={:.2f} z_sig={:.2f} z_std={:.2f}'
                        .format(slot, np.sum(ok), y_sig, y_std, z_sig, z_std))
        else:
            logger.info('Slot {}: no data values selected')
    plt.grid()
    plt.ylim(-5.0, 5.0)
    # plt.title('Obsid {} at {}'.format(obsid, DateTime(dat['time0']).date[:17]))
    plt.show()


def plot_std_change(obsid, sp=None, dp=None, ir=None, ms=None, slots=None, n_samp=150):
    dat = get_obsid_data(obsid)
    plt.clf()
    for slot in slots or dat['slots']:
        dyag = dat['dyag'][slot]
        dzag = dat['dzag'][slot]

        for i0, i1 in zip(np.arange(0, len(dy), n_samp)):
            # Start with OBC flags
            ok = get_flags_match(dat, slot, None, None, None, None)
            ok[:i0] = False
            ok[i1:] = False
            dy = dyag[ok]
            dz = dzag[ok]
            y_mean, y_std, y_sig = get_stats(dy)
            z_mean, z_std, z_sig = get_stats(dz)

            # REFACTOR and check for np.any(ok)
            ok = get_flags_match(dat, slot, sp, dp, ir, ms)
            ok[:i0] = False
            ok[i1:] = False
            dy = dyag[ok]
            dz = dzag[ok]
            y_mean, y_std, y_sig = get_stats(dy)
            z_mean, z_std, z_sig = get_stats(dz)

            if not np.any(ok):
                continue
