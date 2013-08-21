import cPickle as pickle
import os

import matplotlib.pyplot as plt
import numpy as np
from Chandra.Time import DateTime
import update_flags_archive

import pyyaks.logger

logger = pyyaks.logger.get_logger(format='%(asctime)s: %(message)s')


def time_slice_dat(dat, start, stop):
    tstart = DateTime(start).secs
    tstop = DateTime(stop).secs

    out = {}
    out['time0'] = dat['time0']
    out['times'] = {}
    out['vals'] = {}
    out['slots'] = dat['slots']
    out['vals']['dyag'] = {}
    out['vals']['dzag'] = {}

    for pcad_msid in update_flags_archive.PCAD_MSIDS:
        out['times'][pcad_msid] = {}
        out['vals'][pcad_msid] = {}

    for slot in dat['slots']:
        times = dat['times']
        i0, i1 = np.searchsorted(times, [tstart, tstop])
        ok = slice(i0, i1)

        for pcad_msid in update_flags_archive.PCAD_MSIDS:
            out['times'][pcad_msid][slot] = dat['times'][pcad_msid][slot][ok]
            out['vals'][pcad_msid][slot] = dat['vals'][pcad_msid][slot][ok]

        out['dyag'][slot] = dat['dyag'][slot][ok]
        out['dzag'][slot] = dat['dzag'][slot][ok]

    return out


def get_flags_match(dat, slot, sp, dp, ir, ms):
    ok = np.ones(len(dat['vals']['dyag'][slot]), dtype=bool)
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
        dat = update_flags_archive.get_obsid(obsid)
        pickle.dump(dat, open(filename, 'w'), protocol=-1)
        logger.info('Wrote data for {}'.format(obsid))

    return dat


def get_stats(val):
    if len(val) < 20:
        raise ValueError('Not enough')
    p16, p84 = np.percentile(val, [15.87, 84.13])
    sig = (p84 - p16) / 2
    std = np.std(val)
    mean = np.mean(val)
    return mean, std, sig, len(val)


def plot_centroids(dat, sp=None, dp=None, ir=None, ms=None, slots=None):
    """
    The value of 3.0 was semi-empirically derived as the value which minimizes
    the centroid spreads for a few obsids.  It also corresponds roughly to
    2.05 + (2.05 - 1.7 / 2) which could be the center of the ACA integration.
    Some obsids seem to prefer 2.0, others 3.0.
    """
    plt.clf()
    for slot in slots or dat['slots']:
        dyag = dat['vals']['dyag'][slot]
        dzag = dat['vals']['dzag'][slot]
        ok = get_flags_match(dat, slot, sp, dp, ir, ms)

        if np.any(ok):
            ok = ok & ~dat['bads'][slot]
            times = dat['times'][ok]
            dyag = dyag[ok]
            dzag = dzag[ok]
            plt.plot(times, dzag, '.', ms=2.0)
            y_mean, y_std, y_sig, y_n = get_stats(dyag)
            z_mean, z_std, z_sig, z_n = get_stats(dzag)
            logger.info('Slot {}: {} values: y_sig={:.2f} y_std={:.2f} z_sig={:.2f} z_std={:.2f}'
                        .format(slot, np.sum(ok), y_sig, y_std, z_sig, z_std))
        else:
            logger.info('Slot {}: no data values selected')
    plt.grid()
    plt.ylim(-5.0, 5.0)
    # plt.title('Obsid {} at {}'.format(obsid, DateTime(dat['time0']).date[:17]))
    plt.show()


def get_stats_per_sample(obsid, sp=False, dp=False, ir=False, ms=False, slots=None, t_samp=300):
    dat = get_obsid_data(obsid)
    cases = ('obc', 'test')
    stat_types = ('mean', 'std', 'sig', 'n')
    all_stats = {case: {stat_type: [] for stat_type in stat_types} for case in cases}
    stats = {}

    for slot in slots or dat['slots']:
        dyag = dat['vals']['dyag'][slot]
        dzag = dat['vals']['dzag'][slot]
        times = dat['times']['aoacyan'][slot]

        sample_times = np.arange(times[0], times[-1], t_samp)
        for t0, t1 in zip(sample_times[:-1], sample_times[1:]):
            dat_slice = time_slice_dat(dat, t0, t1)

            for case in cases:
                stats[case] = {stat_type: [] for stat_type in stat_types}
                flags = (dict(sp=False, dp=False, ir=False, ms=False) if case == 'obc'
                         else dict(sp=sp, dp=dp, ir=ir, ms=ms))
                ok = get_flags_match(dat_slice, slot, **flags)
                dy = dyag[ok]
                dz = dzag[ok]
                try:
                    y_mean, y_std, y_sig, y_n = get_stats(dy)
                    z_mean, z_std, z_sig, z_n = get_stats(dz)
                except ValueError:
                    logger.info('Too few values {} for case {} slot {} at t={:.0f}:{:.0f}'
                                .format(len(dy), case, slot, t0, t1))
                    break
                stats[case]['mean'] = [y_mean, z_mean]
                stats[case]['std'] = [y_std, z_std]
                stats[case]['sig'] = [y_sig, z_sig]
                stats[case]['n'] = [y_n]
            else:
                # None of the cases had too few values
                for case in cases:
                    for stat_type in stat_types:
                        all_stats[case][stat_type].extend(stats[case][stat_type])

    for case in cases:
        for stat_type in stat_types:
            all_stats[case][stat_type] = np.array(all_stats[case][stat_type])

    return all_stats
