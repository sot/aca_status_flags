import cPickle as pickle
import os

import matplotlib.pyplot as plt
import numpy as np
from Chandra.Time import DateTime
import update_flags_archive
from kadi import events

import pyyaks.logger
import pyyaks.context

logger = pyyaks.logger.get_logger(format='%(asctime)s: %(message)s')

ft = pyyaks.context.ContextDict('ft')
files = pyyaks.context.ContextDict('files', basedir='data')
files.update({'dat': '{ft.obsid}',
              'stats': 'stats / {ft.sp} {ft.dp} {ft.ir} {ft.ms} {ft.t_samp} / {ft.obsid}'
              })


def time_slice_dat(dat, tstart, tstop):
    """
    Get a time slice from a status flags data structure.

    :param tstart: start time in sec relative to dat['time0']
    :param tstop: stop time in sec relative to dat['time0']

    :returns: new status flags structure
    """
    out = {}
    out['time0'] = dat['time0']
    out['vals'] = {}
    out['bads'] = {}
    slots = out['slots'] = dat['slots']

    slot_msids = update_flags_archive.SLOT_MSIDS + ['dyag', 'dzag']

    i0, i1 = np.searchsorted(dat['times'], [tstart, tstop])
    i0_i1 = slice(i0, i1)
    out['times'] = dat['times'][i0_i1]

    for slot in slots:
        out['bads'][slot] = dat['bads'][slot][i0_i1]

    for slot_msid in slot_msids:
        out['vals'][slot_msid] = {}
        for slot in slots:
            out['vals'][slot_msid][slot] = dat['vals'][slot_msid][slot][i0_i1]

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
    if 'obsid' in dat:
        plt.title('Obsid {} at {}'.format(dat['obsid'], DateTime(dat['time0']).date[:17]))
    plt.show()


STAT_CASES = ('obc', 'test')
STAT_TYPES = ('mean', 'std', 'sig', 'n')


def get_stats_per_interval(dat, sp=False, dp=False, ir=False, ms=False, slots=None, t_samp=1000):
    all_stats = {case: {stat_type: [] for stat_type in STAT_TYPES} for case in STAT_CASES}
    stats = {}
    times = dat['times']

    flag_strs = {False: 'f', True: 't', None: 'x'}
    ft['sp'] = flag_strs[sp]
    ft['dp'] = flag_strs[dp]
    ft['ir'] = flag_strs[ir]
    ft['ms'] = flag_strs[ms]
    ft['t_samp'] = str(t_samp)

    dirspec = ''.join(flag_strs[x] for x in (sp, dp, ir, ms)) + str(t_samp)
    rootdir = os.path.join('data', 'stats', dirspec)

    # If this was already computed then return the on-disk version
    outfile = os.path.join(rootdir, '{}.pkl'.format(dat['obsid']))
    failfile = os.path.join(rootdir, '{}.ERR'.format(dat['obsid']))
    if os.path.exists(outfile):
        logger.info('Reading {}'.format(outfile))
        all_stats = pickle.load(open(outfile, 'r'))
        return all_stats

    if os.path.exists(failfile):
        raise ValueError('Known fail: file {} exists'.format(failfile))

    for slot in slots or dat['slots']:
        sample_times = np.arange(times[0], times[-1], t_samp)

        for t0, t1 in zip(sample_times[:-1], sample_times[1:]):
            dat_slice = time_slice_dat(dat, t0, t1)
            goods = ~dat_slice['bads'][slot]

            for case in STAT_CASES:
                stats[case] = {stat_type: [] for stat_type in STAT_TYPES}
                flags = (dict(sp=False, dp=False, ir=False, ms=False) if case == 'obc'
                         else dict(sp=sp, dp=dp, ir=ir, ms=ms))
                ok = get_flags_match(dat_slice, slot, **flags) & goods
                dy = dat_slice['vals']['dyag'][slot][ok]
                dz = dat_slice['vals']['dzag'][slot][ok]
                if np.any(dy < -100):
                    raise ValueError
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
                for case in STAT_CASES:
                    for stat_type in STAT_TYPES:
                        all_stats[case][stat_type].extend(stats[case][stat_type])

    for case in STAT_CASES:
        for stat_type in STAT_TYPES:
            all_stats[case][stat_type] = np.array(all_stats[case][stat_type])

    # If this is a run with all slots included then save the results
    if slots is None:
        if not os.path.exists(rootdir):
            os.makedirs(rootdir)
        logger.info('Writing {}'.format(outfile))
        pickle.dump(all_stats, open(outfile, 'w'), protocol=-1)

    return all_stats


def get_stats_over_time(start, stop=None, sp=False, dp=False, ir=False, ms=False,
                        slots=None, t_samp=1000):
    """
    Equivalent to get_stats_per_interval, but concatenate the results for all
    obsids within the specified time interval.
    """
    # Get obsids in time range and collect all the per-interval statistics
    obsids = events.obsids.filter(start, stop, dur__gt=2000)
    stats_list = []
    for obsid in obsids:
        filename = os.path.join('data', str(obsid.obsid) + '.pkl')
        if not os.path.exists(filename):
            logger.info('Skipping {}: not in archive'.format(obsid))
            continue
        logger.info('Processing obsid {}'.format(obsid))
        dat = pickle.load(open(filename, 'r'))
        try:
            stats = get_stats_per_interval(dat, sp, dp, ir, ms, slots, t_samp)
            stats['obsid'] = obsid.obsid
            stats_list.append(stats)
        except ValueError as err:
            errfile = filename[:-4] + '.ERR'
            logger.warn('ERROR: {}'.format(err))

    stats = {}
    for case in STAT_CASES:
        stats[case] = {}
        for stat_type in STAT_TYPES:
            stats[case][stat_type] = np.hstack([x[case][stat_type] for x in stats_list])
    stats['obsid'] = np.hstack([np.ones(x['obc']['n'], dtype=int) for x in stats_list])

    return stats
