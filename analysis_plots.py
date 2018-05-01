from __future__ import division

import cPickle as pickle
import os
import operator

import matplotlib.pyplot as plt
import numpy as np
from Chandra.Time import DateTime
import update_flags_archive
from kadi import events

import pyyaks.logger
import pyyaks.context

logger = pyyaks.logger.get_logger(format='%(asctime)s: %(message)s')

ft = pyyaks.context.ContextDict('ft')
FILES = pyyaks.context.ContextDict('files', basedir='data')
FILES.update({'dat': '{{ft.obsid}}',
              'stats': ('stats/{{ft.slots}}{{ft.sp}}{{ft.dp}}{{ft.ir}}{{ft.ms}}{{ft.t_samp}}/'
                        '{{ft.obsid}}')
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


def plot_centroids(dat, sp=False, dp=None, ir=False, ms=None, slots=None, **kwargs):
    """
    The value of 3.0 was semi-empirically derived as the value which minimizes
    the centroid spreads for a few obsids.  It also corresponds roughly to
    2.05 + (2.05 - 1.7 / 2) which could be the center of the ACA integration.
    Some obsids seem to prefer 2.0, others 3.0.
    """
    plt.figure(4, figsize=(5, 3.5))
    if isinstance(dat, int):
        dat = get_obsid_data(dat)

    kalman_thresh = get_kalman_threshold(dat['times'][0])  # Time-dependent threshold (20 or 5)
    colors = ['b', 'g', 'r', 'c', 'm', 'BlueViolet', 'k', 'DarkOrange']
    for slot in slots or dat['slots']:
        dyag = dat['vals']['dyag'][slot]
        dzag = dat['vals']['dzag'][slot]
        ok = get_flags_match(dat, slot, sp, dp, ir, ms)

        if np.any(ok):
            if 'markersize' not in kwargs:
                kwargs['markersize'] = 2.0
            ok = ok & ~dat['bads'][slot]
            times = dat['times'][ok]
            dyag = dyag[ok]
            dzag = dzag[ok]
            plt.plot(times / 1000., dzag, '.', color=colors[slot], **kwargs)
            try:
                kalman_ok = (np.abs(dyag) < kalman_thresh) & (np.abs(dzag) < kalman_thresh)
                y_mean, y_std, y_sig, y_n = get_stats(dyag[kalman_ok])
                z_mean, z_std, z_sig, z_n = get_stats(dzag[kalman_ok])
                logger.info('Slot {}: {} values: y_sig={:.2f} y_std={:.2f} z_sig={:.2f} z_std={:.2f}'
                            .format(slot, np.sum(ok), y_sig, y_std, z_sig, z_std))
            except ValueError:
                logger.info('Slot {}: not enough values for statistics'.format(slot))
        else:
            logger.info('Slot {}: no data values selected'.format(slot))
    plt.grid(True)
    plt.ylabel('Centroid residual (arcsec)')
    plt.xlabel('Observation time (ksec)')
    y0, y1 = plt.ylim()
    if y0 > -5 or y1 > 5:
        plt.ylim(min(y0, -5.0), max(y1, 5.0))
    if 'obsid' in dat:
        plt.title('Obsid {} at {}'.format(dat['obsid'], DateTime(dat['time0']).date[:17]))
    plt.tight_layout()
    plt.show()


STAT_CASES = ('obc', 'test')
STAT_TYPES = ('mean', 'std', 'sig', 'n')


def set_FILES_context(obsid, sp, dp, ir, ms, t_samp, slots=None):
    # Set context for FILES
    flag_strs = {False: 'f', True: 't', None: 'x'}
    ft['obsid'] = obsid
    ft['sp'] = flag_strs[sp]
    ft['dp'] = flag_strs[dp]
    ft['ir'] = flag_strs[ir]
    ft['ms'] = flag_strs[ms]
    ft['t_samp'] = str(t_samp)
    ft['slots'] = 'combined_' if slots == 'combined' else ''


class NoStatsFile(IOError):
    pass


class FailedStatsFile(IOError):
    pass


def get_cached_stats(root='stats'):
    stats_file = FILES['{}.pkl'.format(root)].rel
    failfile = FILES['{}.ERR'.format(root)].rel

    # If this was already computed then return the on-disk version
    if os.path.exists(stats_file):
        logger.info('Reading {}'.format(stats_file))
        all_stats = pickle.load(open(stats_file, 'r'))
        return all_stats

    elif os.path.exists(failfile):
        raise FailedStatsFile('Known fail: file {} exists'.format(failfile))

    else:
        raise NoStatsFile


def get_stats_per_interval_per_slot(dat, sp=False, dp=None, ir=False, ms=None, slots=None,
                                    t_samp=1000):
    all_stats = {case: {stat_type: [] for stat_type in STAT_TYPES} for case in STAT_CASES}
    stats = {}

    set_FILES_context(dat['obsid'], sp, dp, ir, ms, t_samp)
    try:
        return get_cached_stats()
    except NoStatsFile:
        pass

    times = dat['times']
    for slot in slots or dat['slots']:
        sample_times = np.arange(times[0], times[-1], t_samp)

        for t0, t1 in zip(sample_times[:-1], sample_times[1:]):
            dat_slice = time_slice_dat(dat, t0, t1)
            goods = ~dat_slice['bads'][slot]

            for case in STAT_CASES:
                stats[case] = {stat_type: [] for stat_type in STAT_TYPES}
                flags = (dict(sp=False, dp=None, ir=False, ms=None) if case == 'obc'
                         else dict(sp=sp, dp=dp, ir=ir, ms=ms))
                ok = get_flags_match(dat_slice, slot, **flags) & goods
                dy = dat_slice['vals']['dyag'][slot][ok]
                dz = dat_slice['vals']['dzag'][slot][ok]
                if np.any((np.abs(dy) > 100) | (np.abs(dz) > 100)):
                    raise ValueError('Filtering inconsistency, unexpected bad values')
                try:
                    # OBC Kalman filter rejects stars outside 20 arcsec
                    ok = (np.abs(dy) < 20) & (np.abs(dz) < 20)
                    dy = dy[ok]
                    dz = dz[ok]
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
    stats_file = FILES['stats.pkl'].rel
    if slots is None:
        rootdir = os.path.dirname(stats_file)
        if not os.path.exists(rootdir):
            os.makedirs(rootdir)
        logger.info('Writing {}'.format(stats_file))
        pickle.dump(all_stats, open(stats_file, 'w'), protocol=-1)

    return all_stats


def get_stats_per_interval_combined(dat, sp=False, dp=None, ir=False, ms=None, t_samp=1000):
    all_stats = {case: {stat_type: [] for stat_type in STAT_TYPES} for case in STAT_CASES}
    stats = {}
    kalman_thresh = get_kalman_threshold(dat['times'][0])

    set_FILES_context(dat['obsid'], sp, dp, ir, ms, t_samp, 'combined')
    try:
        return get_cached_stats()
    except NoStatsFile:
        pass

    times = dat['times']
    sample_times = np.arange(times[0], times[-1], t_samp)

    for t0, t1 in zip(sample_times[:-1], sample_times[1:]):
        dat_slice = time_slice_dat(dat, t0, t1)
        for case in STAT_CASES:
            flags = (dict(sp=False, dp=None, ir=False, ms=None) if case == 'obc'
                     else dict(sp=sp, dp=dp, ir=ir, ms=ms))

            stats[case] = {stat_type: [] for stat_type in STAT_TYPES}
            dys = []
            dzs = []

            for slot in dat['slots']:
                goods = ~dat_slice['bads'][slot]
                ok = get_flags_match(dat_slice, slot, **flags) & goods
                dys.append(dat_slice['vals']['dyag'][slot][ok])
                dzs.append(dat_slice['vals']['dzag'][slot][ok])

            dy = np.concatenate(dys)
            dz = np.concatenate(dzs)
            if np.any((np.abs(dy) > 100) | (np.abs(dz) > 100)):
                raise ValueError('Filtering inconsistency, unexpected bad values')
            try:
                # OBC Kalman filter rejects stars outside threshold
                ok = (np.abs(dy) < kalman_thresh) & (np.abs(dz) < kalman_thresh)
                dy = dy[ok]
                dz = dz[ok]
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

    stats_file = FILES['stats.pkl'].rel
    rootdir = os.path.dirname(stats_file)
    if not os.path.exists(rootdir):
        os.makedirs(rootdir)
    logger.info('Writing {}'.format(stats_file))
    pickle.dump(all_stats, open(stats_file, 'w'), protocol=-1)

    return all_stats


def get_stats_over_time(start, stop=None, sp=False, dp=None, ir=False, ms=None,
                        slots='combined', t_samp=1000):
    """
    Equivalent to get_stats_per_interval, but concatenate the results for all
    obsids within the specified time interval.
    """
    # Get obsids in time range and collect all the per-interval statistics
    obsids = events.obsids.filter(start, stop, dur__gt=2000)
    stats_list = []
    for obsid in obsids:
        set_FILES_context(obsid.obsid, sp, dp, ir, ms, t_samp, slots)

        # First check that there is the raw dat file for this obsid.  Nothing
        # can be done without this.
        dat_file = FILES['dat.pkl'].rel
        if not os.path.exists(dat_file):
            logger.info('Skipping {}: {} not in archive'.format(obsid, dat_file))
            continue

        # Now get the stats for this obsid.  Hopefully it has already been computed and
        # is cached as a file.  If not, try to compute the stats (and cache).  If that
        # fails then press on but touch a file to indicate failure so subsequent attempts
        # don't bother.
        logger.info('Processing obsid {}'.format(obsid))
        try:
            stats = get_cached_stats()  # depends on the context set previously
        except FailedStatsFile:
            # Previously failed
            logger.info('  Skipping {}: failed statistics'.format(obsid.obsid))
            continue
        except NoStatsFile:
            logger.info('  Reading pickled data file {}'.format(dat_file))
            dat = pickle.load(open(dat_file, 'r'))
            try:
                logger.info('  Computing statistics')
                if slots == 'combined':
                    stats = get_stats_per_interval_combined(dat, sp, dp, ir, ms, t_samp)
                else:
                    stats = get_stats_per_interval_per_slot(dat, sp, dp, ir, ms, slots, t_samp)
            except ValueError as err:
                open(FILES['stats.ERR'].rel, 'w')  # touch file to indicate failure to compute stats
                logger.warn('  ERROR: {}'.format(err))

        stats['obsid'] = obsid.obsid
        stats_list.append(stats)

    stats = {}
    for case in STAT_CASES:
        stats[case] = {}
        for stat_type in STAT_TYPES:
            stats[case][stat_type] = np.hstack([x[case][stat_type] for x in stats_list])

    # Set corresponding array of obsids for back-tracing outliers etc
    stats['obsid'] = np.hstack([np.ones(len(x['obc']['std']), dtype=int) * x['obsid']
                                for x in stats_list])

    return stats


def plot_compare_stats_scatter(stats, attr='std',
                               title='Stddev: No DP rejection',
                               xlabel='OBC default (arcsec)',
                               ylabel='No DP filtering (arcsec)',
                               outroot=None):
    """
    Make a scatter plot of centroid degradation for the given
    ``stats`` output from get_stats_over_time().

    stats = get_stats_over_time('2010:001', dp=None)  # No DP filtering
    plot_degradation(stats, 'std',
                     title='Stddev: No DP rejection',
                     xlabel='OBC default (arcsec)',
                     ylabel='No DP filtering (arcsec)',
                     outroot='centr_stats_std_dp')
    """
    plt.figure(1, figsize=(5, 3.5))
    plt.clf()
    plt.plot(stats['obc'][attr], stats['test'][attr], '.')
    plt.plot(stats['obc'][attr], stats['test'][attr], ',y', alpha=0.3)
    plt.plot(stats['obc'][attr], stats['test'][attr], ',r', alpha=0.03)
    xy0 = max([plt.xlim()[1], plt.ylim()[1]])
    plt.plot([0., xy0], [0., xy0], '-g', alpha=0.3)
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    if outroot:
        plt.savefig(outroot + '_scatter.png')


def plot_compare_stats_hist(stats, attr='std',
                            title='Stddev difference: No DP rejection',
                            xlabel='OBC default - No DP (arcsec)',
                            ylabel='Number',
                            op=operator.sub,
                            outroot=None):
    """
    Make a histogram plot of centroid degradation for the given
    ``stats`` output from get_stats_over_time().

    stats = get_stats_over_time('2010:001', dp=None)  # No DP filtering
    plot_degradation(stats, 'std',
                     title='Stddev difference: No DP rejection',
                     xlabel='OBC default - No DP (arcsec)',
                     ylabel='Number',
                     outroot='centr_stats_std_dp')
    """
    plt.figure(2, figsize=(5, 3.5))
    plt.clf()
    plt.hist(op(stats['test'][attr], stats['obc'][attr]), bins=50, log=True)
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    y1 = plt.ylim()[1]
    plt.ylim(0.2, y1)
    if outroot:
        plt.savefig(outroot + '_hist.png')


def get_raw_vals(msid, vals):
    import Ska.tdb
    tsc = Ska.tdb.msids[msid].Tsc
    state_codes = [(x['LOW_RAW_COUNT'], x['STATE_CODE']) for x in tsc]
    raw_vals = np.zeros(len(vals), dtype='int8') - 1
    # CXC state code telem all has same length with trailing spaces
    # so find max length for formatting below.
    max_len = max(len(x[1]) for x in state_codes)
    fmtstr = '{:' + str(max_len) + 's}'
    for raw_val, state_code in state_codes:
        ok = vals == fmtstr.format(state_code)
        raw_vals[ok] = raw_val

    return raw_vals


def get_kalman_threshold(time):
    """
    Kalman star measurement residual threshold was updated from 20 arcsec
    to 5 arcsec on 2017-May-01 (2017:121) with PR-399.  This function returns
    the appropriate value based on the provided ``time``.
    """
    t0 = DateTime(time).secs
    out = 20.0 if (t0 < DateTime('2017:121').secs) else 5.0
    return out


def get_kalman_predicted(dat, sp=False, dp=None, ir=False, ms=None):
    tlm_n_kalman = get_raw_vals('aokalstr', dat['vals']['aokalstr'])
    pred_n_kalman = np.zeros(len(dat['times']), dtype=np.int8)
    kalman_thresh = get_kalman_threshold(dat['times'][0])  # Time-dependent threshold (20 or 5)

    for slot in dat['slots']:
        flags_ok = get_flags_match(dat, slot, sp, dp, ir, ms)
        pos_ok = ((np.abs(dat['vals']['dyag'][slot]) < kalman_thresh) &
                  (np.abs(dat['vals']['dzag'][slot]) < kalman_thresh))
        tlm_ok = ~dat['bads'][slot]
        pred_n_kalman += flags_ok & pos_ok & tlm_ok

    low = logical_intervals(dat['times'], tlm_n_kalman, '<=', 1)
    tlm_drops = low[(low['duration'] < 120) & (low['duration'] > 1)]
    low = logical_intervals(dat['times'], pred_n_kalman, '<=', 1)
    pred_drops = low[(low['duration'] < 120) & (low['duration'] > 1)]

    return tlm_n_kalman, pred_n_kalman, tlm_drops, pred_drops


def get_kalman_predicted_over_time(start, stop=None, sp=False, dp=None, ir=False, ms=None):
    obsids = events.obsids.filter(start, stop, dur__gt=2000)
    tlm_durs = []
    pred_durs = []
    for obsid in obsids:
        logger.info('Reading data for obsid {}'.format(obsid))
        try:
            dat = get_obsid_data(obsid.obsid)
        except Exception as err:
            logger.warn('Failed: {}'.format(err))
            continue
        tlm_drops, pred_drops = get_kalman_predicted(dat, sp, dp, ir, ms)[-2:]
        tlm_durs.append(tlm_drops['duration'])
        pred_durs.append(pred_drops['duration'])
    tlm_durs = np.concatenate(tlm_durs)
    pred_durs = np.concatenate(pred_durs)
    return tlm_durs, pred_durs


def print_flags(dat, start, stop):
    t0 = DateTime(start).secs - dat['time0']
    t1 = DateTime(stop).secs - dat['time0']
    dat = time_slice_dat(dat, t0, t1)
    for ii in xrange(len(dat['times'])):
        outs = []
        for slot in dat['slots']:
            out = ''.join(flag[-2].upper() if dat['vals'][flag][slot][ii] else '.'
                          for flag in ('aoacisp', 'aoaciir', 'aoacims', 'aoacidp'))
            outs.append(out)
        print '{} {}'.format(DateTime(dat['times'][ii] + dat['time0']).date, '  '.join(outs))


def logical_intervals(times, vals, op, val):
    """Determine contiguous intervals during which the logical comparison
    expression "MSID.vals op val" is True.  Allowed values for ``op``
    are::

      ==  !=  >  <  >=  <=

    The intervals are guaranteed to be complete so that the all reported
    intervals had a transition before and after within the telemetry
    interval.

    Returns a structured array table with a row for each interval.
    Columns are:

    * datestart: date of interval start
    * datestop: date of interval stop
    * duration: duration of interval (sec)
    * tstart: time of interval start (CXC sec)
    * tstop: time of interval stop (CXC sec)

    Example::

      dat = fetch.MSID('aomanend', '2010:001', '2010:005')
      manvs = dat.logical_intervals('==', 'NEND')
      manvs['duration']

    :param vals: input values
    :param op: logical operator, one of ==  !=  >  <  >=  <=
    :param val: comparison value
    :returns: structured array table of intervals
    """
    import Ska.Numpy
    import operator
    ops = {'==': operator.eq,
           '!=': operator.ne,
           '>': operator.gt,
           '<': operator.lt,
           '>=': operator.ge,
           '<=': operator.le}
    try:
        op = ops[op]
    except KeyError:
        raise ValueError('op = "{}" is not in allowed values: {}'
                         .format(op, sorted(ops.keys())))

    starts = ~op(vals[:-1], val) & op(vals[1:], val)
    ends = op(vals[:-1], val) & ~op(vals[1:], val)

    # If last telemetry point is not val then the data ends during that
    # interval and there will be an extra start transition that must be
    # removed.
    i_starts = np.flatnonzero(starts)
    if op(vals[-1], val):
        i_starts = i_starts[:-1]

    # If first entry is val then the telemetry starts during an interval
    # and there will be an extra end transition that must be removed.
    i_ends = np.flatnonzero(ends)
    if op(vals[0], val):
        i_ends = i_ends[1:]

    # Specially for the kalman flags intervals we want the time that
    # is one past the end of the logical interval.  That allows
    # distinguishing re-acq events:
    #
    # In [415]: for i, dt in zip(ilng, tlm_drops['duration'][lng]):
    #     print dat['vals']['aokalstr'][i-2:i+2], dt, np.diff(dat['times'][i-2:i+3])
    #    .....:     
    # ['1 ' '1 ' '1 ' '5 '] 86.1001 [  2.04980469  55.35009766   2.05029297   2.04980469]
    # ['1 ' '1 ' '1 ' '5 '] 30.75 [  2.04980469   2.04980469  57.40039062   2.04980469]
    # ['1 ' '1 ' '1 ' '4 '] 30.75 [  2.04980469   2.05078125  57.39941406   2.04980469]
    # ['1 ' '1 ' '1 ' '4 '] 30.75 [  2.05078125   2.04980469  55.34960938   2.04980469]
    # ['1 ' '1 ' '1 ' '2 '] 28.6992 [ 2.05078125  2.04882812  2.05078125  2.04882812]

    i_ends = np.clip(i_ends + 1, 0, len(times) - 1)

    tstarts = times[i_starts]
    tstops = times[i_ends]
    intervals = {'duration': times[i_ends] - times[i_starts],
                 'tstart': tstarts,
                 'tstop':  tstops,
                 'istart': i_starts,
                 'istop': i_ends}

    return Ska.Numpy.structured_array(intervals)
