from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from mica import quaternion
from kadi import events
import Ska.engarchive.fetch as fetch
from Ska.Matplotlib import plot_cxctime
from Quaternion import Quat, normalize
from Chandra.Time import DateTime
import pyyaks.logger
import Ska.Numpy

logger = pyyaks.logger.get_logger()


def get_archive_data(start, stop):
    """
    Get archive data required for analyzing status flag / centroid
    dependencies.  The start and stop time must correspond to a single
    star catalog and be contained within a Kalman dwell.

    :param start: start time (any Chandra DateTime format)
    :param stop: stop time
    :returns: MSIDset object
    """
    start = DateTime(start)
    stop = DateTime(stop)
    slots = range(8)

    # First figure out if any slots (out of 0, 1, 2) are fids and remove
    # from the slots lits if so.
    msids = ['aoimage{}'.format(ii) for ii in range(3)]
    telems = fetch.Msidset(msids, start, start.secs + 100)
    for ii, msid in enumerate(msids):
        if np.any(telems[msid].vals == 'FID '):
            slots.remove(ii)
    logger.info('Using slots {}'.format(slots))

    pcad_msids = ['aoacfct',
                  'aoacisp',
                  'aoacidp',
                  'aoaciir',
                  'aoacims',
                  'aoacyan',
                  'aoaczan']
    slot_msids = [msid + "%s" % slot for msid in pcad_msids for slot in slots]

    att_msids = ['aoattqt1',
                 'aoattqt2',
                 'aoattqt3',
                 'aoattqt4',
                 ]

    msids = slot_msids + att_msids

    # Get telemetry
    logger.info('Fetching telemetry from {} to {}'.format(start.date[:17], stop.date[:17]))
    telems = fetch.MSIDset(msids, start, stop)
    start_interp = max(telems[msid].times[0] for msid in msids)
    stop_interp = min(telems[msid].times[-1] for msid in msids)
    logger.info('Interpolating at 2.05 sec intervals over {} to {}'
                .format(DateTime(start_interp).date[:17], DateTime(stop_interp).date[:17]))
    telems.interpolate(dt=2.05, start=start_interp, stop=stop_interp, filter_bad=False)

    # Select intervals within a kalman dwell with no tsc_moves or dumps
    events.dumps.interval_pad = (10, 500)
    events.tsc_moves.interval_pad = (10, 300)
    good_times = events.dwells & ~(events.tsc_moves | events.dumps)
    for msid in telems:
        telems[msid].select_intervals(good_times)

    # Create a bad filter for any samples with no attitude value
    att_bads = np.zeros(len(telems[att_msids[0]]), dtype=bool)
    for msid in att_msids:
        att_bads |= telems[msid].bads
    for msid in att_msids:
        telems[msid].bads = att_bads
    logger.info('Attitude: found {} / {} bad values'
                .format(np.sum(att_bads), len(att_bads)))

    # Create a bad filter for each slot based on the union of all relevant MSIDs
    for slot in slots:
        slot_msids = [msid + "%s" % slot for msid in pcad_msids]
        slot_bads = att_bads.copy()  # start with attitude bad values
        for msid in slot_msids:
            slot_bads |= telems[msid].bads

        # Only accept samples where the ACA reports tracking
        slot_bads |= telems['aoacfct{}'.format(slot)].vals != 'TRAK'

        for msid in slot_msids:
            telems[msid].bads = slot_bads
        logger.info('Slot {}: found {} / {} bad or not tracking values'
                    .format(slot, np.sum(slot_bads), len(slot_bads)))

    # Finally apply the filtering for each MSID invidivually (note that the
    # MSIDset.filter_bad() method isn't appropriate here).
    for msid in msids:
        telems[msid].filter_bad()

    return telems, slots


def get_q_atts_transforms(telems, slot, dt):
    """
    Get quaternions and associated transforms, matched to the times of yag/zag data
    in slot.  Apply a time offset ``dt`` to account for latencies in telemetry
    and ACA image readout.
    """
    logger.verbose('Interpolating quaternions for slot {}'.format(slot))
    yz_times = telems['aoacyan{}'.format(slot)].times
    q_times = telems['aoattqt1'].times
    qs = np.empty((len(yz_times), 4), dtype=np.float64)

    for ii in range(4):
        q_vals = telems['aoattqt{}'.format(ii + 1)].vals
        qs[:, ii] = Ska.Numpy.interpolate(q_vals, q_times + dt, yz_times, sorted=True)
    q_atts = quaternion.Quat(quaternion.normalize(qs))
    transforms = q_atts.transform  # N x 3 x 3
    return q_atts, transforms


def eci2radec(eci):
    ra = np.degrees(np.arctan2(eci[1], eci[0]))
    dec = np.degrees(np.arctan2(eci[2], np.sqrt(eci[1] ** 2 + eci[0] ** 2)))
    ra = np.mod(ra, 360)
    return ra, dec


def yagzag2radec(yag, zag, q):
    d_aca = np.array([1.0, np.tan(np.radians(yag)), np.tan(np.radians(zag))])
    d_aca *= 1.0 / np.sum(d_aca ** 2)
    eci = np.dot(q.transform, d_aca)
    return eci2radec(eci)


def radec2eci(ra, dec):
    r = np.radians(ra)
    d = np.radians(dec)
    return np.array([np.cos(r) * np.cos(d), np.sin(r) * np.cos(d), np.sin(d)])


def radec2yagzag(ra, dec, transforms_transpose):
    eci = radec2eci(ra, dec)
    d_aca = np.dot(transforms_transpose, eci)
    yag = np.degrees(np.arctan2(d_aca[:, 1], d_aca[:, 0]))
    zag = np.degrees(np.arctan2(d_aca[:, 2], d_aca[:, 0]))
    return yag, zag


def calc_delta_centroids(telems, slot, dt):
    yags = telems['aoacyan{}'.format(slot)].vals / 3600.
    zags = telems['aoaczan{}'.format(slot)].vals / 3600.
    q_atts, transforms = get_q_atts_transforms(telems, slot, dt=dt)

    n_mid = len(q_atts.q) // 2
    q_att0 = quaternion.Quat(q_atts.q[n_mid])
    ra0, dec0 = yagzag2radec(yags[n_mid], zags[n_mid], q_att0)

    transforms_transpose = transforms.transpose(0, 2, 1)

    p_yags, p_zags = radec2yagzag(ra0, dec0, transforms_transpose)
    d_yags = p_yags - yags
    d_zags = p_zags - zags
    d_yags = d_yags - np.median(d_yags)
    d_zags = d_zags - np.median(d_zags)

    return d_yags * 3600, d_zags * 3600


def plot_obsid(obsid, dt=3.0):
    """
    The value of 3.0 was semi-empirically derived as the value which minimizes
    the centroid spreads for a few obsids.  It also corresponds roughly to
    2.05 + (2.05 - 1.7 / 2) which could be the center of the ACA integration.
    Some obsids seem to prefer 2.0, others 3.0.
    """
    obsids = events.obsids.filter(obsid__exact=obsid)
    if len(obsids) == 0:
        raise ValueError('No obsid={} in kadi database'.format(obsid))

    dwells = events.dwells.filter(obsids[0].start, obsids[0].stop)
    obsid_dwells = [dwell for dwell in dwells if dwell.start > obsids[0].start]
    logger.info('Using obsid dwell(s): {}'
                .format(','.join(str(dwell) for dwell in obsid_dwells)))

    telems, slots = get_archive_data(obsid_dwells[0].start, obsid_dwells[-1].stop)
    plt.clf()
    for slot in slots:
        dyag, dzag = calc_delta_centroids(telems, slot, dt)
        plt.plot(dzag, ',')
        p16, p84 = np.percentile(dyag, [15.87, 84.13])
        y_sig = (p84 - p16) / 2
        p16, p84 = np.percentile(dzag, [15.87, 84.13])
        z_sig = (p84 - p16) / 2
        logger.info('Slot {}: yag_sigma={:.2f} zag_sigma={:.2f}'.format(slot, y_sig, z_sig))
    plt.grid()
    plt.ylim(-3.0, 3.0)
    plt.show()


def other():
    radecs = [yagzag2radec(yag, zag, Quat(normalize([q1, q2, q3, q4])))
              for q1, q2, q3, q4, yag, zag in zip(*vals)]
    coords = np.rec.fromrecords(radecs, names=('ra', 'dec'))

    # ok = telems['aoacfct{}'.format(slot)].vals == 'TRAK'

    flags = {'dp': telems['aoacidp%s' % slot].vals != 'OK ',
             'ir': telems['aoaciir%s' % slot].vals != 'OK ',
             'ms': telems['aoacims%s' % slot].vals != 'OK ',
             'sp': telems['aoacisp%s' % slot].vals != 'OK ',
             }

    times = telems['aoacyan%s' % slot].times
    dra = (coords['ra'] - np.mean(coords['ra'])) * 3600 * np.cos(np.radians(coords['dec']))
    ddec = (coords['dec'] - np.mean(coords['dec'])) * 3600
    dr = np.sqrt(dra ** 2 + ddec ** 2)

    # fileroot = 'flags_{}'.format(DateTime(start).date[:14]) if save else None

    # print('Making plots with output fileroot={}'.format(fileroot))

    filt = ((flags['dp'] == False) & (flags['ir'] == False)
            & (flags['ms'] == False) & (flags['sp'] == False))
    if plot:
        plot_axis('dR', times, dr, filt, title='No status flags')
    if np.sum(filt) > 20:
        clean_perc = np.percentile(dr[filt], [50, 84, 90, 95])
        if clean_perc[2] > 1.0:
            print '{} {} : {}'.format(start, stop, str(clean_perc))
    else:
        clean_perc = [-1, -1, -1, -1]
    clean_perc.append(np.sum(filt))

    filt = flags['dp'] == True
    if plot:
        plot_axis('dR', times, dr, filt, title='DP == True')
    if np.sum(filt) > 20:
        dp_perc = np.percentile(dr[filt], [50, 84, 90, 95])
    else:
        dp_perc = [-1, -1, -1, -1]
    dp_perc.append(np.sum(filt))

    return clean_perc, dp_perc


def gather_stats(start, stop, slot=6, root=''):
    cleans = []
    dps = []
    events.dwells.pad_interval = 300
    for interval in events.dwells.intervals(start, stop):
        clean, dp = plot_centroid_resids_by_flag(interval[0], interval[1], slot, plot=False)
        cleans.append(clean)
        dps.append(dp)
        np.savez(root, np.array(cleans), np.array(dps))
    cleans = np.array(cleans)
    dps = np.array(dps)
    return cleans, dps

def plot_axis(label, times, dy, filt, title=''):
    plt.figure(figsize=(6, 4))
    plt.clf()

    plot_cxctime(times, dy, 'r.', markersize=0.3)
    if len(np.flatnonzero(filt)) > 0:
        plot_cxctime(times[filt], dy[filt], 'k.', markersize=2)
    plt.ylabel('Delta %s (arcsec)' % label)
    plt.title(title)
    plt.ylim(-10, 10)
    plt.subplots_adjust(bottom=0.05, top=0.85)
    plt.tight_layout()
