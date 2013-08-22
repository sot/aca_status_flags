#!/usr/bin/env python

from __future__ import division

import cPickle as pickle
import os

import numpy as np

from mica import quaternion
from kadi import events
import Ska.engarchive.fetch as fetch
from Chandra.Time import DateTime
import pyyaks.logger
import Ska.Numpy

logger = pyyaks.logger.get_logger(format='%(asctime)s: %(message)s')

SLOT_MSIDS = ['aoacfct',
              'aoacmag',
              'aoacisp',
              'aoacidp',
              'aoaciir',
              'aoacims',
              'aoacyan',
              'aoaczan']
ATT_MSIDS = ['aoattqt1',
             'aoattqt2',
             'aoattqt3',
             'aoattqt4',
             'aokalstr',
             ]


def filter_bad(msid, bads):
    """
    Filter out any bad values from ``msid`` object.

    :param bads: Bad values mask
    """
    if np.any(bads):
        ok = ~bads
        for colname in msid.colnames:
            setattr(msid, colname, getattr(msid, colname)[ok])


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

    slot_msids = [msid + "%s" % slot for msid in SLOT_MSIDS for slot in slots]
    msids = slot_msids + ATT_MSIDS

    # Get telemetry
    logger.info('Fetching telemetry from {} to {}'.format(start.date[:17], stop.date[:17]))
    telems = fetch.MSIDset(msids, start, stop)
    start_interp = max(telems[msid].times[0] for msid in msids)
    stop_interp = min(telems[msid].times[-1] for msid in msids)
    logger.info('Interpolating at 2.05 sec intervals over {} to {}'
                .format(DateTime(start_interp).date[:17], DateTime(stop_interp).date[:17]))

    # Interpolate everything onto a common uniform time grid
    telems.interpolate(dt=2.05, start=start_interp, stop=stop_interp, filter_bad=False)

    # Select intervals within a kalman dwell with no tsc_moves or dumps
    events.dumps.interval_pad = (10, 500)
    events.tsc_moves.interval_pad = (10, 300)
    good_times = events.dwells & ~(events.tsc_moves | events.dumps)
    for msid in telems:
        telems[msid].select_intervals(good_times)

    # Create a bad filter for any samples with no attitude or n-kalman value
    att_bads = np.zeros(len(telems[ATT_MSIDS[0]]), dtype=bool)
    for msid in ATT_MSIDS:
        att_bads |= telems[msid].bads
    for msid in ATT_MSIDS:
        telems[msid].bads = att_bads
    logger.info('Attitude: found {} / {} bad values'
                .format(np.sum(att_bads), len(att_bads)))

    # Apply the att_bad filtering for each MSID invidivually (note that the
    # MSIDset.filter_bad() method isn't appropriate here)
    for msid in msids:
        filter_bad(telems[msid], att_bads)

    # Set global times to the times of first in MSIDset because they are all
    # the same at this point
    telems.times = telems[msids[0]].times

    # Create a bad filter for each slot based on the union of all relevant MSIDs
    for slot in slots:
        slot_msids = [msid + "%s" % slot for msid in SLOT_MSIDS]
        slot_bads = np.zeros(len(telems.times), dtype=bool)
        for msid in slot_msids:
            slot_bads |= telems[msid].bads

        # Only accept samples where the ACA reports tracking
        slot_bads |= telems['aoacfct{}'.format(slot)].vals != 'TRAK'

        for msid in slot_msids:
            telems[msid].bads = slot_bads
        logger.info('Slot {}: found {} / {} bad or not tracking values'
                    .format(slot, np.sum(slot_bads), len(slot_bads)))

    return telems, slots


def telems_to_struct(telems, slots):
    """
    Convert input MSIDset to an optimized data structure
    """
    out = {}
    time0 = telems.times[0]
    out['time0'] = time0
    out['bads'] = {}
    out['vals'] = {}
    out['slots'] = slots
    out['vals']['dyag'] = {}
    out['vals']['dzag'] = {}

    out['times'] = np.array(telems.times - time0, dtype=np.float32)

    for msid in ATT_MSIDS:
        out['vals'][msid] = telems[msid].vals

    # Set bads array for each slot based the first of the SLOT_MSIDS (they
    # all have the same bads array)
    for slot in slots:
        msid = SLOT_MSIDS[0] + str(slot)
        out['bads'][slot] = telems[msid].bads

    for slot_msid in SLOT_MSIDS:
        out['vals'][slot_msid] = {}
        for slot in slots:
            msid = slot_msid + str(slot)
            tlmsid = telems[msid]
            out['vals'][slot_msid][slot] = (tlmsid.vals if tlmsid.raw_vals is None
                                            else tlmsid.raw_vals)

    for slot in slots:
        dyag, dzag = calc_delta_centroids(telems, slot)
        out['vals']['dyag'][slot] = np.array(dyag, dtype=np.float32)
        out['vals']['dzag'][slot] = np.array(dzag, dtype=np.float32)
        # Force large bad dy/z values at bad telemetry so any downstream
        # filtering mistakes will be obvious
        if np.any(out['bads'][slot]):
            for axis in ('dyag', 'dzag'):
                out['vals'][axis][slot][out['bads'][slot]] = -99999.0

    return out


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


def calc_delta_centroids(telems, slot, dt=3.0):
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


def get_obsid(obsid, dt=3.0):
    """
    Get an obsid
    """
    obsids = events.obsids.filter(obsid__exact=obsid)
    if len(obsids) == 0:
        raise ValueError('No obsid={} in kadi database'.format(obsid))

    dwells = events.dwells.filter(obsids[0].start, obsids[0].stop)
    obsid_dwells = [dwell for dwell in dwells if dwell.start > obsids[0].start]
    logger.info('Using obsid dwell(s): {}'
                .format(','.join(str(dwell) for dwell in obsid_dwells)))

    telems, slots = get_archive_data(obsid_dwells[0].start, obsid_dwells[-1].stop)
    out = telems_to_struct(telems, slots)
    out['obsid'] = obsid

    return out


def process_obsids(start, stop):
    obsids = events.obsids.filter(start, stop)
    for obsid_event in obsids:
        obsid = obsid_event.obsid
        filename = os.path.join('data', str(obsid) + '.pkl')
        if os.path.exists(filename):
            logger.info('Skipping obsid {}, file exists'.format(obsid))
            continue
        else:
            logger.info('**********************************')
            logger.info('Processing obsid {}'.format(obsid))
            logger.info('**********************************')

        try:
            dat, telems = get_obsid(obsid)
        except Exception as err:
            logger.error('ERROR in obsid {}: {}\n'.format(obsid, err))
            open(filename + '.ERR', 'w')
        else:
            pickle.dump(dat, open(filename, 'w'), protocol=-1)
            logger.info('Success for {}\n'.format(obsid))


def main():
    import argparse

    now = DateTime()
    parser = argparse.ArgumentParser(description='Process status flag data')
    parser.add_argument('--start',
                        type=str,
                        default=(now - 10).date,
                        help='Start date')
    parser.add_argument('--stop',
                        type=str,
                        default=now.date,
                        help='Stop date (default=Now)')
    args = parser.parse_args()

    start = DateTime(args.start)
    stop = DateTime(args.stop)
    process_obsids(start, stop)


if __name__ == '__main__':
    # main()
    pass
