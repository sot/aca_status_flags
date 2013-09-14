import numpy as np
import matplotlib.pyplot as plt

from kadi import events
import Ska.engarchive.fetch as fetch
from Ska.Matplotlib import plot_cxctime
from Quaternion import Quat, normalize
from quatutil import yagzag2radec
from Chandra.Time import DateTime


def plot_centroid_resids_by_flag(start, stop, slot, plot=False):
    """
    Plot centroid residuals for a start/stop interval with color
    coding to indicate readouts corresponding to a particular
    combination of DP, IR, MS, and SP.  The specified interval
    must be a stable Kalman dwell at one attitude.

    :param start: start time (any Chandra DateTime format)
    :param stop: stop time
    :param slot: ACA image slot
    :param save: save images as png files
    """

    pcad_msids = ['aoacfct',
                  'aoacisp',
                  'aoacidp',
                  'aoaciir',
                  'aoacims',
                  'aoacyan',
                  'aoaczan']
    slot_msids = [msid + "%s" % slot for msid in pcad_msids]

    msids = ['aoattqt1',
             'aoattqt2',
             'aoattqt3',
             'aoattqt4',
             ]

    msids.extend(slot_msids)

    print('Fetching telemetry from {} to {}'.format(start, stop))
    telems = fetch.MSIDset(msids, start, stop)
    telems.interpolate(dt=2.05, filter_bad=False)

    bads = np.zeros(len(telems.times), dtype=bool)
    for msid in telems:
        bads |= telems[msid].bads
    bads |= telems['aoacfct{}'.format(slot)].vals != 'TRAK'

    telems.bads = bads
    for msid in telems:
        telems[msid].bads = bads
    telems.filter_bad()

    vals = ([telems['aoattqt%d' % i].vals for i in range(1, 5)]
            + [telems['aoacyan{}'.format(slot)].vals / 3600.,
               telems['aoaczan{}'.format(slot)].vals / 3600.])

    print('Interpolating quaternions')
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
